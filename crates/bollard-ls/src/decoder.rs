// Copyright (c) 2025 Felix Kahle.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! Decoding vessel orderings into concrete schedules for local search.
//!
//! This module turns a `VesselPriorityQueue` into a `Schedule` by assigning each
//! vessel to a berth and start time that respect opening windows and processing
//! constraints. The decoder queries an objective evaluator to compare feasible
//! candidates and accumulates the resulting objective value, keeping the decision
//! metric separate from the reported cost when heuristics are in play.
//!
//! The `GreedyDecoder` maintains per‑berth free times and searches for the earliest
//! feasible start within the model’s opening intervals, preferring lower scores,
//! then earlier finishes, then earlier starts to break ties. It is designed for
//! tight inner loops: bounds are asserted in debug builds, internal buffers are
//! reused, and unchecked evaluators can be called once invariants are established.
//!
//! The result is a compact, deterministic routine that decodes a given vessel
//! ordering into a complete schedule suitable for iterative improvement in local
//! search.

use crate::{eval::AssignmentEvaluator, memory::Schedule, queue::VesselPriorityQueue};
use bollard_model::{index::BerthIndex, model::Model};
use bollard_search::num::SolverNumeric;

/// A decoder that transforms a priority queue into a schedule.
pub trait Decoder<T>
where
    T: SolverNumeric,
{
    /// Returns the name of the decoder.
    fn name(&self) -> &str;

    /// Decodes the given priority queue into a schedule.
    ///
    /// The decoder will fill the provided `state` with berth assignments and start times
    /// based on the vessel order in `queue`. It returns `true` if a valid schedule
    /// was constructed, or `false` if any vessel could not be assigned.
    /// Note that if `false` is returned, the contents of `state` may be incomplete or invalid.
    fn decode(
        &mut self,
        model: &Model<T>,
        queue: &VesselPriorityQueue,
        state: &mut Schedule<T>,
    ) -> bool;
}

impl<T> std::fmt::Debug for dyn Decoder<T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Decoder({})", self.name())
    }
}

impl<T> std::fmt::Display for dyn Decoder<T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Decoder({})", self.name())
    }
}

/// Greedy local-search decoder that maps a vessel priority queue to a schedule.
///
/// The decoder tracks per‑berth free times and, for each vessel in queue order,
/// searches the model’s opening intervals for the earliest feasible start on
/// each admissible berth. It consults the provided `AssignmentEvaluator` to
/// compare candidates and applies a deterministic tie‑break (lower score, then
/// earlier finish, then earlier start). Internal buffers are reused between
/// calls to minimize overhead in tight loops.
#[derive(Debug, Clone, Default)]
pub struct GreedyDecoder<T, E>
where
    T: SolverNumeric,
    E: AssignmentEvaluator<T>,
{
    berth_free_times: Vec<T>,
    evaluator: E,
}

impl<T, E> GreedyDecoder<T, E>
where
    T: SolverNumeric,
    E: AssignmentEvaluator<T>,
{
    /// Creates a new `GreedyDecoder` with the specified number of berths and evaluator.
    #[inline(always)]
    pub fn new(num_berths: usize, evaluator: E) -> Self {
        Self {
            berth_free_times: vec![T::zero(); num_berths],
            evaluator,
        }
    }

    /// Finds the earliest feasible start time for a vessel on a given berth.
    ///
    /// Uses a binary search to locate the relevant opening intervals
    /// and iterates through them to find the first one that can accommodate
    /// the vessel's processing duration after its arrival and the berth's free time.
    ///
    /// # Panics
    ///
    /// In debug builds, this function will panic if `berth_idx` is not within
    /// `0..self.berth_free_times.len()`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `berth_idx` is within `0..self.berth_free_times.len()`.
    #[inline]
    unsafe fn find_earliest_start(
        &self,
        model: &Model<T>,
        berth_idx: BerthIndex,
        vessel_arrival: T,
        duration: T,
    ) -> Option<T> {
        debug_assert!(
            berth_idx.get() < self.berth_free_times.len(),
            "called `GreedyDecoder::find_earliest_start` with `berth_idx` out of bounds: the len is {} but the index is {}",
            self.berth_free_times.len(),
            berth_idx.get(),
        );

        let berth_free = unsafe { *self.berth_free_times.get_unchecked(berth_idx.get()) };
        let min_start = if berth_free > vessel_arrival {
            berth_free
        } else {
            vessel_arrival
        };

        let intervals = unsafe { model.berth_opening_times_unchecked(berth_idx) };
        let search_start_index =
            bollard_core::algorithm::lower_bound_start(intervals, min_start).saturating_sub(1);

        for interval in &intervals[search_start_index..] {
            let open_end = interval.end();

            if open_end <= min_start {
                continue;
            }

            let open_start = interval.start();
            let actual_start = if open_start > min_start {
                open_start
            } else {
                min_start
            };

            let finish = actual_start + duration;
            if finish <= open_end {
                return Some(actual_start);
            }
        }

        None
    }
}

/// Determines if the candidate assignment is preferred over the best known assignment
/// based on the multi-criteria decision logic:
/// 1. Lower score is better.
/// 2. If scores are equal, earlier finish time is better.
/// 3. If finish times are equal, earlier start time is better.
#[inline(always)]
fn should_prefer<T: SolverNumeric>(
    candidate_score: T,
    candidate_finish: T,
    candidate_start: T,
    best_score: T,
    best_finish: T,
    best_start: T,
) -> bool {
    candidate_score < best_score
        || (candidate_score == best_score && candidate_finish < best_finish)
        || (candidate_score == best_score
            && candidate_finish == best_finish
            && candidate_start < best_start)
}

impl<T, E> Decoder<T> for GreedyDecoder<T, E>
where
    T: SolverNumeric,
    E: AssignmentEvaluator<T>,
{
    fn name(&self) -> &str {
        "GreedyDecoder"
    }

    fn decode(
        &mut self,
        model: &Model<T>,
        queue: &VesselPriorityQueue,
        state: &mut Schedule<T>,
    ) -> bool {
        let num_berths = model.num_berths();
        let num_vessels = model.num_vessels();
        let (berths_out, starts_out) = state.as_mut_slices();

        debug_assert!(
            berths_out.len() == num_vessels,
            "called `GreedyDecoder::decode` with `berths_out` length mismatch: expected {} vessels but got {}; num_vessels={}, starts_out_len={}",
            num_vessels,
            berths_out.len(),
            num_vessels,
            starts_out.len(),
        );

        debug_assert!(
            starts_out.len() == num_vessels,
            "called `GreedyDecoder::decode` with `starts_out` length mismatch: expected {} vessels but got {}; num_vessels={}, berths_out_len={}",
            num_vessels,
            starts_out.len(),
            num_vessels,
            berths_out.len(),
        );

        debug_assert!(
            queue.len() == num_vessels,
            "called `GreedyDecoder::decode` with `queue` length mismatch: expected {} vessels but got {}",
            num_vessels,
            queue.len(),
        );

        debug_assert!(
            self.berth_free_times.len() == num_berths,
            "called `GreedyDecoder::decode` with `berth_free_times` length mismatch: expected {} berths but got {}",
            num_berths,
            self.berth_free_times.len(),
        );

        self.berth_free_times.fill(T::zero());
        let mut total_objective = T::zero();

        for &vessel_idx in queue.iter() {
            debug_assert!(
                vessel_idx.get() < num_vessels,
                "encountered `vessel_idx` out of bounds in `GreedyDecoder::decode`: the len is {} but the index is {}",
                num_vessels,
                vessel_idx.get(),
            );

            let arrival = unsafe { model.vessel_arrival_time_unchecked(vessel_idx) };

            let mut best_berth = None;
            let mut best_start = T::max_value();
            let mut best_finish = T::max_value();
            let mut best_score = T::max_value();
            let mut best_delta = T::zero();

            for berth_index in 0..num_berths {
                debug_assert!(
                    berth_index < num_berths,
                    "encountered `berth_index` out of bounds in `GreedyDecoder::decode`: the len is {} but the index is {}",
                    num_berths,
                    berth_index,
                );

                let berth = BerthIndex::new(berth_index);

                let pt_opt = unsafe { model.vessel_processing_time_unchecked(vessel_idx, berth) };
                if pt_opt.is_none() {
                    continue;
                }
                let duration = pt_opt.unwrap_unchecked();

                if let Some(start_time) =
                    unsafe { self.find_earliest_start(model, berth, arrival, duration) }
                {
                    let eval_opt = unsafe {
                        self.evaluator
                            .evaluate_unchecked(model, vessel_idx, berth, start_time)
                    };

                    if let Some(eval) = eval_opt {
                        let finish_time = start_time + duration;

                        if should_prefer::<T>(
                            eval.score,
                            finish_time,
                            start_time,
                            best_score,
                            best_finish,
                            best_start,
                        ) {
                            best_finish = finish_time;
                            best_start = start_time;

                            best_score = eval.score;
                            best_delta = eval.objective_delta;
                            best_berth = Some(berth);
                        }
                    }
                }
            }

            if let Some(berth) = best_berth {
                let vessel_index_usize = vessel_idx.get();
                berths_out[vessel_index_usize] = berth;
                starts_out[vessel_index_usize] = best_start;
                self.berth_free_times[berth.get()] = best_finish;

                total_objective = total_objective + best_delta;
            } else {
                return false;
            }
        }

        state.set_objective_value(total_objective);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{eval::WeightedFlowTimeEvaluator, memory::Schedule};
    use bollard_core::math::interval::ClosedOpenInterval;
    use bollard_model::{
        index::{BerthIndex, VesselIndex},
        model::ModelBuilder,
        time::ProcessingTime,
    };

    type Num = i64;

    #[inline]
    fn vi(i: usize) -> VesselIndex {
        VesselIndex::new(i)
    }
    #[inline]
    fn bi(i: usize) -> BerthIndex {
        BerthIndex::new(i)
    }

    fn build_model_simple() -> bollard_model::model::Model<Num> {
        // Two berths, three vessels, open 24/7 by default.
        let mut bldr = ModelBuilder::<Num>::new(2, 3);

        // Arrival times
        bldr.set_vessel_arrival_time(vi(0), 0)
            .set_vessel_arrival_time(vi(1), 0)
            .set_vessel_arrival_time(vi(2), 0);

        // Latest departures far in the future
        bldr.set_vessel_latest_departure_time(vi(0), 1_000)
            .set_vessel_latest_departure_time(vi(1), 1_000)
            .set_vessel_latest_departure_time(vi(2), 1_000);

        // Equal weights
        bldr.set_vessel_weight(vi(0), 1)
            .set_vessel_weight(vi(1), 1)
            .set_vessel_weight(vi(2), 1);

        // Processing times:
        // Vessel 0 can use berth 0 for 5, berth 1 for 10.
        bldr.set_vessel_processing_time(vi(0), bi(0), ProcessingTime::some(5))
            .set_vessel_processing_time(vi(0), bi(1), ProcessingTime::some(10));

        // Vessel 1 can use berth 0 for 7 only.
        bldr.set_vessel_processing_time(vi(1), bi(0), ProcessingTime::some(7));

        // Vessel 2 can use berth 1 for 3 only.
        bldr.set_vessel_processing_time(vi(2), bi(1), ProcessingTime::some(3));

        bldr.build()
    }

    #[test]
    fn test_decode_success_basic_assignment_and_objective() {
        let model = build_model_simple();
        // Priority queue decides visiting order; here [0,1,2].
        let mut queue = crate::queue::VesselPriorityQueue::new();
        queue.extend([vi(0), vi(1), vi(2)]);

        // Prepare state buffers sized to num_vessels.
        let mut schedule = Schedule::<Num>::new(
            0,
            vec![bi(0); model.num_vessels()],
            vec![0; model.num_vessels()],
        );

        // Greedy decoder with WeightedFlowTimeEvaluator (score == objective_delta).
        let evaluator = WeightedFlowTimeEvaluator::<Num>::default();
        let mut decoder = GreedyDecoder::<Num, _>::new(model.num_berths(), evaluator);

        let ok = decoder.decode(&model, &queue, &mut schedule);
        assert!(ok, "decode should succeed with simple feasible model");

        // Validate assignments match earliest feasible selections:
        // Vessel 0: both berths feasible; prefers smaller score (completion*weight).
        // With start 0, durations: b0=5 => finish=5, score=5; b1=10 => finish=10, score=10 => pick b0.
        assert_eq!(schedule.berth_for_vessel(vi(0)), bi(0));
        assert_eq!(schedule.start_time_for_vessel(vi(0)), 0);

        // Vessel 1: only berth 0; berth 0 free time is now 5 (from v0). Arrival 0 -> min_start=5; duration 7 -> finish=12 within deadline.
        assert_eq!(schedule.berth_for_vessel(vi(1)), bi(0));
        assert_eq!(schedule.start_time_for_vessel(vi(1)), 5);

        // Vessel 2: only berth 1; that berth has been unused, free time 0; start 0; duration 3.
        assert_eq!(schedule.berth_for_vessel(vi(2)), bi(1));
        assert_eq!(schedule.start_time_for_vessel(vi(2)), 0);

        // Objective = sum of completion times * weight:
        // v0: finish=5, w=1 -> 5
        // v1: finish=12, w=1 -> 12
        // v2: finish=3, w=1 -> 3
        // total = 20
        assert_eq!(schedule.objective_value(), 20);
    }

    #[test]
    fn test_decode_infeasible_returns_false() {
        // Model with one vessel that cannot dock anywhere (all processing times None).
        let bldr = ModelBuilder::<Num>::new(2, 1);
        // Arrival and deadlines default to feasible, but no processing times set => all None.
        let model = bldr.build();

        let mut queue = crate::queue::VesselPriorityQueue::new();
        queue.extend([vi(0)]);

        let mut schedule = Schedule::<Num>::new(
            0,
            vec![bi(0); model.num_vessels()],
            vec![0; model.num_vessels()],
        );
        let evaluator = WeightedFlowTimeEvaluator::<Num>::default();
        let mut decoder = GreedyDecoder::<Num, _>::new(model.num_berths(), evaluator);

        let ok = decoder.decode(&model, &queue, &mut schedule);
        assert!(
            !ok,
            "decode must return false when no feasible berth exists"
        );
    }

    #[test]
    fn test_tie_breaking_score_then_finish_then_start() {
        // Build a model to force equal score on two berths, but different finish and start.
        let mut bldr = ModelBuilder::<Num>::new(2, 1);

        // Vessel 0: arrival 0, weight 1, deadline large
        bldr.set_vessel_latest_departure_time(vi(0), 1000)
            .set_vessel_weight(vi(0), 1);

        // Opening times:
        // Berth 0 open [0, 1000)
        // Berth 1 open [5, 1000) (delayed opening causes later start if berth_free < 5)
        // We model this by closing [0,5) on berth 1.
        bldr.add_berth_closing_time(bi(1), ClosedOpenInterval::new(0, 5));

        // Processing times:
        // Choose durations so that completion time is equal across both berths if starting at min feasible.
        // b0: duration 10, starting at 0 => completion 10
        // b1: duration 5, starting at 5 => completion 10
        bldr.set_vessel_processing_time(vi(0), bi(0), ProcessingTime::some(10))
            .set_vessel_processing_time(vi(0), bi(1), ProcessingTime::some(5));

        let model = bldr.build();

        let mut queue = crate::queue::VesselPriorityQueue::new();
        queue.extend([vi(0)]);

        let mut schedule = Schedule::<Num>::new(
            0,
            vec![bi(0); model.num_vessels()],
            vec![0; model.num_vessels()],
        );
        let evaluator = WeightedFlowTimeEvaluator::<Num>::default();
        let mut decoder = GreedyDecoder::<Num, _>::new(model.num_berths(), evaluator);

        // First decode: scores are equal (10 vs 10).
        // Secondary tie-breaker prefers earliest finish: equal again.
        // Tertiary tie-breaker prefers earliest start: berth 0 start=0 vs berth 1 start=5 => choose berth 0.
        let ok = decoder.decode(&model, &queue, &mut schedule);
        assert!(ok);
        assert_eq!(schedule.berth_for_vessel(vi(0)), bi(0));
        assert_eq!(schedule.start_time_for_vessel(vi(0)), 0);
        assert_eq!(schedule.objective_value(), 10);

        // Now make berth 0 busy so its earliest finish becomes later, testing secondary tie-breaker.
        // We'll add a second vessel that occupies berth 0 before the first vessel is decoded.
        let mut bldr2 = ModelBuilder::<Num>::new(2, 2);
        bldr2
            .set_vessel_latest_departure_time(vi(0), 1000)
            .set_vessel_latest_departure_time(vi(1), 1000)
            .set_vessel_weight(vi(0), 1)
            .set_vessel_weight(vi(1), 1);

        // Same opening times as above: close [0,5) on berth 1
        bldr2.add_berth_closing_time(bi(1), ClosedOpenInterval::new(0, 5));

        // v0: the target with equal completion candidates as before
        bldr2
            .set_vessel_processing_time(vi(0), bi(0), ProcessingTime::some(10))
            .set_vessel_processing_time(vi(0), bi(1), ProcessingTime::some(5));

        // v1: occupies berth 0 first with long duration to push finish later.
        bldr2.set_vessel_processing_time(vi(1), bi(0), ProcessingTime::some(50));

        let model2 = bldr2.build();

        // Queue: decode v1 first, then v0
        let mut queue2 = crate::queue::VesselPriorityQueue::new();
        queue2.extend([vi(1), vi(0)]);

        let mut schedule2 = Schedule::<Num>::new(
            0,
            vec![bi(0); model2.num_vessels()],
            vec![0; model2.num_vessels()],
        );
        let evaluator2 = WeightedFlowTimeEvaluator::<Num>::default();
        let mut decoder2 = GreedyDecoder::<Num, _>::new(model2.num_berths(), evaluator2);

        let ok2 = decoder2.decode(&model2, &queue2, &mut schedule2);
        assert!(ok2);

        // v1 goes to berth 0 at start 0, finish 50
        assert_eq!(schedule2.berth_for_vessel(vi(1)), bi(0));
        assert_eq!(schedule2.start_time_for_vessel(vi(1)), 0);

        // For v0:
        // - Berth 0 earliest start is 50 due to berth_free_times -> completion 60.
        // - Berth 1 earliest start is 5 due to opening, completion 10.
        // Score difference now decides: 10 < 60 => pick berth 1.
        assert_eq!(schedule2.berth_for_vessel(vi(0)), bi(1));
        assert_eq!(schedule2.start_time_for_vessel(vi(0)), 5);
    }

    #[test]
    fn test_respects_opening_times_and_find_earliest_start() {
        // One berth with a gap in opening, two vessels arriving early.
        let mut bldr = ModelBuilder::<Num>::new(1, 2);
        // Close [0, 10) so the earliest possible start is 10.
        bldr.add_berth_closing_time(bi(0), ClosedOpenInterval::new(0, 10));
        // Both vessels can use the berth with different durations.
        bldr.set_vessel_processing_time(vi(0), bi(0), ProcessingTime::some(3))
            .set_vessel_processing_time(vi(1), bi(0), ProcessingTime::some(4));

        // Arrival times before opening window
        bldr.set_vessel_arrival_time(vi(0), 0)
            .set_vessel_arrival_time(vi(1), 0);

        // Deadlines sufficiently large
        bldr.set_vessel_latest_departure_time(vi(0), 100)
            .set_vessel_latest_departure_time(vi(1), 100);

        let model = bldr.build();

        let mut queue = crate::queue::VesselPriorityQueue::new();
        queue.extend([vi(0), vi(1)]);

        let mut schedule = Schedule::<Num>::new(
            0,
            vec![bi(0); model.num_vessels()],
            vec![0; model.num_vessels()],
        );
        let evaluator = WeightedFlowTimeEvaluator::<Num>::default();
        let mut decoder = GreedyDecoder::<Num, _>::new(model.num_berths(), evaluator);

        let ok = decoder.decode(&model, &queue, &mut schedule);
        assert!(ok);

        // v0: earliest feasible start is at opening start 10; duration 3 => finish 13
        assert_eq!(schedule.berth_for_vessel(vi(0)), bi(0));
        assert_eq!(schedule.start_time_for_vessel(vi(0)), 10);

        // v1: berth_free_times now 13; earliest feasible start is 13 within opening [10,MAX); duration 4 => finish 17
        assert_eq!(schedule.berth_for_vessel(vi(1)), bi(0));
        assert_eq!(schedule.start_time_for_vessel(vi(1)), 13);

        // Objective equals sum of completion times: 13 + 17 = 30
        assert_eq!(schedule.objective_value(), 30);
    }

    #[test]
    fn test_berth_free_times_reset_between_decode_calls() {
        // Ensure consecutive calls start with fresh berth_free_times = 0
        let model = build_model_simple();

        let mut queue = crate::queue::VesselPriorityQueue::new();
        queue.extend([vi(0), vi(1), vi(2)]);

        let mut schedule = Schedule::<Num>::new(
            0,
            vec![bi(0); model.num_vessels()],
            vec![0; model.num_vessels()],
        );
        let evaluator = WeightedFlowTimeEvaluator::<Num>::default();
        let mut decoder = GreedyDecoder::<Num, _>::new(model.num_berths(), evaluator);

        // First decode
        let ok1 = decoder.decode(&model, &queue, &mut schedule);
        assert!(ok1);

        // Second decode into a fresh schedule must not carry over previous berth_free_times.
        let mut schedule2 = Schedule::<Num>::new(
            0,
            vec![bi(0); model.num_vessels()],
            vec![0; model.num_vessels()],
        );
        let ok2 = decoder.decode(&model, &queue, &mut schedule2);
        assert!(ok2);

        // The same solution should be produced again for this deterministic case.
        assert_eq!(schedule2.berths(), schedule.berths());
        assert_eq!(schedule2.start_times(), schedule.start_times());
        assert_eq!(schedule2.objective_value(), schedule.objective_value());
    }

    #[test]
    fn test_decode_fails_on_deadline_violation() {
        let mut bldr = ModelBuilder::<Num>::new(1, 1);
        bldr.set_vessel_arrival_time(vi(0), 0)
            .set_vessel_weight(vi(0), 1)
            // Strict deadline at 5
            .set_vessel_latest_departure_time(vi(0), 5);

        // Process takes 10 (Finish 10 > Deadline 5)
        bldr.set_vessel_processing_time(vi(0), bi(0), ProcessingTime::some(10));

        let model = bldr.build();
        let mut queue = crate::queue::VesselPriorityQueue::new();
        queue.push(vi(0));

        let mut schedule = Schedule::<Num>::new(0, vec![bi(0)], vec![0]);
        let evaluator = WeightedFlowTimeEvaluator::<Num>::default();
        let mut decoder = GreedyDecoder::<Num, _>::new(model.num_berths(), evaluator);

        let ok = decoder.decode(&model, &queue, &mut schedule);

        // Should fail because Evaluator returns None
        assert!(!ok, "Decoder accepted a vessel that violates the deadline");
    }
}
