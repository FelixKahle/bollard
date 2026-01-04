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
///
/// Provides both checked and unchecked decoding methods.
pub trait Decoder<T, E>
where
    T: SolverNumeric,
    E: AssignmentEvaluator<T>,
{
    /// Returns the name of the decoder.
    fn name(&self) -> &str;

    /// Initializes the decoder for the given model.
    fn initialize(&mut self, model: &Model<T>);

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
        evaluator: &E,
    ) -> bool;

    /// Decodes the given priority queue into a schedule.
    ///
    /// The decoder will fill the provided `state` with berth assignments and start times
    /// based on the vessel order in `queue`. It returns `true` if a valid schedule
    /// was constructed, or `false` if any vessel could not be assigned.
    /// Note that if `false` is returned, the contents of `state` may be incomplete or invalid.
    ///
    /// # Safety
    ///
    /// The caller must guarantee all preconditions below. Violating any of them results
    /// in immediate undefined behavior:
    ///
    /// - Model/queue/state coherence:
    ///   * Every `VesselIndex` in `queue` refers to a valid vessel in `model`.
    ///   * `state` has been initialized for all vessels/berths referenced by `queue` and `model`.
    ///   * Any time windows, capacities, and index bounds required by `model` are satisfied.
    /// - Evaluator soundness:
    ///   * The `AssignmentEvaluator` used by this decoder can be invoked without additional
    ///     runtime checks (no panics, no UB) for all inputs produced during decoding.
    /// - Decoder-specific invariants:
    ///   * All invariants documented by this decoder (e.g., monotonic berth free times, non-overlapping
    ///     assignments, deadline feasibility) are already established or will be upheld by the caller.
    ///
    /// In debug builds, some of these conditions may be asserted; in release builds they are not checked.
    unsafe fn decode_unchecked(
        &mut self,
        model: &Model<T>,
        queue: &VesselPriorityQueue,
        state: &mut Schedule<T>,
        evaluator: &E,
    ) -> bool;
}

impl<T, E> std::fmt::Debug for dyn Decoder<T, E>
where
    T: SolverNumeric,
    E: AssignmentEvaluator<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Decoder({})", self.name())
    }
}

impl<T, E> std::fmt::Display for dyn Decoder<T, E>
where
    T: SolverNumeric,
    E: AssignmentEvaluator<T>,
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
#[derive(Debug, Clone)]
pub struct GreedyDecoder<T, E>
where
    T: SolverNumeric,
    E: AssignmentEvaluator<T>,
{
    berth_free_times: Vec<T>, // len = num_berths
    _phantom: std::marker::PhantomData<E>,
}

impl<T, E> Default for GreedyDecoder<T, E>
where
    T: SolverNumeric,
    E: AssignmentEvaluator<T>,
{
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<T, E> GreedyDecoder<T, E>
where
    T: SolverNumeric,
    E: AssignmentEvaluator<T>,
{
    /// Creates a new `GreedyDecoder` with no pre-allocated berths.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            berth_free_times: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new `GreedyDecoder` with the specified number of berths and evaluator.
    #[inline(always)]
    pub fn preallocated(num_berths: usize) -> Self {
        Self {
            berth_free_times: vec![T::zero(); num_berths],
            _phantom: std::marker::PhantomData,
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
    /// In debug builds, this function will panic if `berth_index` is not within
    /// `0..self.berth_free_times.len()`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `berth_index` is within `0..self.berth_free_times.len()`.
    #[inline]
    unsafe fn find_earliest_start(
        &self,
        model: &Model<T>,
        berth_index: BerthIndex,
        vessel_arrival: T,
        duration: T,
    ) -> Option<T> {
        debug_assert!(
            berth_index.get() < self.berth_free_times.len(),
            "called `GreedyDecoder::find_earliest_start` with `berth_index` out of bounds: the len is {} but the index is {}",
            self.berth_free_times.len(),
            berth_index.get(),
        );

        let berth_free = unsafe { *self.berth_free_times.get_unchecked(berth_index.get()) };
        let min_start = if berth_free > vessel_arrival {
            berth_free
        } else {
            vessel_arrival
        };

        let intervals = unsafe { model.berth_opening_times_unchecked(berth_index) };
        // Start from the first interval with start >= min_start, but also check the
        // previous interval (if any). That prior interval may start < min_start yet still
        // cover min_start (i.e., end > min_start) and thus be feasible. If zero `saturating_sub`
        // prevents underflow.
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

impl<T, E> Decoder<T, E> for GreedyDecoder<T, E>
where
    T: SolverNumeric,
    E: AssignmentEvaluator<T>,
{
    fn name(&self) -> &str {
        "GreedyDecoder"
    }

    fn initialize(&mut self, model: &Model<T>) {
        let num_berths = model.num_berths();

        // Make sure the berth free times vector is the correct size.
        if self.berth_free_times.len() != num_berths {
            self.berth_free_times.resize(num_berths, T::zero());
        }
        self.berth_free_times.fill(T::zero());
    }

    fn decode(
        &mut self,
        model: &Model<T>,
        queue: &VesselPriorityQueue,
        state: &mut Schedule<T>,
        evaluator: &E,
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

            let arrival = model.vessel_arrival_time(vessel_idx);

            let mut best_berth = None;
            let mut best_start = T::max_value();
            let mut best_finish = T::max_value();
            let mut best_score = T::max_value();
            let mut best_delta = T::zero();

            for berth_index in 0..num_berths {
                assert!(
                    berth_index < num_berths,
                    "encountered `berth_index` out of bounds in `GreedyDecoder::decode`: the len is {} but the index is {}",
                    num_berths,
                    berth_index,
                );

                let berth = BerthIndex::new(berth_index);

                let pt_opt = model.vessel_processing_time(vessel_idx, berth);
                if pt_opt.is_none() {
                    continue;
                }
                let duration = pt_opt.unwrap_unchecked();

                if let Some(start_time) =
                    // SAFETY: berth_index is in bounds due to loop condition and assert above.
                    unsafe { self.find_earliest_start(model, berth, arrival, duration) }
                {
                    let eval_opt = evaluator.evaluate(model, vessel_idx, berth, start_time);

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

    unsafe fn decode_unchecked(
        &mut self,
        model: &Model<T>,
        queue: &VesselPriorityQueue,
        state: &mut Schedule<T>,
        evaluator: &E,
    ) -> bool {
        let num_berths = model.num_berths();
        let num_vessels = model.num_vessels();
        let (berths_out, starts_out) = state.as_mut_slices();

        debug_assert!(
            berths_out.len() == num_vessels,
            "called `GreedyDecoder::decode_unchecked` with `berths_out` length mismatch: expected {} vessels but got {}; num_vessels={}, starts_out_len={}",
            num_vessels,
            berths_out.len(),
            num_vessels,
            starts_out.len(),
        );

        debug_assert!(
            starts_out.len() == num_vessels,
            "called `GreedyDecoder::decode_unchecked` with `starts_out` length mismatch: expected {} vessels but got {}; num_vessels={}, berths_out_len={}",
            num_vessels,
            starts_out.len(),
            num_vessels,
            berths_out.len(),
        );

        debug_assert!(
            queue.len() == num_vessels,
            "called `GreedyDecoder::decode_unchecked` with `queue` length mismatch: expected {} vessels but got {}",
            num_vessels,
            queue.len(),
        );

        debug_assert!(
            self.berth_free_times.len() == num_berths,
            "called `GreedyDecoder::decode_unchecked` with `berth_free_times` length mismatch: expected {} berths but got {}",
            num_berths,
            self.berth_free_times.len(),
        );

        self.berth_free_times.fill(T::zero());
        let mut total_objective = T::zero();

        for &vessel_idx in queue.iter() {
            debug_assert!(
                vessel_idx.get() < num_vessels,
                "encountered `vessel_idx` out of bounds in `GreedyDecoder::decode_unchecked`: the len is {} but the index is {}",
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
                    "encountered `berth_index` out of bounds in `GreedyDecoder::decode_unchecked`: the len is {} but the index is {}",
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
                        evaluator.evaluate_unchecked(model, vessel_idx, berth, start_time)
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
    use crate::eval::WeightedFlowTimeEvaluator;
    use crate::memory::SearchMemory;
    use bollard_core::math::interval::ClosedOpenInterval;
    use bollard_model::{
        index::{BerthIndex, VesselIndex},
        model::ModelBuilder,
        solution::Solution,
        time::ProcessingTime,
    };
    use num_traits::Zero;

    #[inline]
    fn vi(i: usize) -> VesselIndex {
        VesselIndex::new(i)
    }
    #[inline]
    fn bi(i: usize) -> BerthIndex {
        BerthIndex::new(i)
    }

    #[inline]
    fn new_memory_for_model(model: &bollard_model::model::Model<i64>) -> SearchMemory<i64> {
        let num_vessels = model.num_vessels();
        let mut mem = SearchMemory::<i64>::preallocated(num_vessels);
        let placeholder = Solution::new(
            i64::zero(),
            vec![bi(0); num_vessels],
            vec![i64::zero(); num_vessels],
        );
        mem.initialize(&placeholder);
        mem
    }

    #[inline]
    fn decode_and_accept<T, E>(
        decoder: &mut GreedyDecoder<T, E>,
        model: &bollard_model::model::Model<T>,
        queue: &crate::queue::VesselPriorityQueue,
        memory: &mut SearchMemory<T>,
        evaluator: &E,
    ) -> bool
    where
        T: SolverNumeric,
        E: AssignmentEvaluator<T>,
    {
        let ok = unsafe {
            decoder.decode_unchecked(model, queue, memory.candidate_schedule_mut(), evaluator)
        };
        if ok {
            memory.accept_current();
        }
        ok
    }

    fn build_model_simple() -> bollard_model::model::Model<i64> {
        let mut bldr = ModelBuilder::<i64>::new(2, 3);

        bldr.set_vessel_arrival_time(vi(0), 0)
            .set_vessel_arrival_time(vi(1), 0)
            .set_vessel_arrival_time(vi(2), 0);

        bldr.set_vessel_latest_departure_time(vi(0), 1_000)
            .set_vessel_latest_departure_time(vi(1), 1_000)
            .set_vessel_latest_departure_time(vi(2), 1_000);

        bldr.set_vessel_weight(vi(0), 1)
            .set_vessel_weight(vi(1), 1)
            .set_vessel_weight(vi(2), 1);

        bldr.set_vessel_processing_time(vi(0), bi(0), ProcessingTime::some(5))
            .set_vessel_processing_time(vi(0), bi(1), ProcessingTime::some(10));
        bldr.set_vessel_processing_time(vi(1), bi(0), ProcessingTime::some(7));
        bldr.set_vessel_processing_time(vi(2), bi(1), ProcessingTime::some(3));

        bldr.build()
    }

    #[test]
    fn test_decode_success_basic_assignment_and_objective() {
        let model = build_model_simple();

        let mut queue = crate::queue::VesselPriorityQueue::new();
        queue.extend([vi(0), vi(1), vi(2)]);

        let mut memory = new_memory_for_model(&model);
        let evaluator = WeightedFlowTimeEvaluator::<i64>::default();
        let mut decoder = GreedyDecoder::<i64, _>::preallocated(model.num_berths());

        assert!(decode_and_accept(
            &mut decoder,
            &model,
            &queue,
            &mut memory,
            &evaluator
        ));

        let schedule = memory.current_schedule();

        assert_eq!(schedule.berth_for_vessel(vi(0)), bi(0));
        assert_eq!(schedule.start_time_for_vessel(vi(0)), 0);

        assert_eq!(schedule.berth_for_vessel(vi(1)), bi(0));
        assert_eq!(schedule.start_time_for_vessel(vi(1)), 5);

        assert_eq!(schedule.berth_for_vessel(vi(2)), bi(1));
        assert_eq!(schedule.start_time_for_vessel(vi(2)), 0);

        assert_eq!(schedule.objective_value(), 20);
    }

    #[test]
    fn test_decode_infeasible_returns_false() {
        let bldr = ModelBuilder::<i64>::new(2, 1);
        let model = bldr.build();

        let mut queue = crate::queue::VesselPriorityQueue::new();
        queue.extend([vi(0)]);

        let mut memory = new_memory_for_model(&model);
        let evaluator = WeightedFlowTimeEvaluator::<i64>::default();
        let mut decoder = GreedyDecoder::<i64, _>::preallocated(model.num_berths());

        let ok = unsafe {
            decoder.decode_unchecked(&model, &queue, memory.candidate_schedule_mut(), &evaluator)
        };
        assert!(
            !ok,
            "decode must return false when no feasible berth exists"
        );
    }

    #[test]
    fn test_tie_breaking_score_then_finish_then_start() {
        let mut bldr = ModelBuilder::<i64>::new(2, 1);

        bldr.set_vessel_latest_departure_time(vi(0), 1000)
            .set_vessel_weight(vi(0), 1);

        bldr.add_berth_closing_time(bi(1), ClosedOpenInterval::new(0, 5));

        bldr.set_vessel_processing_time(vi(0), bi(0), ProcessingTime::some(10))
            .set_vessel_processing_time(vi(0), bi(1), ProcessingTime::some(5));

        let model = bldr.build();

        let mut queue = crate::queue::VesselPriorityQueue::new();
        queue.extend([vi(0)]);

        let mut memory = new_memory_for_model(&model);
        let evaluator = WeightedFlowTimeEvaluator::<i64>::default();
        let mut decoder = GreedyDecoder::<i64, _>::preallocated(model.num_berths());

        assert!(decode_and_accept(
            &mut decoder,
            &model,
            &queue,
            &mut memory,
            &evaluator
        ));

        let schedule = memory.current_schedule();
        assert_eq!(schedule.berth_for_vessel(vi(0)), bi(0));
        assert_eq!(schedule.start_time_for_vessel(vi(0)), 0);
        assert_eq!(schedule.objective_value(), 10);

        let mut bldr2 = ModelBuilder::<i64>::new(2, 2);
        bldr2
            .set_vessel_latest_departure_time(vi(0), 1000)
            .set_vessel_latest_departure_time(vi(1), 1000)
            .set_vessel_weight(vi(0), 1)
            .set_vessel_weight(vi(1), 1);
        bldr2.add_berth_closing_time(bi(1), ClosedOpenInterval::new(0, 5));
        bldr2
            .set_vessel_processing_time(vi(0), bi(0), ProcessingTime::some(10))
            .set_vessel_processing_time(vi(0), bi(1), ProcessingTime::some(5));
        bldr2.set_vessel_processing_time(vi(1), bi(0), ProcessingTime::some(50));

        let model2 = bldr2.build();

        let mut queue2 = crate::queue::VesselPriorityQueue::new();
        queue2.extend([vi(1), vi(0)]);

        let mut memory2 = new_memory_for_model(&model2);
        let evaluator2 = WeightedFlowTimeEvaluator::<i64>::default();
        let mut decoder2 = GreedyDecoder::<i64, _>::preallocated(model2.num_berths());

        assert!(decode_and_accept(
            &mut decoder2,
            &model2,
            &queue2,
            &mut memory2,
            &evaluator2
        ));

        let schedule2 = memory2.current_schedule();

        assert_eq!(schedule2.berth_for_vessel(vi(1)), bi(0));
        assert_eq!(schedule2.start_time_for_vessel(vi(1)), 0);

        assert_eq!(schedule2.berth_for_vessel(vi(0)), bi(1));
        assert_eq!(schedule2.start_time_for_vessel(vi(0)), 5);
    }

    #[test]
    fn test_respects_opening_times_and_find_earliest_start() {
        let mut bldr = ModelBuilder::<i64>::new(1, 2);
        bldr.add_berth_closing_time(bi(0), ClosedOpenInterval::new(0, 10));
        bldr.set_vessel_processing_time(vi(0), bi(0), ProcessingTime::some(3))
            .set_vessel_processing_time(vi(1), bi(0), ProcessingTime::some(4));
        bldr.set_vessel_arrival_time(vi(0), 0)
            .set_vessel_arrival_time(vi(1), 0);
        bldr.set_vessel_latest_departure_time(vi(0), 100)
            .set_vessel_latest_departure_time(vi(1), 100);

        let model = bldr.build();

        let mut queue = crate::queue::VesselPriorityQueue::new();
        queue.extend([vi(0), vi(1)]);

        let mut memory = new_memory_for_model(&model);
        let evaluator = WeightedFlowTimeEvaluator::<i64>::default();
        let mut decoder = GreedyDecoder::<i64, _>::preallocated(model.num_berths());

        assert!(decode_and_accept(
            &mut decoder,
            &model,
            &queue,
            &mut memory,
            &evaluator
        ));

        let schedule = memory.current_schedule();

        assert_eq!(schedule.berth_for_vessel(vi(0)), bi(0));
        assert_eq!(schedule.start_time_for_vessel(vi(0)), 10);

        assert_eq!(schedule.berth_for_vessel(vi(1)), bi(0));
        assert_eq!(schedule.start_time_for_vessel(vi(1)), 13);

        assert_eq!(schedule.objective_value(), 30);
    }

    #[test]
    fn test_berth_free_times_reset_between_decode_calls() {
        let model = build_model_simple();

        let mut queue = crate::queue::VesselPriorityQueue::new();
        queue.extend([vi(0), vi(1), vi(2)]);

        let mut memory = new_memory_for_model(&model);
        let evaluator = WeightedFlowTimeEvaluator::<i64>::default();
        let mut decoder = GreedyDecoder::<i64, _>::preallocated(model.num_berths());

        assert!(decode_and_accept(
            &mut decoder,
            &model,
            &queue,
            &mut memory,
            &evaluator
        ));
        let schedule = memory.current_schedule();

        let mut memory2 = new_memory_for_model(&model);
        assert!(decode_and_accept(
            &mut decoder,
            &model,
            &queue,
            &mut memory2,
            &evaluator
        ));
        let schedule2 = memory2.current_schedule();

        assert_eq!(schedule2.berths(), schedule.berths());
        assert_eq!(schedule2.start_times(), schedule.start_times());
        assert_eq!(schedule2.objective_value(), schedule.objective_value());
    }

    #[test]
    fn test_decode_fails_on_deadline_violation() {
        let mut bldr = ModelBuilder::<i64>::new(1, 1);
        bldr.set_vessel_arrival_time(vi(0), 0)
            .set_vessel_weight(vi(0), 1)
            .set_vessel_latest_departure_time(vi(0), 5);
        bldr.set_vessel_processing_time(vi(0), bi(0), ProcessingTime::some(10));

        let model = bldr.build();

        let mut queue = crate::queue::VesselPriorityQueue::new();
        queue.push(vi(0));

        let mut memory = new_memory_for_model(&model);
        let evaluator = WeightedFlowTimeEvaluator::<i64>::default();
        let mut decoder = GreedyDecoder::<i64, _>::preallocated(model.num_berths());

        let ok = unsafe {
            decoder.decode_unchecked(&model, &queue, memory.candidate_schedule_mut(), &evaluator)
        };
        assert!(!ok, "Decoder accepted a vessel that violates the deadline");
    }
}
