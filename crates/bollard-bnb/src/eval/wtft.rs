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

use crate::{eval::evaluator::ObjectiveEvaluator, state::SearchState};
use bollard_core::num::{constants::MinusOne, ops::saturating_arithmetic};
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use num_traits::{Bounded, FromPrimitive, PrimInt, Signed};

/// Computes the earliest finish time for a task of given duration
/// that can start no earlier than `earliest_start` on the specified berth,
/// considering the berth's opening time intervals.
/// Returns `None` if the task cannot be scheduled on the berth.
///
/// # Safety
///
/// The caller must ensure that `berth_index` is within bounds of `0..model.num_berths()`.
#[inline(always)]
unsafe fn earliest_finish_time_unchecked<T>(
    berth_index: BerthIndex,
    earliest_start: T,
    duration: T,
    model: &Model<T>,
) -> Option<T>
where
    T: PrimInt + Signed + saturating_arithmetic::SaturatingAddVal,
{
    debug_assert!(
        berth_index.get() < model.num_berths(),
        "called `wtft::earliest_finish_time_unchecked` with sorted berth out of bounds: the len is {} but the index is {}",
        model.num_berths(),
        berth_index.get()
    );

    let intervals = unsafe { model.berth_opening_times_unchecked(berth_index) };
    for interval in intervals {
        if interval.end() <= earliest_start {
            continue;
        }

        let actual_start = if earliest_start > interval.start() {
            earliest_start
        } else {
            interval.start()
        };

        let finish = actual_start.saturating_add(duration);
        if finish <= interval.end() {
            return Some(finish);
        }
    }

    None
}

/// Internal job representation for the single-machine workload bound.
#[derive(Clone, Copy, Debug)]
struct VesselData<T> {
    processing_time: T,
    weight: T,
}

impl<T> std::fmt::Display for VesselData<T>
where
    T: std::fmt::Display + PrimInt + Signed,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VesselData(processing: {}, weight: {})",
            self.processing_time, self.weight
        )
    }
}

/// Evaluator for the weighted flow time objective.
#[derive(Debug)]
pub struct WeightedFlowTimeEvaluator<T>
where
    T: PrimInt + Signed,
{
    scratch_berths: Vec<T>,
    scratch_vessels: Vec<VesselData<T>>,
}

impl<T> Default for WeightedFlowTimeEvaluator<T>
where
    T: PrimInt + Signed,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> WeightedFlowTimeEvaluator<T>
where
    T: PrimInt + Signed,
{
    /// Creates a new `WeightedFlowTimeEvaluator` with empty scratch buffers.
    #[inline]
    pub fn new() -> Self {
        Self {
            scratch_berths: Vec::new(),
            scratch_vessels: Vec::new(),
        }
    }

    /// Creates a new `WeightedFlowTimeEvaluator` with preallocated scratch buffers.
    #[inline]
    pub fn preallocated(capacity_berths: usize, capacity_vessels: usize) -> Self {
        Self {
            scratch_berths: Vec::with_capacity(capacity_berths),
            scratch_vessels: Vec::with_capacity(capacity_vessels),
        }
    }
}

impl<T> ObjectiveEvaluator<T> for WeightedFlowTimeEvaluator<T>
where
    T: PrimInt + Signed + Bounded,
{
    #[inline]
    fn name(&self) -> &str {
        "WeightedFlowTimeEvaluator"
    }

    #[inline]
    fn evaluate_vessel_assignment(
        &mut self,
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        berth_ready_time: T,
    ) -> Option<T>
    where
        T: PrimInt
            + Signed
            + Bounded
            + MinusOne
            + saturating_arithmetic::SaturatingAddVal
            + saturating_arithmetic::SaturatingMulVal,
    {
        let arrival_time = model.vessel_arrival_time(vessel_index);
        let latest_departure_deadline = model.vessel_latest_departure_time(vessel_index);
        let vessel_weight = model.vessel_weight(vessel_index);

        let processing_time_pt = model.vessel_processing_time(vessel_index, berth_index);
        if processing_time_pt.is_none() {
            return None;
        }
        let processing_time = processing_time_pt.unwrap_unchecked();

        let effective_start_time = if arrival_time > berth_ready_time {
            arrival_time
        } else {
            berth_ready_time
        };
        let completion_time = effective_start_time.saturating_add_val(processing_time);

        if completion_time > latest_departure_deadline {
            return None;
        }

        Some(completion_time.saturating_mul_val(vessel_weight))
    }

    unsafe fn evaluate_vessel_assignment_unchecked(
        &self,
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        berth_ready: T,
    ) -> Option<T>
    where
        T: MinusOne
            + saturating_arithmetic::SaturatingAddVal
            + saturating_arithmetic::SaturatingMulVal,
    {
        let arrival_time = unsafe { model.vessel_arrival_time_unchecked(vessel_index) };
        let weight = unsafe { model.vessel_weight_unchecked(vessel_index) };
        let opt_processing_time =
            unsafe { model.vessel_processing_time_unchecked(vessel_index, berth_index) };
        if opt_processing_time.is_none() {
            return None;
        }
        let processing_time = opt_processing_time.unwrap();

        let effective_start = if arrival_time > berth_ready {
            arrival_time
        } else {
            berth_ready
        };
        let finish_time = effective_start.saturating_add_val(processing_time);
        Some(finish_time.saturating_mul_val(weight))
    }

    fn lower_bound_estimate(&mut self, model: &Model<T>, state: &SearchState<T>) -> Option<T>
    where
        T: MinusOne
            + saturating_arithmetic::SaturatingAddVal
            + saturating_arithmetic::SaturatingMulVal
            + FromPrimitive,
    {
        let num_berths = model.num_berths();
        let num_vessels = model.num_vessels();

        if num_vessels == 0 {
            return Some(T::zero());
        }

        if num_berths == 0 {
            return if state.num_assigned_vessels() == num_vessels {
                Some(T::zero())
            } else {
                None
            };
        }

        // Prepare berth scratch buffer
        self.scratch_berths.clear();
        for b in 0..num_berths {
            self.scratch_berths
                .push(unsafe { state.berth_free_time_unchecked(BerthIndex::new(b)) });
        }

        // Prepare vessel scratch buffer
        self.scratch_vessels.clear();
        let mut lower_bound_independent = T::zero();
        let mut min_unassigned_arrival = T::max_value();

        for i in 0..num_vessels {
            let vessel_index = VesselIndex::new(i);

            if unsafe { state.is_vessel_assigned_unchecked(vessel_index) } {
                continue;
            }

            let arrival = unsafe { model.vessel_arrival_time_unchecked(vessel_index) };

            if arrival < min_unassigned_arrival {
                min_unassigned_arrival = arrival;
            }

            let weight = unsafe { model.vessel_weight_unchecked(vessel_index) };
            let deadline = unsafe { model.vessel_latest_departure_time_unchecked(vessel_index) };

            let mut best_finish_time = T::max_value();
            let mut found_feasible_berth = false;

            for (berth_index, current_free_time) in self.scratch_berths.iter().copied().enumerate()
            {
                let processing_time_opt = unsafe {
                    model.vessel_processing_time_unchecked(
                        vessel_index,
                        BerthIndex::new(berth_index),
                    )
                };

                if processing_time_opt.is_none() {
                    continue;
                }
                let processing_time = processing_time_opt.unwrap_unchecked();
                let tentative_start = arrival.max(current_free_time);

                let possible_finish = unsafe {
                    earliest_finish_time_unchecked(
                        BerthIndex::new(berth_index),
                        tentative_start,
                        processing_time,
                        model,
                    )
                };

                if let Some(finish) = possible_finish {
                    if finish > deadline {
                        continue;
                    }
                    if finish < best_finish_time {
                        best_finish_time = finish;
                        found_feasible_berth = true;
                    }
                }
            }

            if !found_feasible_berth {
                return None;
            }

            lower_bound_independent = lower_bound_independent
                .saturating_add_val(best_finish_time.saturating_mul_val(weight));

            let min_processing_time_opt =
                unsafe { model.vessel_shortest_processing_time_unchecked(vessel_index) };
            if let Some(min_p) = Option::<T>::from(min_processing_time_opt) {
                self.scratch_vessels.push(VesselData {
                    processing_time: min_p,
                    weight,
                });
            }
        }

        let lower_bound_workload = if self.scratch_vessels.is_empty() {
            T::zero()
        } else {
            self.scratch_vessels.sort_unstable_by(|a, b| {
                let lhs = a.processing_time.saturating_mul_val(b.weight);
                let rhs = b.processing_time.saturating_mul_val(a.weight);
                lhs.cmp(&rhs)
            });

            let min_berth_time = self
                .scratch_berths
                .iter()
                .copied()
                .min()
                .unwrap_or(T::zero());
            let start_time = min_berth_time.max(min_unassigned_arrival);
            let mut current_time = start_time;
            let mut total_weighted_completion = T::zero();

            for job in &self.scratch_vessels {
                current_time = current_time.saturating_add_val(job.processing_time);
                let cost = current_time.saturating_mul_val(job.weight);
                total_weighted_completion = total_weighted_completion.saturating_add_val(cost);
            }

            let num_berths_conv = T::from_usize(num_berths).unwrap_or(T::max_value());
            total_weighted_completion / num_berths_conv
        };

        let lower_bound = lower_bound_workload.max(lower_bound_independent);
        Some(lower_bound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::evaluator::ObjectiveEvaluator;
    use bollard_model::{model::ModelBuilder, time::ProcessingTime};

    type IntegerType = i64;

    fn build_model_waiting() -> Model<IntegerType> {
        let mut builder = ModelBuilder::<IntegerType>::new(2, 2);
        builder.set_vessel_arrival_time(VesselIndex::new(0), 5);
        builder.set_vessel_arrival_time(VesselIndex::new(1), 2);
        builder.set_vessel_weight(VesselIndex::new(0), 3);
        builder.set_vessel_weight(VesselIndex::new(1), 4);
        builder.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(0),
            ProcessingTime::some(7),
        );
        builder.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(1),
            ProcessingTime::some(3),
        );
        builder.set_vessel_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(0),
            ProcessingTime::none(),
        );
        builder.set_vessel_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(1),
            ProcessingTime::some(6),
        );
        builder.build()
    }

    #[test]
    fn test_name_and_display_debug() {
        let eval = WeightedFlowTimeEvaluator::<IntegerType>::new();
        assert_eq!(eval.name(), "WeightedFlowTimeEvaluator");

        let dbg = format!("{:?}", &eval as &dyn ObjectiveEvaluator<IntegerType>);
        let disp = format!("{}", &eval as &dyn ObjectiveEvaluator<IntegerType>);
        assert!(dbg.contains("ObjectiveEvaluator(WeightedFlowTimeEvaluator)"));
        assert!(disp.contains("ObjectiveEvaluator(WeightedFlowTimeEvaluator)"));
    }

    #[test]
    fn test_evaluate_respects_waiting_time() {
        let model = build_model_waiting();
        let mut eval = WeightedFlowTimeEvaluator::<IntegerType>::new();

        // Vessel 0, Berth 0: arrival=5, proc=7. Ready=4 -> Start=5 -> Finish=12. Cost=36
        let cost_a =
            eval.evaluate_vessel_assignment(&model, VesselIndex::new(0), BerthIndex::new(0), 4);
        assert_eq!(cost_a, Some(36));

        // Ready=10 -> Start=10 -> Finish=17. Cost=51
        let cost_b =
            eval.evaluate_vessel_assignment(&model, VesselIndex::new(0), BerthIndex::new(0), 10);
        assert_eq!(cost_b, Some(51));
    }

    #[test]
    fn test_evaluate_infeasible_returns_none() {
        let model = build_model_waiting();
        let mut eval = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let cost =
            eval.evaluate_vessel_assignment(&model, VesselIndex::new(1), BerthIndex::new(0), 3);
        assert_eq!(cost, None);
    }

    #[test]
    fn test_lower_bound_independent_projection() {
        // Test Independent Projection logic.
        // V1 (Arrival 2, Dur 6, W 4) -> Needs B1.
        // V0 (Arrival 5, Dur 3 [on B1], W 3) -> Can use B0(7) or B1(3).

        // Calculation:
        // V1: Best is B1. Start max(2,0)=2. Finish 8. Cost 8*4 = 32.
        // V0: Best is B1. Start max(5,0)=5. Finish 8. Cost 8*3 = 24.
        // Total LB = 32 + 24 = 56.

        let model = build_model_waiting();
        let mut eval = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());

        let lb = eval.lower_bound_estimate(&model, &state);

        assert_eq!(lb, Some(56));
    }
}

#[cfg(test)]
mod more_tests {
    use super::*;
    use crate::eval::evaluator::ObjectiveEvaluator;
    use bollard_core::math::interval::ClosedOpenInterval;
    use bollard_model::{model::ModelBuilder, time::ProcessingTime};

    type IntegerType = i64;

    #[test]
    fn test_earliest_finish_time_unchecked_unconstrained() {
        // Default opening is unconstrained [0, MAX)
        let model = ModelBuilder::<IntegerType>::new(1, 0).build();

        unsafe {
            // Start at 5 for duration 3 -> finish at 8
            let finish =
                earliest_finish_time_unchecked::<IntegerType>(BerthIndex::new(0), 5, 3, &model);
            assert_eq!(finish, Some(8));
        }
    }

    #[test]
    fn test_earliest_finish_time_unchecked_respects_openings() {
        // Add a closed interval [10, 20), which induces openings [0,10) and [20, MAX)
        let mut b = ModelBuilder::<IntegerType>::new(1, 0);
        b.add_berth_closing_time(BerthIndex::new(0), ClosedOpenInterval::new(10, 20));
        let model = b.build();

        unsafe {
            // If earliest_start is inside the closed interval, it must jump to 20
            let finish =
                earliest_finish_time_unchecked::<IntegerType>(BerthIndex::new(0), 12, 5, &model);
            assert_eq!(finish, Some(25));

            // If starting at 9 for duration 3, it does not fit in [0,10), so it also jumps to 20
            let finish2 =
                earliest_finish_time_unchecked::<IntegerType>(BerthIndex::new(0), 9, 3, &model);
            assert_eq!(finish2, Some(23));
        }
    }

    #[test]
    fn test_evaluate_unchecked_matches_checked() {
        // One vessel, one berth
        let mut b = ModelBuilder::<IntegerType>::new(1, 1);
        b.set_vessel_arrival_time(VesselIndex::new(0), 5)
            .set_vessel_weight(VesselIndex::new(0), 3)
            .set_vessel_latest_departure_time(VesselIndex::new(0), IntegerType::MAX)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(7),
            );
        let model = b.build();
        let mut eval = WeightedFlowTimeEvaluator::<IntegerType>::new();

        let ready = 4;
        // Checked
        let safe =
            eval.evaluate_vessel_assignment(&model, VesselIndex::new(0), BerthIndex::new(0), ready);
        assert_eq!(safe, Some(36)); // Start=5, Finish=12, Cost=12*3

        // Unchecked
        let unsafe_cost = unsafe {
            ObjectiveEvaluator::<IntegerType>::evaluate_vessel_assignment_unchecked(
                &eval,
                &model,
                VesselIndex::new(0),
                BerthIndex::new(0),
                ready,
            )
        };
        assert_eq!(unsafe_cost, safe);
    }

    #[test]
    fn test_lower_bound_workload_stronger_than_independent_when_m1() {
        // m=1 berth, 3 vessels, arrival=0, weights=1
        // processing times: 2, 3, 5 on the only berth
        let mut b = ModelBuilder::<IntegerType>::new(1, 3);
        for i in 0..3 {
            b.set_vessel_arrival_time(VesselIndex::new(i), 0)
                .set_vessel_latest_departure_time(VesselIndex::new(i), IntegerType::MAX)
                .set_vessel_weight(VesselIndex::new(i), 1);
        }
        b.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(0),
            ProcessingTime::some(2),
        )
        .set_vessel_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(0),
            ProcessingTime::some(3),
        )
        .set_vessel_processing_time(
            VesselIndex::new(2),
            BerthIndex::new(0),
            ProcessingTime::some(5),
        );
        let model = b.build();

        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut eval = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let lb = eval.lower_bound_estimate(&model, &state);

        // LB1 (independent) = sum w*p = 2 + 3 + 5 = 10
        // LB2 (workload on 1 machine, SPT order) = 2 + 5 + 10 = 17
        // max(10,17) = 17
        assert_eq!(lb, Some(17));
    }

    #[test]
    fn test_lower_bound_infeasible_returns_none() {
        // One berth, one vessel: deadline too tight to fit processing time
        let mut b = ModelBuilder::<IntegerType>::new(1, 1);
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(0), 5)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(10),
            );
        let model = b.build();

        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut eval = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let lb = eval.lower_bound_estimate(&model, &state);

        assert_eq!(lb, None);
    }
}
