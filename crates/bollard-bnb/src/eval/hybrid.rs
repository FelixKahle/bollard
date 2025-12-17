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

use crate::{
    berth_availability::BerthAvailability, eval::evaluator::ObjectiveEvaluator, state::SearchState,
};
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use bollard_search::num::SolverNumeric;
use num_traits::{PrimInt, Signed};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Data required for the Workload (Capacity) simulation.
#[derive(Clone, Copy, Debug)]
struct WorkloadJob<T> {
    min_processing_time: T,
    weight: T,
}

/// A hybrid objective evaluator that combines local constraint checking with global capacity analysis.
///
/// This evaluator computes a lower bound by solving two relaxed sub-problems simultaneously
/// and returning the tighter (maximum) value:
///
/// $$ LB = \max(LB_{\text{feasibility}}, LB_{\text{capacity}}) $$
///
/// # Logic
///
/// * **Feasibility (Local Projection):** Calculates the best-case completion time for each vessel
///   individually, respecting complex constraints like maintenance windows, arrival times, and
///   berth availability. This catches bottlenecks where specific ships cannot fit into existing gaps.
/// * **Capacity (Global Workload):** Simulates a parallel machine schedule using a Min-Heap to
///   determine if the total volume of work can physically fit into the remaining time horizon.
///   This catches congestion where too many ships compete for too few berths.
///
/// # Performance
///
/// This implementation uses a **single unified pass** over the unassigned vessels to populate the
/// data structures for both bounds. This minimizes iteration overhead and CPU cache misses compared
/// to running two separate evaluators sequentially.
#[derive(Debug)]
pub struct HybridEvaluator<T>
where
    T: PrimInt + Signed,
{
    workload_jobs: Vec<WorkloadJob<T>>,
    workload_heap: BinaryHeap<Reverse<T>>,
    berth_free_times_cache: Vec<T>,
}

impl<T> Default for HybridEvaluator<T>
where
    T: PrimInt + Signed,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> HybridEvaluator<T>
where
    T: PrimInt + Signed,
{
    /// Creates a new `HybridEvaluator`.
    #[inline]
    pub fn new() -> Self {
        Self {
            workload_jobs: Vec::new(),
            workload_heap: BinaryHeap::new(),
            berth_free_times_cache: Vec::new(),
        }
    }

    /// Creates a preallocated `HybridEvaluator`.
    #[inline]
    pub fn preallocated(capacity_berths: usize, capacity_vessels: usize) -> Self {
        Self {
            workload_jobs: Vec::with_capacity(capacity_vessels),
            workload_heap: BinaryHeap::with_capacity(capacity_berths),
            berth_free_times_cache: Vec::with_capacity(capacity_berths),
        }
    }
}

impl<T> ObjectiveEvaluator<T> for HybridEvaluator<T>
where
    T: SolverNumeric,
{
    #[inline]
    fn name(&self) -> &str {
        "HybridEvaluator"
    }

    fn evaluate_vessel_assignment(
        &mut self,
        model: &Model<T>,
        _berth_availability: &BerthAvailability<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
    ) -> Option<T> {
        let weight = model.vessel_weight(vessel_index);
        let deadline = model.vessel_latest_departure_time(vessel_index);

        let pt_option = model.vessel_processing_time(vessel_index, berth_index);
        if pt_option.is_none() {
            return None;
        }
        let pt = pt_option.unwrap_unchecked();
        let completion_time = start_time.saturating_add_val(pt);

        if completion_time > deadline {
            return None;
        }

        Some(completion_time.saturating_mul_val(weight))
    }

    unsafe fn evaluate_vessel_assignment_unchecked(
        &self,
        model: &Model<T>,
        _berth_availability: &BerthAvailability<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
    ) -> Option<T>
    where
        T: SolverNumeric,
    {
        let weight = unsafe { model.vessel_weight_unchecked(vessel_index) };
        let deadline = unsafe { model.vessel_latest_departure_time_unchecked(vessel_index) };

        let pt_option =
            unsafe { model.vessel_processing_time_unchecked(vessel_index, berth_index) };
        if pt_option.is_none() {
            return None;
        }
        let pt = pt_option.unwrap_unchecked();
        let completion_time = start_time.saturating_add_val(pt);

        if completion_time > deadline {
            return None;
        }

        Some(completion_time.saturating_mul_val(weight))
    }

    fn estimate_remaining_cost(
        &mut self,
        model: &Model<T>,
        berth_availability: &BerthAvailability<T>,
        state: &SearchState<T>,
    ) -> Option<T> {
        let num_berths = model.num_berths();
        let num_vessels = model.num_vessels();

        if state.num_assigned_vessels() == num_vessels {
            return Some(T::zero());
        }

        if num_berths == 0 {
            return None;
        }

        self.workload_heap.clear();
        self.workload_jobs.clear();
        self.berth_free_times_cache.clear();

        for b in 0..num_berths {
            let t = unsafe { state.berth_free_time_unchecked(BerthIndex::new(b)) };
            self.berth_free_times_cache.push(t);
            self.workload_heap.push(Reverse(t));
        }

        let mut lb_feasibility = T::zero();
        let mut min_unassigned_arrival = T::max_value();

        for vessel_index_usize in 0..num_vessels {
            let vessel_index = VesselIndex::new(vessel_index_usize);

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

            for (berth_index_usize, &current_berth_free_time) in
                self.berth_free_times_cache.iter().enumerate()
            {
                let berth_index = BerthIndex::new(berth_index_usize);
                let pt_opt =
                    unsafe { model.vessel_processing_time_unchecked(vessel_index, berth_index) };

                if pt_opt.is_none() {
                    continue;
                }
                let duration = pt_opt.unwrap_unchecked();
                let tentative_start = arrival.max(current_berth_free_time);
                let actual_start_opt = unsafe {
                    berth_availability.earliest_availability_unchecked(
                        berth_index,
                        tentative_start,
                        duration,
                    )
                };

                if let Some(actual_start) = actual_start_opt {
                    let finish = actual_start.saturating_add_val(duration);
                    if finish <= deadline && finish < best_finish_time {
                        best_finish_time = finish;
                        found_feasible_berth = true;
                    }
                }
            }

            if !found_feasible_berth {
                return None;
            }

            let cost = best_finish_time.saturating_mul_val(weight);
            lb_feasibility = lb_feasibility.saturating_add_val(cost);

            let min_pt_opt =
                unsafe { model.vessel_shortest_processing_time_unchecked(vessel_index) };

            if let Some(min_duration) = Option::<T>::from(min_pt_opt) {
                self.workload_jobs.push(WorkloadJob {
                    min_processing_time: min_duration,
                    weight,
                });
            }
        }

        let lb_capacity = if self.workload_jobs.is_empty() {
            T::zero()
        } else {
            self.workload_jobs.sort_unstable_by(|a, b| {
                let score_a = a.weight.saturating_mul_val(b.min_processing_time);
                let score_b = b.weight.saturating_mul_val(a.min_processing_time);
                score_b.cmp(&score_a)
            });

            let mut sim_cost = T::zero();

            for job in &self.workload_jobs {
                if let Some(Reverse(free_time)) = self.workload_heap.pop() {
                    let start = free_time.max(min_unassigned_arrival);
                    let finish = start.saturating_add_val(job.min_processing_time);
                    let cost = finish.saturating_mul_val(job.weight);

                    sim_cost = sim_cost.saturating_add_val(cost);

                    self.workload_heap.push(Reverse(finish));
                }
            }
            sim_cost
        };

        // 4. Return the tighter (maximum) of the two bounds
        Some(lb_feasibility.max(lb_capacity))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bollard_core::math::interval::ClosedOpenInterval;
    use bollard_model::model::ModelBuilder;
    use bollard_model::time::ProcessingTime;

    type IntegerType = i64;

    #[test]
    fn test_hybrid_dominates_when_maintenance_blocks() {
        // Scenario: Maintenance dominates.
        // 1 Berth. 1 Vessel.
        // V0: Arrival 0, Duration 10.
        // Berth 0: Maintenance [0, 100).

        // Capacity Bound: Sees free time 0 (ignores maintenance). Start 0 -> Finish 10. LB = 10.
        // Feasibility Bound: Sees maintenance. Start 100 -> Finish 110. LB = 110.
        // Hybrid Result: 110.

        let mut b = ModelBuilder::<IntegerType>::new(1, 1);
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(10),
            )
            .add_berth_closing_time(BerthIndex::new(0), ClosedOpenInterval::new(0, 100));
        let model = b.build();

        let mut avail = BerthAvailability::new();
        avail.initialize(&model, &[]); // loads closures

        let state = SearchState::new(1, 1);
        let mut eval = HybridEvaluator::<IntegerType>::new();

        let lb = eval.estimate_remaining_cost(&model, &avail, &state);
        assert_eq!(lb, Some(110));
    }

    #[test]
    fn test_hybrid_dominates_when_congestion_blocks() {
        // Scenario: Congestion dominates.
        // 1 Berth. 2 Vessels (Identical).
        // V0, V1: Arrival 0, Duration 10, Weight 1.
        // No maintenance.

        // Feasibility Bound:
        // V0: Best finish 10. Cost 10.
        // V1: Best finish 10. Cost 10. (Ignores that B0 is occupied by V0).
        // Sum = 20.

        // Capacity Bound:
        // Job 1 starts @ 0 -> Finish 10. Cost 10.
        // Job 2 starts @ 10 -> Finish 20. Cost 20.
        // Sum = 30.

        // Hybrid Result: 30.

        let mut b = ModelBuilder::<IntegerType>::new(1, 2);
        for i in 0..2 {
            let v = VesselIndex::new(i);
            b.set_vessel_arrival_time(v, 0)
                .set_vessel_weight(v, 1)
                .set_vessel_processing_time(v, BerthIndex::new(0), ProcessingTime::some(10));
        }
        let model = b.build();
        let mut avail = BerthAvailability::new();
        avail.initialize(&model, &[]);
        let state = SearchState::new(1, 2);
        let mut eval = HybridEvaluator::<IntegerType>::new();

        let lb = eval.estimate_remaining_cost(&model, &avail, &state);
        assert_eq!(lb, Some(30));
    }

    #[test]
    fn test_hybrid_returns_zero_when_all_assigned() {
        // Setup: 2 berths, 3 vessels with arbitrary data.
        let mut b = ModelBuilder::<IntegerType>::new(2, 3);
        for v in 0..3 {
            let vi = VesselIndex::new(v);
            b.set_vessel_arrival_time(vi, 0)
                .set_vessel_weight(vi, 1)
                .set_vessel_processing_time(vi, BerthIndex::new(0), ProcessingTime::some(5))
                .set_vessel_processing_time(vi, BerthIndex::new(1), ProcessingTime::some(7));
        }
        let model = b.build();

        let mut avail = BerthAvailability::new();
        avail.initialize(&model, &[]);

        let mut state = SearchState::<IntegerType>::new(2, 3);
        // Mark all vessels assigned (times and berths arbitrary since code returns early).
        unsafe {
            state.assign_vessel_unchecked(VesselIndex::new(0), BerthIndex::new(0), 0);
            state.assign_vessel_unchecked(VesselIndex::new(1), BerthIndex::new(1), 0);
            state.assign_vessel_unchecked(VesselIndex::new(2), BerthIndex::new(0), 10);
        }

        let mut eval = HybridEvaluator::<IntegerType>::new();
        let lb = eval.estimate_remaining_cost(&model, &avail, &state);
        assert_eq!(
            lb,
            Some(0),
            "When all vessels are assigned, LB must be zero"
        );
    }

    #[test]
    fn test_hybrid_none_when_no_berths() {
        // Setup: 0 berths, 1 vessel.
        let b = ModelBuilder::<IntegerType>::new(0, 1);
        let model = b.build();

        let mut avail = BerthAvailability::new();
        avail.initialize(&model, &[]);

        let state = SearchState::<IntegerType>::new(0, 1);
        let mut eval = HybridEvaluator::<IntegerType>::new();

        let lb = eval.estimate_remaining_cost(&model, &avail, &state);
        assert_eq!(lb, None, "No berths means infeasible lower bound");
    }

    #[test]
    fn test_hybrid_none_when_vessel_forbidden_on_all_berths() {
        // Setup: 1 berth, 1 vessel with NO processing time set (forbidden everywhere).
        let b = ModelBuilder::<IntegerType>::new(1, 1);
        let model = b.build();

        let mut avail = BerthAvailability::new();
        avail.initialize(&model, &[]);

        let state = SearchState::<IntegerType>::new(1, 1);
        let mut eval = HybridEvaluator::<IntegerType>::new();

        let lb = eval.estimate_remaining_cost(&model, &avail, &state);
        assert_eq!(
            lb, None,
            "If a vessel cannot be processed on any berth, LB must be None"
        );
    }

    #[test]
    fn test_hybrid_none_when_deadline_too_tight() {
        // Setup: 1 berth, 1 vessel: duration 10 but deadline 5 -> infeasible.
        let mut b = ModelBuilder::<IntegerType>::new(1, 1);
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(0), 5)
            .set_vessel_weight(VesselIndex::new(0), 3)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(10),
            );
        let model = b.build();

        let mut avail = BerthAvailability::new();
        avail.initialize(&model, &[]);

        let state = SearchState::<IntegerType>::new(1, 1);
        let mut eval = HybridEvaluator::<IntegerType>::new();

        let lb = eval.estimate_remaining_cost(&model, &avail, &state);
        assert_eq!(lb, None, "Finish would exceed deadline -> infeasible");
    }

    #[test]
    fn test_hybrid_capacity_two_berths_three_vessels() {
        // 2 berths, 3 identical vessels: arrival 0, duration 10, weight 1.
        // Feasibility (ignores conflicts): each best finish is 10 -> cost sum 30.
        // Capacity: two finish at 10 (10+10), one finishes at 20 (20) -> sum 40 (dominates).
        let mut b = ModelBuilder::<IntegerType>::new(2, 3);
        for i in 0..3 {
            let v = VesselIndex::new(i);
            b.set_vessel_arrival_time(v, 0)
                .set_vessel_weight(v, 1)
                .set_vessel_processing_time(v, BerthIndex::new(0), ProcessingTime::some(10))
                .set_vessel_processing_time(v, BerthIndex::new(1), ProcessingTime::some(10));
        }
        let model = b.build();

        let mut avail = BerthAvailability::new();
        avail.initialize(&model, &[]);

        let state = SearchState::<IntegerType>::new(2, 3);
        let mut eval = HybridEvaluator::<IntegerType>::new();

        let lb = eval.estimate_remaining_cost(&model, &avail, &state);
        assert_eq!(lb, Some(40));
    }

    #[test]
    fn test_hybrid_capacity_respects_min_arrival() {
        // 2 berths, 3 vessels: arrival 100, duration 10, weight 1.
        // Feasibility: each best finish 110 -> cost sum 330.
        // Capacity: two finish 110, one finishes 120 -> 110+110+120 = 340 (dominates).
        let mut b = ModelBuilder::<IntegerType>::new(2, 3);
        for i in 0..3 {
            let v = VesselIndex::new(i);
            b.set_vessel_arrival_time(v, 100)
                .set_vessel_weight(v, 1)
                .set_vessel_processing_time(v, BerthIndex::new(0), ProcessingTime::some(10))
                .set_vessel_processing_time(v, BerthIndex::new(1), ProcessingTime::some(10));
        }
        let model = b.build();

        let mut avail = BerthAvailability::new();
        avail.initialize(&model, &[]);

        let state = SearchState::<IntegerType>::new(2, 3);
        let mut eval = HybridEvaluator::<IntegerType>::new();

        let lb = eval.estimate_remaining_cost(&model, &avail, &state);
        assert_eq!(lb, Some(340));
    }

    #[test]
    fn test_hybrid_capacity_wspt_weight_ordering() {
        // 1 berth, 2 vessels; different weights and durations
        // v0: p=10, w=1 -> ratio 0.1; v1: p=5, w=10 -> ratio 2.0.
        // Feasibility: costs 10 + 50 = 60.
        // Capacity (WSPT order v1 then v0): finish 5->50, then finish 15->15 => 65 (dominates).
        let mut b = ModelBuilder::<IntegerType>::new(1, 2);
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(10),
            );
        b.set_vessel_arrival_time(VesselIndex::new(1), 0)
            .set_vessel_weight(VesselIndex::new(1), 10)
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(0),
                ProcessingTime::some(5),
            );
        let model = b.build();

        let mut avail = BerthAvailability::new();
        avail.initialize(&model, &[]);

        let state = SearchState::<IntegerType>::new(1, 2);
        let mut eval = HybridEvaluator::<IntegerType>::new();

        let lb = eval.estimate_remaining_cost(&model, &avail, &state);
        assert_eq!(lb, Some(65));
    }

    #[test]
    fn test_hybrid_respects_state_berth_free_time() {
        // 1 berth starts free at time 5 (not 0). 2 vessels, arrival 0, p=10, w=1.
        // Feasibility: each best finish 15 -> sum 30.
        // Capacity: first finishes 15 (15), second finishes 25 (25) -> 40 (dominates).
        let mut b = ModelBuilder::<IntegerType>::new(1, 2);
        for i in 0..2 {
            let v = VesselIndex::new(i);
            b.set_vessel_arrival_time(v, 0)
                .set_vessel_weight(v, 1)
                .set_vessel_processing_time(v, BerthIndex::new(0), ProcessingTime::some(10));
        }
        let model = b.build();

        let mut avail = BerthAvailability::new();
        avail.initialize(&model, &[]);

        let mut state = SearchState::<IntegerType>::new(1, 2);
        state.set_berth_free_time(BerthIndex::new(0), 5);

        let mut eval = HybridEvaluator::<IntegerType>::new();
        let lb = eval.estimate_remaining_cost(&model, &avail, &state);
        assert_eq!(lb, Some(40));
    }

    #[test]
    fn test_evaluate_vessel_assignment_checked() {
        // Simple cost check: start=7, p=3, w=4, deadline large -> completion=10 -> cost=40
        let mut b = ModelBuilder::<IntegerType>::new(1, 1);
        b.set_vessel_weight(VesselIndex::new(0), 4)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(3),
            );
        let model = b.build();

        let avail = BerthAvailability::new(); // not used by this function
        let mut eval = HybridEvaluator::<IntegerType>::new();

        let cost = eval.evaluate_vessel_assignment(
            &model,
            &avail,
            VesselIndex::new(0),
            BerthIndex::new(0),
            7,
        );
        assert_eq!(cost, Some(40));

        // Forbidden processing -> None
        let b2 = ModelBuilder::<IntegerType>::new(1, 1).build();
        let cost2 = eval.evaluate_vessel_assignment(
            &b2,
            &avail,
            VesselIndex::new(0),
            BerthIndex::new(0),
            0,
        );
        assert_eq!(cost2, None);

        // Exceeds deadline -> None
        let mut b3 = ModelBuilder::<IntegerType>::new(1, 1);
        b3.set_vessel_latest_departure_time(VesselIndex::new(0), 9)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(3),
            );
        let model3 = b3.build();
        let cost3 = eval.evaluate_vessel_assignment(
            &model3,
            &avail,
            VesselIndex::new(0),
            BerthIndex::new(0),
            7,
        );
        assert_eq!(cost3, None);
    }

    #[test]
    fn test_evaluate_vessel_assignment_unchecked() {
        // Simple cost check using the unchecked variant.
        let mut b = ModelBuilder::<IntegerType>::new(1, 1);
        b.set_vessel_weight(VesselIndex::new(0), 2)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(8),
            );
        let model = b.build();

        let avail = BerthAvailability::new(); // not used by this function
        let eval = HybridEvaluator::<IntegerType>::new();

        let cost = unsafe {
            eval.evaluate_vessel_assignment_unchecked(
                &model,
                &avail,
                VesselIndex::new(0),
                BerthIndex::new(0),
                3,
            )
        };
        // completion = 3 + 8 = 11; cost = 11 * 2 = 22
        assert_eq!(cost, Some(22));

        // Forbidden processing -> None
        let model2 = ModelBuilder::<IntegerType>::new(1, 1).build();
        let cost2 = unsafe {
            eval.evaluate_vessel_assignment_unchecked(
                &model2,
                &avail,
                VesselIndex::new(0),
                BerthIndex::new(0),
                0,
            )
        };
        assert_eq!(cost2, None);
    }
}
