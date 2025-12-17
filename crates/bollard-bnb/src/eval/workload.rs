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

/// Internal job representation for the workload simulation.
#[derive(Clone, Copy, Debug)]
struct SimulationJob<T> {
    processing_time: T,
    weight: T,
}

/// A "Workload Relaxation" evaluator.
///
/// It calculates a Lower Bound by simulating the future schedule on the actual
/// number of berths using a Min-Heap (Discrete Event Simulation).
///
/// **Relaxation:**
/// 1. **Time:** Assumes all unassigned vessels are available immediately.
/// 2. **Space:** Assumes ALL berths are free at the time of the *earliest* available berth.
///
/// This guarantees the bound is strictly optimistic (Lower Bound <= True Cost),
/// preventing the pruning of optimal solutions while still accounting for
/// the physical inability to process infinite vessels in parallel.
#[derive(Debug)]
pub struct WorkloadEvaluator<T>
where
    T: PrimInt + Signed,
{
    scratch_jobs: Vec<SimulationJob<T>>,
    scratch_heap: BinaryHeap<Reverse<T>>,
}

impl<T> Default for WorkloadEvaluator<T>
where
    T: PrimInt + Signed,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> WorkloadEvaluator<T>
where
    T: PrimInt + Signed,
{
    #[inline]
    pub fn new() -> Self {
        Self {
            scratch_jobs: Vec::new(),
            scratch_heap: BinaryHeap::new(),
        }
    }

    #[inline]
    pub fn preallocated(capacity_berths: usize, capacity_vessels: usize) -> Self {
        Self {
            scratch_jobs: Vec::with_capacity(capacity_vessels),
            scratch_heap: BinaryHeap::with_capacity(capacity_berths),
        }
    }
}

impl<T> ObjectiveEvaluator<T> for WorkloadEvaluator<T>
where
    T: SolverNumeric,
{
    #[inline]
    fn name(&self) -> &str {
        "WorkloadEvaluator"
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
        _berth_availability: &BerthAvailability<T>, // Unused in this relaxation
        state: &SearchState<T>,
    ) -> Option<T> {
        let num_berths = model.num_berths();
        let num_vessels = model.num_vessels();

        self.scratch_heap.clear();
        for b in 0..num_berths {
            let t = unsafe { state.berth_free_time_unchecked(BerthIndex::new(b)) };
            self.scratch_heap.push(Reverse(t));
        }

        self.scratch_jobs.clear();

        for i in 0..num_vessels {
            let vessel_index = VesselIndex::new(i);

            if unsafe { state.is_vessel_assigned_unchecked(vessel_index) } {
                continue;
            }

            let weight = unsafe { model.vessel_weight_unchecked(vessel_index) };
            let mut min_duration = T::max_value();
            let mut feasible = false;

            for b in 0..num_berths {
                let berth_index = BerthIndex::new(b);
                let pt =
                    unsafe { model.vessel_processing_time_unchecked(vessel_index, berth_index) };

                if pt.is_none() {
                    continue;
                }
                let duration = pt.unwrap_unchecked();

                if duration < min_duration {
                    min_duration = duration;
                }
                feasible = true;
            }

            if !feasible {
                return None;
            }

            self.scratch_jobs.push(SimulationJob {
                processing_time: min_duration,
                weight,
            });
        }

        if self.scratch_jobs.is_empty() {
            return Some(T::zero());
        }

        // WSPT ordering for the simulation
        self.scratch_jobs.sort_unstable_by(|a, b| {
            let score_a = a.weight.saturating_mul_val(b.processing_time);
            let score_b = b.weight.saturating_mul_val(a.processing_time);
            score_b.cmp(&score_a)
        });

        let mut simulated_future_cost = T::zero();

        for job in &self.scratch_jobs {
            if let Some(Reverse(free_time)) = self.scratch_heap.pop() {
                // Relaxation: We assume the vessel can start immediately when the berth is free.
                // We do NOT check arrival times or maintenance windows here, preserving the
                // optimistic Lower Bound property.
                let start = free_time;
                let finish = start.saturating_add_val(job.processing_time);
                let cost = finish.saturating_mul_val(job.weight);

                simulated_future_cost = simulated_future_cost.saturating_add_val(cost);
                self.scratch_heap.push(Reverse(finish));
            } else {
                // Should be impossible if num_berths > 0
                return None;
            }
        }

        Some(simulated_future_cost)
    }
}
