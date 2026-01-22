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

//! Workload‑based lower‑bound evaluation for berth scheduling. This module defines
//! `WorkloadEvaluator<T>`, an `ObjectiveEvaluator` that derives an optimistic cost
//! by simulating the remaining workload on the available berths.
//!
//! # Mathematical Note: Completion Time vs. Flow Time
//!
//! This evaluator optimizes **Weighted Completion Time** ($C_j \times w_j$) as a proxy
//! for **Weighted Flow Time** ($(C_j - r_j) \times w_j$). Since the sum of weighted
//! arrival times $\sum (r_j \times w_j)$ is a constant, minimizing completion time
//! yields the exact same schedule as minimizing flow time, but reduces arithmetic
//! overhead in the solver's hot path (the bounds calculation).
//!
//! # Algorithm
//!
//! The evaluator reads each berth’s next free time from the current search state and
//! treats unassigned vessels as immediately available. It selects the fastest feasible
//! processing time for each vessel across all berths to preserve optimism. The
//! resulting jobs are ordered using a WSPT‑style priority (Smith's Rule) so that
//! heavy tasks tend to be completed earlier in the simulation.
//!
//! Local evaluation interprets the provided start time as the actual beginning of
//! service and returns the weighted completion cost when the assignment is feasible.
//! For the bound, maintenance windows and arrivals are relaxed to keep the estimate
//! optimistic, while berth release times are respected to reflect capacity limits.
//!
//! Saturating arithmetic is used when composing times and costs to avoid overflow
//! without violating monotonicity. The implementation favors determinism and low
//! overhead, with optional preallocation to reduce transient allocations.

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

/// Internal job used by the workload‑relaxation simulation. It condenses an
/// unassigned vessel into the duration chosen for the relaxation and its
/// objective weight. The duration is typically the shortest feasible processing
/// time across berths so the bound remains optimistic, and it is consumed by a
/// discrete‑event simulation that enforces berth capacity while relaxing
/// arrivals and maintenance. The weight multiplies the simulated completion time
/// when accumulating weighted completion cost. Feasibility checks and arithmetic
/// are handled by the evaluator; this type only stores the inputs to the
/// simulation.
#[derive(Clone, Copy, Debug)]
struct SimulationJob<T> {
    /// Processing time used by the relaxation (usually the vessel’s shortest
    /// feasible duration across all berths).
    processing_time: T,
    /// Objective weight applied to the simulated completion time.
    weight: T,
}

/// A workload‑relaxation lower bound that simulates the remaining work on
/// parallel machines equal to the number of berths. It performs a discrete‑event
/// simulation using a min‑heap where each berth starts at its current free time,
/// enforcing capacity while deliberately relaxing time‑related constraints.
/// Unassigned vessels are treated as immediately available, each vessel uses its
/// fastest feasible processing time across berths, and deadlines as well as
/// maintenance windows are ignored in the simulation to keep the estimate
/// optimistic. If any vessel has no feasible berth at all, the bound returns
/// `None` to indicate that the branch cannot produce a valid schedule.
///
/// Jobs are ordered with a Smith’s‑rule/WSPT priority so heavier work tends to
/// complete earlier in the simulated schedule, tightening the bound without
/// sacrificing optimism. Weighted completion costs are accumulated using
/// saturating arithmetic to avoid overflow while preserving monotonicity. Scratch
/// buffers hold per‑job data and the berth heap to minimize allocation overhead;
/// `preallocated` can be used when constructing evaluators for large instances.
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

    /// Calculates the objective cost for a vessel assignment.
    ///
    /// # Mathematical Note
    ///
    /// This function calculates **Weighted Completion Time** ($C_j \times w_j$) rather than
    /// strict **Weighted Flow Time** ($(C_j - r_j) \times w_j$).
    ///
    /// Since the term $\sum (r_j \times w_j)$ is constant for a given problem instance, minimizing
    /// Weighted Completion Time yields the **exact same optimal schedule** as minimizing Weighted
    /// Flow Time. Excluding the subtraction of the arrival time $r_j$ allows the solver to perform
    /// fewer arithmetic operations in the hot path.
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

    /// Calculates the objective cost for a vessel assignment.
    ///
    /// # Mathematical Note
    ///
    /// This function calculates **Weighted Completion Time** ($C_j \times w_j$) rather than
    /// strict **Weighted Flow Time** ($(C_j - r_j) \times w_j$).
    ///
    /// Since the term $\sum (r_j \times w_j)$ is constant for a given problem instance, minimizing
    /// Weighted Completion Time yields the **exact same optimal schedule** as minimizing Weighted
    /// Flow Time. Excluding the subtraction of the arrival time $r_j$ allows the solver to perform
    /// fewer arithmetic operations in the hot path.
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

        for vessel_index in 0..num_vessels {
            let vessel = VesselIndex::new(vessel_index);

            if unsafe { state.is_vessel_assigned_unchecked(vessel) } {
                continue;
            }

            let weight = unsafe { model.vessel_weight_unchecked(vessel) };
            let mut min_duration = T::max_value();
            let mut feasible = false;

            for berth_index in 0..num_berths {
                let berth = BerthIndex::new(berth_index);
                let pt = unsafe { model.vessel_processing_time_unchecked(vessel, berth) };

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

        self.scratch_jobs.sort_unstable_by(|a, b| {
            let score_a = a.weight.saturating_mul_val(b.processing_time);
            let score_b = b.weight.saturating_mul_val(a.processing_time);
            score_b.cmp(&score_a)
        });

        let mut simulated_future_cost = T::zero();

        for job in &self.scratch_jobs {
            if let Some(Reverse(free_time)) = self.scratch_heap.pop() {
                let start = free_time;
                let finish = start.saturating_add_val(job.processing_time);
                let cost = finish.saturating_mul_val(job.weight);

                simulated_future_cost = simulated_future_cost.saturating_add_val(cost);
                self.scratch_heap.push(Reverse(finish));
            } else {
                return None;
            }
        }

        Some(simulated_future_cost)
    }
}
