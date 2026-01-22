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

//! This module implements `WeightedCompletionTimeEvaluator<T>`, an `ObjectiveEvaluator` for the
//! weighted completion objective, integrating berth availability to keep both
//! local scoring and global bounds aligned with real operating constraints.
//!
//! # Mathematical Note: Completion Time vs. Flow Time
//!
//! This evaluator optimizes **Weighted Completion Time** ($C_j \times w_j$) as a proxy
//! for **Weighted Flow Time** ($(C_j - r_j) \times w_j$). Since the sum of weighted
//! arrival times $\sum (r_j \times w_j)$ is a constant for any given instance, minimizing
//! completion time yields the **exact same optimal schedule** as minimizing flow time.
//! Excluding the subtraction of the arrival time reduces arithmetic overhead in the
//! solver's hot path and avoids the need to access arrival times during cost evaluation.
//!
//! # Algorithm
//!
//! Local evaluation treats the provided start time as the actual beginning of
//! service and computes a weighted completion cost. It reports infeasibility as
//! `None` when a vessel cannot be processed on the chosen berth or would miss
//! its deadline, preserving the solver’s understanding of the search space.
//!
//! The remaining‑cost estimate blends a feasibility‑aware projection with a
//! lightweight workload relaxation. It first examines each unassigned vessel
//! against every berth’s current free time and earliest usable window so that
//! the best attainable finish time respects closures and deadlines. It then
//! forms a coarse single‑machine schedule over shortest feasible processing
//! times, starts it no earlier than the earliest berth release or arrival, and
//! scales by berth count. Taking the maximum of these two views yields a bound
//! that stays optimistic yet reacts to maintenance and congestion. Scratch
//! buffers are reused for determinism and speed, and saturating arithmetic
//! prevents overflow while maintaining non‑decreasing costs.

use crate::{
    berth_availability::BerthAvailability, eval::evaluator::ObjectiveEvaluator, state::SearchState,
};
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use bollard_search::num::SolverNumeric;
use num_traits::{PrimInt, Signed};

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

/// Evaluator for the weighted completion time objective (sum of weight × completion time),
/// aware of berth availability for both local scoring and global lower bounds.
/// It implements `ObjectiveEvaluator` and is suitable wherever a regular objective
/// is required, meaning costs do not decrease when completion is delayed.
///
/// Local evaluation treats the provided start time as the actual beginning of service
/// and returns a weighted completion cost when the assignment is feasible. If the
/// vessel cannot be processed on the chosen berth or the completion would exceed its
/// deadline, the result is `None`, which cleanly communicates infeasibility to the
/// solver without inflating scores. Saturating arithmetic is used to avoid overflow
/// near numeric limits while preserving monotonicity.
///
/// The remaining‑cost estimate combines two complementary views of the future. It
/// first projects, for each unassigned vessel, the best attainable finish time over
/// all berths given current berth free times, availability windows, arrivals, and
/// deadlines; this captures feasibility against closures and ensures the bound does
/// not assume impossible starts. It then builds a coarse, single‑machine relaxation
/// over each vessel’s shortest feasible processing time, begins no earlier than the
/// earliest berth release or arrival among the remaining vessels, and scales by the
/// berth count to approximate parallel capacity. Taking the maximum of these views
/// yields an optimistic yet availability‑sensitive bound that responds to both
/// maintenance and congestion.
///
/// The scratch buffers store per‑berth free times and per‑vessel summaries used by
/// the bound computation. They are reused across calls to reduce allocation, keep
/// behavior deterministic, and avoid coupling the solver to transient memory traffic.
/// Use `preallocated` when constructing many evaluators or solving large instances to
/// minimize reallocation. This design provides predictable performance and integrates
/// naturally with availability‑aware branching.
#[derive(Debug)]
pub struct WeightedCompletionTimeEvaluator<T>
where
    T: PrimInt + Signed,
{
    scratch_berths: Vec<T>,
    scratch_vessels: Vec<VesselData<T>>,
}

impl<T> Default for WeightedCompletionTimeEvaluator<T>
where
    T: PrimInt + Signed,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> WeightedCompletionTimeEvaluator<T>
where
    T: PrimInt + Signed,
{
    /// Creates a new `WeightedCompletionTimeEvaluator` with empty scratch buffers.
    #[inline]
    pub fn new() -> Self {
        Self {
            scratch_berths: Vec::new(),
            scratch_vessels: Vec::new(),
        }
    }

    /// Creates a new `WeightedCompletionTimeEvaluator` with preallocated scratch buffers.
    #[inline]
    pub fn preallocated(capacity_berths: usize, capacity_vessels: usize) -> Self {
        Self {
            scratch_berths: Vec::with_capacity(capacity_berths),
            scratch_vessels: Vec::with_capacity(capacity_vessels),
        }
    }
}

impl<T> ObjectiveEvaluator<T> for WeightedCompletionTimeEvaluator<T>
where
    T: SolverNumeric,
{
    #[inline]
    fn name(&self) -> &str {
        "WeightedCompletionTimeEvaluator"
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
                let berth_idx = BerthIndex::new(berth_index);
                let processing_time_opt =
                    unsafe { model.vessel_processing_time_unchecked(vessel_index, berth_idx) };

                if processing_time_opt.is_none() {
                    continue;
                }
                let processing_time = processing_time_opt.unwrap_unchecked();
                let tentative_start = arrival.max(current_free_time);

                let possible_finish = unsafe {
                    berth_availability
                        .earliest_availability_unchecked(
                            berth_idx,
                            tentative_start,
                            processing_time,
                        )
                        .map(|start| start.saturating_add_val(processing_time))
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
