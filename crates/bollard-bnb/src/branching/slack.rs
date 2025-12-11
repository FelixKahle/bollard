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
    branching::decision::{Decision, DecisionBuilder},
    eval::evaluator::ObjectiveEvaluator,
    state::SearchState,
};
use bollard_core::num::ops::saturating_arithmetic::SaturatingSubVal;
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use bollard_search::num::SolverNumeric;
use std::cmp::Ordering;
use std::iter::FusedIterator;

/// Represents a feasible option found during the per-vessel scan.
#[derive(Debug, Clone, Copy)]
struct SlackOption<T> {
    cost: T,
    decision: Decision,
}

/// Internal candidate structure for global sorting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SlackCandidate<T> {
    vessel_slack: T, // Primary Key (Ascending)
    cost: T,         // Secondary Key (Ascending)
    decision: Decision,
}

impl<T> SlackCandidate<T> {
    #[inline(always)]
    fn new(vessel_slack: T, cost: T, decision: Decision) -> Self {
        Self {
            vessel_slack,
            cost,
            decision,
        }
    }
}

impl<T: Ord> Ord for SlackCandidate<T> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.vessel_slack
            .cmp(&other.vessel_slack)
            .then_with(|| self.cost.cmp(&other.cost))
            .then_with(|| self.decision.cmp(&other.decision))
    }
}

impl<T: Ord> PartialOrd for SlackCandidate<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A decision builder that implements a **Slack-Guided Best-First** branching heuristic.
///
/// This builder prioritizes vessels with the tightest time slack (least flexibility) to reduce
/// the risk of missing feasible schedules. For each unassigned vessel:
/// 1. It scans all admissible berths and computes the earliest feasible finish time
///    (respecting arrival, berth free time, processing duration, and vessel deadline).
/// 2. If at least one feasible option exists for that vessel, it calculates the vessel’s slack:
///    `slack = deadline - min_finish_time`.
/// 3. It collects all feasible `(vessel, berth)` options for that vessel, each annotated with:
///    - the common vessel slack, and
///    - the immediate objective increase (`cost`) as provided by the evaluator.
/// 4. It sorts all global candidates by:
///    - primary: ascending `vessel_slack` (tightest first),
///    - secondary: ascending `cost` (cheapest first),
///    - tertiary: deterministic `Decision` order (tie-breaker).
///
/// This strategy focuses search on time-critical assignments first, while still preferring
/// cheaper moves among equally critical vessels. It helps find high-quality incumbents early
/// and preserves feasibility by filtering any option whose finish time violates the deadline.
///
/// Notes:
/// - Feasibility requires `finish_time = max(arrival, berth_free) + duration` to be `<= deadline`.
/// - Only vessels with at least one feasible option contribute candidates.
/// - The evaluator is only called for feasible `(vessel, berth)` pairs.
/// - Sorting is stable with respect to the defined keys via `sort_unstable` on a total order.
///
/// Example:
/// - When multiple vessels have options, the vessel with slack=2 will be explored before one with
///   slack=10; within the slack=2 group, options are ordered by increasing cost.
#[derive(Debug, Clone, Default)]
pub struct SlackHeuristicBuilder<T> {
    candidates: Vec<SlackCandidate<T>>,
    scratch_options: Vec<SlackOption<T>>,
}

impl<T> SlackHeuristicBuilder<T> {
    /// Creates a new `SlackHeuristicBuilder` with empty internal buffers.
    ///
    /// Use this when you don't know the instance size in advance. For large instances,
    /// consider `preallocated` to reduce reallocations during branching.
    #[inline]
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
            scratch_options: Vec::new(),
        }
    }

    /// Creates a new `SlackHeuristicBuilder` with preallocated buffers sized for the given
    /// number of berths and vessels.
    ///
    /// Capacity:
    /// - `candidates`: up to `num_berths * num_vessels` global candidates.
    /// - `scratch_options`: up to `num_berths` options per vessel during scanning.
    #[inline]
    pub fn preallocated(num_berths: usize, num_vessels: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(num_berths * num_vessels),
            scratch_options: Vec::with_capacity(num_berths),
        }
    }

    /// Creates a new `SlackHeuristicBuilder` with a candidate buffer of the specified capacity.
    /// You may use `SlackHeuristicBuilder::preallocated` for a more specific preallocation based
    /// on the number of berths and vessels.
    #[inline]
    pub fn with_capacity(size: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(size),
            scratch_options: Vec::with_capacity(size),
        }
    }
}

impl<T, E> DecisionBuilder<T, E> for SlackHeuristicBuilder<T>
where
    T: SolverNumeric + num_traits::Bounded + SaturatingSubVal,
    E: ObjectiveEvaluator<T>,
{
    type DecisionIterator<'a>
        = SlackHeuristicIter<'a, T>
    where
        T: 'a,
        E: 'a,
        Self: 'a;

    fn name(&self) -> &str {
        "SlackHeuristicBuilder"
    }

    fn next_decision<'a>(
        &'a mut self,
        evaluator: &'a mut E,
        model: &'a Model<T>,
        state: &'a SearchState<T>,
    ) -> Self::DecisionIterator<'a> {
        self.candidates.clear();

        let num_vessels = model.num_vessels();
        let num_berths = model.num_berths();

        for v in 0..num_vessels {
            let vessel_index = VesselIndex::new(v);

            if unsafe { state.is_vessel_assigned_unchecked(vessel_index) } {
                continue;
            }

            let arrival = unsafe { model.vessel_arrival_time_unchecked(vessel_index) };
            let deadline = unsafe { model.vessel_latest_departure_time_unchecked(vessel_index) };

            self.scratch_options.clear();
            let mut best_case_finish_time = T::max_value();

            for b in 0..num_berths {
                let berth_index = BerthIndex::new(b);

                if !unsafe { model.vessel_allowed_on_berth_unchecked(vessel_index, berth_index) } {
                    continue;
                }

                if let Some(decision) =
                    unsafe { Decision::try_new_unchecked(vessel_index, berth_index, model, state) }
                {
                    let berth_free = unsafe { state.berth_free_time_unchecked(berth_index) };

                    let start = if arrival > berth_free {
                        arrival
                    } else {
                        berth_free
                    };

                    if let Some(duration) = unsafe {
                        model
                            .vessel_processing_time_unchecked(vessel_index, berth_index)
                            .into()
                    } {
                        let finish = start + duration;

                        if finish <= deadline
                            && let Some(cost) = unsafe {
                                evaluator.evaluate_vessel_assignment_unchecked(
                                    model,
                                    vessel_index,
                                    berth_index,
                                    berth_free,
                                )
                            }
                        {
                            if finish < best_case_finish_time {
                                best_case_finish_time = finish;
                            }
                            self.scratch_options.push(SlackOption { cost, decision });
                        }
                    }
                }
            }

            if self.scratch_options.is_empty() {
                continue;
            }

            let vessel_slack = deadline.saturating_sub_val(best_case_finish_time);

            for option in self.scratch_options.iter() {
                self.candidates.push(SlackCandidate::new(
                    vessel_slack,
                    option.cost,
                    option.decision,
                ));
            }
        }

        self.candidates.sort_unstable();

        SlackHeuristicIter {
            iter: self.candidates.iter(),
        }
    }
}

/// A lightweight iterator that yields `Decision`s from the slack-sorted candidate slice.
///
/// Properties:
/// - Borrowing: holds a reference into the builder’s internal buffer; no allocations.
/// - Order: yields decisions sorted by ascending vessel slack, then cost, then decision.
/// - Fused: once exhausted, subsequent calls return `None`.
pub struct SlackHeuristicIter<'a, T> {
    iter: std::slice::Iter<'a, SlackCandidate<T>>,
}

impl<'a, T: Copy> Iterator for SlackHeuristicIter<'a, T> {
    type Item = Decision;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|c| c.decision)
    }
}

impl<'a, T: Copy> FusedIterator for SlackHeuristicIter<'a, T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::wtft::WeightedFlowTimeEvaluator;
    use bollard_model::{
        index::{BerthIndex, VesselIndex},
        model::ModelBuilder,
        time::ProcessingTime,
    };

    type IntegerType = i64;

    // Build a small model with distinct vessel slacks:
    // - 3 vessels, 2 berths
    // - All arrivals at 0, berths initially free
    // - Deadlines chosen to yield different best-case slacks
    //
    // Vessel 0: deadline 20, durations {b0:10, b1:5} -> best finish 5 -> slack 15
    // Vessel 1: deadline 8,  durations {b0:3,  b1:4} -> best finish 3 -> slack 5
    // Vessel 2: deadline 6,  durations {b0:5,  b1:none} -> best finish 5 -> slack 1
    //
    // Expected slack order: v2 (1) < v1 (5) < v0 (15)
    // The builder should group candidates primarily by slack, then by increasing cost.
    fn build_slack_model() -> bollard_model::model::Model<IntegerType> {
        let mut b = ModelBuilder::<IntegerType>::new(2, 3);

        // Vessel 0
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(0), 20)
            .set_vessel_weight(VesselIndex::new(0), 3) // heavier weight to vary costs
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(10),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::some(5),
            );

        // Vessel 1
        b.set_vessel_arrival_time(VesselIndex::new(1), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(1), 8)
            .set_vessel_weight(VesselIndex::new(1), 2)
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(0),
                ProcessingTime::some(3),
            )
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(1),
                ProcessingTime::some(4),
            );

        // Vessel 2
        b.set_vessel_arrival_time(VesselIndex::new(2), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(2), 6)
            .set_vessel_weight(VesselIndex::new(2), 1)
            .set_vessel_processing_time(
                VesselIndex::new(2),
                BerthIndex::new(0),
                ProcessingTime::some(5),
            )
            .set_vessel_processing_time(
                VesselIndex::new(2),
                BerthIndex::new(1),
                ProcessingTime::none(), // infeasible on b1
            );

        b.build()
    }

    #[test]
    fn test_slack_orders_by_increasing_vessel_slack_then_cost() {
        let model = build_slack_model();
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = SlackHeuristicBuilder::<IntegerType>::new();

        // Collect decisions
        let decisions: Vec<Decision> = builder
            .next_decision(&mut evaluator, &model, &state)
            .by_ref()
            .collect();

        assert!(
            !decisions.is_empty(),
            "Expected non-empty decision list for slack model"
        );

        // Compute per-decision:
        // - vessel index
        // - cost (using evaluator)
        // - derived best-case finish per vessel (for slack verification)
        // - deadline per vessel
        let mut per_decision: Vec<(usize, IntegerType)> = Vec::new();
        for d in &decisions {
            let v = d.vessel_index();
            let b = d.berth_index();
            let ready = unsafe { state.berth_free_time_unchecked(b) };

            let cost = evaluator
                .evaluate_vessel_assignment(&model, v, b, ready)
                .expect("decision must be evaluable");

            per_decision.push((v.get(), cost));
        }

        // Pre-compute the expected vessel slack values from the model definition.
        // We replicate the builder's computation: slack = deadline - best_case_finish_time.
        let mut vessel_best_finish: [IntegerType; 3] = [IntegerType::MAX; 3];
        let deadlines = [20, 8, 6];

        // Berth free times are 0; arrivals are 0; so start=0.
        // Using the same durations as in the build_slack_model:
        // v0: {b0:10, b1:5} -> best finish 5
        vessel_best_finish[0] = 5;
        // v1: {b0:3, b1:4} -> best finish 3
        vessel_best_finish[1] = 3;
        // v2: {b0:5, b1:None} -> best finish 5
        vessel_best_finish[2] = 5;

        let expected_slacks = [
            deadlines[0] - vessel_best_finish[0], // 15
            deadlines[1] - vessel_best_finish[1], // 5
            deadlines[2] - vessel_best_finish[2], // 1
        ];

        // Verify primary ordering: vessel slack ascending.
        // That implies all decisions for v2 first, then v1, then v0.
        let vessel_sequence: Vec<usize> = per_decision.iter().map(|(v, _)| *v).collect();

        // Find the ranges of each vessel in the yielded order
        let _first_v2 = vessel_sequence
            .iter()
            .position(|&vv| vv == 2)
            .expect("v2 must appear");
        let last_v2 = vessel_sequence
            .iter()
            .rposition(|&vv| vv == 2)
            .expect("v2 must appear");
        let first_v1 = vessel_sequence
            .iter()
            .position(|&vv| vv == 1)
            .expect("v1 must appear");
        let last_v1 = vessel_sequence
            .iter()
            .rposition(|&vv| vv == 1)
            .expect("v1 must appear");
        let first_v0 = vessel_sequence
            .iter()
            .position(|&vv| vv == 0)
            .expect("v0 must appear");
        let _last_v0 = vessel_sequence
            .iter()
            .rposition(|&vv| vv == 0)
            .expect("v0 must appear");

        // Ensure contiguous grouping in ascending slack order: [v2...][v1...][v0...]
        assert!(
            last_v2 < first_v1 && last_v1 < first_v0,
            "Expected vessels grouped in ascending slack: v2(1) < v1(5) < v0(15). Got sequence: {:?}",
            vessel_sequence
        );

        // Within each vessel group, verify secondary ordering by increasing cost.
        let costs_v2: Vec<IntegerType> = per_decision
            .iter()
            .enumerate()
            .filter_map(|(i, (v, c))| if *v == 2 { Some((*v, *c, i)) } else { None })
            .map(|(_, c, _)| c)
            .collect();

        let costs_v1: Vec<IntegerType> = per_decision
            .iter()
            .enumerate()
            .filter_map(|(i, (v, c))| if *v == 1 { Some((*v, *c, i)) } else { None })
            .map(|(_, c, _)| c)
            .collect();

        let costs_v0: Vec<IntegerType> = per_decision
            .iter()
            .enumerate()
            .filter_map(|(i, (v, c))| if *v == 0 { Some((*v, *c, i)) } else { None })
            .map(|(_, c, _)| c)
            .collect();

        let mut sorted_v2 = costs_v2.clone();
        let mut sorted_v1 = costs_v1.clone();
        let mut sorted_v0 = costs_v0.clone();
        sorted_v2.sort_unstable();
        sorted_v1.sort_unstable();
        sorted_v0.sort_unstable();

        assert_eq!(
            costs_v2, sorted_v2,
            "Decisions for v2 must be ordered by increasing cost"
        );
        assert_eq!(
            costs_v1, sorted_v1,
            "Decisions for v1 must be ordered by increasing cost"
        );
        assert_eq!(
            costs_v0, sorted_v0,
            "Decisions for v0 must be ordered by increasing cost"
        );

        // Sanity check expected slack ranking values (not strictly necessary for order, but validates setup).
        assert_eq!(expected_slacks[2], 1);
        assert_eq!(expected_slacks[1], 5);
        assert_eq!(expected_slacks[0], 15);

        // Ensure infeasible pair (v2, b1) is not present (ProcessingTime::none).
        assert!(
            !decisions
                .iter()
                .any(|d| d.vessel_index().get() == 2 && d.berth_index().get() == 1),
            "Infeasible (v2,b1) must not be yielded"
        );
    }

    #[test]
    fn test_slack_filters_finish_after_deadline_and_empty_when_none() {
        // One vessel, two berths; both finishes exceed deadline
        let mut b = ModelBuilder::<IntegerType>::new(2, 1);
        // Arrival time 0, tight deadline 3
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(0), 3)
            .set_vessel_weight(VesselIndex::new(0), 1)
            // Durations 4 and 5 -> finishes 4 and 5, both > deadline 3 => infeasible
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(4),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::some(5),
            );
        let model = b.build();

        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = SlackHeuristicBuilder::<IntegerType>::new();

        let mut iter = builder.next_decision(&mut evaluator, &model, &state);
        assert_eq!(
            iter.next(),
            None,
            "Iterator should be empty when all finishes exceed deadline"
        );
    }
}
