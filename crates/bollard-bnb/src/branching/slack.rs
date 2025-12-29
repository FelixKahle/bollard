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

//! Slack‑guided best‑first branching
//!
//! Implements a decision builder that prioritizes vessels with the tightest
//! time slack to reduce the risk of infeasibility later in the search.
//!
//! For each unassigned vessel, feasible `(vessel, berth)` options are collected
//! as “rich decisions”, and the best‑case finish time is computed. Slack is
//! defined as `deadline − min_finish_time`. Vessels with smaller slack are
//! scheduled earlier.
//!
//! Global ordering is by ascending slack, then ascending cost, and finally
//! deterministic decision order. This produces stable, high‑quality branches
//! for branch‑and‑bound, improving pruning compared to naive enumeration.
//!
//! Produces a fused iterator of decisions; once exhausted, `next()` returns `None`.

use crate::{
    berth_availability::BerthAvailability,
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

/// Internal candidate structure for global sorting.
///
/// Stores the calculated slack and the full decision (which includes the cost).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SlackCandidate<T> {
    vessel_slack: T, // Primary Key (Ascending)
    decision: Decision<T>,
}

impl<T> SlackCandidate<T> {
    #[inline(always)]
    fn new(vessel_slack: T, decision: Decision<T>) -> Self {
        Self {
            vessel_slack,
            decision,
        }
    }
}

impl<T: Ord> Ord for SlackCandidate<T>
where
    T: SolverNumeric,
{
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        // Sort order:
        // 1. Ascending Slack (Tightest slack first)
        // 2. Ascending Cost (Cheaper move preferred)
        // 3. Deterministic Decision Order (Tie-break)
        self.vessel_slack
            .cmp(&other.vessel_slack)
            .then_with(|| self.decision.cost_delta().cmp(&other.decision.cost_delta()))
            .then_with(|| self.decision.cmp(&other.decision))
    }
}

impl<T: Ord> PartialOrd for SlackCandidate<T>
where
    T: SolverNumeric,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A decision builder that implements a **Slack-Guided Best-First** branching heuristic.
///
/// This builder prioritizes vessels with the tightest time slack (least flexibility) to reduce
/// the risk of missing feasible schedules. For each unassigned vessel:
/// 1. It scans all admissible berths and computes the earliest feasible finish time
///    (using the Rich Decision's `start_time` + `duration`).
/// 2. If at least one feasible option exists, it calculates the vessel’s slack:
///    `slack = deadline - min_finish_time`.
/// 3. It collects all feasible `(vessel, berth)` options as Rich Decisions.
/// 4. It sorts all global candidates by:
///    - primary: ascending `vessel_slack` (tightest first),
///    - secondary: ascending `cost_delta` (cheapest first),
///    - tertiary: deterministic `Decision` order.
#[derive(Debug, Clone, Default)]
pub struct SlackHeuristicBuilder<T> {
    candidates: Vec<SlackCandidate<T>>,
    scratch_options: Vec<Decision<T>>,
}

impl<T> SlackHeuristicBuilder<T> {
    #[inline]
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
            scratch_options: Vec::new(),
        }
    }

    #[inline]
    pub fn preallocated(num_berths: usize, num_vessels: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(num_berths * num_vessels),
            scratch_options: Vec::with_capacity(num_berths),
        }
    }

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
        berth_availability: &'a BerthAvailability<T>,
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

            let deadline = unsafe { model.vessel_latest_departure_time_unchecked(vessel_index) };

            self.scratch_options.clear();
            let mut best_case_finish_time = T::max_value();

            for b in 0..num_berths {
                let berth_index = BerthIndex::new(b);

                if let Some(decision) = unsafe {
                    Decision::try_new_unchecked(
                        vessel_index,
                        berth_index,
                        model,
                        berth_availability,
                        state,
                        evaluator,
                    )
                }
                    && let Some(duration) = unsafe {
                        model
                            .vessel_processing_time_unchecked(vessel_index, berth_index)
                            .into()
                    } {
                        let finish = decision.start_time() + duration;

                        // Slack-specific constraint: finish must not exceed deadline
                        if finish <= deadline {
                            if finish < best_case_finish_time {
                                best_case_finish_time = finish;
                            }
                            self.scratch_options.push(decision);
                        }
                    }
            }

            if self.scratch_options.is_empty() {
                continue;
            }

            let vessel_slack = deadline.saturating_sub_val(best_case_finish_time);

            for decision in self.scratch_options.iter() {
                self.candidates
                    .push(SlackCandidate::new(vessel_slack, *decision));
            }
        }

        self.candidates.sort_unstable();

        SlackHeuristicIter {
            iter: self.candidates.iter(),
        }
    }
}

/// A lightweight iterator that yields `Decision`s from the slack-sorted candidate slice.
pub struct SlackHeuristicIter<'a, T> {
    iter: std::slice::Iter<'a, SlackCandidate<T>>,
}

impl<'a, T: Copy> Iterator for SlackHeuristicIter<'a, T> {
    type Item = Decision<T>;
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
    fn build_slack_model() -> bollard_model::model::Model<IntegerType> {
        let mut b = ModelBuilder::<IntegerType>::new(2, 3);

        // Vessel 0
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(0), 20)
            .set_vessel_weight(VesselIndex::new(0), 3)
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
                ProcessingTime::none(),
            );

        b.build()
    }

    #[test]
    fn test_slack_orders_by_increasing_vessel_slack_then_cost() {
        let model = build_slack_model();
        let mut berth_availability = BerthAvailability::new();
        berth_availability.initialize(&model, &[]);
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = SlackHeuristicBuilder::<IntegerType>::new();

        let decisions: Vec<Decision<IntegerType>> = builder
            .next_decision(&mut evaluator, &model, &berth_availability, &state)
            .collect();

        assert!(
            !decisions.is_empty(),
            "Expected non-empty decision list for slack model"
        );

        // Gather per-decision vessel index and cost
        let per_decision: Vec<(usize, IntegerType)> = decisions
            .iter()
            .map(|d| (d.vessel_index().get(), d.cost_delta()))
            .collect();

        // Expected sequence logic similar to previous test
        // Verify primary ordering: vessel slack ascending.
        // That implies all decisions for v2 first, then v1, then v0.
        let vessel_sequence: Vec<usize> = per_decision.iter().map(|(v, _)| *v).collect();

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

        // Ensure contiguous grouping in ascending slack order: [v2...][v1...][v0...]
        assert!(
            last_v2 < first_v1 && last_v1 < first_v0,
            "Expected vessels grouped in ascending slack: v2(1) < v1(5) < v0(15). Got sequence: {:?}",
            vessel_sequence
        );

        // Within each vessel group, verify secondary ordering by increasing cost.
        let costs_v2: Vec<IntegerType> = per_decision
            .iter()
            .filter_map(|(v, c)| if *v == 2 { Some(*c) } else { None })
            .collect();

        let costs_v1: Vec<IntegerType> = per_decision
            .iter()
            .filter_map(|(v, c)| if *v == 1 { Some(*c) } else { None })
            .collect();

        let costs_v0: Vec<IntegerType> = per_decision
            .iter()
            .filter_map(|(v, c)| if *v == 0 { Some(*c) } else { None })
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
    }

    #[test]
    fn test_slack_filters_finish_after_deadline_and_empty_when_none() {
        let mut b = ModelBuilder::<IntegerType>::new(2, 1);
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(0), 3)
            .set_vessel_weight(VesselIndex::new(0), 1)
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
        let mut berth_availability = BerthAvailability::new();
        berth_availability.initialize(&model, &[]);
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = SlackHeuristicBuilder::<IntegerType>::new();

        let mut iter = builder.next_decision(&mut evaluator, &model, &berth_availability, &state);
        assert_eq!(
            iter.next(),
            None,
            "Iterator should be empty when all finishes exceed deadline"
        );
    }
}
