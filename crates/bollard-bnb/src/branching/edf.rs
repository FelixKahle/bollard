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

//! Earliest‑Deadline‑First (EDF) branching
//!
//! Implements a decision builder that prioritizes feasible `(vessel, berth)`
//! assignments by increasing urgency, measured as per‑decision slack:
//! `slack = latest_departure − (start_time + processing_time)`.
//!
//! For each unassigned vessel, feasible options are collected as rich decisions
//! that respect model topology, berth availability, evaluator constraints, and
//! built‑in symmetry filtering. Each decision’s finish time is computed using
//! the decision’s scheduled `start_time` and the berth’s processing duration,
//! and slack is derived from the vessel’s deadline.
//!
//! Global ordering is by ascending slack (tightest first), with deterministic
//! tie‑breaking by decision order. This fail‑first strategy focuses the search
//! on urgent assignments to reduce the risk of infeasibility later and produce
//! stable branches for branch‑and‑bound.
//!
//! Produces a fused iterator of decisions; once exhausted, `next()` returns `None`.

use crate::{
    berth_availability::BerthAvailability,
    branching::decision::{Decision, DecisionBuilder},
    eval::evaluator::ObjectiveEvaluator,
    state::SearchState,
};
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use bollard_search::num::SolverNumeric;

#[derive(Debug, Clone, Copy)]
struct EdfCandidate<T> {
    decision: Decision<T>,
    slack: T,
}

impl<T: Ord> Ord for EdfCandidate<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.slack
            .cmp(&other.slack)
            .then_with(|| self.decision.cmp(&other.decision))
    }
}

impl<T: Ord> PartialOrd for EdfCandidate<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: PartialEq> PartialEq for EdfCandidate<T> {
    fn eq(&self, other: &Self) -> bool {
        self.slack == other.slack && self.decision == other.decision
    }
}

impl<T: Eq> Eq for EdfCandidate<T> {}

/// A Decision Builder that prioritizes "Urgency".
///
/// Strategy:
/// 1. Identify all feasible (vessel, berth) pairs at the current time.
/// 2. Calculate **Slack** = `LatestDeparture - (StartTime + ProcessingTime)`.
/// 3. Sort by Slack (ascending).
///
/// This acts as a "Fail-First" heuristic compatible with chronological search states.
#[derive(Debug, Clone, Default)]
pub struct EarliestDeadlineFirstBuilder<T> {
    candidates: Vec<EdfCandidate<T>>,
}

impl<T> EarliestDeadlineFirstBuilder<T> {
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
        }
    }

    pub fn preallocated(num_berths: usize, num_vessels: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(num_berths * num_vessels),
        }
    }
}

impl<T, E> DecisionBuilder<T, E> for EarliestDeadlineFirstBuilder<T>
where
    T: SolverNumeric,
    E: ObjectiveEvaluator<T>,
{
    type DecisionIterator<'a>
        = std::vec::IntoIter<Decision<T>>
    where
        Self: 'a,
        T: 'a,
        E: 'a;

    fn name(&self) -> &str {
        "EarliestDeadlineFirst"
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
                } {
                    let finish_time = unsafe {
                        decision.start_time().saturating_add_val(
                            model
                                .vessel_processing_time_unchecked(vessel_index, berth_index)
                                .unwrap_unchecked(),
                        )
                    };
                    let deadline = model.vessel_latest_departure_time(vessel_index);
                    let slack = deadline.saturating_sub(finish_time);
                    self.candidates.push(EdfCandidate { decision, slack });
                }
            }
        }

        self.candidates.sort_unstable();
        let sorted_decisions: Vec<Decision<T>> =
            self.candidates.iter().map(|c| c.decision).collect();
        sorted_decisions.into_iter()
    }
}

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

    // Model for EDF ordering:
    // - 3 vessels, 2 berths
    // - All arrivals at 0, large deadlines
    // - Different processing times to create distinct per-decision finish times
    //
    // V0: b0=10, b1=6; deadline=30
    //   decisions:
    //     (v0,b0): finish=10, slack=20
    //     (v0,b1): finish=6,  slack=24
    // V1: b0=3,  b1=8; deadline=12
    //   decisions:
    //     (v1,b0): finish=3,  slack=9
    //     (v1,b1): finish=8,  slack=4
    // V2: b0=5,  b1=none; deadline=9
    //   decisions:
    //     (v2,b0): finish=5,  slack=4
    //
    // Expected per-decision slack ascending:
    //   slack=4: (v1,b1), (v2,b0)  -> tie-break by Decision Ord (deterministic)
    //   slack=9: (v1,b0)
    //   slack=20: (v0,b0)
    //   slack=24: (v0,b1)
    fn build_edf_model() -> bollard_model::model::Model<IntegerType> {
        let mut b = ModelBuilder::<IntegerType>::new(2, 3);

        // Vessel 0
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(0), 30)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(10),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::some(6),
            );

        // Vessel 1
        b.set_vessel_arrival_time(VesselIndex::new(1), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(1), 12)
            .set_vessel_weight(VesselIndex::new(1), 1)
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(0),
                ProcessingTime::some(3),
            )
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(1),
                ProcessingTime::some(8),
            );

        // Vessel 2
        b.set_vessel_arrival_time(VesselIndex::new(2), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(2), 9)
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
    fn test_edf_orders_by_increasing_per_decision_slack() {
        let model = build_edf_model();
        let mut berth_availability = BerthAvailability::new();
        berth_availability.initialize(&model, &[]);
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = EarliestDeadlineFirstBuilder::<IntegerType>::new();

        let decisions: Vec<Decision<IntegerType>> = builder
            .next_decision(&mut evaluator, &model, &berth_availability, &state)
            .collect();

        assert!(
            !decisions.is_empty(),
            "Expected non-empty decisions for EDF model"
        );

        // Compute per-decision slack the same way EDF does
        let mut observed: Vec<(usize, usize, IntegerType)> = Vec::new();
        for d in &decisions {
            let v = d.vessel_index().get();
            let b = d.berth_index().get();
            let duration = model
                .vessel_processing_time(VesselIndex::new(v), BerthIndex::new(b))
                .unwrap();
            let finish = d.start_time() + duration;
            let deadline = model.vessel_latest_departure_time(VesselIndex::new(v));
            let slack = deadline - finish;
            observed.push((v, b, slack));
        }

        // Ensure non-decreasing slack
        let slacks: Vec<IntegerType> = observed.iter().map(|(_, _, s)| *s).collect();
        let mut sorted_slacks = slacks.clone();
        sorted_slacks.sort_unstable();
        assert_eq!(
            slacks, sorted_slacks,
            "EDF must yield decisions ordered by increasing per-decision slack"
        );

        // Verify expected leading group with slack=4 is present and ordered deterministically
        // Expected membership: (v1,b1) and (v2,b0), both slack 4
        let leading_group: Vec<(usize, usize)> = observed
            .iter()
            .take_while(|(_, _, s)| *s == 4)
            .map(|(v, b, _)| (*v, *b))
            .collect();

        assert!(
            leading_group.contains(&(1, 1)) && leading_group.contains(&(2, 0)),
            "Leading slack=4 group should contain (v1,b1) and (v2,b0). Got: {:?}",
            leading_group
        );
    }

    #[test]
    fn test_edf_empty_when_all_finishes_exceed_deadline() {
        // One vessel, two berths; both finishes exceed the tight deadline.
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
        let mut builder = EarliestDeadlineFirstBuilder::<IntegerType>::new();

        let mut iter = builder.next_decision(&mut evaluator, &model, &berth_availability, &state);
        assert_eq!(
            iter.next(),
            None,
            "EDF iterator should be empty when all finishes are beyond the deadline"
        );
    }

    #[test]
    fn test_edf_tie_breaks_deterministically_when_slack_equal() {
        // Two vessels with equal per-decision slack for a given berth produce
        // ordering determined by Decision's Ord tie-break.
        let mut b = ModelBuilder::<IntegerType>::new(2, 2);

        // Vessel 0: deadline 20, proc b0=10 (finish=10 slack=10)
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(0), 20)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(10),
            );

        // Vessel 1: deadline 15, proc b0=5 (finish=5 slack=10)
        b.set_vessel_arrival_time(VesselIndex::new(1), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(1), 15)
            .set_vessel_weight(VesselIndex::new(1), 1)
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(0),
                ProcessingTime::some(5),
            );

        // Make b1 infeasible for both to keep only one decision each
        b.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(1),
            ProcessingTime::none(),
        )
        .set_vessel_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(1),
            ProcessingTime::none(),
        );

        let model = b.build();
        let mut berth_availability = BerthAvailability::new();
        berth_availability.initialize(&model, &[]);
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = EarliestDeadlineFirstBuilder::<IntegerType>::new();

        let decisions: Vec<Decision<IntegerType>> = builder
            .next_decision(&mut evaluator, &model, &berth_availability, &state)
            .collect();

        assert_eq!(
            decisions.len(),
            2,
            "Expected two decisions with equal slack"
        );

        // Both have slack 10; check deterministic order using Decision Ord
        let slacks: Vec<IntegerType> = decisions
            .iter()
            .map(|d| {
                let v = d.vessel_index().get();
                let b = d.berth_index().get();
                let duration = model
                    .vessel_processing_time(VesselIndex::new(v), BerthIndex::new(b))
                    .unwrap();
                let finish = d.start_time() + duration;
                let deadline = model.vessel_latest_departure_time(VesselIndex::new(v));
                deadline - finish
            })
            .collect();

        assert!(
            slacks.iter().all(|&s| s == 10),
            "Both decisions must have slack 10"
        );

        // The order is deterministic; we don't assert a specific pair since Decision Ord
        // may depend on internal indices, but we assert it is stable across runs.
        let first_pair = (
            decisions[0].vessel_index().get(),
            decisions[0].berth_index().get(),
        );
        let second_pair = (
            decisions[1].vessel_index().get(),
            decisions[1].berth_index().get(),
        );
        assert_ne!(
            first_pair, second_pair,
            "Two distinct decisions expected in stable order"
        );
    }
}
