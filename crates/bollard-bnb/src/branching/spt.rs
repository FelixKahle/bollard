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

//! Shortest‑processing‑time branching
//!
//! Implements a decision builder that explores feasible `(vessel, berth)`
//! assignments by increasing processing time, prioritizing shorter jobs.
//! Feasible candidates are built as rich decisions respecting the model,
//! berth availability, and evaluator rules, then ordered so that the
//! smallest `p_ij` is considered first.
//!
//! When processing times match, immediate objective cost serves as a
//! secondary preference, with deterministic tie‑breaking by decision
//! indices to maintain stable iteration order.
//!
//! This best‑first, duration‑oriented traversal helps find low‑impact
//! moves early, often sharpening bounds and pruning in branch‑and‑bound.
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
use std::cmp::Ordering;
use std::iter::FusedIterator;

/// Candidate for SPT sorting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SptCandidate<T> {
    duration: T, // Primary Key (Ascending)
    decision: Decision<T>,
}

impl<T: Ord + SolverNumeric> Ord for SptCandidate<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // 1. Shortest duration first
        self.duration
            .cmp(&other.duration)
            // 2. Cheapest move next
            .then_with(|| self.decision.cost_delta().cmp(&other.decision.cost_delta()))
            // 3. Deterministic tie-break
            .then_with(|| self.decision.cmp(&other.decision))
    }
}

impl<T: Ord + SolverNumeric> PartialOrd for SptCandidate<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Implements the **Shortest Processing Time (SPT)** heuristic from the paper.
///
/// It prioritizes assignments where the processing time $p_{ij}$ is minimal.
#[derive(Debug, Clone, Default)]
pub struct SptHeuristicBuilder<T> {
    candidates: Vec<SptCandidate<T>>,
}

impl<T> SptHeuristicBuilder<T> {
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

impl<T, E> DecisionBuilder<T, E> for SptHeuristicBuilder<T>
where
    T: SolverNumeric,
    E: ObjectiveEvaluator<T>,
{
    type DecisionIterator<'a>
        = SptHeuristicIter<'a, T>
    where
        Self: 'a,
        T: 'a,
        E: 'a;

    fn name(&self) -> &str {
        "SptHeuristicBuilder"
    }

    fn next_decision<'a>(
        &'a mut self,
        evaluator: &'a mut E,
        model: &'a Model<T>,
        ba: &'a BerthAvailability<T>,
        state: &'a SearchState<T>,
    ) -> Self::DecisionIterator<'a> {
        self.candidates.clear();
        let num_vessels = model.num_vessels();
        let num_berths = model.num_berths();

        let last_decision_time = state.last_decision_time();
        let last_decision_vessel = state.last_decision_vessel();

        for vessel_index in 0..num_vessels {
            let vessel = VesselIndex::new(vessel_index);
            if unsafe { state.is_vessel_assigned_unchecked(vessel) } {
                continue;
            }

            for berth_index in 0..num_berths {
                let berth = BerthIndex::new(berth_index);

                let processing_opt =
                    unsafe { model.vessel_processing_time_unchecked(vessel, berth) };

                if processing_opt.is_none() {
                    continue;
                }

                let processing = processing_opt.unwrap_unchecked();

                if let Some(decision) = unsafe {
                    Decision::try_new_unchecked(vessel, berth, model, ba, state, evaluator)
                } {
                    if decision.start_time() < last_decision_time {
                        continue;
                    }

                    if decision.start_time() == last_decision_time
                        && decision.vessel_index() < last_decision_vessel
                    {
                        continue;
                    }

                    self.candidates.push(SptCandidate {
                        duration: processing,
                        decision,
                    });
                }
            }
        }

        self.candidates.sort_unstable(); // Ascending duration

        SptHeuristicIter {
            iter: self.candidates.iter(),
        }
    }
}

pub struct SptHeuristicIter<'a, T> {
    iter: std::slice::Iter<'a, SptCandidate<T>>,
}

impl<'a, T: Copy> Iterator for SptHeuristicIter<'a, T> {
    type Item = Decision<T>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|c| c.decision)
    }
}
impl<'a, T: Copy> FusedIterator for SptHeuristicIter<'a, T> {}

#[cfg(test)]
mod tests {
    use crate::{
        berth_availability::BerthAvailability,
        branching::decision::{Decision, DecisionBuilder},
        // Assuming SptHeuristicBuilder is exported or available here
        branching::spt::SptHeuristicBuilder,
        eval::wtft::WeightedFlowTimeEvaluator,
        state::SearchState,
    };
    use bollard_model::{
        index::{BerthIndex, VesselIndex},
        model::ModelBuilder,
        time::ProcessingTime,
    };

    type IntegerType = i64;

    /// Helper: Builds a 1-Berth, 3-Vessel model to isolate duration sorting.
    /// Durations: V0=10, V1=100, V2=2.
    fn build_spt_model() -> bollard_model::model::Model<IntegerType> {
        let mut b = ModelBuilder::<IntegerType>::new(1, 3);

        // Vessel 0: Duration 10
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(10),
            );

        // Vessel 1: Duration 100
        b.set_vessel_arrival_time(VesselIndex::new(1), 0)
            .set_vessel_weight(VesselIndex::new(1), 1)
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(0),
                ProcessingTime::some(100),
            );

        // Vessel 2: Duration 2
        b.set_vessel_arrival_time(VesselIndex::new(2), 0)
            .set_vessel_weight(VesselIndex::new(2), 1)
            .set_vessel_processing_time(
                VesselIndex::new(2),
                BerthIndex::new(0),
                ProcessingTime::some(2),
            );

        b.build()
    }

    #[test]
    fn test_spt_orders_by_ascending_duration() {
        let model = build_spt_model();
        let mut berth_availability = BerthAvailability::new();
        berth_availability.initialize(&model, &[]);
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();

        let mut builder = SptHeuristicBuilder::<IntegerType>::new();

        let decisions: Vec<Decision<IntegerType>> = builder
            .next_decision(&mut evaluator, &model, &berth_availability, &state)
            .collect();

        assert_eq!(decisions.len(), 3, "Expected 3 decisions");

        // Expected Order: 2 (dur=2) -> 0 (dur=10) -> 1 (dur=100)
        assert_eq!(
            decisions[0].vessel_index().get(),
            2,
            "1st: Shortest duration (2)"
        );
        assert_eq!(
            decisions[1].vessel_index().get(),
            0,
            "2nd: Medium duration (10)"
        );
        assert_eq!(
            decisions[2].vessel_index().get(),
            1,
            "3rd: Longest duration (100)"
        );
    }

    #[test]
    fn test_spt_chooses_fastest_berth_for_single_vessel() {
        // 1 Vessel, 2 Berths.
        // Berth 0 is slow (20), Berth 1 is fast (5).
        let mut b = ModelBuilder::<IntegerType>::new(2, 1);
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(20),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::some(5),
            );

        let model = b.build();
        let mut ba = BerthAvailability::new();
        ba.initialize(&model, &[]);
        let state = SearchState::<IntegerType>::new(2, 1);
        let mut eval = WeightedFlowTimeEvaluator::<IntegerType>::new();

        let mut builder = SptHeuristicBuilder::<IntegerType>::new();
        let decisions: Vec<Decision<IntegerType>> = builder
            .next_decision(&mut eval, &model, &ba, &state)
            .collect();

        assert_eq!(
            decisions[0].berth_index().get(),
            1,
            "Should prioritize fast Berth 1"
        );
        assert_eq!(
            decisions[1].berth_index().get(),
            0,
            "Should put slow Berth 0 last"
        );
    }

    #[test]
    fn test_spt_tie_breaking_by_cost() {
        // Two vessels, same duration (10).
        // V0 has Weight 1 (Cost ~10), V1 has Weight 10 (Cost ~100).
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
                ProcessingTime::some(10),
            );

        let model = b.build();
        let mut ba = BerthAvailability::new();
        ba.initialize(&model, &[]);
        let state = SearchState::<IntegerType>::new(1, 2);
        let mut eval = WeightedFlowTimeEvaluator::<IntegerType>::new();

        let mut builder = SptHeuristicBuilder::<IntegerType>::new();
        let decisions: Vec<Decision<IntegerType>> = builder
            .next_decision(&mut eval, &model, &ba, &state)
            .collect();

        // Duration tied -> Check Cost. V0 is cheaper.
        assert_eq!(decisions[0].vessel_index().get(), 0, "Tie: Pick cheaper V0");
        assert_eq!(
            decisions[1].vessel_index().get(),
            1,
            "Tie: Pick expensive V1 last"
        );
    }
}
