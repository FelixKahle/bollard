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

//! Longest‑processing‑time branching
//!
//! Implements a decision builder that explores feasible `(vessel, berth)`
//! assignments by decreasing processing time, favoring longer jobs first.
//! Decisions are constructed through the same rich pipeline that enforces
//! model feasibility, berth availability, and objective evaluator checks,
//! then globally ordered so that the largest `p_ij` takes precedence.
//!
//! When processing times coincide, immediate objective cost acts as a
//! secondary guide, and ties are broken deterministically by decision
//! indices to ensure stable traversal.
//!
//! This strategy can accelerate discovery of strong incumbents when long
//! tasks dominate the objective landscape, improving pruning compared to
//! arbitrary ordering.
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

/// Candidate for LPT sorting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LptCandidate<T> {
    duration: T, // Primary Key (Descending)
    decision: Decision<T>,
}

impl<T: Ord + SolverNumeric> Ord for LptCandidate<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // 1. Longest duration first (Descending)
        other
            .duration
            .cmp(&self.duration)
            // 2. Cheapest move next
            .then_with(|| self.decision.cost_delta().cmp(&other.decision.cost_delta()))
            // 3. Deterministic tie-break
            .then_with(|| self.decision.cmp(&other.decision))
    }
}

impl<T: Ord + SolverNumeric> PartialOrd for LptCandidate<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Implements the **Longest Processing Time (LPT)** heuristic.
///
/// It prioritizes assignments where the processing time $p_{ij}$ is maximal.
#[derive(Debug, Clone, Default)]
pub struct LptHeuristicBuilder<T> {
    candidates: Vec<LptCandidate<T>>,
}

impl<T> LptHeuristicBuilder<T> {
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

impl<T, E> DecisionBuilder<T, E> for LptHeuristicBuilder<T>
where
    T: SolverNumeric,
    E: ObjectiveEvaluator<T>,
{
    type DecisionIterator<'a>
        = LptHeuristicIter<'a, T>
    where
        Self: 'a,
        T: 'a,
        E: 'a;

    fn name(&self) -> &str {
        "LptHeuristicBuilder"
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
                    self.candidates.push(LptCandidate {
                        duration: processing,
                        decision,
                    });
                }
            }
        }
        self.candidates.sort_unstable(); // Sorts descending due to Ord impl
        LptHeuristicIter {
            iter: self.candidates.iter(),
        }
    }
}

pub struct LptHeuristicIter<'a, T> {
    iter: std::slice::Iter<'a, LptCandidate<T>>,
}

impl<'a, T: Copy> Iterator for LptHeuristicIter<'a, T> {
    type Item = Decision<T>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|c| c.decision)
    }
}
impl<'a, T: Copy> FusedIterator for LptHeuristicIter<'a, T> {}

#[cfg(test)]
mod tests {
    use crate::{
        berth_availability::BerthAvailability,
        branching::decision::{Decision, DecisionBuilder},
        // Assuming LptHeuristicBuilder is exported or available here
        branching::lpt::LptHeuristicBuilder,
        eval::wtft::WeightedFlowTimeEvaluator,
        state::SearchState,
    };
    use bollard_model::{
        index::{BerthIndex, VesselIndex},
        model::ModelBuilder,
        time::ProcessingTime,
    };

    type IntegerType = i64;

    /// Helper: Same structure as SPT, but utilized to verify DESCENDING order.
    /// Durations: V0=10, V1=100, V2=2.
    fn build_lpt_model() -> bollard_model::model::Model<IntegerType> {
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
    fn test_lpt_orders_by_descending_duration() {
        let model = build_lpt_model();
        let mut berth_availability = BerthAvailability::new();
        berth_availability.initialize(&model, &[]);
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();

        let mut builder = LptHeuristicBuilder::<IntegerType>::new();

        let decisions: Vec<Decision<IntegerType>> = builder
            .next_decision(&mut evaluator, &model, &berth_availability, &state)
            .collect();

        assert_eq!(decisions.len(), 3, "Expected 3 decisions");

        // Expected Order: 1 (dur=100) -> 0 (dur=10) -> 2 (dur=2)
        assert_eq!(
            decisions[0].vessel_index().get(),
            1,
            "1st: Longest duration (100)"
        );
        assert_eq!(
            decisions[1].vessel_index().get(),
            0,
            "2nd: Medium duration (10)"
        );
        assert_eq!(
            decisions[2].vessel_index().get(),
            2,
            "3rd: Shortest duration (2)"
        );
    }

    #[test]
    fn test_lpt_chooses_longest_berth_for_single_vessel() {
        // 1 Vessel, 2 Berths.
        // Berth 0 is slow (20), Berth 1 is fast (5).
        // LPT should prioritize the SLOW berth (Duration 20) over the fast one.
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

        let mut builder = LptHeuristicBuilder::<IntegerType>::new();
        let decisions: Vec<Decision<IntegerType>> = builder
            .next_decision(&mut eval, &model, &ba, &state)
            .collect();

        assert_eq!(
            decisions[0].berth_index().get(),
            0,
            "Should prioritize slow Berth 0 (Duration 20)"
        );
        assert_eq!(
            decisions[1].berth_index().get(),
            1,
            "Should put fast Berth 1 last"
        );
    }

    #[test]
    fn test_lpt_handles_infeasible_processing_times() {
        // V0 has Some(10) on B0, but None on B1.
        let mut b = ModelBuilder::<IntegerType>::new(2, 1);
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(10),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::none(),
            );

        let model = b.build();
        let mut ba = BerthAvailability::new();
        ba.initialize(&model, &[]);
        let state = SearchState::<IntegerType>::new(2, 1);
        let mut eval = WeightedFlowTimeEvaluator::<IntegerType>::new();

        let mut builder = LptHeuristicBuilder::<IntegerType>::new();
        let decisions: Vec<Decision<IntegerType>> = builder
            .next_decision(&mut eval, &model, &ba, &state)
            .collect();

        assert_eq!(decisions.len(), 1, "Should only have 1 valid decision");
        assert_eq!(decisions[0].berth_index().get(), 0);
    }
}
