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

/// Internal candidate structure for global sorting.
///
/// Stores the calculated regret and the full decision (which includes the cost).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RegretCandidate<T> {
    vessel_regret: T, // Primary Key (Descending)
    decision: Decision<T>,
}

impl<T> RegretCandidate<T> {
    #[inline(always)]
    fn new(vessel_regret: T, decision: Decision<T>) -> Self {
        Self {
            vessel_regret,
            decision,
        }
    }
}

impl<T> Ord for RegretCandidate<T>
where
    T: SolverNumeric,
{
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        // Sort order:
        // 1. Descending Regret (Higher regret = more critical to assign now)
        // 2. Ascending Cost (Cheaper move preferred)
        // 3. Deterministic Decision Order (Tie-break)
        other
            .vessel_regret
            .cmp(&self.vessel_regret)
            .then_with(|| self.decision.cost_delta().cmp(&other.decision.cost_delta()))
            .then_with(|| self.decision.cmp(&other.decision))
    }
}

impl<T> PartialOrd for RegretCandidate<T>
where
    T: SolverNumeric,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A decision builder that implements a **Regret-Guided Best-First** branching heuristic.
///
/// This builder focuses on vessels where choosing suboptimally would hurt most.
/// For each unassigned vessel:
/// 1. It collects all feasible `(vessel, berth)` options as Rich Decisions.
/// 2. It sorts the vessel’s local options by ascending `cost`.
/// 3. It computes the vessel’s regret:
///    - If >= 2 options: `regret = second_best_cost - best_cost`.
///    - If 1 option: `regret = T::max_value()` (Must assign now).
/// 4. It promotes all options to a global list, sorted by descending regret.
#[derive(Debug, Clone, Default)]
pub struct RegretHeuristicBuilder<T> {
    candidates: Vec<RegretCandidate<T>>,
    scratch_options: Vec<Decision<T>>,
}

impl<T> RegretHeuristicBuilder<T> {
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
}

impl<T, E> DecisionBuilder<T, E> for RegretHeuristicBuilder<T>
where
    T: SolverNumeric,
    E: ObjectiveEvaluator<T>,
{
    type DecisionIterator<'a>
        = RegretHeuristicIter<'a, T>
    where
        T: 'a,
        E: 'a,
        Self: 'a;

    fn name(&self) -> &str {
        "RegretHeuristicBuilder"
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

            self.scratch_options.clear();

            for b in 0..num_berths {
                let berth_index = BerthIndex::new(b);

                // Use Rich Decision pipeline to validate, schedule, and price.
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
                    self.scratch_options.push(decision);
                }
            }

            if self.scratch_options.is_empty() {
                continue;
            }

            // Sort local options by cost to find best and second-best
            self.scratch_options.sort_unstable_by_key(|a| a.cost_delta());

            let best_cost = self.scratch_options[0].cost_delta();

            let regret = if self.scratch_options.len() > 1 {
                self.scratch_options[1].cost_delta() - best_cost
            } else {
                T::max_value()
            };

            for decision in self.scratch_options.iter() {
                self.candidates
                    .push(RegretCandidate::new(regret, *decision));
            }
        }

        self.candidates.sort_unstable();

        RegretHeuristicIter {
            iter: self.candidates.iter(),
        }
    }
}

/// A lightweight iterator that yields `Decision`s from the regret-sorted candidate slice.
pub struct RegretHeuristicIter<'a, T> {
    iter: std::slice::Iter<'a, RegretCandidate<T>>,
}

impl<'a, T: Copy> Iterator for RegretHeuristicIter<'a, T> {
    type Item = Decision<T>;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|c| c.decision)
    }
}

impl<'a, T: Copy> FusedIterator for RegretHeuristicIter<'a, T> {}

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

    // Build a small model to produce distinct regret values:
    // - 3 vessels, 3 berths
    // - All arrivals at 0, berths initially free
    //
    // Vessel 0: two feasible options with close costs -> small regret
    // Vessel 1: two feasible options with large gap    -> large regret
    // Vessel 2: only one feasible option               -> regret = MAX
    fn build_regret_model() -> bollard_model::model::Model<IntegerType> {
        let mut b = ModelBuilder::<IntegerType>::new(3, 3);

        // Vessel 0 (small regret): weight 2
        // b0: proc 5 -> cost ~ 10
        // b1: proc 6 -> cost ~ 12
        // b2: none
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_weight(VesselIndex::new(0), 2)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(5),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::some(6),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(2),
                ProcessingTime::none(),
            );

        // Vessel 1 (large regret): weight 3
        // b0: proc 2 -> cost ~ 6
        // b1: proc 10 -> cost ~ 30
        // b2: none
        b.set_vessel_arrival_time(VesselIndex::new(1), 0)
            .set_vessel_weight(VesselIndex::new(1), 3)
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(0),
                ProcessingTime::some(2),
            )
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(1),
                ProcessingTime::some(10),
            )
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(2),
                ProcessingTime::none(),
            );

        // Vessel 2 (single option -> regret MAX): weight 1
        // b0: none
        // b1: proc 4 -> cost ~ 4
        // b2: none
        b.set_vessel_arrival_time(VesselIndex::new(2), 0)
            .set_vessel_weight(VesselIndex::new(2), 1)
            .set_vessel_processing_time(
                VesselIndex::new(2),
                BerthIndex::new(0),
                ProcessingTime::none(),
            )
            .set_vessel_processing_time(
                VesselIndex::new(2),
                BerthIndex::new(1),
                ProcessingTime::some(4),
            )
            .set_vessel_processing_time(
                VesselIndex::new(2),
                BerthIndex::new(2),
                ProcessingTime::none(),
            );

        b.build()
    }

    #[test]
    fn test_regret_orders_by_descending_regret_then_ascending_cost() {
        let model = build_regret_model();
        let mut berth_availability = BerthAvailability::new();
        berth_availability.initialize(&model, &[]);
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = RegretHeuristicBuilder::<IntegerType>::new();

        let decisions: Vec<Decision<IntegerType>> = builder
            .next_decision(&mut evaluator, &model, &berth_availability, &state)
            .collect();

        assert!(
            !decisions.is_empty(),
            "Expected non-empty decision list for regret model"
        );

        // Gather per-decision vessel index and cost delta directly from decision
        let per_decision: Vec<(usize, IntegerType)> = decisions
            .iter()
            .map(|d| (d.vessel_index().get(), d.cost_delta()))
            .collect();

        // Expected Regrets:
        // v0: cost {10, 12} -> regret = 2
        // v1: cost {6, 30}  -> regret = 24
        // v2: cost {4}      -> regret = MAX

        // Expected global order: v2 -> v1 -> v0
        let vessel_sequence: Vec<usize> = per_decision.iter().map(|(v, _)| *v).collect();

        let last_v2 = vessel_sequence
            .iter()
            .rposition(|&v| v == 2)
            .expect("v2 present");
        let first_v1 = vessel_sequence
            .iter()
            .position(|&v| v == 1)
            .expect("v1 present");
        let last_v1 = vessel_sequence
            .iter()
            .rposition(|&v| v == 1)
            .expect("v1 present");
        let first_v0 = vessel_sequence
            .iter()
            .position(|&v| v == 0)
            .expect("v0 present");

        // Verify grouping: [v2...][v1...][v0...]
        assert!(
            last_v2 < first_v1 && last_v1 < first_v0,
            "Expected vessels grouped by descending regret: v2(MAX) > v1(24) > v0(2). Got: {:?}",
            vessel_sequence
        );

        // Verify internal sort (ascending cost)
        // v1 options: cost 6 (b0) then cost 30 (b1)
        let v1_costs: Vec<IntegerType> = per_decision
            .iter()
            .filter(|(v, _)| *v == 1)
            .map(|(_, c)| *c)
            .collect();
        assert_eq!(v1_costs, vec![6, 30], "v1 options should be sorted by cost");
    }

    #[test]
    fn test_regret_empty_when_no_feasible_candidates() {
        // One vessel, two berths, all infeasible processing times
        let mut b = ModelBuilder::<IntegerType>::new(2, 1);
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::none(),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::none(),
            );
        let model = b.build();
        let mut berth_availability = BerthAvailability::new();
        berth_availability.initialize(&model, &[]);

        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = RegretHeuristicBuilder::<IntegerType>::new();

        let mut iter = builder.next_decision(&mut evaluator, &model, &berth_availability, &state);
        assert_eq!(
            iter.next(),
            None,
            "Iterator should be empty when no feasible assignments exist"
        );
    }
}
