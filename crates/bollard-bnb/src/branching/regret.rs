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
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use bollard_search::num::SolverNumeric;
use std::cmp::Ordering;
use std::iter::FusedIterator;

/// Represents a single feasible assignment option for a specific vessel.
#[derive(Debug, Clone, Copy)]
struct FeasibleOption<T> {
    cost: T,
    decision: Decision,
}

/// Internal candidate structure for global sorting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RegretCandidate<T> {
    vessel_regret: T, // Primary Key (Descending)
    cost: T,          // Secondary Key (Ascending)
    decision: Decision,
}

impl<T> RegretCandidate<T> {
    #[inline(always)]
    fn new(vessel_regret: T, cost: T, decision: Decision) -> Self {
        Self {
            vessel_regret,
            cost,
            decision,
        }
    }
}

impl<T: Ord> Ord for RegretCandidate<T> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .vessel_regret
            .cmp(&self.vessel_regret)
            .then_with(|| self.cost.cmp(&other.cost))
            .then_with(|| self.decision.cmp(&other.decision))
    }
}

impl<T: Ord> PartialOrd for RegretCandidate<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A decision builder that implements a **Regret-Guided Best-First** branching heuristic.
///
/// This builder focuses on vessels where choosing suboptimally would hurt most.
/// For each unassigned vessel:
/// 1. It collects all feasible `(vessel, berth)` options, each with an immediate `cost` as
///    reported by the evaluator (e.g., weighted flow time increment).
/// 2. It sorts the vessel’s local options by ascending `cost`.
/// 3. It computes the vessel’s regret:
///    - If the vessel has at least two feasible options:
///      `regret = second_best_cost - best_cost` (a larger gap implies higher risk).
///    - If the vessel has only one feasible option:
///      `regret = T::max_value()` (must-assign now; no alternative).
/// 4. It promotes all of the vessel’s feasible options into a global candidate list,
///    annotated with the common vessel regret and their individual costs.
/// 5. It globally sorts candidates by:
///    - primary: descending `vessel_regret` (most critical vessels first),
///    - secondary: ascending `cost` (prefer cheaper options within a vessel),
///    - tertiary: deterministic `Decision` order (tie-breaker).
///
/// This “choose the most critical vessel first” strategy helps avoid painting the
/// search into corners where future choices become infeasible or expensive.
///
/// Notes:
/// - Feasibility is determined by `Decision::try_new_unchecked` and evaluator returns.
/// - Vessels with no feasible options contribute nothing to the candidates.
/// - The regret calculation is robust to single-option vessels (treated as highest regret).
/// - Sorting uses `sort_unstable` over a total order and yields deterministic iteration.
///
/// See also:
/// - `WsptHeuristicBuilder`: purely cost-guided best-first branching.
/// - `SlackHeuristicBuilder`: prioritizes vessels by tightest time slack.
#[derive(Debug, Clone, Default)]
pub struct RegretHeuristicBuilder<T> {
    candidates: Vec<RegretCandidate<T>>,
    scratch_options: Vec<FeasibleOption<T>>,
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
    T: SolverNumeric + num_traits::Bounded,
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

                if !unsafe { model.vessel_allowed_on_berth_unchecked(vessel_index, berth_index) } {
                    continue;
                }

                if let Some(decision) =
                    unsafe { Decision::try_new_unchecked(vessel_index, berth_index, model, state) }
                {
                    let berth_free_time = unsafe { state.berth_free_time_unchecked(berth_index) };

                    if let Some(cost) = unsafe {
                        evaluator.evaluate_vessel_assignment_unchecked(
                            model,
                            vessel_index,
                            berth_index,
                            berth_free_time,
                        )
                    } {
                        self.scratch_options.push(FeasibleOption { cost, decision });
                    }
                }
            }

            if self.scratch_options.is_empty() {
                continue;
            }

            self.scratch_options
                .sort_unstable_by(|a, b| a.cost.cmp(&b.cost));

            let best_cost = self.scratch_options[0].cost;

            let regret = if self.scratch_options.len() > 1 {
                self.scratch_options[1].cost - best_cost
            } else {
                T::max_value()
            };

            for option in self.scratch_options.iter() {
                self.candidates
                    .push(RegretCandidate::new(regret, option.cost, option.decision));
            }
        }

        self.candidates.sort_unstable();

        RegretHeuristicIter {
            iter: self.candidates.iter(),
        }
    }
}

/// A lightweight iterator that yields `Decision`s from the regret-sorted candidate slice.
///
/// Properties:
/// - Borrowing: holds a reference into the builder’s internal buffer; no allocations.
/// - Order: yields decisions sorted by descending vessel regret, then cost, then decision.
/// - Fused: once exhausted, subsequent calls return `None`.
pub struct RegretHeuristicIter<'a, T> {
    iter: std::slice::Iter<'a, RegretCandidate<T>>,
}

impl<'a, T: Copy> Iterator for RegretHeuristicIter<'a, T> {
    type Item = Decision;
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
    //
    // Costs are shaped via weights and processing times under WeightedFlowTimeEvaluator.
    fn build_regret_model() -> bollard_model::model::Model<IntegerType> {
        let mut b = ModelBuilder::<IntegerType>::new(3, 3);

        // Vessel 0 (small regret): weight 2
        // b0: proc 5 -> finish 5 -> cost ~ (finish * weight) => around 10
        // b1: proc 6 -> finish 6 -> cost ~ 12
        // b2: none
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(0), IntegerType::MAX)
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
            .set_vessel_latest_departure_time(VesselIndex::new(1), IntegerType::MAX)
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
            .set_vessel_latest_departure_time(VesselIndex::new(2), IntegerType::MAX)
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
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = RegretHeuristicBuilder::<IntegerType>::new();

        let decisions: Vec<Decision> = builder
            .next_decision(&mut evaluator, &model, &state)
            .by_ref()
            .collect();

        assert!(
            !decisions.is_empty(),
            "Expected non-empty decision list for regret model"
        );

        // Gather per-decision vessel index and cost
        let per_decision: Vec<(usize, IntegerType)> = decisions
            .iter()
            .map(|d| {
                let v = d.vessel_index();
                let b = d.berth_index();
                let ready = unsafe { state.berth_free_time_unchecked(b) };
                let cost = evaluator
                    .evaluate_vessel_assignment(&model, v, b, ready)
                    .expect("decision must be evaluable");
                (v.get(), cost)
            })
            .collect();

        // Determine the expected regret per vessel from the model setup:
        // v0 options costs ~ {10, 12} -> regret = 2
        // v1 options costs ~ {6, 30} -> regret = 24
        // v2 only one option         -> regret = MAX
        let regret_v0 = 12 - 10; // 2
        let regret_v1 = 30 - 6; // 24
        let regret_v2 = IntegerType::MAX;

        // Expected ordering by regret descending: v2(MAX) first, then v1(24), then v0(2)
        let vessel_sequence: Vec<usize> = per_decision.iter().map(|(v, _)| *v).collect();

        let _first_v2 = vessel_sequence
            .iter()
            .position(|&vv| vv == 2)
            .expect("v2 present");
        let last_v2 = vessel_sequence
            .iter()
            .rposition(|&vv| vv == 2)
            .expect("v2 present");
        let first_v1 = vessel_sequence
            .iter()
            .position(|&vv| vv == 1)
            .expect("v1 present");
        let last_v1 = vessel_sequence
            .iter()
            .rposition(|&vv| vv == 1)
            .expect("v1 present");
        let first_v0 = vessel_sequence
            .iter()
            .position(|&vv| vv == 0)
            .expect("v0 present");
        let _last_v0 = vessel_sequence
            .iter()
            .rposition(|&vv| vv == 0)
            .expect("v0 present");

        // Ensure contiguous grouping and descending regret: [v2...][v1...][v0...]
        assert!(
            last_v2 < first_v1 && last_v1 < first_v0,
            "Expected vessels grouped by descending regret: v2(MAX) > v1(24) > v0(2). Got sequence: {:?}",
            vessel_sequence
        );

        // Within each vessel group, verify ascending cost
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

        // Ensure infeasible pairs (ProcessingTime::none) are not present
        assert!(
            !decisions
                .iter()
                .any(|d| d.vessel_index().get() == 0 && d.berth_index().get() == 2),
            "Infeasible (v0,b2) must not be yielded"
        );
        assert!(
            !decisions
                .iter()
                .any(|d| d.vessel_index().get() == 1 && d.berth_index().get() == 2),
            "Infeasible (v1,b2) must not be yielded"
        );
        assert!(
            !decisions
                .iter()
                .any(|d| d.vessel_index().get() == 2 && d.berth_index().get() == 0),
            "Infeasible (v2,b0) must not be yielded"
        );
        assert!(
            !decisions
                .iter()
                .any(|d| d.vessel_index().get() == 2 && d.berth_index().get() == 2),
            "Infeasible (v2,b2) must not be yielded"
        );

        // Sanity check computed regrets
        assert_eq!(regret_v0, 2);
        assert_eq!(regret_v1, 24);
        assert_eq!(regret_v2, IntegerType::MAX);
    }

    #[test]
    fn test_regret_empty_when_no_feasible_candidates() {
        // One vessel, two berths, all infeasible processing times
        let mut b = ModelBuilder::<IntegerType>::new(2, 1);
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(0), IntegerType::MAX)
            .set_vessel_weight(VesselIndex::new(0), 1)
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

        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = RegretHeuristicBuilder::<IntegerType>::new();

        let mut iter = builder.next_decision(&mut evaluator, &model, &state);
        assert_eq!(
            iter.next(),
            None,
            "Iterator should be empty when no feasible assignments exist"
        );
    }
}
