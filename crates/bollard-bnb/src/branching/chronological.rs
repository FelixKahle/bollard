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
use bollard_core::num::constants::MinusOne;
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use num_traits::{PrimInt, Signed};
use std::iter::FusedIterator;

/// A decision builder that generates a **chronological, canonical, multi-branch
/// search tree** of feasible `(vessel, berth)` assignments.
///
/// The tree is built incrementally: each node represents a partial schedule,
/// and each outgoing edge represents assigning the next admissible
/// `(vessel → berth)` pair according to chronological ordering and symmetry rules.
///
/// ### Structural Guarantees
///
/// - **Non-decreasing time along every branch**
///   (`start_time(next) ≥ start_time(previous)`).
/// - **Symmetry breaking between identical berths**
///   (equivalent schedules are never generated).
/// - **Deterministic ordering** of generated children (row-major vessel × berth).
/// - **Non-binary branching**: each node may have 0..B children depending on feasibility.
///
/// The resulting search space is significantly smaller than the naive full
/// permutation tree, while still covering all canonical optimal schedules.
///
/// ---
///
/// ### Example Search Tree (Unicode Box Layout)
///
/// Instance:
/// ```text
/// Vessels:  v0, v1
/// Berths:   b0, b1
/// Arrival:  all at t=0
/// Processing times: identical across berths
/// Both berths initially free
/// ```
///
/// Resulting exploration tree (ignoring pruning and cost bounds):
///
/// ```text
/// (root)
/// ├── v0 → b0
/// │   ├── v1 → b0
/// │   └── v1 → b1
/// └── v1 → b0
///      └── v0 → b1
/// ```
///
/// **Interpretation:**
///
/// - At the root, `v0→b1` and `v1→b1` are *not generated* because `b0` and `b1`
///   are symmetric at time zero with identical processing rules. Only the first
///   canonical representative berth (`b0`) is allowed.
///
/// - After scheduling `v0→b0`, symmetry disappears (berths now differ in free
///   time), so both placements of `v1` become distinct and valid branches.
///
/// - In the right subtree (`v1→b0`), the only valid follow-up is `v0→b1`,
///   because the chronological/tie-breaking rule rejects equal-time assignments
///   where a lower-indexed vessel would appear *after* a higher-indexed one.
///
/// ---
///
/// ### Why this Matters
///
/// This builder creates a **reduced exact search tree**:
///
/// - large symmetric subtrees are eliminated at the branching level, not later;
/// - chronological monotonicity prevents infeasible temporal reorderings;
/// - deterministic traversal ensures reproducible search and stable pruning.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ChronologicalExhaustiveBuilder;

impl<T, E> DecisionBuilder<T, E> for ChronologicalExhaustiveBuilder
where
    T: PrimInt + Signed + MinusOne,
    E: ObjectiveEvaluator<T>,
{
    type DecisionIterator<'a>
        = ExhaustiveIter<'a, T>
    where
        T: 'a,
        E: 'a,
        Self: 'a;

    fn name(&self) -> &str {
        "ChronologicalExhaustiveBuilder"
    }

    fn next_decision<'a>(
        &'a mut self,
        _evaluator: &'a mut E,
        model: &'a Model<T>,
        state: &'a SearchState<T>,
    ) -> Self::DecisionIterator<'a> {
        ExhaustiveIter {
            current_vessel: VesselIndex::new(0),
            current_berth: BerthIndex::new(0),
            model,
            state,
        }
    }
}

/// Iterator over feasible `(vessel, berth)` decisions in a chronological,
/// canonical, exhaustive order.
///
/// This iterator traverses the assignment space in row-major order
/// (vessels × berths), skipping already-assigned vessels and yielding only
/// decisions deemed feasible by the model and current search state.
/// It is the concrete iterator returned by
/// `ChronologicalExhaustiveBuilder::next_decision`.
///
/// Properties:
/// - Row-major iteration: vessels are advanced outer-most; berths wrap modulo `num_berths`.
/// - Skips vessels already assigned in `state`.
/// - Yields only feasible `Decision`s (as determined by `Decision::try_new_unchecked`).
/// - Fused: once it returns `None`, subsequent calls also return `None`.
///
/// Chronological guarantee:
/// - Time moves forward along any branch. Feasibility checks in `Decision`
///   enforce non-decreasing start times and tie-breaking rules
///   (see `branching/decision.rs`: `try_new`, `try_new_unchecked`,
///   and tests such as `test_try_new_enforces_chronological_order_by_start_time`).
#[derive(Debug, Clone)]
pub struct ExhaustiveIter<'a, T>
where
    T: PrimInt + Signed + MinusOne,
{
    current_vessel: VesselIndex,
    current_berth: BerthIndex,
    model: &'a Model<T>,
    state: &'a SearchState<T>,
}

impl<'a, T> Iterator for ExhaustiveIter<'a, T>
where
    T: PrimInt + Signed + MinusOne,
{
    type Item = Decision;

    fn next(&mut self) -> Option<Self::Item> {
        let num_vessels = self.model.num_vessels();
        let num_berths = self.model.num_berths();

        // Iterate vessels × berths in row-major order using typed indices
        while self.current_vessel.get() < num_vessels {
            let vessel_index = self.current_vessel;
            let berth_index = self.current_berth;

            // Advance for the next call using index arithmetic with wrap on berths
            self.current_berth = (self.current_berth + 1usize) % num_berths;
            if self.current_berth.get() == 0 {
                self.current_vessel += 1usize;
            }

            // Skip already assigned vessels
            if unsafe { self.state.is_vessel_assigned_unchecked(vessel_index) } {
                continue;
            }

            if let Some(decision) = unsafe {
                Decision::try_new_unchecked(vessel_index, berth_index, self.model, self.state)
            } {
                return Some(decision);
            }
        }

        None
    }
}

impl<'a, T> FusedIterator for ExhaustiveIter<'a, T> where T: PrimInt + Signed + MinusOne {}

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

    // Build a small model: 2 berths × 3 vessels with explicit feasibility.
    // v0: feasible on b0, b1
    // v1: infeasible on b0, feasible on b1
    // v2: feasible on b0, infeasible on b1
    fn build_small_model() -> bollard_model::model::Model<IntegerType> {
        let mut mb = ModelBuilder::<IntegerType>::new(2, 3);

        // Vessel 0
        mb.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(0), IntegerType::MAX)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(5),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::some(7),
            );

        // Vessel 1
        mb.set_vessel_arrival_time(VesselIndex::new(1), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(1), IntegerType::MAX)
            .set_vessel_weight(VesselIndex::new(1), 1)
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(0),
                ProcessingTime::none(), // infeasible on b0
            )
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(1),
                ProcessingTime::some(3),
            );

        // Vessel 2
        mb.set_vessel_arrival_time(VesselIndex::new(2), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(2), IntegerType::MAX)
            .set_vessel_weight(VesselIndex::new(2), 1)
            .set_vessel_processing_time(
                VesselIndex::new(2),
                BerthIndex::new(0),
                ProcessingTime::some(4),
            )
            .set_vessel_processing_time(
                VesselIndex::new(2),
                BerthIndex::new(1),
                ProcessingTime::none(), // infeasible on b1
            );

        mb.build()
    }

    #[test]
    fn test_chronological_iter_yields_row_major_feasible_pairs() {
        let model = build_small_model();
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());

        // Evaluator is unused by the chronological builder but required by the trait.
        let mut eval = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = ChronologicalExhaustiveBuilder;

        let mut iter = builder.next_decision(&mut eval, &model, &state);
        let decisions: Vec<Decision> = iter.by_ref().collect();
        assert!(
            !decisions.is_empty(),
            "Expected at least one feasible decision"
        );

        // Expect row-major order over vessels×berths with feasibility filtering applied.
        let actual_pairs: Vec<(usize, usize)> = decisions
            .iter()
            .map(|d| (d.vessel_index().get(), d.berth_index().get()))
            .collect();

        // v0,b0 (feasible), v0,b1 (feasible),
        // v1,b0 (infeasible, skipped), v1,b1 (feasible),
        // v2,b0 (feasible), v2,b1 (infeasible, skipped)
        let expected_pairs = vec![(0, 0), (0, 1), (1, 1), (2, 0)];
        assert_eq!(
            actual_pairs, expected_pairs,
            "Iterator must yield feasible decisions in row-major order"
        );
    }

    #[test]
    fn test_chronological_iter_is_fused() {
        let model = build_small_model();
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());

        let mut eval = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = ChronologicalExhaustiveBuilder;
        let mut it = builder.next_decision(&mut eval, &model, &state);

        // Exhaust the iterator
        while it.next().is_some() {}
        // Subsequent calls must return None
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None);
    }
}
