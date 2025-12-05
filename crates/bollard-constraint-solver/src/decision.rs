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

//! # Search strategy and branching (Decision Builder pattern)
//!
//! This module defines the pluggable search policy for the constraint solver. It cleanly
//! separates the mechanics of search and propagation (owned by `Solver`/`SearchState`) from
//! the strategy for exploring the search space (owned by a `DecisionBuilder`).
//!
//! In other words:
//! - `Solver` + `SearchState`: maintain reversible state, propagate constraints, backtrack.
//! - `DecisionBuilder`: chooses which move(s) to try next and in what order.
//!
//! ## The Decision Builder pattern
//!
//! The `DecisionBuilder` trait produces an iterator of `Decision` values for the current
//! state. The solver tries them in order and relies on propagation/backtracking to explore
//! the induced search tree.
//!
//! This lets you plug in different exploration strategies without touching propagation:
//! - Exhaustive search (DFS/branch-and-bound): yield all feasible decisions at a node.
//! - Greedy/heuristic dives: yield only the best-looking decision to get fast incumbents.
//! - Randomized or adaptive strategies: keep RNG or scores in `self` and vary ordering.
//! - Meta-heuristics (e.g., LNS): restrict the iterator to a neighborhood of moves.
//!
//! Contract:
//! - The iterator may yield zero or more `Decision`s. Empty means “no move from here”.
//! - The solver will try decisions in the yielded order; ordering encodes your heuristic.
//! - `next_decision` must not mutate the `SearchState`; it only reads `model/state`.
//! - `&mut self` lets builders keep internal state (e.g., RNG seeds, caches, counters).
//!
//! ## Minimal `Decision`
//!
//! A `Decision` is just `(VesselIndex, BerthIndex)`. It intentionally omits times, costs,
//! or sequencing metadata:
//! - Implicit time via active schedule generation: after assigning a vessel to a berth,
//!   the solver schedules it at the earliest feasible start time (max(arrival, berth free)).
//! - Implicit sequence: the order vessels appear on a berth follows the order decisions
//!   are applied during search.
//! - Performance: two small integers copy cheaply and remain cache- and register-friendly.
//!
//! This keeps branching lightweight and makes state transitions fast.
//!
//! ## Returning iterators (not `Vec` or `Box<dyn Iterator>`)
//!
//! `DecisionBuilder` uses an associated type for its iterator:
//! - Memory efficiency: decisions are produced lazily, avoiding per-node heap allocation.
//! - Zero-cost abstraction: no dynamic dispatch or boxing; the compiler can inline and
//!   optimize the concrete iterator type.
//!
//! Practically, this reduces peak memory from O(width × depth) to O(depth) during search.

use crate::state::SearchState;
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use num_traits::{PrimInt, Signed};
use std::iter::FusedIterator;

/// A destinct decision in the decision tree.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Decision {
    vessel_index: VesselIndex,
    berth_index: BerthIndex,
}

impl Decision {
    /// Creates a new decision.
    #[inline]
    pub const fn new(vessel_index: VesselIndex, berth_index: BerthIndex) -> Self {
        Self {
            vessel_index,
            berth_index,
        }
    }

    /// Returns the vessel index of the decision.
    #[inline]
    pub const fn vessel_index(&self) -> VesselIndex {
        self.vessel_index
    }

    /// Returns the berth index of the decision.
    #[inline]
    pub const fn berth_index(&self) -> BerthIndex {
        self.berth_index
    }

    /// Consumes the decision and returns its inner indices.
    #[inline]
    pub const fn into_inner(self) -> (VesselIndex, BerthIndex) {
        (self.vessel_index, self.berth_index)
    }
}

impl std::fmt::Display for Decision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Decision(vessel_index: {}, berth_index: {})",
            self.vessel_index, self.berth_index
        )
    }
}

pub trait DecisionBuilder<T>
where
    T: PrimInt + Signed,
{
    /// The iterator over decisions of this decision builder.
    type DecisionIterator: Iterator<Item = Decision> + FusedIterator;

    /// Returns the name of the decision builder.
    /// This is mostly used for logging and debugging purposes.
    fn name(&self) -> &str;

    /// Returns the next decision to be made in the search.
    fn next_decision(
        &mut self,
        model: &Model<T>,
        search_state: &SearchState<T>,
    ) -> Self::DecisionIterator;
}

#[cfg(test)]
mod tests {
    use super::*;
    use bollard_model::model::ModelBuilder;

    type Int = i64;

    #[test]
    fn decision_new_getters_display_and_ordering() {
        let d1 = Decision::new(VesselIndex::new(0), BerthIndex::new(1));
        let d2 = Decision::new(VesselIndex::new(0), BerthIndex::new(2));
        let d3 = Decision::new(VesselIndex::new(1), BerthIndex::new(0));

        // Getters
        assert_eq!(d1.vessel_index().get(), 0);
        assert_eq!(d1.berth_index().get(), 1);

        // Display contains indices (TypedIndex prints as TagName(n))
        let s = d1.to_string();
        assert!(s.contains("Decision("));
        assert!(s.contains("vessel_index: VesselIndex(0)"));
        assert!(s.contains("berth_index: BerthIndex(1)"));

        // Equality
        assert_eq!(d1, Decision::new(VesselIndex::new(0), BerthIndex::new(1)));
        assert_ne!(d1, d2);

        // Ordering (derived: first by vessel_index, then by berth_index)
        let mut v = vec![d3.clone(), d1.clone(), d2.clone()];
        v.sort();
        assert_eq!(v, vec![d1, d2, d3]);
    }

    // A builder that yields no decisions.
    struct NoDecisionBuilder;

    impl<T: PrimInt + Signed> DecisionBuilder<T> for NoDecisionBuilder {
        type DecisionIterator = std::iter::Empty<Decision>;

        fn name(&self) -> &str {
            "None"
        }

        fn next_decision<'m, 's>(
            &mut self,
            _model: &'m Model<T>,
            _search_state: &'s SearchState<T>,
        ) -> Self::DecisionIterator {
            std::iter::empty()
        }
    }

    // A builder that yields exactly one decision.
    struct SingleDecisionBuilder {
        decision: Decision,
    }

    impl<T: PrimInt + Signed> DecisionBuilder<T> for SingleDecisionBuilder {
        type DecisionIterator = std::option::IntoIter<Decision>;

        fn name(&self) -> &str {
            "Single"
        }

        fn next_decision<'m, 's>(
            &mut self,
            _model: &'m Model<T>,
            _search_state: &'s SearchState<T>,
        ) -> Self::DecisionIterator {
            Some(self.decision.clone()).into_iter()
        }
    }

    // A builder that yields two decisions in a fixed order.
    struct TwoDecisionBuilder {
        a: Decision,
        b: Decision,
    }

    impl<T: PrimInt + Signed> DecisionBuilder<T> for TwoDecisionBuilder {
        type DecisionIterator = std::array::IntoIter<Decision, 2>;

        fn name(&self) -> &str {
            "Two"
        }

        fn next_decision<'m, 's>(
            &mut self,
            _model: &'m Model<T>,
            _search_state: &'s SearchState<T>,
        ) -> Self::DecisionIterator {
            let array = [self.a.clone(), self.b.clone()];
            IntoIterator::into_iter(array)
        }
    }

    fn small_model_and_state() -> (Model<Int>, SearchState<Int>) {
        // Build a tiny model; details are irrelevant for these tests,
        // but we need concrete references to satisfy the trait signature.
        let model = ModelBuilder::<Int>::new(2, 2).build();
        let state = SearchState::<Int>::new(model.num_berths(), model.num_vessels());
        (model, state)
        // Note: We don't rely on any particular field of model/state here.
    }

    #[test]
    fn builder_empty_iterator_is_fused() {
        let (model, state) = small_model_and_state();
        let mut builder = NoDecisionBuilder;

        let mut it = builder.next_decision(&model, &state);
        assert!(it.next().is_none());
        // FusedIterator: once None, always None
        assert!(it.next().is_none());
    }

    #[test]
    fn builder_single_decision_iterator_is_fused_and_yields_one() {
        let (model, state) = small_model_and_state();
        let d = Decision::new(VesselIndex::new(1), BerthIndex::new(0));
        let mut builder = SingleDecisionBuilder {
            decision: d.clone(),
        };

        let mut it = builder.next_decision(&model, &state);
        assert_eq!(it.next(), Some(d));
        assert!(it.next().is_none()); // Fused after exhaustion
    }

    #[test]
    fn builder_multiple_decisions_ordering_preserved() {
        let (model, state) = small_model_and_state();
        let d1 = Decision::new(VesselIndex::new(0), BerthIndex::new(1));
        let d2 = Decision::new(VesselIndex::new(1), BerthIndex::new(0));
        let mut builder = TwoDecisionBuilder {
            a: d1.clone(),
            b: d2.clone(),
        };

        let it = builder.next_decision(&model, &state);
        let decisions: Vec<_> = it.collect();
        assert_eq!(decisions, vec![d1, d2]);
    }

    #[test]
    fn builder_names_are_exposed_for_logging() {
        let none = NoDecisionBuilder;
        let single = SingleDecisionBuilder {
            decision: Decision::new(VesselIndex::new(0), BerthIndex::new(0)),
        };
        let two = TwoDecisionBuilder {
            a: Decision::new(VesselIndex::new(0), BerthIndex::new(0)),
            b: Decision::new(VesselIndex::new(0), BerthIndex::new(1)),
        };

        assert_eq!(
            <NoDecisionBuilder as DecisionBuilder<Int>>::name(&none),
            "None"
        );
        assert_eq!(
            <SingleDecisionBuilder as DecisionBuilder<Int>>::name(&single),
            "Single"
        );
        assert_eq!(
            <TwoDecisionBuilder as DecisionBuilder<Int>>::name(&two),
            "Two"
        );
    }
}
