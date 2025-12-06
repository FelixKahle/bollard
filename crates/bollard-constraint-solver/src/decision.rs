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

// decision.rs
pub trait DecisionBuilder<T>
where
    T: PrimInt + Signed,
{
    // The iterator output must live for 'a
    type DecisionIterator<'a>: Iterator<Item = Decision> + FusedIterator + 'a
    where
        Self: 'a,
        T: 'a;

    fn name(&self) -> &str;

    // We pass references that live for at least 'a.
    // The implementation will tie the returned iterator to these lifetimes.
    fn next_decision<'a>(
        &mut self,
        model: &'a Model<T>,
        search_state: &'a SearchState<T>,
    ) -> Self::DecisionIterator<'a>;
}
