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

//! Adapter for external search monitors
//!
//! `WrapperMonitor` bridges this crate’s `TreeSearchMonitor` with a generic
//! `SearchMonitor` from `bollard_search`. It forwards lifecycle events and
//! commands to the inner monitor while ignoring tree‑specific callbacks.
//!
//! Behavior
//! - Delegates: enter, step, solution, exit, and `search_command`.
//! - No‑ops: prune, descend, backtrack, lower‑bound, and decisions‑enqueued.
//! - `name()` is `WrapperMonitor(inner.name())`.
//! - Holds `&mut dyn SearchMonitor<T>`; lifetime‑bound, single owner.

use crate::{
    branching::decision::Decision,
    monitor::tree_search_monitor::{PruneReason, TreeSearchMonitor},
    state::SearchState,
    stats::BnbSolverStatistics,
};
use bollard_model::{model::Model, solution::Solution};
use bollard_search::monitor::search_monitor::{SearchCommand, SearchMonitor};
use num_traits::{PrimInt, Signed};

/// A wrapper tree search monitor, that wraps a general
/// search monitor.
pub struct WrapperMonitor<'a, T> {
    inner: &'a mut dyn SearchMonitor<T>,
    name: String,
}

impl<'a, T> WrapperMonitor<'a, T> {
    /// Creates a new `WrapperMonitor` that wraps the given
    /// search monitor.
    #[inline(always)]
    pub fn new(inner: &'a mut dyn SearchMonitor<T>) -> Self
    where
        T: PrimInt + Signed,
    {
        let name = format!("WrapperMonitor({})", inner.name());
        Self { inner, name }
    }
}

impl<'a, T> TreeSearchMonitor<T> for WrapperMonitor<'a, T>
where
    T: PrimInt + Signed,
{
    #[inline(always)]
    fn name(&self) -> &str {
        &self.name
    }

    #[inline(always)]
    fn on_enter_search(&mut self, model: &Model<T>, _statistics: &BnbSolverStatistics) {
        self.inner.on_enter_search(model);
    }

    #[inline(always)]
    fn on_solution_found(&mut self, solution: &Solution<T>, _statistics: &BnbSolverStatistics) {
        self.inner.on_solution_found(solution);
    }

    #[inline(always)]
    fn on_backtrack(&mut self, _state: &SearchState<T>, _statistics: &BnbSolverStatistics) {}

    #[inline(always)]
    fn on_descend(
        &mut self,
        _state: &SearchState<T>,
        _decision: Decision<T>,
        _statistics: &BnbSolverStatistics,
    ) {
    }

    #[inline(always)]
    fn on_exit_search(&mut self, _statistics: &BnbSolverStatistics) {
        self.inner.on_exit_search();
    }

    #[inline(always)]
    fn search_command(
        &mut self,
        _state: &SearchState<T>,
        _statistics: &BnbSolverStatistics,
    ) -> SearchCommand {
        self.inner.search_command()
    }

    #[inline(always)]
    fn on_step(&mut self, _state: &SearchState<T>, _statistics: &BnbSolverStatistics) {
        self.inner.on_step();
    }

    #[inline(always)]
    fn on_lower_bound_computed(
        &mut self,
        _state: &SearchState<T>,
        _lower_bound: T,
        _estimated_remaining: T,
        _statistics: &BnbSolverStatistics,
    ) {
    }

    #[inline(always)]
    fn on_prune(
        &mut self,
        _state: &SearchState<T>,
        _reason: PruneReason,
        _statistics: &BnbSolverStatistics,
    ) {
    }

    #[inline(always)]
    fn on_decisions_enqueued(
        &mut self,
        _state: &SearchState<T>,
        _count: usize,
        _statistics: &BnbSolverStatistics,
    ) {
    }
}
