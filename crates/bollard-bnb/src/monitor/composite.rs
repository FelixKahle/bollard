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

//! Monitoring combinators for tree search
//!
//! Provides `CompositeTreeSearchMonitor`, a fan‑out monitor that forwards every
//! event to its children. This lets you mix logging, metrics, visualization,
//! and early‑stopping without coupling them to the solver.
//!
//! Behavior
//! - Events are dispatched to child monitors in insertion order.
//! - `search_command` short‑circuits on the first non‑`Continue` response;
//!   put stricter stop conditions first.
//! - Other callbacks always fan out to all children.

use crate::{
    branching::decision::Decision,
    monitor::tree_search_monitor::{PruneReason, TreeSearchMonitor},
    state::SearchState,
    stats::BnbSolverStatistics,
};
use bollard_model::{model::Model, solution::Solution};
use bollard_search::monitor::search_monitor::SearchCommand;
use num_traits::{PrimInt, Signed};

/// A tree search monitor that aggregates multiple monitors and forwards events to all of them.
/// This allows combining different monitoring behaviors into a single monitor.
pub struct CompositeTreeSearchMonitor<'a, T>
where
    T: PrimInt + Signed,
{
    monitors: Vec<Box<dyn TreeSearchMonitor<T> + 'a>>,
}

impl<'a, T> Default for CompositeTreeSearchMonitor<'a, T>
where
    T: PrimInt + Signed,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> CompositeTreeSearchMonitor<'a, T>
where
    T: PrimInt + Signed,
{
    /// Creates a new empty `CompositeTreeSearchMonitor`.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            monitors: Vec::new(),
        }
    }

    /// Creates a new `CompositeTreeSearchMonitor` with the specified capacity.
    /// This pre-allocates space for the given number of monitors.
    #[inline(always)]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            monitors: Vec::with_capacity(capacity),
        }
    }

    /// Creates a new `CompositeTreeSearchMonitor` from a vector of boxed monitors.
    #[inline(always)]
    pub fn from_vec(monitors: Vec<Box<dyn TreeSearchMonitor<T>>>) -> Self {
        Self { monitors }
    }

    /// Adds a new monitor to the composite monitor.
    #[inline(always)]
    pub fn add_monitor<M>(&mut self, monitor: M)
    where
        M: TreeSearchMonitor<T> + 'a,
    {
        self.monitors.push(Box::new(monitor));
    }

    /// Adds a boxed monitor to the composite monitor.
    #[inline(always)]
    pub fn add_monitor_boxed(&mut self, monitor: Box<dyn TreeSearchMonitor<T> + 'a>) {
        self.monitors.push(monitor);
    }

    /// Returns a slice of the monitors contained in the composite monitor.
    #[inline(always)]
    pub fn monitors(&self) -> &[Box<dyn TreeSearchMonitor<T> + 'a>] {
        &self.monitors
    }

    /// Returns a mutable slice of the monitors contained in the composite monitor.
    #[inline(always)]
    pub fn monitors_mut(&mut self) -> &mut [Box<dyn TreeSearchMonitor<T> + 'a>] {
        &mut self.monitors
    }

    /// Clears all monitors from the composite monitor.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.monitors.clear();
    }

    /// Returns the number of monitors contained in the composite monitor.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.monitors.len()
    }

    /// Returns `true` if the composite monitor contains no monitors,
    /// `false` otherwise.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.monitors.is_empty()
    }
}

impl<'a, T> FromIterator<Box<dyn TreeSearchMonitor<T> + 'a>> for CompositeTreeSearchMonitor<'a, T>
where
    T: PrimInt + Signed,
{
    #[inline(always)]
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Box<dyn TreeSearchMonitor<T> + 'a>>,
    {
        Self {
            monitors: iter.into_iter().collect(),
        }
    }
}

impl<'a, T> TreeSearchMonitor<T> for CompositeTreeSearchMonitor<'a, T>
where
    T: PrimInt + Signed,
{
    #[inline(always)]
    fn name(&self) -> &str {
        "CompositeTreeSearchMonitor"
    }

    #[inline(always)]
    fn on_enter_search(&mut self, model: &Model<T>, statistics: &BnbSolverStatistics) {
        for monitor in &mut self.monitors {
            monitor.on_enter_search(model, statistics);
        }
    }

    #[inline(always)]
    fn on_exit_search(&mut self, statistics: &BnbSolverStatistics) {
        for monitor in &mut self.monitors {
            monitor.on_exit_search(statistics);
        }
    }

    #[inline(always)]
    fn search_command(
        &mut self,
        state: &SearchState<T>,
        statistics: &BnbSolverStatistics,
    ) -> SearchCommand {
        for monitor in &mut self.monitors {
            let cmd = monitor.search_command(state, statistics);
            // Short-circuit on the first non-Continue command
            if !matches!(cmd, SearchCommand::Continue) {
                return cmd;
            }
        }
        SearchCommand::Continue
    }

    #[inline(always)]
    fn on_step(&mut self, state: &SearchState<T>, statistics: &BnbSolverStatistics) {
        for monitor in &mut self.monitors {
            monitor.on_step(state, statistics);
        }
    }

    #[inline(always)]
    fn on_lower_bound_computed(
        &mut self,
        state: &SearchState<T>,
        lower_bound: T,
        estimated_remaining: T,
        statistics: &BnbSolverStatistics,
    ) {
        for monitor in &mut self.monitors {
            monitor.on_lower_bound_computed(state, lower_bound, estimated_remaining, statistics);
        }
    }

    #[inline(always)]
    fn on_prune(
        &mut self,
        state: &SearchState<T>,
        reason: PruneReason,
        statistics: &BnbSolverStatistics,
    ) {
        for monitor in &mut self.monitors {
            monitor.on_prune(state, reason.clone(), statistics);
        }
    }

    #[inline(always)]
    fn on_decisions_enqueued(
        &mut self,
        state: &SearchState<T>,
        count: usize,
        statistics: &BnbSolverStatistics,
    ) {
        for monitor in &mut self.monitors {
            monitor.on_decisions_enqueued(state, count, statistics);
        }
    }

    #[inline(always)]
    fn on_descend(
        &mut self,
        state: &SearchState<T>,
        decision: Decision<T>,
        statistics: &BnbSolverStatistics,
    ) {
        for monitor in &mut self.monitors {
            monitor.on_descend(state, decision, statistics);
        }
    }

    #[inline(always)]
    fn on_solution_found(&mut self, solution: &Solution<T>, statistics: &BnbSolverStatistics) {
        for monitor in &mut self.monitors {
            monitor.on_solution_found(solution, statistics);
        }
    }

    #[inline(always)]
    fn on_backtrack(&mut self, state: &SearchState<T>, statistics: &BnbSolverStatistics) {
        for monitor in &mut self.monitors {
            monitor.on_backtrack(state, statistics);
        }
    }
}
