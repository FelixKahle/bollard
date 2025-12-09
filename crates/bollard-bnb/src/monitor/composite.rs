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
    monitor::search_monitor::TreeSearchMonitor, state::SearchState, stats::BnbSolverStatistics,
};
use bollard_search::monitor::search_monitor::SearchCommand;
use num_traits::{PrimInt, Signed};

/// A composite monitor that aggregates multiple search monitors.
pub struct CompositeMonitor<'a, T> {
    monitors: Vec<Box<dyn TreeSearchMonitor<T> + 'a>>,
}

impl<'a, T> std::fmt::Debug for CompositeMonitor<'a, T>
where
    T: PrimInt + Signed,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let monitors = self.monitors.iter().map(|m| m.name()).collect::<Vec<_>>();
        f.debug_struct("CompositeMonitor")
            .field("monitors", &monitors)
            .finish()
    }
}

impl<'a, T> std::fmt::Display for CompositeMonitor<'a, T>
where
    T: PrimInt + Signed,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let monitors = self.monitors.iter().map(|m| m.name()).collect::<Vec<_>>();
        write!(f, "CompositeMonitor({})", monitors.join(", "))
    }
}

impl<'a, T> Default for CompositeMonitor<'a, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> CompositeMonitor<'a, T> {
    /// Creates a new composite monitor.
    #[inline]
    pub fn new() -> Self {
        Self {
            monitors: Vec::new(),
        }
    }

    /// Creates a new composite monitor with the specified capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            monitors: Vec::with_capacity(capacity),
        }
    }

    /// Adds a monitor to the composite.
    #[inline]
    pub fn add_monitor<M>(mut self, monitor: M) -> Self
    where
        T: PrimInt + Signed,
        M: TreeSearchMonitor<T> + 'a,
    {
        self.monitors.push(Box::new(monitor));
        self
    }

    /// Returns a reference to the list of monitors.
    #[inline]
    pub fn monitors(&self) -> &[Box<dyn TreeSearchMonitor<T> + 'a>] {
        &self.monitors
    }

    /// Returns an iterator over the monitors.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, Box<dyn TreeSearchMonitor<T> + 'a>> {
        self.monitors.iter()
    }
}

impl<'a, T> TreeSearchMonitor<T> for CompositeMonitor<'a, T>
where
    T: PrimInt + Signed,
{
    fn on_enter_search(&mut self, model: &bollard_model::model::Model<T>) {
        for monitor in &mut self.monitors {
            monitor.on_enter_search(model);
        }
    }

    fn check_termination(
        &mut self,
        state: &SearchState<T>,
        stats: &BnbSolverStatistics,
    ) -> SearchCommand {
        for monitor in &mut self.monitors {
            let command = monitor.check_termination(state, stats);
            if command != SearchCommand::Continue {
                return command;
            }
        }
        SearchCommand::Continue
    }

    fn on_solution(
        &mut self,
        solution: &bollard_model::solution::Solution<T>,
        stats: &BnbSolverStatistics,
    ) {
        for monitor in &mut self.monitors {
            monitor.on_solution(solution, stats);
        }
    }

    fn on_backtrack(&mut self, state: &SearchState<T>, stats: &BnbSolverStatistics) {
        for monitor in &mut self.monitors {
            monitor.on_backtrack(state, stats);
        }
    }

    fn on_descend(&mut self, state: &SearchState<T>, stats: &BnbSolverStatistics) {
        for monitor in &mut self.monitors {
            monitor.on_descend(state, stats);
        }
    }

    fn on_exit_search(&mut self, stats: &BnbSolverStatistics) {
        for monitor in &mut self.monitors {
            monitor.on_exit_search(stats);
        }
    }

    fn name(&self) -> &str {
        "CompositeMonitor"
    }
}
