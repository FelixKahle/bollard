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

use crate::{monitor::local_search_monitor::LocalSearchMonitor, stats::LocalSearchStatistics};
use bollard_model::{model::Model, solution::Solution};
use bollard_search::{
    monitor::search_monitor::{SearchCommand, SearchMonitor},
    num::SolverNumeric,
};

/// A wrapper that adapts a generic `SearchMonitor` to the
/// `LocalSearchMonitor` trait.
///
/// This struct holds a mutable reference to a `SearchMonitor` and
/// forwards local search events to it, translating the lifecycle
/// callbacks appropriately. Some local search specific events are
/// no-ops since they do not have direct equivalents in the generic
/// monitor interface.
pub struct LocalSearchMonitorWrapper<'a, T>
where
    T: SolverNumeric,
{
    inner: &'a mut dyn SearchMonitor<T>,
    name: String,
}

impl<'a, T> LocalSearchMonitorWrapper<'a, T>
where
    T: SolverNumeric,
{
    /// Creates a new `LocalSearchMonitorWrapper` that wraps the given
    /// search monitor.
    #[inline]
    pub fn new(monitor: &'a mut dyn SearchMonitor<T>) -> Self {
        let name = format!("LocalSearchMonitorWrapper({})", monitor.name());
        Self {
            inner: monitor,
            name,
        }
    }

    /// Returns a reference to the inner `SearchMonitor`.
    #[inline]
    pub fn inner(&self) -> &dyn SearchMonitor<T> {
        self.inner
    }
}

impl<'a, T> LocalSearchMonitor<T> for LocalSearchMonitorWrapper<'a, T>
where
    T: SolverNumeric,
{
    #[inline(always)]
    fn name(&self) -> &str {
        &self.name
    }

    #[inline(always)]
    fn on_start(&mut self, model: &Model<T>, _initial_solution: &Solution<T>) {
        self.inner.on_enter_search(model);
    }

    #[inline(always)]
    fn on_end(
        &mut self,
        _model: &Model<T>,
        _best_solution: &Solution<T>,
        _statistics: &LocalSearchStatistics,
    ) {
        self.inner.on_exit_search();
    }

    #[inline(always)]
    fn on_iteration(
        &mut self,
        _model: &Model<T>,
        _current_solution: &Solution<T>,
        _statistics: &LocalSearchStatistics,
    ) {
        self.inner.on_step();
    }

    #[inline(always)]
    fn on_solution_found(
        &mut self,
        _model: &Model<T>,
        solution: &Solution<T>,
        _statistics: &LocalSearchStatistics,
    ) {
        self.inner.on_solution_found(solution);
    }

    #[inline(always)]
    fn on_solution_accepted(
        &mut self,
        _model: &Model<T>,
        _solution: &Solution<T>,
        _statistics: &LocalSearchStatistics,
    ) {
        // No op
    }

    #[inline(always)]
    fn on_solution_rejected(
        &mut self,
        _model: &Model<T>,
        _solution: &Solution<T>,
        _statistics: &LocalSearchStatistics,
    ) {
        // No op
    }

    #[inline(always)]
    fn on_best_solution_updated(
        &mut self,
        _model: &Model<T>,
        solution: &Solution<T>,
        _statistics: &LocalSearchStatistics,
    ) {
        self.inner.on_improvement_found(solution);
    }

    #[inline(always)]
    fn search_command(
        &mut self,
        _model: &Model<T>,
        _statistics: &LocalSearchStatistics,
    ) -> SearchCommand {
        self.inner.search_command()
    }
}
