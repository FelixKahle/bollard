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

pub struct LocalSearchMonitorWrapper<T, M>
where
    T: bollard_search::num::SolverNumeric,
    M: SearchMonitor<T>,
{
    inner: M,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, M> LocalSearchMonitorWrapper<T, M>
where
    T: SolverNumeric,
    M: SearchMonitor<T>,
{
    #[inline]
    pub fn new(monitor: M) -> Self {
        Self {
            inner: monitor,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn inner(&self) -> &M {
        &self.inner
    }
}

impl<T, M> LocalSearchMonitor<T> for LocalSearchMonitorWrapper<T, M>
where
    T: SolverNumeric,
    M: SearchMonitor<T> + Send + Sync,
{
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn on_start(&mut self, model: &Model<T>, _initial_solution: &Solution<T>) {
        self.inner.on_enter_search(model);
    }

    fn on_end(
        &mut self,
        _model: &Model<T>,
        _best_solution: &Solution<T>,
        _statistics: &LocalSearchStatistics,
    ) {
        self.inner.on_exit_search();
    }

    fn on_iteration(
        &mut self,
        _model: &Model<T>,
        _current_solution: &Solution<T>,
        _statistics: &LocalSearchStatistics,
    ) {
        self.inner.on_step();
    }

    fn on_solution_found(
        &mut self,
        _model: &Model<T>,
        solution: &Solution<T>,
        _statistics: &LocalSearchStatistics,
    ) {
        self.inner.on_solution_found(solution);
    }

    fn on_solution_accepted(
        &mut self,
        _model: &Model<T>,
        _solution: &Solution<T>,
        _statistics: &LocalSearchStatistics,
    ) {
        // No op
    }

    fn on_solution_rejected(
        &mut self,
        _model: &Model<T>,
        _solution: &Solution<T>,
        _statistics: &LocalSearchStatistics,
    ) {
        // No op
    }

    fn on_best_solution_updated(
        &mut self,
        _model: &Model<T>,
        _solution: &Solution<T>,
        _statistics: &LocalSearchStatistics,
    ) {
        // No op
    }

    fn search_command(
        &mut self,
        _model: &Model<T>,
        _statistics: &LocalSearchStatistics,
    ) -> SearchCommand {
        self.inner.search_command()
    }
}
