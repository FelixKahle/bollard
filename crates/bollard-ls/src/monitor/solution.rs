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
    memory::Schedule, monitor::local_search_monitor::LocalSearchMonitor,
    stats::LocalSearchStatistics,
};
use bollard_search::monitor::search_monitor::SearchCommand;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SolutionLimitMonitor {
    limit: u64,
}

impl SolutionLimitMonitor {
    /// Create a new monitor that terminates once `total_solutions` >= `limit`.
    pub fn new(limit: u64) -> Self {
        Self { limit }
    }

    /// Returns the configured limit.
    pub fn limit(&self) -> u64 {
        self.limit
    }
}

impl<T> LocalSearchMonitor<T> for SolutionLimitMonitor
where
    T: bollard_search::num::SolverNumeric,
{
    fn name(&self) -> &str {
        "SolutionLimitMonitor"
    }

    fn on_start(&mut self, _initial_solution: &Schedule<T>) {
        // No-op
    }

    fn on_end(&mut self, _best_solution: &Schedule<T>, _statistics: &LocalSearchStatistics) {
        // No-op
    }

    fn on_iteration(
        &mut self,
        _current_solution: &Schedule<T>,
        _statistics: &LocalSearchStatistics,
    ) {
        // No-op
    }

    fn on_solution_found(&mut self, _solution: &Schedule<T>, _statistics: &LocalSearchStatistics) {
        // No-op
    }

    fn on_solution_accepted(
        &mut self,
        _solution: &Schedule<T>,
        _statistics: &LocalSearchStatistics,
    ) {
        // No-op
    }

    fn on_solution_rejected(
        &mut self,
        _solution: &Schedule<T>,
        _statistics: &LocalSearchStatistics,
    ) {
        // No-op
    }

    fn on_best_solution_updated(
        &mut self,
        _solution: &Schedule<T>,
        _statistics: &LocalSearchStatistics,
    ) {
        // No-op
    }

    fn search_command(&mut self, statistics: &LocalSearchStatistics) -> SearchCommand {
        if statistics.total_solutions >= self.limit {
            SearchCommand::Terminate(format!(
                "Solution limit reached: {} (total_solutions={})",
                self.limit, statistics.total_solutions
            ))
        } else {
            SearchCommand::Continue
        }
    }
}
