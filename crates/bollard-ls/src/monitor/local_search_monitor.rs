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

//! Monitoring interface for local search runs.
//!
//! This module defines callbacks for observing the lifecycle of the solver,
//! including start/end events, per‑iteration updates, and notifications on
//! solutions found, accepted, or rejected. Implementations can stream logs,
//! collect metrics, or trigger early termination by returning a search
//! command to the engine. The default `search_command` continues execution,
//! allowing monitors to remain lightweight unless an explicit limit or
//! condition is reached.

use crate::{memory::Schedule, stats::LocalSearchStatistics};
use bollard_search::{monitor::search_monitor::SearchCommand, num::SolverNumeric};

/// A monitor for local search algorithms.
pub trait LocalSearchMonitor<T>
where
    T: SolverNumeric,
{
    /// Returns the name of the monitor.
    fn name(&self) -> &str;

    /// Called at the start of the local search.
    fn on_start(&mut self, initial_solution: &Schedule<T>);

    /// Called at the end of the local search.
    fn on_end(&mut self, best_solution: &Schedule<T>, statistics: &LocalSearchStatistics);

    /// Called at each iteration of the local search.
    fn on_iteration(&mut self, current_solution: &Schedule<T>, statistics: &LocalSearchStatistics);

    /// Called when a solution is found.
    fn on_solution_found(&mut self, solution: &Schedule<T>, statistics: &LocalSearchStatistics);

    /// Called when a solution is accepted.
    fn on_solution_accepted(&mut self, solution: &Schedule<T>, statistics: &LocalSearchStatistics);

    /// Called when a solution is rejected.
    fn on_solution_rejected(&mut self, solution: &Schedule<T>, statistics: &LocalSearchStatistics);

    fn on_best_solution_updated(
        &mut self,
        solution: &Schedule<T>,
        statistics: &LocalSearchStatistics,
    );

    /// Determines the command for the next step of the local search.
    fn search_command(&mut self, _statistics: &LocalSearchStatistics) -> SearchCommand {
        SearchCommand::Continue
    }
}

impl<T> std::fmt::Debug for dyn LocalSearchMonitor<T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LocalSearchMonitor {{ name: {} }}", self.name())
    }
}

impl<T> std::fmt::Display for dyn LocalSearchMonitor<T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LocalSearchMonitor: {}", self.name())
    }
}
