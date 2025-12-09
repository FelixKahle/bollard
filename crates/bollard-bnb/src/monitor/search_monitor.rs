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

use crate::{state::SearchState, stats::BnbSolverStatistics};
use bollard_model::{model::Model, solution::Solution};
use num_traits::{PrimInt, Signed};

/// Command returned by the monitor to control the search process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchCommand {
    /// Stop the search process.
    Continue,
    /// Continue the search process.
    Stop,
}

impl std::fmt::Display for SearchCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchCommand::Continue => write!(f, "Continue"),
            SearchCommand::Stop => write!(f, "Stop"),
        }
    }
}

/// Trait for monitoring and controlling the search process of the solver.
pub trait TreeSearchMonitor<T>: Send + Sync
where
    T: PrimInt + Signed,
{
    /// Called once before the search loop begins.
    fn on_enter_search(&mut self, model: &Model<T>);

    /// Called at the beginning of every search loop iteration.
    /// This is the primary place to check for time limits or iteration limits.
    fn check_termination(
        &mut self,
        _state: &SearchState<T>,
        _stats: &BnbSolverStatistics,
    ) -> SearchCommand {
        SearchCommand::Continue
    }

    /// Called when a complete valid solution is found.
    /// The monitor can decide to stop the search here (e.g., "find first")
    /// by returning `SearchCommand::Stop`.
    fn on_solution(&mut self, solution: &Solution<T>, stats: &BnbSolverStatistics);

    /// Called when the solver backtracks (moves up the tree).
    fn on_backtrack(&mut self, state: &SearchState<T>, stats: &BnbSolverStatistics);

    /// Called just before a new child node is pushed onto the stack.
    /// Useful for visualizing the tree shape.
    fn on_descend(&mut self, state: &SearchState<T>, stats: &BnbSolverStatistics);

    /// Called when the search is finished (either exhausted or stopped).
    fn on_exit_search(&mut self, stats: &BnbSolverStatistics);

    /// Returns the name of the monitor.
    fn name(&self) -> &str;
}

impl<T> std::fmt::Debug for dyn TreeSearchMonitor<T>
where
    T: PrimInt + Signed,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SearchMonitor({})", self.name())
    }
}

impl<T> std::fmt::Display for dyn TreeSearchMonitor<T>
where
    T: PrimInt + Signed,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SearchMonitor({})", self.name())
    }
}
