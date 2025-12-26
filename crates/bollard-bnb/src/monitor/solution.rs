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

//! Solution limit monitor for tree search
//!
//! `SolutionLimitMonitor` implements `TreeSearchMonitor` and stops the search
//! once a configured number of solutions has been found. It observes the solver
//! statistics and returns a termination command when the global count reaches
//! the limit, remaining otherwise unobtrusive during the search.
//!
use crate::{
    monitor::tree_search_monitor::{PruneReason, TreeSearchMonitor},
    state::SearchState,
    stats::BnbSolverStatistics,
};
use bollard_model::model::Model;
use bollard_search::monitor::search_monitor::SearchCommand;
use num_traits::{PrimInt, Signed};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolutionLimitMonitor<T> {
    solution_limit: u64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> SolutionLimitMonitor<T> {
    /// Creates a new `SolutionLimitMonitor` with the specified solution limit.
    pub fn new(solution_limit: u64) -> Self {
        Self {
            solution_limit,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> TreeSearchMonitor<T> for SolutionLimitMonitor<T>
where
    T: PrimInt + Signed,
{
    fn name(&self) -> &str {
        "TimeLimitMonitor"
    }

    fn on_enter_search(&mut self, _model: &Model<T>, _statistics: &BnbSolverStatistics) {}

    fn on_exit_search(&mut self, _statistics: &BnbSolverStatistics) {}

    fn on_step(&mut self, _state: &SearchState<T>, _statistics: &BnbSolverStatistics) {}

    fn on_lower_bound_computed(
        &mut self,
        _state: &SearchState<T>,
        _lower_bound: T,
        _estimated_remaining: T,
        _statistics: &BnbSolverStatistics,
    ) {
    }

    fn on_prune(
        &mut self,
        _state: &SearchState<T>,
        _reason: PruneReason,
        _statistics: &BnbSolverStatistics,
    ) {
    }

    fn on_decisions_enqueued(
        &mut self,
        _state: &SearchState<T>,
        _count: usize,
        _statistics: &BnbSolverStatistics,
    ) {
    }

    fn on_descend(
        &mut self,
        _state: &SearchState<T>,
        _decision: crate::branching::decision::Decision<T>,
        _statistics: &BnbSolverStatistics,
    ) {
    }

    fn on_backtrack(&mut self, _state: &SearchState<T>, _statistics: &BnbSolverStatistics) {}

    fn on_solution_found(
        &mut self,
        _solution: &bollard_model::solution::Solution<T>,
        _statistics: &BnbSolverStatistics,
    ) {
    }

    fn search_command(
        &mut self,
        _state: &SearchState<T>,
        statistics: &BnbSolverStatistics,
    ) -> SearchCommand {
        if statistics.solutions_found >= self.solution_limit {
            SearchCommand::Terminate("Solution limit reached".to_string())
        } else {
            SearchCommand::Continue
        }
    }
}
