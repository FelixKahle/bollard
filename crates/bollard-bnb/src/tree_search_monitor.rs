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

use crate::{branching::decision::Decision, state::SearchState, stats::BnbSolverStatistics};
use bollard_model::{model::Model, solution::Solution};
use bollard_search::monitor::search_monitor::SearchCommand;
use num_traits::{PrimInt, Signed};

/// Reasons for pruning a search state.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum PruneReason {
    /// The subtree is infeasible.
    Infeasible,
    /// The subtree is dominated by the current bound.
    BoundDominated,
}

impl std::fmt::Display for PruneReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PruneReason::Infeasible => write!(f, "Infeasible"),
            PruneReason::BoundDominated => write!(f, "BoundDominated"),
        }
    }
}

/// Trait for monitoring and controlling the search process of the solver.
pub trait TreeSearchMonitor<T>
where
    T: PrimInt + Signed,
{
    /// Returns the name of the monitor.
    fn name(&self) -> &str;
    /// Called when the search starts.
    fn on_enter_search(&mut self, model: &Model<T>, statistics: &BnbSolverStatistics);
    /// Called when the search ends.
    fn on_exit_search(&mut self, statistics: &BnbSolverStatistics);
    /// Called to determine the next action of the search.
    fn search_command(
        &mut self,
        _state: &SearchState<T>,
        _statistics: &BnbSolverStatistics,
    ) -> SearchCommand {
        SearchCommand::Continue
    }
    /// Called at each step of the search.
    fn on_step(&mut self, state: &SearchState<T>, statistics: &BnbSolverStatistics);
    /// Called when a lower bound is computed for a search state.
    /// `lower_bound` is the computed lower bound,
    /// `estimated_remaining` is an estimate of the remaining cost
    /// to reach a solution from this state.
    fn on_lower_bound_computed(
        &mut self,
        state: &SearchState<T>,
        lower_bound: T,
        estimated_remaining: T,
        statistics: &BnbSolverStatistics,
    );
    /// Called when a search state is pruned.
    fn on_prune(
        &mut self,
        state: &SearchState<T>,
        reason: PruneReason,
        statistics: &BnbSolverStatistics,
    );
    /// Called when decisions are enqueued for exploration.
    fn on_decisions_enqueued(
        &mut self,
        state: &SearchState<T>,
        count: usize,
        statistics: &BnbSolverStatistics,
    );
    /// Called when descending into a child state.
    fn on_descend(
        &mut self,
        state: &SearchState<T>,
        decision: Decision,
        statistics: &BnbSolverStatistics,
    );
    /// Called when backtracking to a parent state.
    fn on_backtrack(&mut self, state: &SearchState<T>, statistics: &BnbSolverStatistics);
    /// Called when a new solution is found.
    fn on_solution_found(&mut self, solution: &Solution<T>, statistics: &BnbSolverStatistics);
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

/// A no-operation monitor that implements the `TreeSearchMonitor` trait
/// but does nothing on any of the events, always returning `Continue` for the
/// search command.
#[repr(transparent)]
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct NoOperationMonitor<T>
where
    T: PrimInt + Signed,
{
    _phantom: std::marker::PhantomData<T>,
}

impl<T> NoOperationMonitor<T>
where
    T: PrimInt + Signed,
{
    /// Creates a new `NoOperationMonitor`.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> TreeSearchMonitor<T> for NoOperationMonitor<T>
where
    T: PrimInt + Signed,
{
    #[inline(always)]
    fn name(&self) -> &str {
        "NoOperationMonitor"
    }

    #[inline(always)]
    fn on_enter_search(&mut self, _model: &Model<T>, _statistics: &BnbSolverStatistics) {}

    #[inline(always)]
    fn on_solution_found(&mut self, _solution: &Solution<T>, _statistics: &BnbSolverStatistics) {}

    #[inline(always)]
    fn on_backtrack(&mut self, _state: &SearchState<T>, _statistics: &BnbSolverStatistics) {}

    #[inline(always)]
    fn on_descend(
        &mut self,
        _state: &SearchState<T>,
        _decision: Decision,
        _statistics: &BnbSolverStatistics,
    ) {
    }

    #[inline(always)]
    fn on_exit_search(&mut self, _statistics: &BnbSolverStatistics) {}

    #[inline(always)]
    fn search_command(
        &mut self,
        _state: &SearchState<T>,
        _statistics: &BnbSolverStatistics,
    ) -> SearchCommand {
        SearchCommand::Continue
    }

    #[inline(always)]
    fn on_step(&mut self, _state: &SearchState<T>, _statistics: &BnbSolverStatistics) {}

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
