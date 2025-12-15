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
    branching::decision::Decision,
    monitor::tree_search_monitor::{PruneReason, TreeSearchMonitor},
    state::SearchState,
    stats::BnbSolverStatistics,
};
use bollard_model::{model::Model, solution::Solution};
use bollard_search::monitor::search_monitor::SearchCommand;
use num_traits::{PrimInt, Signed};

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
        _decision: Decision<T>,
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
