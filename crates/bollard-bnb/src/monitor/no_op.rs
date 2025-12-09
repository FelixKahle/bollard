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
use bollard_model::{model::Model, solution::Solution};
use bollard_search::monitor::search_monitor::SearchCommand;
use num_traits::{PrimInt, Signed};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoOpMonitor<T>
where
    T: PrimInt + Signed,
{
    _marker: std::marker::PhantomData<T>,
}

impl<T> NoOpMonitor<T>
where
    T: PrimInt + Signed,
{
    pub fn new() -> Self {
        NoOpMonitor {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T> Default for NoOpMonitor<T>
where
    T: PrimInt + Signed,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TreeSearchMonitor<T> for NoOpMonitor<T>
where
    T: PrimInt + Signed,
{
    fn on_enter_search(&mut self, _model: &Model<T>) {}
    fn check_termination(
        &mut self,
        _state: &SearchState<T>,
        _stats: &BnbSolverStatistics,
    ) -> SearchCommand {
        SearchCommand::Continue
    }
    fn on_solution(&mut self, _solution: &Solution<T>, _stats: &BnbSolverStatistics) {}
    fn on_backtrack(&mut self, _state: &SearchState<T>, _stats: &BnbSolverStatistics) {}
    fn on_descend(&mut self, _state: &SearchState<T>, _stats: &BnbSolverStatistics) {}
    fn on_exit_search(&mut self, _stats: &BnbSolverStatistics) {}
    fn name(&self) -> &str {
        "NoOpMonitor"
    }
}
