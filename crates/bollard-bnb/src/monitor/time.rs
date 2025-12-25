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

//! Time limit monitor for tree search
//!
//! `TimeLimitMonitor` implements `TreeSearchMonitor` and enforces a wall‑clock
//! time budget for the search. It resets its clock at the start, checks
//! elapsed time at masked step intervals to minimize overhead, and signals
//! termination when the configured limit is reached.
//!
//! Construct with `new(limit)` or `with_clock_check_mask(limit, mask)` to
//! tune how frequently the clock is checked versus search throughput.

use crate::{
    monitor::tree_search_monitor::{PruneReason, TreeSearchMonitor},
    state::SearchState,
    stats::BnbSolverStatistics,
};
use bollard_search::monitor::search_monitor::SearchCommand;
use num_traits::{PrimInt, Signed};
use std::time::{Duration, Instant};

/// A tree search monitor that enforces a time limit on the search process.
/// If the time limit is exceeded, the monitor will signal to terminate the search.
/// It uses a step mask to limit clock checks, reducing overhead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeLimitMonitor<T> {
    start_time: Instant,
    time_limit: Duration,
    clock_check_mask: u64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> TimeLimitMonitor<T> {
    /// Default mask for clock checks to avoid excessive time checks.
    /// This mask checks the clock every 16384 steps.
    const DEFAULT_STEP_CLOCK_CHECK_MASK: u64 = 0x3FFF;

    /// Creates a new `TimeLimitMonitor` with the specified time limit.
    pub fn new(time_limit: Duration) -> Self {
        Self {
            start_time: Instant::now(),
            time_limit,
            clock_check_mask: Self::DEFAULT_STEP_CLOCK_CHECK_MASK,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new `TimeLimitMonitor` with the specified time limit and clock check mask.
    pub fn with_clock_check_mask(time_limit: Duration, mask: u64) -> Self {
        Self {
            start_time: Instant::now(),
            time_limit,
            clock_check_mask: mask,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> TreeSearchMonitor<T> for TimeLimitMonitor<T>
where
    T: PrimInt + Signed,
{
    fn name(&self) -> &str {
        "TimeLimitMonitor"
    }

    fn on_enter_search(
        &mut self,
        _model: &bollard_model::model::Model<T>,
        _statistics: &BnbSolverStatistics,
    ) {
        self.start_time = Instant::now();
    }

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
        if (statistics.steps & self.clock_check_mask) == 0 {
            let elapsed = self.start_time.elapsed();
            if elapsed >= self.time_limit {
                return SearchCommand::Terminate("time limit exceeded".to_string());
            }
        }
        SearchCommand::Continue
    }
}
