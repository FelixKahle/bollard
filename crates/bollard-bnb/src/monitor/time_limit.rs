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

use crate::monitor::tree_search_monitor::TreeSearchMonitor;
use crate::state::SearchState;
use crate::stats::BnbSolverStatistics;
use bollard_core::num::ops::saturating_arithmetic::SaturatingAddVal;
use bollard_model::{model::Model, solution::Solution};
use bollard_search::monitor::search_monitor::SearchCommand;
use num_traits::{PrimInt, Signed};
use std::marker::PhantomData;
use std::time::{Duration, Instant};

/// A monitor that terminates the search after a specified duration.
///
/// Checks the clock only every `check_interval` nodes to minimize overhead.
pub struct TimeLimitMonitor<T>
where
    T: PrimInt + Signed,
{
    time_limit: Duration,
    start_time: Option<Instant>,
    check_interval: u64,
    ops_since_last_check: u64,
    _marker: PhantomData<T>,
}

impl<T> TimeLimitMonitor<T>
where
    T: PrimInt + Signed,
{
    /// Creates a new `TimeLimitMonitor` with the specified duration and check interval.
    /// `check_interval` specifies how many steps to take between time checks.
    /// A higher value reduces overhead but may lead to slightly exceeding the time limit.
    pub fn new(duration: Duration, check_interval: u64) -> Self {
        Self {
            time_limit: duration,
            start_time: None,
            check_interval,
            ops_since_last_check: 0,
            _marker: PhantomData,
        }
    }

    /// Creates a new `TimeLimitMonitor` with the specified duration and a default check interval of 10,000.
    pub fn with_default_check_interval(duration: Duration) -> Self {
        Self::new(duration, 10_000)
    }
}

impl<T> TreeSearchMonitor<T> for TimeLimitMonitor<T>
where
    T: PrimInt + Signed,
{
    fn on_enter_search(&mut self, _model: &Model<T>) {
        self.start_time = Some(Instant::now());
        self.ops_since_last_check = 0;
    }

    fn check_termination(
        &mut self,
        _state: &SearchState<T>,
        _stats: &BnbSolverStatistics,
    ) -> SearchCommand {
        self.ops_since_last_check = self.ops_since_last_check.saturating_add_val(1);

        if self.ops_since_last_check >= self.check_interval {
            self.ops_since_last_check = 0;

            if let Some(start) = self.start_time
                && start.elapsed() > self.time_limit
            {
                return SearchCommand::Terminate(format!(
                    "Time limit of {} seconds exceeded",
                    self.time_limit.as_secs()
                ));
            }
        }

        SearchCommand::Continue
    }

    fn on_solution(&mut self, _solution: &Solution<T>, _stats: &BnbSolverStatistics) {}
    fn on_backtrack(&mut self, _state: &SearchState<T>, _stats: &BnbSolverStatistics) {}
    fn on_descend(&mut self, _state: &SearchState<T>, _stats: &BnbSolverStatistics) {}
    fn on_exit_search(&mut self, _stats: &BnbSolverStatistics) {
        self.start_time = None;
    }

    fn name(&self) -> &str {
        "TimeLimitMonitor"
    }
}
