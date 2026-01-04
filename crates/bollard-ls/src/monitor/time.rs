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

//! Time-based termination for local search.
//!
//! This module provides `TimeLimitMonitor`, a lightweight monitor that stops a local
//! search after a configurable wall-clock duration. It integrates with the
//! `LocalSearchMonitor` trait and issues a `SearchCommand::Terminate` when the
//! elapsed time exceeds the configured limit.
//!
//! To minimize overhead, clock checks are throttled using a step mask. The mask is
//! applied to the iteration counter and only when the masked value is zero the clock
//! is queried. A mask of `0x0FFF` yields a check roughly every 4096 iterations, which
//! offers a balance between responsiveness and performance. The mask can be customized
//! via `with_mask`, allowing tighter or looser checking based on the problem scale.
//!
//! The monitor resets its start time on `on_start`, ensuring each search run is measured
//! independently. No state is mutated during `search_command` beyond timing checks, and
//! the termination message includes a short reason indicating that the time limit was
//! exceeded.

use crate::{
    memory::Schedule, monitor::local_search_monitor::LocalSearchMonitor,
    stats::LocalSearchStatistics,
};
use bollard_search::{monitor::search_monitor::SearchCommand, num::SolverNumeric};
use std::time::{Duration, Instant};

/// A lightweight wall-clock monitor that terminates a local search after a fixed duration.
///
/// This monitor records the start time at `on_start` and periodically checks the elapsed time
/// during `search_command`. To reduce overhead, time checks are throttled using `clock_check_mask`,
/// which masks the iteration counter and only queries the clock when the masked value is zero.
/// When `elapsed >= time_limit`, it issues a termination command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeLimitMonitor {
    start_time: Instant,
    time_limit: Duration,
    clock_check_mask: u64,
}

impl TimeLimitMonitor {
    /// Default mask for clock checks to avoid excessive time checks.
    /// This mask checks the clock every 4096 steps.
    const DEFAULT_STEP_CLOCK_CHECK_MASK: u64 = 0x0FFF;

    /// Creates a new `TimeLimitMonitor` with the specified time limit.
    pub fn new(time_limit: Duration) -> Self {
        Self {
            start_time: Instant::now(),
            time_limit,
            clock_check_mask: Self::DEFAULT_STEP_CLOCK_CHECK_MASK,
        }
    }

    /// Creates a new `TimeLimitMonitor` with a custom step clock check mask.
    /// Lower mask values check more often; higher values check less often.
    pub fn with_mask(time_limit: Duration, clock_check_mask: u64) -> Self {
        Self {
            start_time: Instant::now(),
            time_limit,
            clock_check_mask,
        }
    }
}

impl<T> LocalSearchMonitor<T> for TimeLimitMonitor
where
    T: SolverNumeric,
{
    fn name(&self) -> &str {
        "TimeLimitMonitor"
    }

    fn on_start(&mut self, _initial_solution: &Schedule<T>) {
        self.start_time = Instant::now();
    }

    fn on_end(&mut self, _best_solution: &Schedule<T>, _statistics: &LocalSearchStatistics) {}

    fn on_iteration(
        &mut self,
        _current_solution: &Schedule<T>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn on_solution_found(&mut self, _solution: &Schedule<T>, _statistics: &LocalSearchStatistics) {}

    fn on_solution_accepted(
        &mut self,
        _solution: &Schedule<T>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn on_solution_rejected(
        &mut self,
        _solution: &Schedule<T>,
        _statistics: &LocalSearchStatistics,
    ) {
    }

    fn search_command(&mut self, statistics: &LocalSearchStatistics) -> SearchCommand {
        if (statistics.iterations & self.clock_check_mask) == 0 {
            let elapsed = self.start_time.elapsed();
            if elapsed >= self.time_limit {
                return SearchCommand::Terminate("time limit exceeded".to_string());
            }
        }
        SearchCommand::Continue
    }
}
