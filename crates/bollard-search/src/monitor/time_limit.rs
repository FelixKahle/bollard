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

//! # Time Limit Monitor
//!
//! A lightweight monitor that enforces a wall-clock time budget on the search.
//! It periodically checks elapsed time (using a bitmask-based step filter) and
//! requests termination once the configured `Duration` has been exceeded.
//!
//! ## Motivation
//!
//! Exact search can be compute-intensive. Many applications need predictable
//! time-bounded behavior for responsiveness or SLA compliance. This monitor
//! provides a low-overhead way to cap runtime without checking the clock at
//! every step.
//!
//! ## Highlights
//!
//! - `TimeLimitMonitor<T>` stores a `time_limit`, `start_time`, and `steps` counter.
//! - Bitmask-driven clock checks: `(steps & clock_check_mask) == 0` triggers a check.
//!   The default mask (`0x3FFF`) checks approximately every 16,384 steps.
//! - `on_step()` uses `wrapping_add` to increment steps at minimal cost.
//! - `search_command()` returns `Terminate("time limit reached")` once elapsed time
//!   exceeds the limit at a check point; otherwise `Continue`.
//! - Constructors: `new(time_limit)` and `with_clock_check_mask(time_limit, mask)`.
//!
//! ## Usage
//!
//! ```rust
//! use bollard_search::monitor::time_limit::TimeLimitMonitor;
//! use bollard_search::monitor::search_monitor::{SearchMonitor, SearchCommand};
//! use std::time::Duration;
//!
//! let mut mon = TimeLimitMonitor::<i64>::new(Duration::from_secs(5));
//! // In the search loop:
//! mon.on_step(); // periodically
//! match mon.search_command() {
//!     SearchCommand::Continue => { /* keep searching */ }
//!     SearchCommand::Terminate(reason) => { /* stop: reason */ }
//! }
//! ```

use crate::monitor::search_monitor::{SearchCommand, SearchMonitor};
use bollard_model::model::Model;
use num_traits::{PrimInt, Signed};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeLimitMonitor<T> {
    clock_check_mask: u64,
    steps: u64,
    time_limit: std::time::Duration,
    start_time: std::time::Instant,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> TimeLimitMonitor<T> {
    /// Default mask: Check every 16,384 steps (2^14).
    /// 16384 - 1 = 16383 = 0x3FFF
    const DEFAULT_STEP_CLOCK_CHECK_MASK: u64 = 0x3FFF;

    #[inline]
    pub fn new(time_limit: std::time::Duration) -> Self {
        Self {
            clock_check_mask: Self::DEFAULT_STEP_CLOCK_CHECK_MASK,
            steps: 0,
            time_limit,
            start_time: std::time::Instant::now(),
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn with_clock_check_mask(time_limit: std::time::Duration, clock_check_mask: u64) -> Self {
        Self {
            clock_check_mask,
            steps: 0,
            time_limit,
            start_time: std::time::Instant::now(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> SearchMonitor<T> for TimeLimitMonitor<T>
where
    T: PrimInt + Signed,
{
    fn name(&self) -> &str {
        "TimeLimitMonitor"
    }

    fn on_enter_search(&mut self, _model: &Model<T>) {
        self.start_time = std::time::Instant::now();
        self.steps = 0;
    }

    fn on_exit_search(&mut self) {}

    fn on_solution_found(&mut self, _solution: &bollard_model::solution::Solution<T>) {}

    #[inline(always)]
    fn on_step(&mut self) {
        self.steps = self.steps.wrapping_add(1);
    }

    #[inline(always)]
    fn search_command(&self) -> SearchCommand {
        if (self.steps & self.clock_check_mask) == 0 && self.start_time.elapsed() >= self.time_limit
        {
            return SearchCommand::Terminate("time limit reached".to_string());
        }
        SearchCommand::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::search_monitor::SearchCommand;
    use std::time::{Duration, Instant};

    type IntegerType = i64;

    fn new_monitor_with_limit(ms: u64) -> TimeLimitMonitor<IntegerType> {
        TimeLimitMonitor::<IntegerType>::new(Duration::from_millis(ms))
    }

    #[test]
    fn test_default_mask_is_power_of_two_minus_one() {
        // Ensure the default mask matches the documented 0x3FFF (16383).
        assert_eq!(
            TimeLimitMonitor::<IntegerType>::DEFAULT_STEP_CLOCK_CHECK_MASK,
            0x3FFF
        );
    }

    #[test]
    fn test_search_command_terminates_after_time_limit_when_mask_condition_met() {
        let mut mon = new_monitor_with_limit(10);
        // Make elapsed exceed limit by setting start_time sufficiently in the past.
        mon.start_time = Instant::now() - Duration::from_millis(50);

        // steps = 0 => (steps & mask) == 0, so clock check runs
        mon.steps = 0;
        match mon.search_command() {
            SearchCommand::Terminate(msg) => {
                assert!(msg.contains("time limit"), "unexpected message: {msg}");
            }
            other => panic!("expected Terminate, got {:?}", other),
        }
    }

    #[test]
    fn test_search_command_continues_when_mask_condition_not_met_even_if_time_exceeded() {
        let mut mon = new_monitor_with_limit(1);
        mon.start_time = Instant::now() - Duration::from_millis(50);

        // With default mask 0x3FFF, any nonzero steps with low bits set will skip the check.
        mon.steps = 1; // 1 & 0x3FFF != 0
        match mon.search_command() {
            SearchCommand::Continue => {}
            other => panic!("expected Continue, got {:?}", other),
        }
    }

    #[test]
    fn test_search_command_respects_custom_mask_zero_always_checks() {
        let mut mon =
            TimeLimitMonitor::<IntegerType>::with_clock_check_mask(Duration::from_millis(1), 0);
        // If mask is 0, (steps & mask) == 0 is always true, so we always check the clock.
        mon.start_time = Instant::now() - Duration::from_millis(50);

        mon.steps = 12345;
        match mon.search_command() {
            SearchCommand::Terminate(_) => {}
            other => panic!("expected Terminate due to exceeded time, got {:?}", other),
        }
    }

    #[test]
    fn test_search_command_continues_before_time_limit_when_mask_condition_met() {
        let mut mon = new_monitor_with_limit(1000);
        // Elapsed is small, below limit
        mon.start_time = Instant::now();
        mon.steps = 0; // check will run

        match mon.search_command() {
            SearchCommand::Continue => {}
            other => panic!("expected Continue, got {:?}", other),
        }
    }

    #[test]
    fn test_on_step_increments_steps_wrapping() {
        let mut mon = new_monitor_with_limit(1000);
        let before = mon.steps;
        mon.on_step();
        assert_eq!(mon.steps, before.wrapping_add(1));

        // Simulate near-overflow boundary
        mon.steps = u64::MAX;
        mon.on_step();
        assert_eq!(mon.steps, 0); // wrapping_add semantics
    }

    #[test]
    fn test_new_initializes_with_zero_steps_and_recent_start() {
        let mon = new_monitor_with_limit(1000);
        assert_eq!(mon.steps, 0);
        // We can’t reliably assert exact start_time, but it should be “recent”:
        // not a strict check; just ensure elapsed is small.
        assert!(
            mon.start_time.elapsed() < Duration::from_secs(10),
            "start_time seems too old"
        );
    }

    #[test]
    fn test_with_clock_check_mask_initializes_fields() {
        let mask = 0xFF;
        let mon = TimeLimitMonitor::<IntegerType>::with_clock_check_mask(
            Duration::from_millis(500),
            mask,
        );
        assert_eq!(mon.clock_check_mask, mask);
        assert_eq!(mon.steps, 0);
        assert!(
            mon.start_time.elapsed() < Duration::from_secs(10),
            "start_time seems too old"
        );
    }

    #[test]
    fn test_mask_condition_triggers_every_2_pow_k_steps() {
        // With mask = 0x3 (binary 11), the check should happen when low 2 bits are zero,
        // i.e., at steps that are multiples of 4: 0,4,8,12,...
        let mut mon =
            TimeLimitMonitor::<IntegerType>::with_clock_check_mask(Duration::from_secs(3600), 0x3);
        mon.start_time = Instant::now();

        // Steps where (steps & 0x3) == 0 should run the check; here time limit is large, so Continue.
        for s in [0u64, 4, 8, 12, 16, 20] {
            mon.steps = s;
            match mon.search_command() {
                SearchCommand::Continue => {}
                other => panic!("expected Continue for steps={s}, got {:?}", other),
            }
        }

        // Steps where (steps & 0x3) != 0 should skip the check entirely and continue as well.
        for s in [1u64, 2, 3, 5, 6, 7, 9, 10, 11] {
            mon.steps = s;
            match mon.search_command() {
                SearchCommand::Continue => {}
                other => panic!("expected Continue for steps={s}, got {:?}", other),
            }
        }
    }
}
