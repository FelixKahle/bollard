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

//! # Solution Count Monitor
//!
//! A search monitor that tracks the number of solutions discovered using a
//! shared `AtomicU64` counter, and optionally terminates the search when a
//! configured global limit is reached. Multiple monitors can share the same
//! counter to enforce cross-component limits.
//!
//! ## Motivation
//!
//! In exact search (e.g., branch-and-bound), you may want to:
//! - Stop after N solutions for sampling or portfolio strategies.
//! - Collect only a bounded set of feasible solutions.
//! - Coordinate termination across threads or monitor instances.
//!
//! This monitor provides a lightweight, thread-friendly mechanism to do so.
//!
//! ## Highlights
//!
//! - `SolutionMonitor<'a, T>` accepts a shared `&AtomicU64` and an optional
//!   `solution_limit`.
//! - Increments the counter on `on_solution_found`.
//! - `search_command()` returns `Terminate("global solution limit reached")`
//!   once the shared counter meets or exceeds the limit; otherwise `Continue`.
//! - Convenience constructors: `new`, `with_limit`, and `without_limit`.
//!
//! ## Usage
//!
//! ```rust
//! use bollard_search::monitor::solution::SolutionMonitor;
//! use bollard_search::monitor::search_monitor::{SearchMonitor, SearchCommand};
//! use std::sync::atomic::{AtomicU64, Ordering};
//!
//! let global_count = AtomicU64::new(0);
//! let limit = 3;
//! let mut monitor = SolutionMonitor::<i64>::with_limit(&global_count, limit);
//!
//! // After each discovered solution:
//! global_count.fetch_add(1, Ordering::Relaxed);
//! // or, equivalently: monitor.on_solution_found(&solution);
//!
//! match monitor.search_command() {
//!     SearchCommand::Continue => { /* keep searching */ }
//!     SearchCommand::Terminate(reason) => { /* stop: reason */ }
//! }
//! ```

use crate::{
    monitor::search_monitor::{SearchCommand, SearchMonitor},
    num::SolverNumeric,
};
use bollard_model::{model::Model, solution::Solution};
use std::sync::atomic::{AtomicU64, Ordering};

/// A monitor that terminates the search when a specified number of solutions has been found,
/// or continues indefinitely if no limit is set just updating the solution count.
/// This monitor keeps track of the number of solutions found using an atomic counter
/// and compares it against a predefined solution limit if provided.
#[derive(Debug)]
pub struct SolutionMonitor<'a, T> {
    solutions_found: &'a AtomicU64,
    solution_limit: Option<u64>,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T> SolutionMonitor<'a, T>
where
    T: SolverNumeric,
{
    /// Creates a new `SolutionLimitMonitor`.
    #[inline]
    pub fn new(solutions_found: &'a AtomicU64, solution_limit: Option<u64>) -> Self {
        Self {
            solutions_found,
            solution_limit,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new `SolutionLimitMonitor` with a specified solution limit.
    #[inline]
    pub fn with_limit(solutions_found: &'a AtomicU64, limit: u64) -> Self {
        Self::new(solutions_found, Some(limit))
    }

    /// Creates a new `SolutionLimitMonitor` without a solution limit.
    #[inline]
    pub fn without_limit(solutions_found: &'a AtomicU64) -> Self {
        Self::new(solutions_found, None)
    }

    /// Checks if the solution limit has been reached.
    #[inline]
    fn reached_limit(&self) -> bool {
        if let Some(limit) = self.solution_limit {
            return self.solutions_found.load(Ordering::Relaxed) >= limit;
        }
        false
    }
}

impl<'a, T> SearchMonitor<T> for SolutionMonitor<'a, T>
where
    T: SolverNumeric,
{
    fn name(&self) -> &str {
        "SolutionMonitor"
    }

    fn on_enter_search(&mut self, _model: &Model<T>) {}

    fn on_exit_search(&mut self) {}

    fn on_solution_found(&mut self, _solution: &Solution<T>) {
        self.solutions_found.fetch_add(1, Ordering::Relaxed);
    }

    fn on_step(&mut self) {}

    fn search_command(&self) -> SearchCommand {
        if self.reached_limit() {
            SearchCommand::Terminate("global solution limit reached".to_string())
        } else {
            SearchCommand::Continue
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SolutionMonitor;
    use crate::monitor::search_monitor::{SearchCommand, SearchMonitor};
    use crate::num::SolverNumeric;
    use bollard_model::solution::Solution;
    use std::sync::atomic::AtomicU64;

    fn dummy_solution<T: SolverNumeric>(objective: T) -> Solution<T> {
        // Construct a trivial solution. The monitor ignores it.
        // Solution::new(objective, berths: Vec<BerthIndex>, start_times: Vec<T>)
        // Use empty vectors to avoid extra setup.
        Solution::new(objective, Vec::new(), Vec::new())
    }

    #[test]
    fn test_continue_before_limit_and_terminate_at_limit() {
        let counter = AtomicU64::new(0);
        let limit = 3;
        let mut monitor = SolutionMonitor::<i64>::new(&counter, Some(limit));

        // Before any solution, command is Continue
        assert!(matches!(monitor.search_command(), SearchCommand::Continue));

        // Feed 2 solutions (< limit)
        monitor.on_solution_found(&dummy_solution(10));
        assert!(matches!(monitor.search_command(), SearchCommand::Continue));

        monitor.on_solution_found(&dummy_solution(20));
        assert!(matches!(monitor.search_command(), SearchCommand::Continue));

        // Hitting the limit
        monitor.on_solution_found(&dummy_solution(30));
        assert!(matches!(
            monitor.search_command(),
            SearchCommand::Terminate(_)
        ));

        // Further calls still report Terminate
        assert!(matches!(
            monitor.search_command(),
            SearchCommand::Terminate(_)
        ));
    }

    #[test]
    fn test_multiple_monitors_share_global_counter() {
        let counter = AtomicU64::new(0);
        let limit = 4;

        let mut m1 = SolutionMonitor::<i64>::new(&counter, Some(limit));
        let mut m2 = SolutionMonitor::<i64>::new(&counter, Some(limit));

        // m1 finds 2 solutions
        m1.on_solution_found(&dummy_solution(1));
        m1.on_solution_found(&dummy_solution(2));
        assert!(matches!(m1.search_command(), SearchCommand::Continue));
        assert!(matches!(m2.search_command(), SearchCommand::Continue));

        // m2 finds 2 solutions -> reaches global limit
        m2.on_solution_found(&dummy_solution(3));
        assert!(matches!(m1.search_command(), SearchCommand::Continue));
        m2.on_solution_found(&dummy_solution(4));

        // Both now observe termination
        assert!(matches!(m1.search_command(), SearchCommand::Terminate(_)));
        assert!(matches!(m2.search_command(), SearchCommand::Terminate(_)));
    }

    #[test]
    fn test_concurrent_increment_reaches_limit() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicU64;
        use std::thread;

        use crate::monitor::search_monitor::{SearchCommand, SearchMonitor};

        let counter = Arc::new(AtomicU64::new(0));
        let limit = 10usize as u64;

        let mut handles = Vec::new();
        for _ in 0..4 {
            let c = Arc::clone(&counter);
            handles.push(thread::spawn(move || {
                // Monitor is constructed inside the thread, borrowing from the cloned Arc.
                let mut m = super::SolutionMonitor::<i64>::new(Arc::as_ref(&c), Some(limit));

                // Simulate this thread finding 3 solutions
                let dummy = bollard_model::solution::Solution::new(0i64, Vec::new(), Vec::new());
                m.on_solution_found(&dummy);
                m.on_solution_found(&dummy);
                m.on_solution_found(&dummy);

                // Return the observed command for aggregation
                m.search_command()
            }));
        }

        // Join threads and collect commands
        let commands = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>();

        // At least one thread should have observed termination
        assert!(
            commands
                .iter()
                .any(|c| matches!(c, SearchCommand::Terminate(_))),
            "expected at least one termination command across threads"
        );

        // Global counter should be >= limit (may overshoot depending on interleaving)
        assert!(
            counter.load(std::sync::atomic::Ordering::Relaxed) >= limit,
            "global counter did not reach the limit"
        );
    }
}
