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
    monitor::search_monitor::{SearchCommand, SearchMonitor},
    num::SolverNumeric,
};
use bollard_model::{model::Model, solution::Solution};
use std::sync::atomic::{AtomicU64, Ordering};

/// A monitor that terminates the search when a specified number of solutions has been found.
/// This monitor keeps track of the number of solutions found using an atomic counter
/// and compares it against a predefined solution limit.
#[derive(Debug)]
pub struct SolutionLimitMonitor<'a, T> {
    solutions_found: &'a AtomicU64,
    solution_limit: u64,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T> SolutionLimitMonitor<'a, T>
where
    T: SolverNumeric,
{
    /// Creates a new `SolutionLimitMonitor`.
    #[inline]
    pub fn new(solutions_found: &'a AtomicU64, solution_limit: u64) -> Self {
        Self {
            solutions_found,
            solution_limit,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Checks if the solution limit has been reached.
    #[inline]
    fn reached_limit(&self) -> bool {
        self.solutions_found.load(Ordering::Relaxed) >= self.solution_limit
    }
}

impl<'a, T> SearchMonitor<T> for SolutionLimitMonitor<'a, T>
where
    T: SolverNumeric,
{
    fn name(&self) -> &str {
        "SolutionLimitMonitor"
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
    use super::SolutionLimitMonitor;
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
        let mut monitor = SolutionLimitMonitor::<i64>::new(&counter, limit);

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

        let mut m1 = SolutionLimitMonitor::<i64>::new(&counter, limit);
        let mut m2 = SolutionLimitMonitor::<i64>::new(&counter, limit);

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
                let mut m = super::SolutionLimitMonitor::<i64>::new(Arc::as_ref(&c), limit);

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
