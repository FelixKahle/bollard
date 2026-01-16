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

use crate::{monitor::local_search_monitor::LocalSearchMonitor, stats::LocalSearchStatistics};
use bollard_model::{model::Model, solution::Solution};
use bollard_search::num::SolverNumeric;
use num_traits::{PrimInt, Signed};
use std::time::{Duration, Instant};

/// Console logger for local search.
///
/// Prints a compact progress table in regular time intervals:
/// - elapsed wall-clock time
/// - search steps
/// - best objective value (if any)
///
/// It is intentionally lightweight and uses a step mask to avoid
/// checking the clock on every step.
pub struct LogLocalSearchMonitor<T>
where
    T: std::fmt::Display + std::fmt::Debug + PrimInt + Signed,
{
    start_time: Instant,
    last_log_time: Instant,
    log_interval: Duration,
    clock_check_mask: u64,
    best_objective: Option<T>,
}

impl<T> LogLocalSearchMonitor<T>
where
    T: std::fmt::Display + std::fmt::Debug + PrimInt + Signed,
{
    /// Default mask for clock checks to avoid excessive time checks.
    /// This mask checks the clock every 512 steps.
    const DEFAULT_STEP_CLOCK_CHECK_MASK: u64 = 0x1FF; // 511 ≈ 512 steps

    /// Separator rule for header / footer.
    const HEADER_FOOTER_RULE: &'static str =
        "------------------------------------------------------------";

    /// Create a new local search log monitor that prints every `log_interval`.
    #[inline]
    pub fn new(log_interval: Duration) -> Self {
        let now = Instant::now();
        Self {
            start_time: now,
            last_log_time: now,
            log_interval,
            clock_check_mask: Self::DEFAULT_STEP_CLOCK_CHECK_MASK,
            best_objective: None,
        }
    }

    /// Create with both a custom log interval and a custom clock-check mask.
    #[inline]
    pub fn with_clock_check_mask(log_interval: Duration, mask: u64) -> Self {
        let now = Instant::now();
        Self {
            start_time: now,
            last_log_time: now,
            log_interval,
            clock_check_mask: mask,
            best_objective: None,
        }
    }

    #[inline(always)]
    fn print_header(&self) {
        println!(
            "{:<9} | {:>14} | {:>14}",
            "Elapsed", "Steps", "Best Solution"
        );
        println!("{}", Self::HEADER_FOOTER_RULE);
    }

    #[inline(always)]
    fn log_line(&mut self, stats: &LocalSearchStatistics) {
        let elapsed = self.start_time.elapsed().as_secs_f32();

        println!(
            "{:<9} | {:>14} | {:>14}",
            format!("{:.1}s", elapsed),
            stats.iterations,
            self.best_objective
                .as_ref()
                .map_or("Inf".to_string(), |obj| obj.to_string()),
        );

        self.last_log_time = Instant::now();
    }

    #[inline]
    fn update_best_from_solution(&mut self, solution: &Solution<T>) {
        let obj = solution.objective_value();
        self.best_objective = Some(obj);
    }
}

impl<T> Default for LogLocalSearchMonitor<T>
where
    T: std::fmt::Display + std::fmt::Debug + PrimInt + Signed,
{
    #[inline]
    fn default() -> Self {
        Self::new(Duration::from_secs(1))
    }
}

impl<T> LocalSearchMonitor<T> for LogLocalSearchMonitor<T>
where
    T: SolverNumeric,
{
    fn name(&self) -> &str {
        "LogLocalSearchMonitor"
    }

    fn on_start(&mut self, _model: &Model<T>, initial_solution: &Solution<T>) {
        let now = Instant::now();
        self.start_time = now;
        self.last_log_time = now;
        self.best_objective = None;
        self.print_header();
        // Record objective of the initial solution as current best.
        self.update_best_from_solution(initial_solution);
    }

    fn on_end(&mut self, best_solution: &Solution<T>, statistics: &LocalSearchStatistics) {
        // Ensure we log the final best objective and a footer.
        self.update_best_from_solution(best_solution);
        self.log_line(statistics);
        println!("{}", Self::HEADER_FOOTER_RULE);
        println!("Local search finished.");
    }

    fn on_iteration(
        &mut self,
        _current_solution: &Solution<T>,
        statistics: &LocalSearchStatistics,
    ) {
        // Throttle clock checks via bitmask, mirroring the BnB monitor.
        if (statistics.iterations & self.clock_check_mask) == 0
            && self.last_log_time.elapsed() >= self.log_interval
        {
            self.log_line(statistics);
        }
    }

    fn on_solution_found(&mut self, solution: &Solution<T>, _statistics: &LocalSearchStatistics) {
        // First feasible or any discovered solution (metaheuristic dependent).
        if self.best_objective.is_none() {
            self.update_best_from_solution(solution);
        }
    }

    fn on_solution_accepted(
        &mut self,
        _solution: &Solution<T>,
        _statistics: &LocalSearchStatistics,
    ) {
        // For now we do not log every accepted move; logging every iteration
        // via `on_iteration` keeps output compact.
    }

    fn on_solution_rejected(
        &mut self,
        _solution: &Solution<T>,
        _statistics: &LocalSearchStatistics,
    ) {
        // No-op for logging; can be extended if you want rejection statistics.
    }

    fn on_best_solution_updated(
        &mut self,
        solution: &Solution<T>,
        _statistics: &LocalSearchStatistics,
    ) {
        // Always track the latest best objective.
        self.update_best_from_solution(solution);
    }
}
