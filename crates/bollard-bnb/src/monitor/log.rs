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
use bollard_model::model::Model;
use num_traits::{PrimInt, Signed};
use std::time::{Duration, Instant};

/// A tree search monitor that logs progress to the console at regular intervals.
/// It prints a header at the start of the search and logs the current state
/// of the search at specified time intervals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogTreeSearchMonitor<T>
where
    T: std::fmt::Display + std::fmt::Debug + PrimInt + Signed,
{
    start_time: Instant,
    last_log_time: Instant,
    log_interval: Duration,
    clock_check_mask: u64,
    best_objective: Option<T>,
}

impl<T> Default for LogTreeSearchMonitor<T>
where
    T: std::fmt::Display + std::fmt::Debug + PrimInt + Signed,
{
    #[inline]
    fn default() -> Self {
        Self::new(Duration::from_secs(1))
    }
}

impl<T> LogTreeSearchMonitor<T>
where
    T: std::fmt::Display + std::fmt::Debug + PrimInt + Signed,
{
    /// Default mask for clock checks to avoid excessive time checks.
    /// This mask checks the clock every 16384 steps.
    const DEFAULT_STEP_CLOCK_CHECK_MASK: u64 = 0x3FFF;

    /// Creates a new `LogTreeSearchMonitor` with the specified log interval.
    /// The monitor will log progress every `log_interval` duration.
    #[inline]
    pub fn new(log_interval: Duration) -> Self {
        Self {
            start_time: Instant::now(),
            last_log_time: Instant::now(),
            log_interval,
            clock_check_mask: Self::DEFAULT_STEP_CLOCK_CHECK_MASK,
            best_objective: None,
        }
    }

    /// Creates a new `LogTreeSearchMonitor` with the specified log interval
    /// and clock check mask. The monitor will log progress every `log_interval`
    /// duration, and will check the clock every time the number of steps
    /// matches the mask.
    #[inline]
    pub fn with_clock_check_mask(log_duration: Duration, mask: u64) -> Self {
        Self {
            start_time: Instant::now(),
            last_log_time: Instant::now(),
            log_interval: log_duration,
            clock_check_mask: mask,
            best_objective: None,
        }
    }

    const HEADER_FOOTER_RULE: &str =
        "----------------------------------------------------------------------------------";

    #[inline(always)]
    fn print_header(&self) {
        println!(
            "{:<9} | {:>14} | {:>7} | {:>14} | {:>10} | {:>13}",
            "Elapsed", "Nodes", "Depth", "Best Solution", "Backtracks", "Pruned"
        );
        println!("{}", Self::HEADER_FOOTER_RULE);
    }

    #[inline(always)]
    fn log_line(&mut self, state: &SearchState<T>, stats: &BnbSolverStatistics) {
        let elapsed = self.start_time.elapsed().as_secs_f32();

        let nodes = stats.nodes_explored;
        let depth = state.num_assigned_vessels();
        let backtracks = stats.backtracks;
        let pruned = stats.prunings_infeasible + stats.prunings_bound;

        println!(
            "{:<9} | {:>14} | {:>7} | {:>14} | {:>10} | {:>13}",
            format!("{:.1}s", elapsed),
            nodes,
            depth,
            self.best_objective
                .as_ref()
                .map_or("Inf".to_string(), |sol| sol.to_string()),
            backtracks,
            pruned
        );

        self.last_log_time = Instant::now();
    }
}

impl<T> TreeSearchMonitor<T> for LogTreeSearchMonitor<T>
where
    T: std::fmt::Display + std::fmt::Debug + PrimInt + Signed,
{
    fn name(&self) -> &str {
        "LogTreeSearchMonitor"
    }

    fn on_enter_search(&mut self, _model: &Model<T>, _statistics: &BnbSolverStatistics) {
        self.start_time = Instant::now();
        self.last_log_time = self.start_time;
        self.best_objective = None; // Reset
        self.print_header();
    }

    fn on_exit_search(&mut self, _statistics: &BnbSolverStatistics) {
        println!("{}", Self::HEADER_FOOTER_RULE);
        println!("Search finished.");
    }

    fn on_step(&mut self, state: &SearchState<T>, statistics: &BnbSolverStatistics) {
        if (statistics.steps & self.clock_check_mask) == 0
            && self.last_log_time.elapsed() >= self.log_interval
        {
            self.log_line(state, statistics);
        }
    }

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
        _decision: Decision,
        _statistics: &BnbSolverStatistics,
    ) {
    }

    fn on_backtrack(&mut self, _state: &SearchState<T>, _statistics: &BnbSolverStatistics) {}

    fn on_solution_found(
        &mut self,
        solution: &bollard_model::solution::Solution<T>,
        _statistics: &BnbSolverStatistics,
    ) {
        let obj = solution.objective_value();
        self.best_objective = Some(obj);
    }
}
