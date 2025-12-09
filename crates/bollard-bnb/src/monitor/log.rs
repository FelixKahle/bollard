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

use crate::monitor::search_monitor::TreeSearchMonitor;
use crate::state::SearchState;
use crate::stats::BnbSolverStatistics;
use bollard_model::{model::Model, solution::Solution};
use num_traits::{PrimInt, Signed};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct LogMonitor<T>
where
    T: std::fmt::Display + std::fmt::Debug + PrimInt + Signed,
{
    start_time: Instant,
    last_log_time: Instant,
    log_interval: Duration,
    clock_check_mask: u64,
    best_objective: Option<T>,
}

impl<T> LogMonitor<T>
where
    T: std::fmt::Display + std::fmt::Debug + PrimInt + Signed,
{
    pub fn new(log_interval: Duration, clock_check_mask: u64) -> Self {
        Self {
            start_time: Instant::now(),
            last_log_time: Instant::now(),
            log_interval,
            clock_check_mask,
            best_objective: None,
        }
    }

    #[inline(always)]
    fn print_header(&self) {
        println!(
            "{:<9} | {:<14} | {:<7} | {:<14} | {:<17} | {:<10} | {:<13}",
            "Elapsed",
            "Nodes",
            "Depth",
            "Best Solution",
            "Current Objective",
            "Backtracks",
            "Pruned (Local)"
        );
        println!("{}", "-".repeat(102));
    }

    #[inline(always)]
    fn log_line(&mut self, state: &SearchState<T>, stats: &BnbSolverStatistics) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.start_time).as_secs_f32();

        let nodes = stats.nodes_explored;
        let depth = state.num_assigned_vessels();
        let backtracks = stats.backtracks;
        let pruned_local = stats.prunings_local;
        let current_obj = state.current_objective();

        let best_obj_str = if let Some(sol) = &self.best_objective {
            format!("{}", sol)
        } else {
            "Inf".to_string()
        };

        let elapsed_field = format!("{:.1}s", elapsed);

        println!(
            "{:<9} | {:<14} | {:<7} | {:<14} | {:<17} | {:<10} | {:<13}",
            elapsed_field, nodes, depth, best_obj_str, current_obj, backtracks, pruned_local
        );

        self.last_log_time = now;
    }
}

impl<T> Default for LogMonitor<T>
where
    T: std::fmt::Display + std::fmt::Debug + PrimInt + Signed,
{
    fn default() -> Self {
        Self::new(Duration::from_secs(1), 4095)
    }
}

impl<T> std::fmt::Display for LogMonitor<T>
where
    T: std::fmt::Display + std::fmt::Debug + PrimInt + Signed,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LogMonitor(log_interval: {}s, clock_check_mask: {})",
            self.log_interval.as_secs(),
            self.clock_check_mask
        )
    }
}

impl<T> TreeSearchMonitor<T> for LogMonitor<T>
where
    T: std::fmt::Display + std::fmt::Debug + PrimInt + Signed,
{
    fn on_enter_search(&mut self, _model: &Model<T>) {
        self.start_time = Instant::now();
        self.last_log_time = self.start_time;
        self.best_objective = None; // Reset
        self.print_header();
    }

    fn on_descend(&mut self, state: &SearchState<T>, stats: &BnbSolverStatistics) {
        if (stats.nodes_explored & self.clock_check_mask) == 0
            && self.last_log_time.elapsed() >= self.log_interval
        {
            self.log_line(state, stats);
        }
    }

    fn on_solution(&mut self, solution: &Solution<T>, _stats: &BnbSolverStatistics) {
        let obj = solution.objective_value();
        self.best_objective = Some(obj);
    }

    fn on_backtrack(&mut self, _state: &SearchState<T>, _stats: &BnbSolverStatistics) {}

    fn on_exit_search(&mut self, _stats: &BnbSolverStatistics) {
        println!("{}", "-".repeat(102));
        println!("Search finished.");
    }

    fn name(&self) -> &str {
        "LogMonitor"
    }
}
