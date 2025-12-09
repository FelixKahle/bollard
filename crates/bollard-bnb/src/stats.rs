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

use bollard_core::num::ops::saturating_arithmetic::SaturatingAddVal;
use std::time::Duration;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BnbSolverStatistics {
    pub nodes_explored: u64,
    pub backtracks: u64,
    pub max_depth: u64,
    pub decisions_generated: u64,
    pub decisions_applied: u64,
    pub prunings_local: u64,
    pub prunings_infeasible: u64,
    pub prunings_global: u64,
    pub solutions_found: u64,
    pub time_total: Duration,
}

impl BnbSolverStatistics {
    #[inline]
    pub fn on_node_explored(&mut self) {
        self.nodes_explored = self.nodes_explored.saturating_add_val(1);
    }

    #[inline]
    pub fn on_backtrack(&mut self) {
        self.backtracks = self.backtracks.saturating_add_val(1);
    }

    #[inline]
    pub fn on_solution_found(&mut self) {
        self.solutions_found = self.solutions_found.saturating_add_val(1);
    }

    #[inline]
    pub fn on_depth_update(&mut self, depth: u64) {
        self.max_depth = self.max_depth.max(depth);
    }

    #[inline]
    pub fn on_decision_generated(&mut self) {
        self.decisions_generated = self.decisions_generated.saturating_add_val(1);
    }

    #[inline]
    pub fn on_decision_applied(&mut self) {
        self.decisions_applied = self.decisions_applied.saturating_add_val(1);
    }

    #[inline]
    pub fn on_pruning_local(&mut self) {
        self.prunings_local = self.prunings_local.saturating_add_val(1);
    }

    #[inline]
    pub fn on_pruning_infeasible(&mut self) {
        self.prunings_infeasible = self.prunings_infeasible.saturating_add_val(1);
    }

    #[inline]
    pub fn on_pruning_global(&mut self) {
        self.prunings_global = self.prunings_global.saturating_add_val(1);
    }

    #[inline]
    pub fn set_total_time(&mut self, duration: Duration) {
        self.time_total = duration;
    }
}

impl std::fmt::Display for BnbSolverStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Bollard-BnB Solver Statistics:")?;
        writeln!(f, "  Nodes explored: {}", self.nodes_explored)?;
        writeln!(f, "  Backtracks: {}", self.backtracks)?;
        writeln!(f, "  Max depth reached: {}", self.max_depth)?;
        writeln!(f, "  Decisions generated: {}", self.decisions_generated)?;
        writeln!(f, "  Decisions applied: {}", self.decisions_applied)?;
        writeln!(f, "  Prunings (local): {}", self.prunings_local)?;
        writeln!(f, "  Prunings (infeasible): {}", self.prunings_infeasible)?;
        writeln!(f, "  Prunings (global): {}", self.prunings_global)?;
        writeln!(f, "  Solutions found: {}", self.solutions_found)?;
        writeln!(f, "  Total time: {:.2?}", self.time_total)?;
        Ok(())
    }
}
