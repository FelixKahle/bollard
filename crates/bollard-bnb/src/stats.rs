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

use bollard_core::num::{constants::Zero, ops::saturating_arithmetic::SaturatingAddVal};
use std::time::Duration;

/// Statistics collected during the execution of the Bollard-BnB solver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BnbSolverStatistics<T> {
    /// Total nodes visited.
    pub nodes_explored: u64,
    /// Total leaf nodes reached or dead-ends hit.
    pub backtracks: u64,
    /// Total distinct branching choices generated.
    pub decisions_generated: u64,
    /// The deepest level reached in the tree.
    pub max_depth: u64,
    /// Pruned because the move was structurally impossible (e.g., vessel too big).
    pub prunings_infeasible: u64,
    /// Pruned because the move cost + heuristics exceeded the bound immediately.
    /// This combines both local (node-level) and global (incumbent) pruning.
    pub prunings_bound: u64,
    /// Total solutions found during the search.
    pub solutions_found: u64,
    /// Total time spent in the solver.
    pub time_total: Duration,
    /// The lower bound at the root node. Used to calculate the "Optimality Gap".
    /// Stored as i64 for simplicity; assumes the solver uses i64-compatible objectives.
    pub root_lower_bound: T,
}

impl<T> Default for BnbSolverStatistics<T>
where
    T: Zero,
{
    fn default() -> Self {
        Self {
            nodes_explored: 0,
            backtracks: 0,
            decisions_generated: 0,
            max_depth: 0,
            prunings_infeasible: 0,
            prunings_bound: 0,
            solutions_found: 0,
            time_total: Duration::ZERO,
            root_lower_bound: T::ZERO,
        }
    }
}

impl<T> BnbSolverStatistics<T> {
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
    pub fn on_pruning_infeasible(&mut self) {
        self.prunings_infeasible = self.prunings_infeasible.saturating_add_val(1);
    }

    /// Records a pruning event caused by the objective bound (either local or global).
    #[inline]
    pub fn on_pruning_bound(&mut self) {
        self.prunings_bound = self.prunings_bound.saturating_add_val(1);
    }

    #[inline]
    pub fn set_total_time(&mut self, duration: Duration) {
        self.time_total = duration;
    }

    #[inline]
    pub fn set_root_lower_bound(&mut self, bound: T) {
        self.root_lower_bound = bound;
    }
}

impl<T> std::fmt::Display for BnbSolverStatistics<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Bollard-BnB Solver Statistics:")?;
        writeln!(f, "  Nodes explored:       {}", self.nodes_explored)?;
        writeln!(f, "  Backtracks:           {}", self.backtracks)?;
        writeln!(f, "  Max depth reached:    {}", self.max_depth)?;
        writeln!(f, "  Decisions generated:  {}", self.decisions_generated)?;
        writeln!(f, "  Prunings (infeasible):{}", self.prunings_infeasible)?;
        writeln!(f, "  Prunings (bound):     {}", self.prunings_bound)?;
        writeln!(f, "  Solutions found:      {}", self.solutions_found)?;
        writeln!(f, "  Root Lower Bound:     {}", self.root_lower_bound)?;
        writeln!(f, "  Total time:           {:.2?}", self.time_total)?;
        Ok(())
    }
}
