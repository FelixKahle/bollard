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

use crate::stats::BnbSolverStatistics;
use bollard_model::solution::Solution;
use num_traits::{PrimInt, Signed};

/// Reason for termination of the solver.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TerminationReason {
    /// The solver found and proved optimality of a solution.
    OptimalityProven,
    /// The solver proved that the problem is infeasible.
    InfeasibilityProven,
    /// The solver aborted due to a search limit (time, iterations, etc.).
    Aborted,
}

impl std::fmt::Display for TerminationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TerminationReason::OptimalityProven => write!(f, "Optimality Proven"),
            TerminationReason::InfeasibilityProven => write!(f, "Infeasibility Proven"),
            TerminationReason::Aborted => write!(f, "Aborted"),
        }
    }
}

/// Outcome of the search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchOutcome {
    /// The solver found an optimal solution.
    Optimal,
    /// The solver found a feasible solution, but optimality is not proven.
    Feasible,
    /// The solver proved that the problem is infeasible.
    Infeasible,
    /// The outcome of the search is unknown (e.g., due to abortion).
    Unknown,
}

impl std::fmt::Display for SearchOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchOutcome::Optimal => write!(f, "Optimal"),
            SearchOutcome::Feasible => write!(f, "Feasible"),
            SearchOutcome::Infeasible => write!(f, "Infeasible"),
            SearchOutcome::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Result of the solver after termination.
#[derive(Debug, Clone)]
pub struct SolverResult<T> {
    /// Outcome of the search.
    pub outcome: SearchOutcome,
    /// Reason for termination.
    pub reason: TerminationReason,
    /// Best solution found (if any).
    pub solution: Option<Solution<T>>,
    /// Statistics collected during the search.
    pub statistics: BnbSolverStatistics,
    /// Name of the objective evaluator used.
    pub objective_evaluator: String,
    /// Name of the tree builder used.
    pub tree_builder: String,
}

impl<T> std::fmt::Display for SolverResult<T>
where
    T: PrimInt + Signed + std::fmt::Display + Copy,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Bollard-BnB Solving Report")?;

        let write_kv = |f: &mut std::fmt::Formatter, key: &str, val: String| {
            writeln!(f, "  {:<18} {}", key, val)
        };

        write_kv(f, "Status", format!("{:?}", self.outcome))?;
        write_kv(f, "Reason", self.reason.to_string())?;

        if let Some(sol) = &self.solution {
            write_kv(f, "Objective", sol.objective_value().to_string())?;
        } else {
            write_kv(f, "Objective", "-".to_string())?;
        }

        writeln!(f, "  Configuration")?;
        write_kv(f, "  Evaluator", self.objective_evaluator.clone())?;
        write_kv(f, "  Builder", self.tree_builder.clone())?;

        let total_time_s = self.statistics.time_total.as_secs_f64();
        let nodes = self.statistics.nodes_explored;
        let nodes_per_sec = if total_time_s > 0.0 {
            (nodes as f64 / total_time_s) as u64
        } else {
            0
        };

        writeln!(f, "  Timing             {:.2} s", total_time_s)?;
        write_kv(f, "Nodes", nodes.to_string())?;
        write_kv(f, "Nodes/Sec", nodes_per_sec.to_string())?;
        write_kv(f, "Max Depth", self.statistics.max_depth.to_string())?;
        write_kv(f, "Backtracks", self.statistics.backtracks.to_string())?;
        write_kv(
            f,
            "Solutions Found",
            self.statistics.solutions_found.to_string(),
        )?;

        writeln!(f, "  Pruning")?;
        write_kv(f, "  Local", self.statistics.prunings_local.to_string())?;
        write_kv(f, "  Global", self.statistics.prunings_global.to_string())?;
        write_kv(
            f,
            "  Infeasible",
            self.statistics.prunings_infeasible.to_string(),
        )?;

        Ok(())
    }
}
