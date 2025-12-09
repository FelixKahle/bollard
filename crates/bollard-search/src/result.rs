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

use crate::stats::SolverStatistics;
use bollard_model::solution::Solution;
use num_traits::{PrimInt, Signed};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverResult<T> {
    /// We have proven that the problem is infeasible.
    Infeasible,
    /// We have found a solution and proven its optimality.
    Optimal(Solution<T>),
    /// We have found a feasible solution, but not proven its optimality.
    Feasible(Solution<T>),
    /// The solver terminated without finding a solution and
    /// without proving infeasibility.
    Unknown,
}

impl<T> std::fmt::Display for SolverResult<T>
where
    T: PrimInt + Signed + Copy + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverResult::Infeasible => write!(f, "Infeasible"),
            SolverResult::Optimal(solution) => {
                write!(f, "Optimal(objective={})", solution.objective_value())
            }
            SolverResult::Feasible(solution) => {
                write!(f, "Feasible(objective={})", solution.objective_value())
            }
            SolverResult::Unknown => write!(f, "Unknown"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminationReason {
    /// The solver found and proved optimality of a solution.
    OptimalityProven,
    /// The solver proved that the problem is infeasible.
    InfeasibilityProven,
    /// The solver aborted due to a search limit (time, iterations, etc.).
    /// The string contains information about the reason for abortion.
    Aborted(String),
}

impl std::fmt::Display for TerminationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TerminationReason::OptimalityProven => write!(f, "Optimality Proven"),
            TerminationReason::InfeasibilityProven => write!(f, "Infeasibility Proven"),
            TerminationReason::Aborted(index) => write!(f, "Aborted: {}", *index),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverOutcome<T>
where
    T: PrimInt + Signed + Copy,
{
    pub result: SolverResult<T>,
    pub reason: TerminationReason,
    pub statistics: SolverStatistics,
}

impl<T> SolverOutcome<T>
where
    T: PrimInt + Signed + Copy,
{
    #[inline]
    pub fn new(
        result: SolverResult<T>,
        reason: TerminationReason,
        statistics: SolverStatistics,
    ) -> Self {
        Self {
            result,
            reason,
            statistics,
        }
    }

    #[inline]
    pub fn is_optimal(&self) -> bool {
        matches!(self.result, SolverResult::Optimal(_))
    }

    #[inline]
    pub fn is_feasible(&self) -> bool {
        matches!(self.result, SolverResult::Feasible(_))
    }

    #[inline]
    pub fn is_infeasible(&self) -> bool {
        matches!(self.result, SolverResult::Infeasible)
    }

    #[inline]
    pub fn has_solution(&self) -> bool {
        matches!(
            self.result,
            SolverResult::Optimal(_) | SolverResult::Feasible(_)
        )
    }
}

impl<T> std::fmt::Display for SolverOutcome<T>
where
    T: PrimInt + Signed + Copy + std::fmt::Display,
{
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}
