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

/// The result of the solver after termination.
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

impl<T> SolverResult<T>
where
    T: PrimInt + Signed + Copy,
{
    /// Returns the objective value if a solution was found (optimal or feasible).
    #[inline]
    pub fn objective_value(&self) -> Option<T> {
        match self {
            SolverResult::Optimal(sol) | SolverResult::Feasible(sol) => Some(sol.objective_value()),
            _ => None,
        }
    }

    #[inline]
    pub fn unwrap_optimal(&self) -> &Solution<T> {
        match self {
            SolverResult::Optimal(sol) => sol,
            _ => panic!("called `SolverResult::unwrap_optimal()` on a non-optimal result"),
        }
    }
}

/// The reason for the solver's termination.
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

/// The complete outcome of the solver after termination,
/// including result, termination reason, and statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverOutcome<T>
where
    T: PrimInt + Signed + Copy,
{
    result: SolverResult<T>,
    reason: TerminationReason,
    statistics: SolverStatistics,
}

impl<T> SolverOutcome<T>
where
    T: PrimInt + Signed + Copy,
{
    /// Creates a new `SolverOutcome` representing an optimal solution.
    #[inline]
    pub fn optimal(solution: Solution<T>, statistics: SolverStatistics) -> Self {
        Self {
            result: SolverResult::Optimal(solution),
            reason: TerminationReason::OptimalityProven,
            statistics,
        }
    }

    /// Creates a new `SolverOutcome` representing an infeasible problem.
    #[inline]
    pub fn infeasible(statistics: SolverStatistics) -> Self {
        Self {
            result: SolverResult::Infeasible,
            reason: TerminationReason::InfeasibilityProven,
            statistics,
        }
    }

    /// Creates a new `SolverOutcome` representing a feasible solution
    /// found before abortion. It may or may not be optimal.
    #[inline]
    pub fn feasible<R>(
        solution: bollard_model::solution::Solution<T>,
        abort_reason: R,
        statistics: SolverStatistics,
    ) -> Self
    where
        R: Into<String>,
    {
        Self {
            result: SolverResult::Feasible(solution),
            reason: TerminationReason::Aborted(abort_reason.into()),
            statistics,
        }
    }

    /// Creates a new `SolverOutcome` representing an unknown result
    /// due to abortion without any solution found.
    #[inline]
    pub fn unknown<R>(abort_reason: R, statistics: SolverStatistics) -> Self
    where
        R: Into<String>,
    {
        Self {
            result: SolverResult::Unknown,
            reason: TerminationReason::Aborted(abort_reason.into()),
            statistics,
        }
    }

    /// Returns the solver result.
    #[inline]
    pub fn result(&self) -> &SolverResult<T> {
        &self.result
    }

    /// Returns the termination reason.
    #[inline]
    pub fn reason(&self) -> &TerminationReason {
        &self.reason
    }

    /// Returns the solver statistics.
    #[inline]
    pub fn statistics(&self) -> &SolverStatistics {
        &self.statistics
    }

    /// Returns `true` if the solver found an optimal solution
    /// and proved its optimality, `false` otherwise.
    #[inline]
    pub fn is_optimal(&self) -> bool {
        matches!(self.result, SolverResult::Optimal(_))
    }

    /// Returns `true` if the solver found a feasible solution
    /// (optimal or not), `false` otherwise.
    #[inline]
    pub fn is_feasible(&self) -> bool {
        matches!(self.result, SolverResult::Feasible(_))
    }

    /// Returns `true` if the solver proved that the problem is infeasible,
    /// `false` otherwise.
    #[inline]
    pub fn is_infeasible(&self) -> bool {
        matches!(self.result, SolverResult::Infeasible)
    }

    /// Returns `true` if the solver found any solution
    /// (optimal or feasible), `false` otherwise.
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Result: {}", self.result)?;
        writeln!(f, "Termination: {}", self.reason)?;
        write!(f, "{}", self.statistics)?;
        Ok(())
    }
}
