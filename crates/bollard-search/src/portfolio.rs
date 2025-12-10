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
    incumbent::SharedIncumbent,
    monitor::search_monitor::SearchMonitor,
    result::{SolverResult, TerminationReason},
};
use bollard_core::num::constants::MinusOne;
use bollard_model::{model::Model, solution::Solution};
use num_traits::{PrimInt, Signed};

/// Context provided to a portfolio solver during its `solve` method.
pub struct PortfolioSolverContext<'a, T>
where
    T: PrimInt + Signed,
{
    /// The model to be solved.
    pub model: &'a Model<T>,
    /// The shared incumbent solution.
    pub incumbent: &'a SharedIncumbent<T>,
    /// The search monitor for reporting progress.
    pub monitor: &'a mut dyn SearchMonitor<T>,
}

impl<'a, T> PortfolioSolverContext<'a, T>
where
    T: PrimInt + Signed,
{
    /// Creates a new `PortfolioSolverContext`.
    #[inline(always)]
    pub fn new(
        model: &'a Model<T>,
        incumbent: &'a SharedIncumbent<T>,
        monitor: &'a mut dyn SearchMonitor<T>,
    ) -> Self {
        Self {
            model,
            incumbent,
            monitor,
        }
    }
}

impl<'a, T> std::fmt::Debug for PortfolioSolverContext<'a, T>
where
    T: PrimInt + Signed + Copy + MinusOne + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PortfolioSolverContext")
            .field("model", &self.model)
            .field("incumbent", &self.incumbent)
            .field("monitor", &self.monitor.name())
            .finish()
    }
}

impl<'a, T> std::fmt::Display for PortfolioSolverContext<'a, T>
where
    T: PrimInt + Signed + Copy + MinusOne + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PortfolioSolverContext(model: {}, monitor: {})",
            self.model,
            self.monitor.name(),
        )
    }
}

/// The result of a portfolio solver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortfolioSolverResult<T>
where
    T: PrimInt + Signed,
{
    result: SolverResult<T>,
    termination_reason: TerminationReason,
}

impl<T> PortfolioSolverResult<T>
where
    T: PrimInt + Signed,
{
    /// Creates a new `PortfolioSolverResult` representing an optimal solution.
    #[inline]
    pub fn optimal(solution: Solution<T>) -> Self {
        Self {
            result: SolverResult::Optimal(solution),
            termination_reason: TerminationReason::OptimalityProven,
        }
    }

    /// Creates a new `PortfolioSolverResult` representing a feasible solution.
    #[inline]
    pub fn infeasible() -> Self {
        Self {
            result: SolverResult::Infeasible,
            termination_reason: TerminationReason::InfeasibilityProven,
        }
    }

    /// Creates a new `PortfolioSolverResult` representing an aborted solve.
    /// If a solution is provided, it is treated as feasible; otherwise, the result is infeasible.
    #[inline]
    pub fn aborted<R>(solution: Option<Solution<T>>, reason: R) -> Self
    where
        R: Into<String>,
    {
        let termination_reason = TerminationReason::Aborted(reason.into());

        let result = match solution {
            Some(sol) => SolverResult::Feasible(sol),
            None => SolverResult::Infeasible,
        };

        Self {
            result,
            termination_reason,
        }
    }

    /// Returns a reference to the solver result.
    #[inline]
    pub fn result(&self) -> &SolverResult<T> {
        &self.result
    }

    /// Returns a reference to the termination reason.
    #[inline]
    pub fn termination_reason(&self) -> &TerminationReason {
        &self.termination_reason
    }
}

impl<T> std::fmt::Display for PortfolioSolverResult<T>
where
    T: PrimInt + Signed + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PortfolioSolverResult(result: {}, termination_reason: {})",
            self.result, self.termination_reason
        )
    }
}

/// A trait representing a portfolio solver.
/// Such solvers can be runned in parallel as part
/// of a portfolio approach to solving the Berth Allocation Problem.
pub trait PortofolioSolver<T>
where
    T: PrimInt + Signed,
{
    /// Solves the given model within the provided context.
    fn invoke<'a>(&mut self, context: PortfolioSolverContext<'a, T>) -> PortfolioSolverResult<T>;
    /// Returns the name of the portfolio solver.
    fn name(&self) -> &str;
}
