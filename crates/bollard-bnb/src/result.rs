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
use bollard_search::result::{SolverResult, TerminationReason};

/// Result of the solver after termination.
#[derive(Debug, Clone)]
pub struct BnbSolverOutcome<T> {
    result: SolverResult<T>,
    termination_reason: TerminationReason,
    statistics: BnbSolverStatistics,
}

impl<T> BnbSolverOutcome<T> {
    #[inline]
    pub fn optimal(solution: Solution<T>, statistics: BnbSolverStatistics) -> Self {
        Self {
            result: SolverResult::Optimal(solution),
            termination_reason: TerminationReason::OptimalityProven,
            statistics,
        }
    }

    #[inline]
    pub fn infeasible(statistics: BnbSolverStatistics) -> Self {
        Self {
            result: SolverResult::Infeasible,
            termination_reason: TerminationReason::InfeasibilityProven,
            statistics,
        }
    }

    #[inline]
    pub fn aborted<R>(
        solution: Option<Solution<T>>,
        reason: R,
        statistics: BnbSolverStatistics,
    ) -> Self
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
    pub fn termination_reason(&self) -> &TerminationReason {
        &self.termination_reason
    }

    /// Returns the solver statistics.
    #[inline]
    pub fn statistics(&self) -> &BnbSolverStatistics {
        &self.statistics
    }
}
