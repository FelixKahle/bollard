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

//! Local search outcome and termination reporting.
//!
//! This module encapsulates the final result produced by the local search engine,
//! including the best solution discovered, aggregate run statistics, and a concise
//! termination reason. The `LocalSearchEngineOutcome` serves as a single transport
//! object for downstream consumers such as monitors, CLI tools, or higher-level
//! orchestration logic. Termination reasons distinguish between reaching a local
//! optimum, a metaheuristic-requested stop, and solver-imposed limits, making it
//! straightforward to audit the end state of a run and to integrate with broader
//! experiment pipelines or production workflows.

use crate::stats::LocalSearchStatistics;
use bollard_model::solution::Solution;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LocalSearchTerminationReason {
    /// The neighborhood has been fully explored without accepting any new solution.
    LocalOptimum,

    /// The metaheuristic decided to abort the search.
    Metaheuristic(String),

    /// The solver aborted due to a search limit (time, iterations, etc.).
    /// The string contains information about the reason for abortion.
    Aborted(String),
}

impl std::fmt::Display for LocalSearchTerminationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LocalSearchTerminationReason::LocalOptimum => write!(f, "Local Optimum Reached"),
            LocalSearchTerminationReason::Metaheuristic(msg) => {
                write!(f, "Metaheuristic Termination: {}", msg)
            }
            LocalSearchTerminationReason::Aborted(msg) => write!(f, "Aborted: {}", msg),
        }
    }
}

/// Result of the solver after termination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalSearchEngineOutcome<T> {
    termination_reason: LocalSearchTerminationReason,
    solution: Solution<T>,
    statistics: LocalSearchStatistics,
}

impl<T> LocalSearchEngineOutcome<T> {
    /// Creates a new local optimum outcome.
    #[inline]
    pub fn local_optimum(solution: Solution<T>, statistics: LocalSearchStatistics) -> Self {
        Self {
            termination_reason: LocalSearchTerminationReason::LocalOptimum,
            solution,
            statistics,
        }
    }

    /// Creates a new metaheuristic-initiated termination outcome.
    #[inline]
    pub fn metaheuristic<R>(
        solution: Solution<T>,
        name: R,
        statistics: LocalSearchStatistics,
    ) -> Self
    where
        R: Into<String>,
    {
        Self {
            termination_reason: LocalSearchTerminationReason::Metaheuristic(name.into()),
            solution,
            statistics,
        }
    }

    /// Creates a new aborted outcome.
    #[inline]
    pub fn aborted<R>(solution: Solution<T>, reason: R, statistics: LocalSearchStatistics) -> Self
    where
        R: Into<String>,
    {
        Self {
            termination_reason: LocalSearchTerminationReason::Aborted(reason.into()),
            solution,
            statistics,
        }
    }

    /// Returns the termination reason.
    #[inline]
    pub fn termination_reason(&self) -> &LocalSearchTerminationReason {
        &self.termination_reason
    }

    /// Returns the solution.
    #[inline]
    pub fn solution(&self) -> &Solution<T> {
        &self.solution
    }

    /// Returns the statistics.
    #[inline]
    pub fn statistics(&self) -> &LocalSearchStatistics {
        &self.statistics
    }
}
