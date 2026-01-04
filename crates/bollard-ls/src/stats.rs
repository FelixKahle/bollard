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

//! Statistics reporting for local search runs.
//!
//! This module defines a lightweight container for tracking aggregate metrics
//! during a local search, including iteration count, number of candidates found,
//! number of accepted moves, and total elapsed time. The interface is optimized
//! for hot-loop usage: updates rely on saturating arithmetic to avoid overflow
//! traps and expose clear, inline methods for per-iteration and per-event
//! accounting. The resulting `LocalSearchStatistics` can be consumed by monitors,
//! metaheuristics, and result reporting to provide visibility into solver progress
//! and convergence behavior without imposing measurable overhead on the inner loop.

use bollard_core::num::ops::saturating_arithmetic::SaturatingAddVal;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocalSearchStatistics {
    /// Number of iterations performed by the local search.
    pub iterations: u64,

    /// Total number of solutions found during the local search.
    pub total_solutions: u64,

    /// Number of accepted solutions during the local search.
    pub accepted_solutions: u64,

    /// Total time taken by the local search.
    pub time_total: Duration,
}

impl Default for LocalSearchStatistics {
    fn default() -> Self {
        Self {
            iterations: 0,
            total_solutions: 0,
            accepted_solutions: 0,
            time_total: Duration::ZERO,
        }
    }
}

impl LocalSearchStatistics {
    /// Called at each iteration of the local search.
    #[inline]
    pub fn on_iteration(&mut self) {
        self.iterations = self.iterations.saturating_add_val(1);
    }

    /// Called when a solution is found.
    #[inline]
    pub fn on_found_solution(&mut self) {
        self.total_solutions = self.total_solutions.saturating_add_val(1);
    }

    /// Called when a solution is accepted.
    #[inline]
    pub fn on_accepted_solution(&mut self) {
        self.accepted_solutions = self.accepted_solutions.saturating_add_val(1);
    }

    /// Sets the total time taken by the local search.
    #[inline]
    pub fn set_total_time(&mut self, duration: Duration) {
        self.time_total = duration;
    }

    #[inline]
    pub fn rejected_solutions(&self) -> u64 {
        self.total_solutions.saturating_sub(self.accepted_solutions)
    }
}

impl std::fmt::Display for LocalSearchStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Bollard-LS-Solver Statistics:")?;
        writeln!(f, "   Iterations:           {}", self.iterations)?;
        writeln!(f, "   Total Solutions:     {}", self.total_solutions)?;
        writeln!(f, "   Accepted Solutions:  {}", self.accepted_solutions)?;
        writeln!(f, "   Rejected Solutions:  {}", self.rejected_solutions())?;
        writeln!(f, "   Total Time:         {:?}", self.time_total)?;
        Ok(())
    }
}
