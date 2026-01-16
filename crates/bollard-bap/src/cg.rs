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

use crate::pricing::PricingOracle;
use crate::simplex::{OptimizationPhase, SimplexOptimizer};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CgStatus {
    Optimal(f64),
    Infeasible,
    LimitReached,
}

/// The Column Generation Solver.
pub struct ColumnGenerator {
    /// The Linear Programming solver (Master Problem).
    simplex: SimplexOptimizer,
    /// Maximum number of CG iterations.
    max_iterations: usize,
}

impl ColumnGenerator {
    /// Creates a new Column Generator.
    /// Note: No Oracle reference needed here anymore.
    pub fn new(num_vessels: usize) -> Self {
        Self {
            simplex: SimplexOptimizer::new(num_vessels),
            max_iterations: 10_000,
        }
    }

    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Solves the Linear Relaxation using Column Generation.
    ///
    /// # Generic Oracle
    /// We use generics `<O: PricingOracle>` here to allow static dispatch (inlining)
    /// of the pricing logic, just like we did for the Trail.
    pub fn solve<O: PricingOracle>(&mut self, oracle: &mut O) -> CgStatus {
        let mut iter_count = 0;

        loop {
            if iter_count >= self.max_iterations {
                return CgStatus::LimitReached;
            }
            iter_count += 1;

            self.simplex.recompute_state();

            if self.simplex.phase() == OptimizationPhase::Feasibility
                && self.simplex.try_transition_phase() {
                    continue;
                }

            let duals = self.simplex.duals();
            let is_phase_2 = self.simplex.phase() == OptimizationPhase::Optimality;
            let new_columns = oracle.solve_pricing(duals, is_phase_2);

            if new_columns.is_empty() {
                if self.simplex.phase() == OptimizationPhase::Feasibility {
                    if self.simplex.objective_value() > 1e-6 {
                        return CgStatus::Infeasible;
                    }
                    if self.simplex.try_transition_phase() {
                        continue;
                    }
                }
                return CgStatus::Optimal(self.simplex.objective_value());
            }

            let mut pivoted = false;
            for col in new_columns {
                if self.simplex.perform_pivot(col) {
                    pivoted = true;
                    break;
                }
            }

            if !pivoted {
                return CgStatus::Optimal(self.simplex.objective_value());
            }
        }
    }

    pub fn simplex(&self) -> &SimplexOptimizer {
        &self.simplex
    }
}
