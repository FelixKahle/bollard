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

use crate::{column::Column, constraint::BranchConstraint};

/// A trait for solvers that can generate columns with negative reduced cost.
///
/// This is the bridge between the Master Problem (Simplex) and the Subproblem (CP/RCSPP).
pub trait PricingOracle {
    /// Solves the Pricing Subproblem.
    ///
    /// # Arguments
    /// * `duals`: The current vector of shadow prices ($\pi$) from the Simplex.
    /// * `use_real_costs`:
    ///     - `false` (Phase 1): Minimize $0 - \pi A_j$.
    ///     - `true` (Phase 2): Minimize $c_j - \pi A_j$.
    fn solve_pricing(&mut self, duals: &[f64], use_real_costs: bool) -> Vec<Column>;

    /// Adds a branching constraint to the subproblem.
    ///
    /// The oracle must respect this constraint in all future calls to `solve_pricing`
    /// until it is removed via `remove_constraint`.
    fn add_constraint(&mut self, constraint: BranchConstraint);

    /// Removes a previously added branching constraint.
    ///
    /// This is used during backtracking to restore the oracle's state.
    fn remove_constraint(&mut self, constraint: BranchConstraint);
}
