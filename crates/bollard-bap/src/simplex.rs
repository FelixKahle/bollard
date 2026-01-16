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

use crate::column::{Column, ColumnKind};
use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::partial_pivoting::{factor, solve};
use faer::perm::PermRef;
use faer::prelude::*;
use faer::{Conj, Par};

/// Numerical tolerance for floating point comparisons.
const NUMERICAL_TOLERANCE: f64 = 1e-9;

/// Defines the current operating mode of the Simplex solver (Two-Phase Method).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationPhase {
    /// Phase 1: Minimize the sum of Artificial Variables to find a valid schedule.
    Feasibility,
    /// Phase 2: Minimize the actual Weighted Flow Time.
    Optimality,
}

/// The Low-Level Optimized Revised Simplex Engine.
///
/// This struct manages the linear algebra state for the Column Generation process.
/// It uses `faer`'s low-level API with persistent memory buffers to perform
/// decompositions with zero heap allocations in the hot loop.
pub struct SimplexOptimizer {
    /// The number of constraints (Rows), equivalent to the number of vessels.
    num_constraints: usize,

    // --- Solver State (Private) ---
    /// The pool of ALL columns (variables) generated so far.
    columns: Vec<Column>,

    /// The indices of the columns currently in the Basis.
    basis: Vec<usize>,

    /// The current values of the Basic Variables ($x_B = B^{-1} \mathbf{1}$).
    primal_solution: Vec<f64>,

    /// The current Dual Values ($\pi = c_B B^{-1}$).
    duals: Vec<f64>,

    /// The current phase of the Two-Phase Simplex method.
    phase: OptimizationPhase,

    // --- Persistent Workspaces (Private Arena) ---
    /// The Basis Matrix workspace ($N \times N$).
    basis_matrix_workspace: Mat<f64>,

    /// The Cost Vector workspace ($N \times 1$).
    basis_cost_workspace: Mat<f64>,

    /// The Reusable RHS Vector workspace ($N \times 1$).
    rhs_workspace: Mat<f64>,

    /// Forward permutation indices for Partial Pivoting LU.
    row_permutation: Vec<usize>,

    /// Inverse permutation indices.
    row_permutation_inverse: Vec<usize>,

    /// Single block of pre-allocated scratch memory for faer algorithms.
    memory_buffer: MemBuffer,
}

impl SimplexOptimizer {
    /// Initializes the optimizer for a problem with `num_vessels` constraints.
    pub fn new(num_vessels: usize) -> Self {
        let mut columns = Vec::with_capacity(num_vessels);
        let mut basis = Vec::with_capacity(num_vessels);

        // 1. Initialize with Artificial Columns (Identity Basis)
        for i in 0..num_vessels {
            columns.push(Column::new_artificial(i, num_vessels));
            basis.push(i);
        }

        // 2. Compute Scratch Memory Requirements
        // Using Default::default() to let Rust infer the correct Spec type for params.

        let req_factor = factor::lu_in_place_scratch::<usize, f64>(
            num_vessels,
            num_vessels,
            Par::Seq,
            Default::default(),
        );

        let req_solve = solve::solve_in_place_scratch::<usize, f64>(num_vessels, 1, Par::Seq);

        let req_solve_transpose =
            solve::solve_transpose_in_place_scratch::<usize, f64>(num_vessels, 1, Par::Seq);

        let max_requirement = req_factor.or(req_solve).or(req_solve_transpose);
        let memory_buffer = MemBuffer::new(max_requirement);

        Self {
            num_constraints: num_vessels,
            columns,
            basis,
            primal_solution: vec![1.0; num_vessels],
            duals: vec![0.0; num_vessels],
            phase: OptimizationPhase::Feasibility,

            basis_matrix_workspace: Mat::zeros(num_vessels, num_vessels),
            basis_cost_workspace: Mat::zeros(num_vessels, 1),
            rhs_workspace: Mat::zeros(num_vessels, 1),

            row_permutation: vec![0; num_vessels],
            row_permutation_inverse: vec![0; num_vessels],

            memory_buffer,
        }
    }

    // --- Public Getters (Read-Only API) ---

    /// Returns the current Dual Values (Shadow Prices).
    /// Used by the Pricing Oracle to calculate reduced costs.
    #[inline(always)]
    pub fn duals(&self) -> &[f64] {
        &self.duals
    }

    /// Returns the current optimization phase (Feasibility or Optimality).
    #[inline(always)]
    pub fn phase(&self) -> OptimizationPhase {
        self.phase
    }

    /// Returns all columns generated so far.
    #[inline(always)]
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// Returns the number of constraints (vessels).
    #[inline(always)]
    pub fn num_constraints(&self) -> usize {
        self.num_constraints
    }

    /// Calculates the current Objective Function value.
    pub fn objective_value(&self) -> f64 {
        let mut objective = 0.0;

        for (basis_index, &column_index) in self.basis.iter().enumerate() {
            debug_assert!(
                basis_index < self.basis.len(),
                "`Simplex::objective_value`: basis_index {} is out of bounds for basis (len = {})",
                basis_index,
                self.basis.len(),
            );
            debug_assert!(
                basis_index < self.primal_solution.len(),
                "`Simplex::objective_value`: basis_index {} is out of bounds for primal_solution (len = {})",
                basis_index,
                self.primal_solution.len(),
            );
            debug_assert!(
                column_index < self.columns.len(),
                "`Simplex::objective_value`: basis column index {} is out of bounds (len = {})",
                column_index,
                self.columns.len(),
            );

            // SAFETY: `column_index` is asserted to be strictly less than `self.columns.len()`
            // above. The basis is maintained so that every entry is a valid column index.
            let column = unsafe { self.columns.get_unchecked(column_index) };

            // SAFETY: `self.primal_solution` is kept in sync with `self.basis`, so `basis_index`
            // (coming from `self.basis.iter().enumerate()`) is always a valid index.
            let solution = unsafe { self.primal_solution.get_unchecked(basis_index) };

            objective += self.get_phase_aware_cost(column) * solution;
        }
        objective
    }

    pub fn iter_active_columns(&self) -> impl Iterator<Item = (&Column, f64)> + '_ {
        self.basis
            .iter()
            .enumerate()
            .map(move |(basis_index, &basis_column_index)| {
                debug_assert!(
                    basis_column_index < self.columns.len(),
                    "`Simplex::iter_active_columns`: basis column index {} is out of bounds for columns (len = {})",
                    basis_column_index,
                    self.columns.len(),
                );
                debug_assert!(
                    basis_index < self.primal_solution.len(),
                    "`Simplex::iter_active_columns`: basis_index {} is out of bounds for primal_solution (len = {})",
                    basis_index,
                    self.primal_solution.len(),
                );

                // SAFETY:
                // - `basis_column_index` is asserted in-bounds for `self.columns`.
                // - `basis_index` is asserted in-bounds for `self.primal_solution`.
                let column = unsafe { self.columns.get_unchecked(basis_column_index) };
                let value = unsafe { *self.primal_solution.get_unchecked(basis_index) };

                (column, value)
            })
    }

    #[inline(always)]
    fn get_phase_aware_cost(&self, col: &Column) -> f64 {
        match (self.phase, col.kind()) {
            (OptimizationPhase::Feasibility, ColumnKind::Artificial) => 1.0,
            (OptimizationPhase::Feasibility, ColumnKind::Regular) => 0.0,
            (OptimizationPhase::Optimality, ColumnKind::Artificial) => f64::INFINITY,
            (OptimizationPhase::Optimality, ColumnKind::Regular) => col.cost(),
        }
    }

    /// Re-computes the Primal ($x$) and Dual ($\pi$) solutions from scratch.
    ///
    /// This rebuilds the basis matrix, factorizes it, and solves the linear systems.
    /// Must be called after `perform_pivot` or `try_transition_phase`.
    pub fn recompute_state(&mut self) {
        let num_constraints = self.num_constraints;

        // 1. Rebuild Dense Basis Matrix
        self.basis_matrix_workspace.fill(0.0);

        for (basis_index, &basis_column_index) in self.basis.iter().enumerate() {
            debug_assert!(
                basis_column_index < self.columns.len(),
                "`Simplex::recompute_state`: basis column index {} is out of bounds (len = {})",
                basis_column_index,
                self.columns.len(),
            );

            // SAFETY: `basis_column_index` was checked above to be in-bounds for `self.columns`.
            // The basis is maintained so that every entry is a valid column index.
            let basis_column = unsafe { self.columns.get_unchecked(basis_column_index) };

            self.basis_cost_workspace[(basis_index, 0)] = self.get_phase_aware_cost(basis_column);

            for vessel_row_index in basis_column.covered_vessels().ones() {
                debug_assert!(
                    vessel_row_index < num_constraints,
                    "`Simplex::recompute_state`: vessel_row_index {} is out of bounds (num_constraints = {})",
                    vessel_row_index,
                    num_constraints,
                );

                // faer matrix indexing uses (row, col)
                self.basis_matrix_workspace[(vessel_row_index, basis_index)] = 1.0;
            }
        }

        // 2. LU Factorization (In-Place)
        {
            let stack = MemStack::new(&mut self.memory_buffer);
            factor::lu_in_place(
                self.basis_matrix_workspace.as_mut(),
                &mut self.row_permutation,
                &mut self.row_permutation_inverse,
                Par::Seq,
                stack,
                Default::default(),
            );
        }

        let permutation = PermRef::new_checked(
            &self.row_permutation,
            &self.row_permutation_inverse,
            num_constraints,
        );

        // 3. Solve Primal (B * x = 1)
        self.rhs_workspace.fill(1.0);
        {
            let stack = MemStack::new(&mut self.memory_buffer);
            solve::solve_in_place(
                self.basis_matrix_workspace.as_ref(), // L
                self.basis_matrix_workspace.as_ref(), // U
                permutation,
                self.rhs_workspace.as_mut(),
                Par::Seq,
                stack,
            );
        }

        // Update Primal Solution (Vector Copy)
        for constraint_index in 0..num_constraints {
            debug_assert!(
                constraint_index < self.primal_solution.len(),
                "`Simplex::recompute_state`: constraint_index {} is out of bounds for primal_solution (len = {})",
                constraint_index,
                self.primal_solution.len(),
            );

            // SAFETY: `constraint_index` is strictly less than `self.primal_solution.len()`
            // (asserted above), and the loop is bounded by `num_constraints` which matches
            // the logical size of `primal_solution`.
            unsafe {
                *self.primal_solution.get_unchecked_mut(constraint_index) =
                    self.rhs_workspace[(constraint_index, 0)];
            }
        }

        // 4. Solve Dual (B^T * pi = c_B)
        self.rhs_workspace.copy_from(&self.basis_cost_workspace);
        {
            let stack = MemStack::new(&mut self.memory_buffer);
            solve::solve_transpose_in_place_with_conj(
                self.basis_matrix_workspace.as_ref(), // L
                self.basis_matrix_workspace.as_ref(), // U
                permutation,
                Conj::No,
                self.rhs_workspace.as_mut(),
                Par::Seq,
                stack,
            );
        }

        // Update Duals (Vector Copy)
        for constraint_index in 0..num_constraints {
            debug_assert!(
                constraint_index < self.duals.len(),
                "`Simplex::recompute_state`: constraint_index {} is out of bounds for duals (len = {})",
                constraint_index,
                self.duals.len(),
            );

            // SAFETY: `constraint_index` is strictly less than `self.duals.len()`
            // (asserted above), and the loop is bounded by `num_constraints` which matches
            // the logical size of `duals`.
            unsafe {
                *self.duals.get_unchecked_mut(constraint_index) =
                    self.rhs_workspace[(constraint_index, 0)];
            }
        }
    }

    /// Attempts to transition from Phase 1 to Phase 2.
    ///
    /// Returns `true` if the transition was successful (feasible solution found).
    pub fn try_transition_phase(&mut self) -> bool {
        if self.phase == OptimizationPhase::Optimality {
            return false;
        }

        // Check Phase 1 Objective (Sum of artificials)
        let mut artificial_sum = 0.0;
        for (basis_index, &basis_column_index) in self.basis.iter().enumerate() {
            debug_assert!(
                basis_index < self.basis.len(),
                "`Simplex::try_transition_phase`: basis_index {} is out of bounds for basis (len = {})",
                basis_index,
                self.basis.len(),
            );
            debug_assert!(
                basis_index < self.primal_solution.len(),
                "`Simplex::try_transition_phase`: basis_index {} is out of bounds for primal_solution (len = {})",
                basis_index,
                self.primal_solution.len(),
            );
            debug_assert!(
                basis_column_index < self.columns.len(),
                "`Simplex::try_transition_phase`: basis column index {} is out of bounds for columns (len = {})",
                basis_column_index,
                self.columns.len(),
            );

            // SAFETY: `basis_column_index` is asserted to be strictly less than `self.columns.len()`.
            //         `basis_index` is asserted to be strictly less than `self.primal_solution.len()`.
            let column = unsafe { self.columns.get_unchecked(basis_column_index) };
            let primal_value = unsafe { *self.primal_solution.get_unchecked(basis_index) };

            if column.kind() == ColumnKind::Artificial {
                artificial_sum += primal_value;
            }
        }

        if artificial_sum.abs() < NUMERICAL_TOLERANCE {
            self.phase = OptimizationPhase::Optimality;
            // Costs changed (Artificial -> Infinity), must recompute.
            self.recompute_state();
            return true;
        }

        false
    }

    /// Performs a Simplex Pivot operation to introduce a new column into the basis.
    ///
    /// Returns `true` if the pivot was successful.
    pub fn perform_pivot(&mut self, new_column: Column) -> bool {
        let num_constraints = self.num_constraints;

        // 1. Construct Entering Vector (A_j) into workspace
        self.rhs_workspace.fill(0.0);
        for vessel_row_index in new_column.covered_vessels().ones() {
            debug_assert!(
                vessel_row_index < num_constraints,
                "`Simplex::perform_pivot`: vessel_row_index {} is out of bounds (num_constraints = {})",
                vessel_row_index,
                num_constraints,
            );

            // faer matrix indexing uses (row, col)
            self.rhs_workspace[(vessel_row_index, 0)] = 1.0;
        }

        // 2. Solve for Direction d (B * d = A_j)
        // Reuse LU factors from basis_matrix_workspace
        let permutation = PermRef::new_checked(
            &self.row_permutation,
            &self.row_permutation_inverse,
            num_constraints,
        );

        {
            let stack = MemStack::new(&mut self.memory_buffer);
            solve::solve_in_place(
                self.basis_matrix_workspace.as_ref(), // L
                self.basis_matrix_workspace.as_ref(), // U
                permutation,
                self.rhs_workspace.as_mut(), // Destination (d)
                Par::Seq,
                stack,
            );
        }

        // 3. Ratio Test
        let mut minimum_theta = f64::MAX;
        let mut leaving_basis_index: Option<usize> = None;

        for constraint_index in 0..num_constraints {
            debug_assert!(
                constraint_index < self.rhs_workspace.nrows(),
                "`Simplex::perform_pivot`: constraint_index {} is out of bounds for rhs_workspace rows (nrows = {})",
                constraint_index,
                self.rhs_workspace.nrows(),
            );
            debug_assert!(
                constraint_index < self.primal_solution.len(),
                "`Simplex::perform_pivot`: constraint_index {} is out of bounds for primal_solution (len = {})",
                constraint_index,
                self.primal_solution.len(),
            );

            let direction_component = self.rhs_workspace[(constraint_index, 0)];

            if direction_component > NUMERICAL_TOLERANCE {
                let primal_value = self.primal_solution[constraint_index];
                let theta = primal_value / direction_component;

                if theta < minimum_theta {
                    minimum_theta = theta;
                    leaving_basis_index = Some(constraint_index);
                }
            }
        }

        if let Some(leaving_index) = leaving_basis_index {
            debug_assert!(
                leaving_index < self.basis.len(),
                "`Simplex::perform_pivot`: leaving_index {} is out of bounds for basis (len = {})",
                leaving_index,
                self.basis.len(),
            );

            let new_global_index = self.columns.len();

            // The new entering column is appended to `columns` and replaces the leaving basis
            // column index in `basis`.
            self.columns.push(new_column);
            self.basis[leaving_index] = new_global_index;

            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use bollard_model::index::{BerthIndex, VesselIndex};

    fn create_regular_col(total_vessels: usize, vessels: &[usize], cost: f64) -> Column {
        let mut assignments = Vec::with_capacity(vessels.len());

        for &vessel_index in vessels {
            assert!(
                vessel_index < total_vessels,
                "`Simplex::create_regular_col`: vessel_index {} is out of bounds (total_vessels = {})",
                vessel_index,
                total_vessels,
            );

            // We use a dummy start time of 0.0 here; the pricing/oracle layer is responsible
            // for assigning meaningful schedule times.
            assignments.push((VesselIndex::new(vessel_index), 0.0));
        }

        // Berth index is not relevant for this helper; use berth 0 as a sentinel.
        // SAFETY: `Column::new_regular` will rebuild the `covered_vessels` bitset from
        // `assignments` and assert the same bounds invariants again in debug builds.
        let berth_index = BerthIndex::new(0);
        Column::new_regular(berth_index, assignments, cost, total_vessels)
    }

    #[test]
    fn test_initialization() {
        let n = 3;
        let opt = SimplexOptimizer::new(n);

        assert_eq!(opt.phase(), OptimizationPhase::Feasibility);
        assert_eq!(opt.num_constraints(), n);
        assert_eq!(opt.columns().len(), n); // N artificials

        // Check duals (initially 0.0 until updated)
        assert_eq!(opt.duals().len(), n);
    }

    #[test]
    fn test_phase_transition() {
        // N=1 case.
        // Initial: Basis=[Artificial], x=1.0, cost=1.0.
        // We pivot in a Regular column with cost 10.0.
        let n = 1;
        let mut opt = SimplexOptimizer::new(n);

        opt.recompute_state();
        assert_eq!(opt.phase(), OptimizationPhase::Feasibility);

        // Pivot in valid column
        let reg_col = create_regular_col(n, &[0], 10.0);
        let success = opt.perform_pivot(reg_col);
        assert!(success);

        // Update state to reflect pivot
        opt.recompute_state();

        // Transition
        let transitioned = opt.try_transition_phase();
        assert!(transitioned);
        assert_eq!(opt.phase(), OptimizationPhase::Optimality);

        // Check duals in Optimality phase (should be 10.0)
        assert!((opt.duals()[0] - 10.0).abs() < NUMERICAL_TOLERANCE);
        assert!((opt.objective_value() - 10.0).abs() < NUMERICAL_TOLERANCE);
    }
}
