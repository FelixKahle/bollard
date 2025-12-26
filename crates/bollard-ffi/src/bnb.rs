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

//! # Foreign Function Interface (FFI) for the Bollard Branch-and-Bound Solver
//!
//! This module provides a stable, C-compatible ABI for interacting with the
//! **Bollard Branch-and-Bound (BnB)** solver. It handles the solver instantiation,
//! execution strategies, and the inspection of results.
//!
//! ## Overview
//!
//! This module exposes two main components:
//!
//! 1.  **Solver API**: Functions to instantiate `BnbSolver` and execute searches using
//!     combinatoric strategies (Objective Evaluators + Search Heuristics).
//! 2.  **Outcome API**: A set of accessors for `BnbSolverFfiOutcome` to inspect optimization
//!     results, extraction solutions, and analyze search statistics (e.g., nodes explored).
//!
//! ## Usage Lifecycle
//!
//! 1.  **Solver Instantiation**:
//!     * Create a `BnbSolver` instance using `bollard_bnb_solver_new` or `bollard_bnb_solver_preallocated`.
//! 2.  **Execution**:
//!     * Call a specific solve function (e.g., `bollard_bnb_solver_solve_with_hybrid_evaluator_and_regret_heuristic_builder`).
//!     * Requires a valid `Model` pointer (created via the separate Model API).
//!     * Accepts constraints (time limits, solution limits) and optional fixed assignments.
//! 3.  **Inspection**:
//!     * The solve function returns a `BnbSolverFfiOutcome` pointer.
//!     * Use `bollard_bnb_outcome_*` functions to check status and read data.
//! 4.  **Cleanup**:
//!     * Free the solver via `bollard_bnb_solver_free`.
//!     * Free the outcome via `bollard_bnb_outcome_free`.
//!
//! ## Safety
//!
//! This module uses `unsafe` code to handle raw pointers. Callers **must** ensure:
//!
//! * **Pointer Validity**: Pointers must be valid and allocated by this library.
//! * **Ownership**: `_free` functions invalidate the passed pointer immediately.
//! * **Null Pointers**: Passing `NULL` will strictly **panic** (abort the process).
//!
//! ## Exported Functions
//!
//! ### 1. Solver Lifecycle
//! * `bollard_bnb_solver_new`
//! * `bollard_bnb_solver_preallocated`
//! * `bollard_bnb_solver_free`
//!
//! ### 2. Solve Functions (Hybrid Evaluator)
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_chronological_exhaustive_builder`
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_chronological_exhaustive_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_fcfs_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_fcfs_heuristic_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_regret_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_regret_heuristic_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_slack_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_slack_heuristic_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_wspt_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_wspt_heuristic_builder_with_fixed`
//!
//! ### 3. Solve Functions (Workload Evaluator)
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_chronological_exhaustive_builder`
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_chronological_exhaustive_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_fcfs_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_fcfs_heuristic_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_regret_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_regret_heuristic_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_slack_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_slack_heuristic_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_wspt_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_wspt_heuristic_builder_with_fixed`
//!
//! ### 4. Solve Functions (Weighted Flow Time Evaluator)
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_chronological_exhaustive_builder`
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_chronological_exhaustive_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_fcfs_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_fcfs_heuristic_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_regret_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_regret_heuristic_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_slack_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_slack_heuristic_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_wspt_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_wspt_heuristic_builder_with_fixed`
//!
//! ### 5. Outcome API
//!
//! **Lifecycle**
//! * `bollard_bnb_outcome_free`
//!
//! **Status & Metadata**
//! * `bollard_bnb_outcome_has_solution`
//! * `bollard_bnb_outcome_get_status`
//! * `bollard_bnb_outcome_get_status_str`
//! * `bollard_bnb_outcome_get_termination_reason_enum`
//! * `bollard_bnb_outcome_get_termination_reason`
//!
//! **Solution Data**
//! * `bollard_bnb_outcome_get_objective`
//! * `bollard_bnb_outcome_get_num_vessels`
//! * `bollard_bnb_outcome_get_berth`
//! * `bollard_bnb_outcome_get_start_time`
//! * `bollard_bnb_outcome_get_berths` (Direct pointer access)
//! * `bollard_bnb_outcome_get_start_times` (Direct pointer access)
//! * `bollard_bnb_outcome_copy_solution` (Batch copy to user buffers)
//!
//! **Statistics**
//! * `bollard_bnb_outcome_get_nodes_explored`
//! * `bollard_bnb_outcome_get_backtracks`
//! * `bollard_bnb_outcome_get_decisions_generated`
//! * `bollard_bnb_outcome_get_max_depth`
//! * `bollard_bnb_outcome_get_prunings_infeasible`
//! * `bollard_bnb_outcome_get_prunings_bound`
//! * `bollard_bnb_outcome_get_solutions_found`
//! * `bollard_bnb_outcome_get_steps`
//! * `bollard_bnb_outcome_get_time_total_ms`

use bollard_bnb::result::BnbSolverOutcome;
use bollard_bnb::{
    bnb::BnbSolver,
    branching::{
        chronological::ChronologicalExhaustiveBuilder, decision::DecisionBuilder,
        fcfs::FcfsHeuristicBuilder, regret::RegretHeuristicBuilder, slack::SlackHeuristicBuilder,
        wspt::WsptHeuristicBuilder,
    },
    eval::{
        evaluator::ObjectiveEvaluator, hybrid::HybridEvaluator, workload::WorkloadEvaluator,
        wtft::WeightedFlowTimeEvaluator,
    },
    fixed::FixedAssignment,
    monitor::{
        composite::CompositeTreeSearchMonitor, log::LogTreeSearchMonitor,
        solution::SolutionLimitMonitor, time::TimeLimitMonitor,
    },
};
use bollard_model::index::VesselIndex;
use bollard_model::model::Model;
use bollard_search::result::{SolverResult, TerminationReason};
use libc::c_char;
use std::ffi::CString;
use std::time::Duration;

/// FFI-compatible representation of the solver outcome.
/// Includes C strings for termination reason and status.
/// The C strings are owned by this struct
/// and valid until the struct is freed.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct BnbSolverFfiOutcome {
    inner: BnbSolverOutcome<i64>,     // Internal outcome with i64 objective
    termination_reason_cstr: CString, // Owned C string for termination reason
    status_cstr: CString,             // Owned C string for status
}

/// FFI-compatible enum for solver status.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BnbSolverFfiStatus {
    Optimal = 0,
    Feasible = 1,
    Infeasible = 2,
    Unknown = 3,
}

impl BnbSolverFfiStatus {
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Optimal => "Optimal",
            Self::Feasible => "Feasible",
            Self::Infeasible => "Infeasible",
            Self::Unknown => "Unknown",
        }
    }
}

impl From<&SolverResult<i64>> for BnbSolverFfiStatus {
    #[inline]
    fn from(result: &SolverResult<i64>) -> Self {
        match result {
            SolverResult::Optimal(_) => BnbSolverFfiStatus::Optimal,
            SolverResult::Feasible(_) => BnbSolverFfiStatus::Feasible,
            SolverResult::Infeasible => BnbSolverFfiStatus::Infeasible,
            SolverResult::Unknown => BnbSolverFfiStatus::Unknown,
        }
    }
}

impl std::fmt::Display for BnbSolverFfiStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// FFI-compatible enum for termination reason.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BnbSolverFfiTerminationReason {
    OptimalityProven = 0,
    InfeasibilityProven = 1,
    Aborted = 2,
}

impl BnbSolverFfiTerminationReason {
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::OptimalityProven => "OptimalityProven",
            Self::InfeasibilityProven => "InfeasibilityProven",
            Self::Aborted => "Aborted",
        }
    }
}

impl From<&TerminationReason> for BnbSolverFfiTerminationReason {
    #[inline]
    fn from(reason: &TerminationReason) -> Self {
        match reason {
            TerminationReason::OptimalityProven => BnbSolverFfiTerminationReason::OptimalityProven,
            TerminationReason::InfeasibilityProven => {
                BnbSolverFfiTerminationReason::InfeasibilityProven
            }
            TerminationReason::Aborted(_) => BnbSolverFfiTerminationReason::Aborted,
        }
    }
}

impl std::fmt::Display for BnbSolverFfiTerminationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl BnbSolverFfiOutcome {
    /// Constructs a `BnbSolverFFIOutcome` from a `BnbSolverOutcome<i64>`.
    ///
    /// # Panics
    ///
    /// This function will panic if the C strings cannot be created,
    /// which should not happen for valid termination reasons and statuses.
    #[inline]
    pub fn new(inner: BnbSolverOutcome<i64>) -> Self {
        let term_str = match inner.termination_reason() {
            TerminationReason::Aborted(s) => s.as_str(),
            reason => BnbSolverFfiTerminationReason::from(reason).as_str(),
        };

        // Kill the process if CString creation fails (should not happen)
        let termination_reason_cstr =
            CString::new(term_str).expect("`CString::new` should create valid C string");

        let status_str = BnbSolverFfiStatus::from(inner.result()).as_str();

        // Kill the process if CString creation fails (should not happen)
        let status_cstr =
            CString::new(status_str).expect("`CString::new` should create valid C string");

        Self {
            inner,
            termination_reason_cstr,
            status_cstr,
        }
    }

    /// Returns a reference to the inner `BnbSolverOutcome<i64>`.
    #[inline]
    pub fn inner(&self) -> &BnbSolverOutcome<i64> {
        &self.inner
    }

    /// Returns a reference to the termination reason C string.
    #[inline]
    pub fn termination_reason(&self) -> &CString {
        &self.termination_reason_cstr
    }

    /// Returns a reference to the status C string.
    #[inline]
    pub fn status(&self) -> &CString {
        &self.status_cstr
    }
}

impl From<BnbSolverOutcome<i64>> for BnbSolverFfiOutcome {
    #[inline]
    fn from(outcome: BnbSolverOutcome<i64>) -> Self {
        Self::new(outcome)
    }
}

/// Frees a `BnbSolverFFIOutcome` pointer allocated on the heap.
///
/// # Panics
///
/// This function will panic if called with a null pointer.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_free(ptr: *mut BnbSolverFfiOutcome) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// Returns the termination reason as a C string.
/// The pointer is valid until the outcome is freed.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_get_termination_reason(
    ptr: *const BnbSolverFfiOutcome,
) -> *const c_char {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_termination_reason` with null pointer"
    );

    (*ptr).termination_reason_cstr.as_ptr()
}

/// Returns the status as a C string.
/// The pointer is valid until the outcome is freed.
///
/// # Panics
///
/// This function will panic if called with a null pointer.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_get_status_str(
    ptr: *const BnbSolverFfiOutcome,
) -> *const c_char {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_status_str` with null pointer"
    );
    (*ptr).status_cstr.as_ptr()
}

/// Retrieves the status enum from the solver outcome.
///
/// # Panics
///
/// This function will panic if called with a null pointer.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_get_status(
    ptr: *const BnbSolverFfiOutcome,
) -> BnbSolverFfiStatus {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_status` with null pointer"
    );

    let outcome = &(*ptr).inner;
    BnbSolverFfiStatus::from(outcome.result())
}

/// Retrieves the termination reason enum from the solver outcome.
///
/// # Panics
///
/// This function will panic if called with a null pointer.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_get_termination_reason_enum(
    ptr: *const BnbSolverFfiOutcome,
) -> BnbSolverFfiTerminationReason {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_termination_reason_enum` with null pointer"
    );

    let outcome = &(*ptr).inner;
    BnbSolverFfiTerminationReason::from(outcome.termination_reason())
}

/// Checks if the solver outcome has a feasible or optimal solution.
///
/// # Panics
///
/// This function will panic if called with a null pointer.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_has_solution(ptr: *const BnbSolverFfiOutcome) -> bool {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_has_solution` with null pointer"
    );

    let outcome = &(*ptr).inner;
    matches!(
        outcome.result(),
        SolverResult::Optimal(_) | SolverResult::Feasible(_)
    )
}

/// Retrieves the objective value from the solver outcome.
///
/// # Panics
///
/// This function will panic if called with a null pointer or
/// if called on an outcome without a solution.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_get_objective(ptr: *const BnbSolverFfiOutcome) -> i64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_objective` with null pointer"
    );

    let outcome = &(*ptr).inner;
    match outcome.result() {
        SolverResult::Optimal(sol) | SolverResult::Feasible(sol) => sol.objective_value(),
        _ => panic!("called `bollard_bnb_outcome_get_objective` on an outcome with no solution"),
    }
}

/// Retrieves the number of vessels in the solver outcome.
///
/// # Panics
///
/// This function will panic if called with a null pointer or
/// if called on an outcome without a solution.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_get_num_vessels(
    ptr: *const BnbSolverFfiOutcome,
) -> usize {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_num_vessels` with null pointer"
    );

    let outcome = &(*ptr).inner;
    match outcome.result() {
        SolverResult::Optimal(sol) | SolverResult::Feasible(sol) => sol.num_vessels(),
        _ => panic!("called `bollard_bnb_outcome_get_num_vessels` on an outcome with no solution"),
    }
}

/// Retrieves the berth index assigned to a specific vessel in the solver outcome.
///
/// # Panics
///
/// This function will panic if called with a null pointer,
/// if called on an outcome without a solution,
/// or if the vessel index is out of bounds.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_get_berth(
    ptr: *const BnbSolverFfiOutcome,
    vessel_idx: usize,
) -> usize {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_berth` with null pointer"
    );

    let outcome = &(*ptr).inner;
    let solution = match outcome.result() {
        SolverResult::Optimal(sol) | SolverResult::Feasible(sol) => sol,
        _ => panic!("called `bollard_bnb_outcome_get_berth` on an outcome with no solution"),
    };

    assert!(
        vessel_idx < solution.num_vessels(),
        "called `bollard_bnb_outcome_get_berth` with vessel index out of bounds: the len is {} but the index is {}",
        solution.num_vessels(),
        vessel_idx
    );

    solution
        .berth_for_vessel(VesselIndex::new(vessel_idx))
        .get()
}

/// Retrieves the start time assigned to a specific vessel in the solver outcome.
///
/// # Panics
///
/// This function will panic if called with a null pointer,
/// if called on an outcome without a solution,
/// or if the vessel index is out of bounds.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_get_start_time(
    ptr: *const BnbSolverFfiOutcome,
    vessel_idx: usize,
) -> i64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_start_time` with null pointer"
    );
    let outcome = &(*ptr).inner;

    let solution = match outcome.result() {
        SolverResult::Optimal(sol) | SolverResult::Feasible(sol) => sol,
        _ => panic!("called `bollard_bnb_outcome_get_start_time` on an outcome with no solution"),
    };

    assert!(vessel_idx < solution.num_vessels(),
        "called `bollard_bnb_outcome_get_start_time` with vessel index out of bounds: the len is {} but the index is {}",
        solution.num_vessels(),
        vessel_idx
    );

    solution.start_time_for_vessel(VesselIndex::new(vessel_idx))
}

/// Retrieves a pointer to the array of start times from the solver outcome.
///
/// # Panics
///
/// This function will panic if called with a null pointer or
/// if called on an outcome without a solution.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_get_start_times(
    ptr: *const BnbSolverFfiOutcome,
) -> *const i64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_start_times` with null pointer"
    );
    let outcome = &(*ptr).inner;

    let solution = match outcome.result() {
        SolverResult::Optimal(sol) | SolverResult::Feasible(sol) => sol,
        _ => panic!("called `bollard_bnb_outcome_get_start_times` on an outcome with no solution"),
    };

    solution.start_times().as_ptr()
}

/// Retrieves a pointer to the array of berth indices from the solver outcome.
///
/// # Panics
///
/// This function will panic if called with a null pointer or
/// if called on an outcome without a solution.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_get_berths(
    ptr: *const BnbSolverFfiOutcome,
) -> *const usize {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_berths` with null pointer"
    );
    let outcome = &(*ptr).inner;
    let solution = match outcome.result() {
        SolverResult::Optimal(sol) | SolverResult::Feasible(sol) => sol,
        _ => panic!("called `bollard_bnb_outcome_get_berths` on an outcome with no solution"),
    };

    debug_assert_eq!(
        std::mem::size_of::<bollard_model::index::BerthIndex>(),
        std::mem::size_of::<usize>()
    );
    debug_assert_eq!(
        std::mem::align_of::<bollard_model::index::BerthIndex>(),
        std::mem::align_of::<usize>()
    );

    solution.berths().as_ptr().cast::<usize>()
}

/// Copies the solution data into the provided output arrays.
///
/// This allows the FFI caller to populate their own data structures with the result.
///
/// # Panics
///
/// Panics if the outcome does not contain a valid solution.
///
/// # Safety
///
/// This function is unsafe because it dereferences raw pointers.
/// The caller must ensure that the pointers are valid and
/// that the output arrays have sufficient space to hold the solution data.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_copy_solution(
    ptr: *const BnbSolverFfiOutcome,
    out_berths: *mut usize,
    out_start_times: *mut i64,
) {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_copy_solution` with null pointer"
    );
    assert!(
        !out_berths.is_null(),
        "called `bollard_bnb_outcome_copy_solution` with null out_berths pointer"
    );
    assert!(
        !out_start_times.is_null(),
        "called `bollard_bnb_outcome_copy_solution` with null out_start_times pointer"
    );

    let outcome = &(*ptr).inner;
    let solution = match outcome.result() {
        SolverResult::Optimal(sol) | SolverResult::Feasible(sol) => sol,
        _ => panic!("called `bollard_bnb_outcome_copy_solution` on an outcome with no solution"),
    };

    for vessel_index in 0..solution.num_vessels() {
        let vessel = VesselIndex::new(vessel_index);

        let berth_idx = solution.berth_for_vessel(vessel).get();
        let start_time = solution.start_time_for_vessel(vessel);

        *out_berths.add(vessel_index) = berth_idx;
        *out_start_times.add(vessel_index) = start_time;
    }
}

/// Returns the number of nodes explored during the solver run.
///
/// # Panics
///
/// This function will panic if called with a null pointer.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_get_nodes_explored(
    ptr: *const BnbSolverFfiOutcome,
) -> u64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_nodes_explored` with null pointer"
    );
    let outcome = &(*ptr).inner;
    outcome.statistics().nodes_explored
}

/// Returns the number of backtracks during the solver run.
///
/// # Panics
///
/// This function will panic if called with a null pointer.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_get_backtracks(
    ptr: *const BnbSolverFfiOutcome,
) -> u64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_backtracks` with null pointer"
    );
    let outcome = &(*ptr).inner;
    outcome.statistics().backtracks
}

/// Returns the number of decisions generated during the solver run.
///
/// # Panics
///
/// This function will panic if called with a null pointer.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_get_decisions_generated(
    ptr: *const BnbSolverFfiOutcome,
) -> u64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_decisions_generated` with null pointer"
    );
    let outcome = &(*ptr).inner;
    outcome.statistics().decisions_generated
}

/// Returns the maximum depth reached during the solver run.
///
/// # Panics
///
/// This function will panic if called with a null pointer.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_get_max_depth(ptr: *const BnbSolverFfiOutcome) -> u64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_max_depth` with null pointer"
    );
    let outcome = &(*ptr).inner;
    outcome.statistics().max_depth
}

/// Returns the number of infeasible prunings during the solver run.
///
/// # Panics
///
/// This function will panic if called with a null pointer.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_get_prunings_infeasible(
    ptr: *const BnbSolverFfiOutcome,
) -> u64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_prunings_infeasible` with null pointer"
    );
    let outcome = &(*ptr).inner;
    outcome.statistics().prunings_infeasible
}

/// Returns the number of bound prunings during the solver run.
///
/// # Panics
///
/// This function will panic if called with a null pointer.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_get_prunings_bound(
    ptr: *const BnbSolverFfiOutcome,
) -> u64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_prunings_bound` with null pointer"
    );
    let outcome = &(*ptr).inner;
    outcome.statistics().prunings_bound
}

/// Returns the number of solutions found during the solver run.
///
/// # Panics
///
/// This function will panic if called with a null pointer.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_get_solutions_found(
    ptr: *const BnbSolverFfiOutcome,
) -> u64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_solutions_found` with null pointer"
    );
    let outcome = &(*ptr).inner;
    outcome.statistics().solutions_found
}

/// Returns the number of steps taken during the solver run.
///
/// # Panics
///
/// This function will panic if called with a null pointer.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_get_steps(ptr: *const BnbSolverFfiOutcome) -> u64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_steps` with null pointer"
    );
    let outcome = &(*ptr).inner;
    outcome.statistics().steps
}

/// Returns the total time taken during the solver run in milliseconds.
///
/// # Panics
///
/// This function will panic if called with a null pointer.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_get_time_total_ms(
    ptr: *const BnbSolverFfiOutcome,
) -> u64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_time_total_ms` with null pointer"
    );
    let outcome = &(*ptr).inner;
    let dur = outcome.statistics().time_total;
    dur.as_millis().try_into().unwrap_or(u64::MAX)
}

/// Creates a new Branch-and-Bound solver instance.
#[no_mangle]
pub extern "C" fn bollard_bnb_solver_new() -> *mut BnbSolver<i64> {
    let solver = BnbSolver::<i64>::new();
    Box::into_raw(Box::new(solver))
}

/// Creates a new Branch-and-Bound solver instance with preallocated memory.
#[no_mangle]
pub extern "C" fn bollard_bnb_solver_preallocated(
    num_berths: usize,
    num_vessels: usize,
) -> *mut BnbSolver<i64> {
    let solver = BnbSolver::<i64>::preallocated(num_berths, num_vessels);
    Box::into_raw(Box::new(solver))
}

/// Frees the memory allocated for the BnbSolver.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was allocated
/// by `bollard_bnb_solver_new` or `bollard_bnb_solver_preallocated`.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_solver_free(ptr: *mut BnbSolver<i64>) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn solve<B, E>(
    solver: &mut BnbSolver<i64>,
    model: &Model<i64>,
    mut builder: B,
    mut evaluator: E,
    solution_limit: usize,
    time_limit_ms: i64,
    enable_log: bool,
    fixed_assignments: &[FixedAssignment<i64>], // Added argument
) -> BnbSolverOutcome<i64>
where
    E: ObjectiveEvaluator<i64>,
    B: DecisionBuilder<i64, E>,
{
    let capacity =
        (solution_limit > 0) as usize + (time_limit_ms > 0) as usize + (enable_log as usize);

    let mut monitor = CompositeTreeSearchMonitor::with_capacity(capacity);

    if solution_limit > 0 {
        monitor.add_monitor(SolutionLimitMonitor::new(solution_limit as u64));
    }
    if time_limit_ms > 0 {
        let duration = Duration::from_millis(time_limit_ms as u64);
        monitor.add_monitor(TimeLimitMonitor::new(duration));
    }
    if enable_log {
        monitor.add_monitor(LogTreeSearchMonitor::default());
    }

    // We now always use solve_with_fixed.
    // If the slice is empty, it behaves exactly like the standard solve().
    solver.solve_with_fixed(
        model,
        &mut builder,
        &mut evaluator,
        monitor,
        fixed_assignments,
    )
}

/// Macro for generating FFI solve functions for different evaluator and builder combinations.
macro_rules! generate_solve {
    (
        $fn_name:ident,
        $eval_ty:ty,
        $builder_ty:ty,
        $eval_init:expr,
        $builder_init:expr
    ) => {
        // Use the paste! macro to generate the new function name dynamically
        paste::paste! {
            /// Solves the given model.
            ///
            /// # Safety
            ///
            /// This function is unsafe because it dereferences raw pointers.
            /// The caller must ensure that the pointers are valid and were allocated
            /// by the appropriate functions.
            #[no_mangle]
            pub unsafe extern "C" fn $fn_name(
                solver_ptr: *mut BnbSolver<i64>,
                model_ptr: *const Model<i64>,
                solution_limit: usize,
                time_limit_ms: i64,
                enable_log: bool,
            ) -> *mut BnbSolverFfiOutcome {
                assert!(!solver_ptr.is_null(), "called `{}` with null solver pointer", stringify!($fn_name));
                assert!(!model_ptr.is_null(), "called `{}` with null model pointer", stringify!($fn_name));

                let solver: &mut BnbSolver<i64> = &mut *solver_ptr;
                let model: &Model<i64> = &*model_ptr;

                let evaluator: $eval_ty = ($eval_init)(model);
                let builder: $builder_ty = ($builder_init)(model);

                // Pass empty slice for standard solve
                let outcome = solve(
                    solver,
                    model,
                    builder,
                    evaluator,
                    solution_limit,
                    time_limit_ms,
                    enable_log,
                    &[],
                );

                let ffi_outcome = BnbSolverFfiOutcome::from(outcome);
                Box::into_raw(Box::new(ffi_outcome))
            }

            /// Solves the given model using fixed assignments.
            /// Expects `fixed_ptr` to point to an array of `FixedAssignment` of length `fixed_len`.
            ///
            /// # Safety
            ///
            /// This function is unsafe because it dereferences raw pointers.
            /// The caller must ensure that the pointers are valid and were allocated
            /// by the appropriate functions.
            /// Also ensures that if `fixed_len` > 0, `fixed_ptr` is not null.
            #[no_mangle]
            pub unsafe extern "C" fn [<$fn_name _with_fixed>](
                solver_ptr: *mut BnbSolver<i64>,
                model_ptr: *const Model<i64>,
                fixed_ptr: *const FixedAssignment<i64>, // Pointer to array
                fixed_len: usize,                       // Length of array
                solution_limit: usize,
                time_limit_ms: i64,
                enable_log: bool,
            ) -> *mut BnbSolverFfiOutcome {
                assert!(!solver_ptr.is_null(), "called `{}` with null solver pointer", stringify!([<$fn_name _with_fixed>]));
                assert!(!model_ptr.is_null(), "called `{}` with null model pointer", stringify!([<$fn_name _with_fixed>]));

                // Safety check for fixed array if len > 0
                if fixed_len > 0 {
                    assert!(!fixed_ptr.is_null(),
                    "called `{}` with null fixed_ptr but non-zero length",
                    stringify!([<$fn_name _with_fixed>]));
                }

                let solver: &mut BnbSolver<i64> = &mut *solver_ptr;
                let model: &Model<i64> = &*model_ptr;

                // fixed len can not be longer than number of vessels in the model
                assert!(fixed_len <= model.num_vessels(),
                    "called `{}` with fixed_len greater than number of vessels in model",
                    stringify!([<$fn_name _with_fixed>])
                );

                // Create slice from raw pointer
                let fixed_slice = if fixed_len > 0 {
                    std::slice::from_raw_parts(fixed_ptr, fixed_len)
                } else {
                    &[]
                };

                let evaluator: $eval_ty = ($eval_init)(model);
                let builder: $builder_ty = ($builder_init)(model);

                let outcome = solve(
                    solver,
                    model,
                    builder,
                    evaluator,
                    solution_limit,
                    time_limit_ms,
                    enable_log,
                    fixed_slice,
                );

                let ffi_outcome = BnbSolverFfiOutcome::from(outcome);
                Box::into_raw(Box::new(ffi_outcome))
            }
        }
    };
}

generate_solve!(
    bollard_bnb_solver_solve_with_hybrid_evaluator_and_chronological_exhaustive_builder,
    HybridEvaluator<i64>,
    ChronologicalExhaustiveBuilder<i64>,
    |m: &Model<i64>| HybridEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |_| ChronologicalExhaustiveBuilder::new()
);

generate_solve!(
    bollard_bnb_solver_solve_with_hybrid_evaluator_and_fcfs_heuristic_builder,
    HybridEvaluator<i64>,
    FcfsHeuristicBuilder<i64>,
    |m: &Model<i64>| HybridEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| FcfsHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

generate_solve!(
    bollard_bnb_solver_solve_with_hybrid_evaluator_and_regret_heuristic_builder,
    HybridEvaluator<i64>,
    RegretHeuristicBuilder<i64>,
    |m: &Model<i64>| HybridEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| RegretHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

generate_solve!(
    bollard_bnb_solver_solve_with_hybrid_evaluator_and_slack_heuristic_builder,
    HybridEvaluator<i64>,
    SlackHeuristicBuilder<i64>,
    |m: &Model<i64>| HybridEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| SlackHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

generate_solve!(
    bollard_bnb_solver_solve_with_hybrid_evaluator_and_wspt_heuristic_builder,
    HybridEvaluator<i64>,
    WsptHeuristicBuilder<i64>,
    |m: &Model<i64>| HybridEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| WsptHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

// --- Workload Evaluator Combinations ---

generate_solve!(
    bollard_bnb_solver_solve_with_workload_evaluator_and_chronological_exhaustive_builder,
    WorkloadEvaluator<i64>,
    ChronologicalExhaustiveBuilder<i64>,
    |m: &Model<i64>| WorkloadEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |_| ChronologicalExhaustiveBuilder::new()
);

generate_solve!(
    bollard_bnb_solver_solve_with_workload_evaluator_and_fcfs_heuristic_builder,
    WorkloadEvaluator<i64>,
    FcfsHeuristicBuilder<i64>,
    |m: &Model<i64>| WorkloadEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| FcfsHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

generate_solve!(
    bollard_bnb_solver_solve_with_workload_evaluator_and_regret_heuristic_builder,
    WorkloadEvaluator<i64>,
    RegretHeuristicBuilder<i64>,
    |m: &Model<i64>| WorkloadEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| RegretHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

generate_solve!(
    bollard_bnb_solver_solve_with_workload_evaluator_and_slack_heuristic_builder,
    WorkloadEvaluator<i64>,
    SlackHeuristicBuilder<i64>,
    |m: &Model<i64>| WorkloadEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| SlackHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

generate_solve!(
    bollard_bnb_solver_solve_with_workload_evaluator_and_wspt_heuristic_builder,
    WorkloadEvaluator<i64>,
    WsptHeuristicBuilder<i64>,
    |m: &Model<i64>| WorkloadEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| WsptHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

// --- Weighted Flow Time Evaluator Combinations ---

generate_solve!(
    bollard_bnb_solver_solve_with_wtft_evaluator_and_chronological_exhaustive_builder,
    WeightedFlowTimeEvaluator<i64>,
    ChronologicalExhaustiveBuilder<i64>,
    |m: &Model<i64>| WeightedFlowTimeEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |_| ChronologicalExhaustiveBuilder::new()
);

generate_solve!(
    bollard_bnb_solver_solve_with_wtft_evaluator_and_fcfs_heuristic_builder,
    WeightedFlowTimeEvaluator<i64>,
    FcfsHeuristicBuilder<i64>,
    |m: &Model<i64>| WeightedFlowTimeEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| FcfsHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

generate_solve!(
    bollard_bnb_solver_solve_with_wtft_evaluator_and_regret_heuristic_builder,
    WeightedFlowTimeEvaluator<i64>,
    RegretHeuristicBuilder<i64>,
    |m: &Model<i64>| WeightedFlowTimeEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| RegretHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

generate_solve!(
    bollard_bnb_solver_solve_with_wtft_evaluator_and_slack_heuristic_builder,
    WeightedFlowTimeEvaluator<i64>,
    SlackHeuristicBuilder<i64>,
    |m: &Model<i64>| WeightedFlowTimeEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| SlackHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

generate_solve!(
    bollard_bnb_solver_solve_with_wtft_evaluator_and_wspt_heuristic_builder,
    WeightedFlowTimeEvaluator<i64>,
    WsptHeuristicBuilder<i64>,
    |m: &Model<i64>| WeightedFlowTimeEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| WsptHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

#[cfg(test)]
mod tests {
    use crate::{
        bnb::{
            bollard_bnb_outcome_free, bollard_bnb_outcome_get_status_str, bollard_bnb_solver_free,
            bollard_bnb_solver_preallocated,
            bollard_bnb_solver_solve_with_hybrid_evaluator_and_regret_heuristic_builder,
        },
        model::{
            bollard_model_builder_build, bollard_model_builder_new,
            bollard_model_builder_set_arrival_time,
            bollard_model_builder_set_latest_departure_time,
            bollard_model_builder_set_processing_time, bollard_model_builder_set_vessel_weight,
            bollard_model_free,
        },
    };

    #[test]
    fn test_solver_computes_optimal_solution() {
        unsafe {
            let builder = bollard_model_builder_new(2, 5);

            // First vessel
            bollard_model_builder_set_arrival_time(builder, 0, 0);
            bollard_model_builder_set_latest_departure_time(builder, 0, 100);
            bollard_model_builder_set_processing_time(builder, 0, 0, 4);
            bollard_model_builder_set_processing_time(builder, 0, 1, 6);
            bollard_model_builder_set_vessel_weight(builder, 0, 1);

            // Second vessel
            bollard_model_builder_set_arrival_time(builder, 1, 2);
            bollard_model_builder_set_latest_departure_time(builder, 1, 100);
            bollard_model_builder_set_processing_time(builder, 1, 0, 3);
            bollard_model_builder_set_processing_time(builder, 1, 1, 5);
            bollard_model_builder_set_vessel_weight(builder, 1, 2);

            // Third vessel
            bollard_model_builder_set_arrival_time(builder, 2, 4);
            bollard_model_builder_set_latest_departure_time(builder, 2, 100);
            bollard_model_builder_set_processing_time(builder, 2, 0, 2);
            bollard_model_builder_set_processing_time(builder, 2, 1, 4);
            bollard_model_builder_set_vessel_weight(builder, 2, 3);

            // Fourth vessel
            bollard_model_builder_set_arrival_time(builder, 3, 6);
            bollard_model_builder_set_latest_departure_time(builder, 3, 100);
            bollard_model_builder_set_processing_time(builder, 3, 0, 5);
            bollard_model_builder_set_processing_time(builder, 3, 1, 3);
            bollard_model_builder_set_vessel_weight(builder, 3, 4);

            // Fifth vessel
            bollard_model_builder_set_arrival_time(builder, 4, 8);
            bollard_model_builder_set_latest_departure_time(builder, 4, 100);
            bollard_model_builder_set_processing_time(builder, 4, 0, 4);
            bollard_model_builder_set_processing_time(builder, 4, 1, 2);
            bollard_model_builder_set_vessel_weight(builder, 4, 5);

            let model = bollard_model_builder_build(builder);
            let solver = bollard_bnb_solver_preallocated(2, 5);
            let outcome =
                bollard_bnb_solver_solve_with_hybrid_evaluator_and_regret_heuristic_builder(
                    solver, model, 0, 0, false,
                );

            assert_eq!(
                std::ffi::CStr::from_ptr(bollard_bnb_outcome_get_status_str(outcome))
                    .to_str()
                    .unwrap(),
                "Optimal"
            );

            // Clean up
            bollard_model_free(model);
            bollard_bnb_solver_free(solver);
            bollard_bnb_outcome_free(outcome);
        }
    }
}
