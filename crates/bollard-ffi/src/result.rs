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

//! # Foreign Function Interface (FFI) for Solver Outcomes
//!
//! This module provides a C-compatible ABI for inspecting the results produced by the
//! Bollard Branch-and-Bound solver. It wraps the internal Rust outcome types into
//! an opaque, FFI-friendly structure that acts as a stable boundary between
//! Rust's rich types and C-style consumers.
//!
//! ## Overview
//!
//! The core type, `BnbSolverFfiOutcome`, serves two purposes:
//! 1.  **Data Container**: It holds the actual solver result (objective, assignments, statistics).
//! 2.  **Memory Owner**: It owns FFI-compatible C-strings for the "Status" and "Termination Reason".
//!     This ensures that `char*` pointers returned to the caller remain valid for the lifetime
//!     of the outcome object.
//!
//! ## Usage Lifecycle
//!
//! 1.  **Acquisition**: An instance of `BnbSolverFfiOutcome` is obtained as the return value
//!     from one of the `solve` functions in the `solver` module.
//! 2.  **Status Check**: Check `bollard_bnb_outcome_get_status` or `bollard_bnb_outcome_has_solution`
//!     to determine if a valid schedule was found.
//! 3.  **Data Extraction**:
//!     * Use accessors like `bollard_bnb_outcome_get_objective` for scalar values.
//!     * Use `bollard_bnb_outcome_copy_solution` to efficiently populate client-side arrays
//!         with berth assignments and start times.
//! 4.  **Telemetry**: Inspect solver performance using the statistics functions (e.g., nodes explored).
//! 5.  **Cleanup**: When finished, pass the pointer to `bollard_bnb_outcome_free` to release all
//!     associated memory (including the internal C-strings).
//!
//! ## Safety
//!
//! This module uses `unsafe` code to dereference raw pointers. Callers **must** ensure:
//!
//! * **Pointer Validity**: Pointers must be valid and returned by a Bollard solve function.
//! * **Null Pointers**: Functions marked with `# Panics` will abort the process if passed `NULL`.
//! * **Lifetime**: Pointers returned by string accessors (e.g., `get_status_str`) are only valid
//!     as long as the parent `BnbSolverFfiOutcome` is not freed.
//!
//! ## Exported Functions
//!
//! ### Lifecycle
//! * `bollard_bnb_outcome_free`
//!
//! ### Status & Metadata
//! * `bollard_bnb_outcome_has_solution`
//! * `bollard_bnb_outcome_get_status`
//! * `bollard_bnb_outcome_get_status_str`
//! * `bollard_bnb_outcome_get_termination_reason_enum`
//! * `bollard_bnb_outcome_get_termination_reason`
//!
//! ### Solution Data
//! * `bollard_bnb_outcome_get_objective`
//! * `bollard_bnb_outcome_get_num_vessels`
//! * `bollard_bnb_outcome_get_berth`
//! * `bollard_bnb_outcome_get_start_time`
//! * `bollard_bnb_outcome_get_berths` (Direct pointer access)
//! * `bollard_bnb_outcome_get_start_times` (Direct pointer access)
//! * `bollard_bnb_outcome_copy_solution` (Batch copy to user buffers)
//!
//! ### Statistics
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
use bollard_model::index::VesselIndex;
use bollard_search::result::{SolverResult, TerminationReason};
use libc::c_char;
use std::ffi::CString;

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
