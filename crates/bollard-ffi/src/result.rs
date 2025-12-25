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

//! FFI bindings for solver outcomes
//!
//! This module exposes a stable C ABI for reading results produced by the
//! Branch‑and‑Bound solver. It wraps the internal outcome in an FFI‑friendly
//! struct that owns C strings for status and termination reason, and provides
//! accessor functions for solution fields such as objective value, berths, and
//! start times. Pointers returned by accessors remain valid while the outcome
//! is alive; ownership is released via a dedicated free function.
//!
//! Designed for consumers who need predictable memory lifetimes and string
//! representations without depending on Rust data structures, while preserving
//! the solver’s semantics and safety checks at the FFI boundary.

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
pub struct BnbSolverFFIOutcome {
    inner: BnbSolverOutcome<i64>,     // Internal outcome with i64 objective
    termination_reason_cstr: CString, // Owned C string for termination reason
    status_cstr: CString,             // Owned C string for status
}

impl BnbSolverFFIOutcome {
    /// Constructs a `BnbSolverFFIOutcome` from a `BnbSolverOutcome<i64>`.
    #[inline]
    pub fn new(inner: BnbSolverOutcome<i64>) -> Self {
        let term_str = match inner.termination_reason() {
            TerminationReason::OptimalityProven => "OptimalityProven",
            TerminationReason::InfeasibilityProven => "InfeasibilityProven",
            TerminationReason::Aborted(s) => s.as_str(),
        };
        let termination_reason_cstr = CString::new(term_str).unwrap_or_default();

        let status_str = match inner.result() {
            SolverResult::Optimal(_) => "Optimal",
            SolverResult::Feasible(_) => "Feasible",
            SolverResult::Infeasible => "Infeasible",
            SolverResult::Unknown => "Unknown",
        };
        let status_cstr = CString::new(status_str).unwrap_or_default();

        Self {
            inner,
            termination_reason_cstr,
            status_cstr,
        }
    }
}

impl From<BnbSolverOutcome<i64>> for BnbSolverFFIOutcome {
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
pub unsafe extern "C" fn bollard_bnb_outcome_free(ptr: *mut BnbSolverFFIOutcome) {
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
    ptr: *const BnbSolverFFIOutcome,
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
    ptr: *const BnbSolverFFIOutcome,
) -> *const c_char {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_status_str` with null pointer"
    );
    (*ptr).status_cstr.as_ptr()
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
pub unsafe extern "C" fn bollard_bnb_outcome_has_solution(ptr: *const BnbSolverFFIOutcome) -> bool {
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
pub unsafe extern "C" fn bollard_bnb_outcome_get_objective(ptr: *const BnbSolverFFIOutcome) -> i64 {
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
    ptr: *const BnbSolverFFIOutcome,
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
    ptr: *const BnbSolverFFIOutcome,
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
    ptr: *const BnbSolverFFIOutcome,
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
    ptr: *const BnbSolverFFIOutcome,
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
    ptr: *const BnbSolverFFIOutcome,
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
    ptr: *const BnbSolverFFIOutcome,
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
    ptr: *const BnbSolverFFIOutcome,
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
    ptr: *const BnbSolverFFIOutcome,
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
    ptr: *const BnbSolverFFIOutcome,
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
pub unsafe extern "C" fn bollard_bnb_outcome_get_max_depth(ptr: *const BnbSolverFFIOutcome) -> u64 {
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
    ptr: *const BnbSolverFFIOutcome,
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
    ptr: *const BnbSolverFFIOutcome,
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
    ptr: *const BnbSolverFFIOutcome,
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
pub unsafe extern "C" fn bollard_bnb_outcome_get_steps(ptr: *const BnbSolverFFIOutcome) -> u64 {
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
    ptr: *const BnbSolverFFIOutcome,
) -> u64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_bnb_outcome_get_time_total_ms` with null pointer"
    );
    let outcome = &(*ptr).inner;
    let dur = outcome.statistics().time_total;
    dur.as_millis().try_into().unwrap_or(u64::MAX)
}
