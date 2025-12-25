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

use bollard_bnb::result::BnbSolverOutcome;
use bollard_model::index::VesselIndex;
use bollard_search::result::SolverResult;

/// Frees a BnbSolverOutcome pointer allocated on the heap.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_outcome_free(ptr: *mut BnbSolverOutcome<i64>) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
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
pub unsafe extern "C" fn bollard_outcome_has_solution(ptr: *const BnbSolverOutcome<i64>) -> bool {
    assert!(
        !ptr.is_null(),
        "called `bollard_outcome_has_solution` with null pointer"
    );

    let outcome = &*ptr;
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
pub unsafe extern "C" fn bollard_outcome_get_objective(ptr: *const BnbSolverOutcome<i64>) -> i64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_outcome_get_objective` with null pointer"
    );

    let outcome = &*ptr;
    match outcome.result() {
        SolverResult::Optimal(sol) | SolverResult::Feasible(sol) => sol.objective_value(),
        _ => panic!("called `bollard_outcome_get_objective` on an outcome with no solution"),
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
pub unsafe extern "C" fn bollard_outcome_get_num_vessels(
    ptr: *const BnbSolverOutcome<i64>,
) -> usize {
    assert!(
        !ptr.is_null(),
        "called `bollard_outcome_get_num_vessels` with null pointer"
    );

    let outcome = &*ptr;
    match outcome.result() {
        SolverResult::Optimal(sol) | SolverResult::Feasible(sol) => sol.num_vessels(),
        _ => panic!("called `bollard_outcome_get_num_vessels` on an outcome with no solution"),
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
pub unsafe extern "C" fn bollard_outcome_get_berth(
    ptr: *const BnbSolverOutcome<i64>,
    vessel_idx: usize,
) -> usize {
    assert!(
        !ptr.is_null(),
        "called `bollard_outcome_get_berth` with null pointer"
    );

    let outcome = &*ptr;
    let solution = match outcome.result() {
        SolverResult::Optimal(sol) | SolverResult::Feasible(sol) => sol,
        _ => panic!("called `bollard_outcome_get_berth` on an outcome with no solution"),
    };

    assert!(
        vessel_idx < solution.num_vessels(),
        "called `bollard_outcome_get_berth` with vessel index out of bounds: the len is {} but the index is {}",
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
pub unsafe extern "C" fn bollard_outcome_get_start_time(
    ptr: *const BnbSolverOutcome<i64>,
    vessel_idx: usize,
) -> i64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_outcome_get_start_time` with null pointer"
    );
    let outcome = &*ptr;

    let solution = match outcome.result() {
        SolverResult::Optimal(sol) | SolverResult::Feasible(sol) => sol,
        _ => panic!("called `bollard_outcome_get_start_time` on an outcome with no solution"),
    };

    assert!(vessel_idx < solution.num_vessels(),
        "called `bollard_outcome_get_start_time` with vessel index out of bounds: the len is {} but the index is {}",
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
pub unsafe extern "C" fn bollard_outcome_get_start_times(
    ptr: *const BnbSolverOutcome<i64>,
) -> *const i64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_outcome_get_start_times` with null pointer"
    );
    let outcome = &*ptr;

    let solution = match outcome.result() {
        SolverResult::Optimal(sol) | SolverResult::Feasible(sol) => sol,
        _ => panic!("called `bollard_outcome_get_start_times` on an outcome with no solution"),
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
pub unsafe extern "C" fn bollard_outcome_get_berths(
    ptr: *const BnbSolverOutcome<i64>,
) -> *const usize {
    assert!(
        !ptr.is_null(),
        "called `bollard_outcome_get_berths` with null pointer"
    );
    let outcome = &*ptr;
    let solution = match outcome.result() {
        SolverResult::Optimal(sol) | SolverResult::Feasible(sol) => sol,
        _ => panic!("called `bollard_outcome_get_berths` on an outcome with no solution"),
    };
    solution.berths().as_ptr().cast::<usize>() // Cast BerthIndex to usize pointer
}
