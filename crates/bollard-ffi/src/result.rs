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

use crate::solution::BollardFfiSolution;
use bollard_search::result::SolverResult;
use std::ffi::{c_schar, CString};

/// The status of the solver result.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BollardFfiStatus {
    Optimal = 0,
    Feasible = 1,
    Infeasible = 2,
    Unknown = 3,
}

/// FFI-compatible representation of a solver result.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct BollardFfiSolverResult {
    pub status_string: CString,
    pub status: BollardFfiStatus,
    pub solution: *mut BollardFfiSolution,
    pub has_solution: bool,
}

impl From<SolverResult<i64>> for BollardFfiSolverResult {
    fn from(result: SolverResult<i64>) -> Self {
        match result {
            SolverResult::Optimal(sol) => BollardFfiSolverResult {
                status_string: CString::new("Optimal").expect("`CString::new` should not fail"),
                status: BollardFfiStatus::Optimal,
                solution: Box::into_raw(Box::new(BollardFfiSolution::from(sol))),
                has_solution: true,
            },
            SolverResult::Feasible(sol) => BollardFfiSolverResult {
                status_string: CString::new("Feasible").expect("`CString::new` should not fail"),
                status: BollardFfiStatus::Feasible,
                solution: Box::into_raw(Box::new(BollardFfiSolution::from(sol))),
                has_solution: true,
            },
            SolverResult::Infeasible => BollardFfiSolverResult {
                status_string: CString::new("Infeasible").expect("`CString::new` should not fail"),
                status: BollardFfiStatus::Infeasible,
                solution: std::ptr::null_mut(),
                has_solution: false,
            },
            SolverResult::Unknown => BollardFfiSolverResult {
                status_string: CString::new("Unknown").expect("`CString::new` should not fail"),
                status: BollardFfiStatus::Unknown,
                solution: std::ptr::null_mut(),
                has_solution: false,
            },
        }
    }
}

/// Frees a `BollardFfiSolverResult` previously allocated by Bollard.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `BollardFfiSolverResult`
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_ffi_solver_result_free(ptr: *mut BollardFfiSolverResult) {
    if ptr.is_null() {
        return;
    }
    drop(Box::from_raw(ptr));
}

/// Returns the status of the solver result.
///
/// # Panics
///
/// This function will panic if `result` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `result` is a valid pointer to a `BollardFfiSolverResult`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ffi_solver_result_status(
    result: *const BollardFfiSolverResult,
) -> BollardFfiStatus {
    assert!(
        !result.is_null(),
        "called `bollard_ffi_solver_result_status` with `ptr` as null pointer"
    );
    (*result).status
}

/// Returns the status string of the solver result.
///
/// # Panics
///
/// This function will panic if `result` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `result` is a valid pointer to a `BollardFfiSolverResult`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ffi_solver_result_status_string(
    result: *const BollardFfiSolverResult,
) -> *const c_schar {
    assert!(
        !result.is_null(),
        "called `bollard_ffi_solver_result_status_string` with `ptr` as null pointer"
    );
    (*result).status_string.as_ptr()
}

/// Returns whether the solver result has a solution.
///
/// # Panics
///
/// This function will panic if `result` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `result` is a valid pointer to a `BollardFfiSolverResult`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ffi_solver_result_has_solution(
    result: *const BollardFfiSolverResult,
) -> bool {
    assert!(
        !result.is_null(),
        "called `bollard_ffi_solver_result_has_solution` with `ptr` as null pointer"
    );
    (*result).has_solution
}

/// Returns the solution of the solver result.
///
/// # Panics
///
/// This function will panic if `result` is a null pointer or if the result does not have a solution.
///
/// # Safety
///
/// The caller must ensure that `result` is a valid pointer to a `BollardFfiSolverResult`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ffi_solver_result_solution(
    result: *const BollardFfiSolverResult,
) -> *mut BollardFfiSolution {
    assert!(
        !result.is_null(),
        "called `bollard_ffi_solver_result_solution` with `ptr` as null pointer"
    );
    assert!(
        (*result).has_solution,
        "called `bollard_ffi_solver_result_solution` on a result without solution"
    );
    (*result).solution
}

// ----------------------------------------------------------------
// Termination reason
// ---------------------------------------------------------------

/// The reason for solver termination.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BollardFfiTerminationReason {
    OptimalityProven = 0,
    InfeasibilityProven = 1,
    Converged = 2,
    Aborted = 3,
}

/// FFI-compatible representation of a solver termination reason.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct BollardFfiTermination {
    pub reason: BollardFfiTerminationReason,
    pub message: CString,
}

impl From<bollard_search::result::TerminationReason> for BollardFfiTermination {
    fn from(reason: bollard_search::result::TerminationReason) -> Self {
        match reason {
            bollard_search::result::TerminationReason::OptimalityProven => BollardFfiTermination {
                reason: BollardFfiTerminationReason::OptimalityProven,
                message: CString::new("Optimality Proven").expect("`CString::new` should not fail"),
            },
            bollard_search::result::TerminationReason::InfeasibilityProven => {
                BollardFfiTermination {
                    reason: BollardFfiTerminationReason::InfeasibilityProven,
                    message: CString::new("Infeasibility Proven")
                        .expect("`CString::new` should not fail"),
                }
            }
            bollard_search::result::TerminationReason::Aborted(msg) => BollardFfiTermination {
                reason: BollardFfiTerminationReason::Aborted,
                message: CString::new(msg).expect("`CString::new` should not fail"),
            },
            bollard_search::result::TerminationReason::Converged(msg) => BollardFfiTermination {
                reason: BollardFfiTerminationReason::Converged,
                message: CString::new(msg).expect("`CString::new` should not fail"),
            },
        }
    }
}

/// Frees a `BollardFfiTermination` previously allocated by Bollard.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `BollardFfiTermination`
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_ffi_termination_free(ptr: *mut BollardFfiTermination) {
    if ptr.is_null() {
        return;
    }
    drop(Box::from_raw(ptr));
}

/// Returns the termination reason.
///
/// # Panics
///
/// This function will panic if `termination` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `termination` is a valid pointer to a `BollardFfiTermination`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ffi_termination_reason(
    termination: *const BollardFfiTermination,
) -> BollardFfiTerminationReason {
    assert!(
        !termination.is_null(),
        "called `bollard_ffi_termination_reason` with `ptr` as null pointer"
    );
    (*termination).reason
}

/// Returns the termination message.
///
/// # Panics
///
/// This function will panic if `termination` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `termination` is a valid pointer to a `BollardFfiTermination`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ffi_termination_message(
    termination: *const BollardFfiTermination,
) -> *const c_schar {
    assert!(
        !termination.is_null(),
        "called `bollard_ffi_termination_message` with `ptr` as null pointer"
    );
    (*termination).message.as_ptr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bollard_model::index::BerthIndex;
    use bollard_model::solution::Solution;
    use bollard_search::result::{SolverResult, TerminationReason};
    use std::ffi::CStr;

    // Helper to read the status string via the FFI accessor.
    unsafe fn status_str_via_ffi_ptr(res: *const BollardFfiSolverResult) -> &'static str {
        let ptr = bollard_ffi_solver_result_status_string(res);
        // Cast the `*const c_schar` to `*const c_char` for CStr
        let cstr = CStr::from_ptr(ptr as *const std::os::raw::c_char);
        cstr.to_str().expect("status string should be valid UTF-8")
    }

    // Helper to read the termination message via the FFI accessor.
    unsafe fn termination_msg_str_via_ffi_ptr(term: *const BollardFfiTermination) -> String {
        let ptr = bollard_ffi_termination_message(term);
        let cstr = CStr::from_ptr(ptr as *const std::os::raw::c_char);
        cstr.to_str()
            .expect("termination message should be valid UTF-8")
            .to_string()
    }

    #[test]
    fn test_from_optimal_and_accessors() {
        // Build a small Solution<i64>
        let berths = vec![BerthIndex::new(0), BerthIndex::new(1)];
        let starts = vec![10_i64, 20_i64];
        let sol = Solution::<i64>::new(123_i64, berths, starts);

        // Convert to FFI result
        let ffi = BollardFfiSolverResult::from(SolverResult::Optimal(sol));

        // Check raw fields
        assert_eq!(ffi.status, BollardFfiStatus::Optimal);
        assert!(ffi.has_solution);
        assert!(!ffi.solution.is_null());

        // Check FFI accessors
        unsafe {
            let res_ptr = &ffi as *const BollardFfiSolverResult;
            assert_eq!(
                bollard_ffi_solver_result_status(res_ptr),
                BollardFfiStatus::Optimal
            );
            assert!(bollard_ffi_solver_result_has_solution(res_ptr));
            assert_eq!(status_str_via_ffi_ptr(res_ptr), "Optimal");

            // Validate the solution pointer by checking objective and num vessels
            use crate::solution::{
                bollard_solution_free, bollard_solution_num_vessels, bollard_solution_objective,
            };

            let sol_ptr = ffi.solution as *const BollardFfiSolution;
            assert_eq!(bollard_solution_objective(sol_ptr), 123_i64);
            assert_eq!(bollard_solution_num_vessels(sol_ptr), 2);

            // Free the solution pointer to avoid leaking
            bollard_solution_free(ffi.solution);
        }

        // `ffi` drops here; it will drop the CString but not the solution (already freed).
    }

    #[test]
    fn test_from_feasible_and_accessors() {
        let berths = vec![BerthIndex::new(2)];
        let starts = vec![7_i64];
        let sol = Solution::<i64>::new(77_i64, berths, starts);

        let ffi = BollardFfiSolverResult::from(SolverResult::Feasible(sol));

        assert_eq!(ffi.status, BollardFfiStatus::Feasible);
        assert!(ffi.has_solution);
        assert!(!ffi.solution.is_null());

        unsafe {
            let res_ptr = &ffi as *const BollardFfiSolverResult;
            assert_eq!(
                bollard_ffi_solver_result_status(res_ptr),
                BollardFfiStatus::Feasible
            );
            assert!(bollard_ffi_solver_result_has_solution(res_ptr));
            assert_eq!(status_str_via_ffi_ptr(res_ptr), "Feasible");

            use crate::solution::{
                bollard_solution_free, bollard_solution_num_vessels, bollard_solution_objective,
            };

            let sol_ptr = ffi.solution as *const BollardFfiSolution;
            assert_eq!(bollard_solution_objective(sol_ptr), 77_i64);
            assert_eq!(bollard_solution_num_vessels(sol_ptr), 1);

            // Free the solution pointer to avoid leaking
            bollard_solution_free(ffi.solution);
        }
    }

    #[test]
    fn test_from_infeasible_and_accessors() {
        let ffi = BollardFfiSolverResult::from(SolverResult::<i64>::Infeasible);

        assert_eq!(ffi.status, BollardFfiStatus::Infeasible);
        assert!(!ffi.has_solution);
        assert!(ffi.solution.is_null());

        unsafe {
            let res_ptr = &ffi as *const BollardFfiSolverResult;
            assert_eq!(
                bollard_ffi_solver_result_status(res_ptr),
                BollardFfiStatus::Infeasible
            );
            assert!(!bollard_ffi_solver_result_has_solution(res_ptr));
            assert_eq!(status_str_via_ffi_ptr(res_ptr), "Infeasible");
        }
        // No heap allocations to free (solution is null, result is stack).
    }

    #[test]
    fn test_from_unknown_and_accessors() {
        let ffi = BollardFfiSolverResult::from(SolverResult::<i64>::Unknown);

        assert_eq!(ffi.status, BollardFfiStatus::Unknown);
        assert!(!ffi.has_solution);
        assert!(ffi.solution.is_null());

        unsafe {
            let res_ptr = &ffi as *const BollardFfiSolverResult;
            assert_eq!(
                bollard_ffi_solver_result_status(res_ptr),
                BollardFfiStatus::Unknown
            );
            assert!(!bollard_ffi_solver_result_has_solution(res_ptr));
            assert_eq!(status_str_via_ffi_ptr(res_ptr), "Unknown");
        }
        // No heap allocations to free (solution is null, result is stack).
    }

    #[test]
    fn test_free_infeasible_via_ffi() {
        // Use a case with no solution to avoid manual solution deallocation.
        let ffi = BollardFfiSolverResult::from(SolverResult::<i64>::Infeasible);
        unsafe {
            let ptr = Box::into_raw(Box::new(ffi));
            // Should not crash and should deallocate the result container.
            bollard_ffi_solver_result_free(ptr);
        }
    }

    #[test]
    fn test_free_feasible_via_ffi_with_solution_cleanup() {
        // Build a small feasible result and ensure both the solution and result are freed.
        let berths = vec![BerthIndex::new(0)];
        let starts = vec![5_i64];
        let sol = Solution::<i64>::new(42_i64, berths, starts);
        let ffi = BollardFfiSolverResult::from(SolverResult::Feasible(sol));

        unsafe {
            // Clean up the solution first, as the result free does not free the solution pointer.
            use crate::solution::bollard_solution_free;
            assert!(!ffi.solution.is_null());
            bollard_solution_free(ffi.solution);

            // Now free the heap-allocated result container.
            let ptr = Box::into_raw(Box::new(ffi));
            bollard_ffi_solver_result_free(ptr);
        }
    }

    // ----------------------------
    // Additional termination tests
    // ----------------------------

    #[test]
    fn test_termination_from_optimality_proven_and_accessors() {
        let term = BollardFfiTermination::from(TerminationReason::OptimalityProven);
        assert_eq!(term.reason, BollardFfiTerminationReason::OptimalityProven);

        unsafe {
            let term_ptr = &term as *const BollardFfiTermination;
            assert_eq!(
                bollard_ffi_termination_reason(term_ptr),
                BollardFfiTerminationReason::OptimalityProven
            );
            let msg = termination_msg_str_via_ffi_ptr(term_ptr);
            assert_eq!(msg, "Optimality Proven");
        }
        // `term` is stack-allocated and drops cleanly.
    }

    #[test]
    fn test_termination_from_infeasibility_proven_and_accessors() {
        let term = BollardFfiTermination::from(TerminationReason::InfeasibilityProven);
        assert_eq!(
            term.reason,
            BollardFfiTerminationReason::InfeasibilityProven
        );

        unsafe {
            let term_ptr = &term as *const BollardFfiTermination;
            assert_eq!(
                bollard_ffi_termination_reason(term_ptr),
                BollardFfiTerminationReason::InfeasibilityProven
            );
            let msg = termination_msg_str_via_ffi_ptr(term_ptr);
            assert_eq!(msg, "Infeasibility Proven");
        }
        // `term` is stack-allocated and drops cleanly.
    }

    #[test]
    fn test_termination_from_aborted_and_accessors_message_roundtrip() {
        let reason_msg = "time limit exceeded";
        let term = BollardFfiTermination::from(TerminationReason::Aborted(reason_msg.to_string()));
        assert_eq!(term.reason, BollardFfiTerminationReason::Aborted);

        unsafe {
            let term_ptr = &term as *const BollardFfiTermination;
            assert_eq!(
                bollard_ffi_termination_reason(term_ptr),
                BollardFfiTerminationReason::Aborted
            );
            let msg = termination_msg_str_via_ffi_ptr(term_ptr);
            assert_eq!(msg, reason_msg);
        }
        // `term` is stack-allocated and drops cleanly.
    }

    #[test]
    fn test_termination_from_aborted_with_non_ascii_message() {
        // Include non-ASCII Unicode characters to ensure UTF-8 handling is correct.
        let reason_msg = "aborted: límite de tiempo ✓";
        let term = BollardFfiTermination::from(TerminationReason::Aborted(reason_msg.to_string()));
        assert_eq!(term.reason, BollardFfiTerminationReason::Aborted);

        unsafe {
            let term_ptr = &term as *const BollardFfiTermination;
            let msg = termination_msg_str_via_ffi_ptr(term_ptr);
            assert_eq!(msg, reason_msg);
        }
        // `term` is stack-allocated and drops cleanly.
    }

    #[test]
    fn test_bollard_ffi_termination_free() {
        // Allocate on the heap and free with the FFI function.
        let term = BollardFfiTermination::from(TerminationReason::Aborted("cleanup".into()));
        unsafe {
            let ptr = Box::into_raw(Box::new(term));
            bollard_ffi_termination_free(ptr);
        }
    }

    #[test]
    fn test_solver_result_status_string_bytes_are_nul_terminated() {
        let ffi = BollardFfiSolverResult::from(SolverResult::<i64>::Unknown);
        // Ensure the CString is properly NUL-terminated by round-tripping via CStr.
        unsafe {
            let res_ptr = &ffi as *const BollardFfiSolverResult;
            let c_ptr = bollard_ffi_solver_result_status_string(res_ptr);
            let cstr = CStr::from_ptr(c_ptr as *const std::os::raw::c_char);
            assert_eq!(cstr.to_str().unwrap(), "Unknown");
            // Validate the last byte is NUL in the underlying bytes representation.
            let bytes = cstr.to_bytes_with_nul();
            assert_eq!(bytes.last().copied(), Some(0));
        }
        // `ffi` is stack-allocated and drops cleanly.
    }

    #[test]
    fn test_solver_result_solution_pointer_is_null_for_non_solution_statuses() {
        let infeasible = BollardFfiSolverResult::from(SolverResult::<i64>::Infeasible);
        let unknown = BollardFfiSolverResult::from(SolverResult::<i64>::Unknown);
        assert!(infeasible.solution.is_null());
        assert!(unknown.solution.is_null());

        unsafe {
            let inf_ptr = &infeasible as *const BollardFfiSolverResult;
            let unk_ptr = &unknown as *const BollardFfiSolverResult;
            assert!(!bollard_ffi_solver_result_has_solution(inf_ptr));
            assert!(!bollard_ffi_solver_result_has_solution(unk_ptr));
        }
        // Both are stack-allocated; no extra cleanup required.
    }
}
