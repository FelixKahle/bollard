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

use bollard_model::index::{BerthIndex, VesselIndex};
use bollard_model::solution::Solution;

/// FFI-compatible wrapper around `Solution<i64>`.
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct BollardFfiSolution {
    inner: Solution<i64>,
}

impl From<Solution<i64>> for BollardFfiSolution {
    fn from(sol: Solution<i64>) -> Self {
        BollardFfiSolution { inner: sol }
    }
}

impl From<&Solution<i64>> for BollardFfiSolution {
    fn from(sol: &Solution<i64>) -> Self {
        BollardFfiSolution { inner: sol.clone() }
    }
}

/// Creates a new `BollardFfiSolution`.
///
/// # Panics
///
/// This function will panic if `berths_ptr` or `start_times_ptr` is null.
///
/// # Safety
///
/// The caller must ensure that `berths_ptr` points to an array of `len` valid `usize` values,
/// and `start_times_ptr` points to an array of `len` valid `i64` values.
#[no_mangle]
pub unsafe extern "C" fn bollard_solution_new(
    objective: i64,
    berths_ptr: *const usize,
    start_times_ptr: *const i64,
    len: usize,
) -> *mut BollardFfiSolution {
    assert!(
        !berths_ptr.is_null(),
        "called `bollard_solution_new` with `berths_ptr` as null pointer"
    );
    assert!(
        !start_times_ptr.is_null(),
        "called `bollard_solution_new` with `start_times_ptr` as null pointer"
    );

    let berths_usize = std::slice::from_raw_parts(berths_ptr, len);
    let start_times = std::slice::from_raw_parts(start_times_ptr, len);

    let berths: Vec<BerthIndex> = berths_usize.iter().map(|&b| BerthIndex::new(b)).collect();

    let inner = Solution::<i64>::new(objective, berths, start_times.to_vec());
    Box::into_raw(Box::new(BollardFfiSolution { inner }))
}

/// Frees a `BollardFfiSolution`.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `BollardFfiSolution`
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_solution_free(ptr: *mut BollardFfiSolution) {
    if ptr.is_null() {
        return;
    }
    drop(Box::from_raw(ptr));
}

/// Accesses the objective value of the solution.
///
/// # Panics
///
/// This function will panic if `ptr` is null.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `BollardFfiSolution`.
#[no_mangle]
pub unsafe extern "C" fn bollard_solution_objective(ptr: *const BollardFfiSolution) -> i64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_solution_objective` with `ptr` as null pointer"
    );
    (&*ptr).inner.objective_value()
}

/// Accesses the number of vessels in the solution.
///
/// # Panics
///
/// This function will panic if `ptr` is null.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `BollardFfiSolution`.
#[no_mangle]
pub unsafe extern "C" fn bollard_solution_num_vessels(ptr: *const BollardFfiSolution) -> usize {
    assert!(
        !ptr.is_null(),
        "called `bollard_solution_num_vessels` with `ptr` as null pointer"
    );
    (&*ptr).inner.num_vessels()
}

/// Accesses the berth assigned to a specific vessel.
///
/// # Panics
///
/// This function will panic if `ptr` is null or if `vessel_index` is out of bounds.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `BollardFfiSolution`.
#[no_mangle]
pub unsafe extern "C" fn bollard_solution_berth(
    ptr: *const BollardFfiSolution,
    vessel_index: usize,
) -> usize {
    assert!(
        !ptr.is_null(),
        "called `bollard_solution_berth` with null pointer"
    );
    let sol = &(&*ptr).inner;

    assert!(
        vessel_index < sol.num_vessels(),
        "called `bollard_solution_berth` with vessel index out of bounds: the len is {} but the index is {}",
        sol.num_vessels(),
        vessel_index
    );

    sol.berth_for_vessel(VesselIndex::new(vessel_index)).get()
}

/// Accesses the start time assigned to a specific vessel.
///
/// # Panics
///
/// This function will panic if `ptr` is null or if `vessel_idx` is out of bounds.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `BollardFfiSolution`.
#[no_mangle]
pub unsafe extern "C" fn bollard_solution_start_time(
    ptr: *const BollardFfiSolution,
    vessel_idx: usize,
) -> i64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_solution_start_time` with `ptr` as null pointer"
    );
    let sol = &(&*ptr).inner;

    assert!(
        vessel_idx < sol.num_vessels(),
        "called `bollard_solution_start_time` with vessel index out of bounds: the len is {} but the index is {}",
        sol.num_vessels(),
        vessel_idx
    );

    sol.start_time_for_vessel(VesselIndex::new(vessel_idx))
}

/// Accesses a pointer to the array of berth assignments.
///
/// # Panics
///
/// This function will panic if `ptr` is null.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `BollardFfiSolution`.
#[no_mangle]
pub unsafe extern "C" fn bollard_solution_berths(ptr: *const BollardFfiSolution) -> *const usize {
    assert!(
        !ptr.is_null(),
        "called `bollard_solution_berths` with `ptr` as null pointer"
    );

    let sol = &(&*ptr).inner;

    debug_assert_eq!(
        std::mem::size_of::<BerthIndex>(),
        std::mem::size_of::<usize>()
    );
    debug_assert_eq!(
        std::mem::align_of::<BerthIndex>(),
        std::mem::align_of::<usize>()
    );

    sol.berths().as_ptr().cast::<usize>()
}

/// Accesses a pointer to the array of start times.
///
/// # Panics
///
/// This function will panic if `ptr` is null.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `BollardFfiSolution`.
#[no_mangle]
pub unsafe extern "C" fn bollard_solution_start_times(
    ptr: *const BollardFfiSolution,
) -> *const i64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_solution_start_times` with `ptr` as null pointer"
    );

    let sol = &(&*ptr).inner;
    sol.start_times().as_ptr()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_free_minimal() {
        unsafe {
            let berths = [0_usize, 1_usize];
            let starts = [10_i64, 20_i64];

            let ptr = bollard_solution_new(100, berths.as_ptr(), starts.as_ptr(), 2);
            assert!(!ptr.is_null());

            // Free
            bollard_solution_free(ptr);
        }
    }

    #[test]
    fn test_objective_and_counts() {
        unsafe {
            let berths = [0_usize, 1_usize, 0_usize];
            let starts = [10_i64, 20_i64, 30_i64];

            let ptr = bollard_solution_new(123, berths.as_ptr(), starts.as_ptr(), 3);
            assert_eq!(bollard_solution_objective(ptr), 123);
            assert_eq!(bollard_solution_num_vessels(ptr), 3);
            bollard_solution_free(ptr);
        }
    }

    #[test]
    fn test_per_vessel_accessors() {
        unsafe {
            let berths = [2_usize, 1_usize];
            let starts = [7_i64, 9_i64];

            let ptr = bollard_solution_new(0, berths.as_ptr(), starts.as_ptr(), 2);

            assert_eq!(bollard_solution_berth(ptr, 0), 2);
            assert_eq!(bollard_solution_berth(ptr, 1), 1);
            assert_eq!(bollard_solution_start_time(ptr, 0), 7);
            assert_eq!(bollard_solution_start_time(ptr, 1), 9);

            bollard_solution_free(ptr);
        }
    }
}
