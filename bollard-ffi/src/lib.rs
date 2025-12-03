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

use bollard_core::math::interval::ClosedOpenInterval;
use bollard_model::{BerthIndex, Model, ModelBuilder, Solution, VesselIndex};
use std::slice;

#[no_mangle]
pub extern "C" fn bollard_model_new(
    num_vessels: usize,
    num_berths: usize,
) -> *mut ModelBuilder<i64> {
    let builder = ModelBuilder::<i64>::new(num_vessels, num_berths);
    Box::into_raw(Box::new(builder))
}

/// Frees a `ModelBuilder<i64>` previously created by [`bollard_model_new`].
///
/// # Safety
/// - `ptr` must be a valid, non-dangling pointer returned by `bollard_model_new`.
/// - `ptr` must not have been freed already, and must not be aliased elsewhere.
/// - After this call, `ptr` must not be used again.
/// - Passing an invalid pointer or double-freeing is undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_free(ptr: *mut ModelBuilder<i64>) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// Sets the arrival time for a vessel in the builder.
///
/// # Safety
/// - `ptr` must be a valid, non-null pointer to a `ModelBuilder<i64>` created
///   by `bollard_model_new` and not yet freed.
/// - `vessel_index` must be within `0..builder.num_vessels()`; otherwise this may panic.
/// - Caller must ensure no concurrent aliasing mutable access to the same builder.
#[no_mangle]
pub unsafe extern "C" fn bollard_set_arrival(
    ptr: *mut ModelBuilder<i64>,
    vessel_index: usize,
    time: i64,
) {
    let builder = &mut *ptr;
    builder.vessel_arrival(VesselIndex::new(vessel_index), time);
}

/// Sets both arrival and latest departure (window) for a vessel.
///
/// # Safety
/// - `ptr` must be a valid, non-null pointer to a `ModelBuilder<i64>` created
///   by `bollard_model_new` and not yet freed.
/// - `vessel_index` must be within `0..builder.num_vessels()`; otherwise this may panic.
/// - Caller must ensure no concurrent aliasing mutable access to the same builder.
/// - `arr` and `departure` are interpreted as times; no overflow checking is performed.
#[no_mangle]
pub unsafe extern "C" fn bollard_set_window(
    ptr: *mut ModelBuilder<i64>,
    vessel_index: usize,
    arr: i64,
    departure: i64,
) {
    let builder = &mut *ptr;
    builder.vessel_window(VesselIndex::new(vessel_index), arr, departure);
}

/// Sets the processing time for a vessel at a berth. Negative `time` marks unavailability.
///
/// # Safety
/// - `ptr` must be a valid, non-null pointer to a `ModelBuilder<i64>` created
///   by `bollard_model_new` and not yet freed.
/// - `vessel_index` must be within `0..builder.num_vessels()` and `berth_index` within `0..builder.num_berths()`;
///   otherwise this may panic.
/// - Caller must ensure no concurrent aliasing mutable access to the same builder.
/// - `time` is interpreted as an i64; negative means unavailable, non-negative must be valid.
#[no_mangle]
pub unsafe extern "C" fn bollard_set_processing(
    ptr: *mut ModelBuilder<i64>,
    vessel_index: usize,
    berth_index: usize,
    time: i64,
) {
    let builder = &mut *ptr;
    if time < 0 {
        builder
            .processing_unavailable(VesselIndex::new(vessel_index), BerthIndex::new(berth_index));
    } else {
        builder.processing_time(
            VesselIndex::new(vessel_index),
            BerthIndex::new(berth_index),
            time,
        );
    }
}

/// Adds (sets) a single opening interval for a berth.
///
/// # Safety
/// - `ptr` must be a valid, non-null pointer to a `ModelBuilder<i64>` created
///   by `bollard_model_new` and not yet freed.
/// - `berth_index` must be within `0..builder.num_berths()`; otherwise this may panic.
/// - Caller must ensure no concurrent aliasing mutable access to the same builder.
/// - `start` and `end` define a closed-open interval `[start, end)`; if `start >= end`,
///   model validation will fail later during `build()`.
#[no_mangle]
pub unsafe extern "C" fn bollard_add_opening_interval(
    ptr: *mut ModelBuilder<i64>,
    berth_index: usize,
    start: i64,
    end: i64,
) {
    let builder = &mut *ptr;
    let interval = ClosedOpenInterval::new(start, end);
    builder.berth_interval(BerthIndex::new(berth_index), interval);
}

/// Solves the model and writes the solution into Julia-owned output arrays.
///
/// On success, returns `1` and fills `out_berths[i]` with the assigned berth index,
/// and `out_starts[i]` with the start time for vessel `i`. Returns `0` if infeasible
/// (no solution), or `-1` on input errors (e.g., `len` mismatch or build failure).
///
/// # Safety
/// - `ptr` must be a valid, non-null pointer to a `ModelBuilder<i64>` created
///   by `bollard_model_new` and not yet freed.
/// - `out_berths` must point to a writable array of length at least `len` (Julia-owned),
///   with element type compatible with `usize`.
/// - `out_starts` must point to a writable array of length at least `len` (Julia-owned),
///   with element type compatible with `i64`.
/// - The caller must ensure these pointers are properly aligned and not aliased in a way
///   that violates Rust’s aliasing rules.
/// - `len` must equal `builder.num_vessels()`; otherwise `-1` is returned without writing.
/// - No concurrent mutation of the builder should occur during this call.
/// - Behavior is undefined if any pointer is invalid or the buffers are too small.
#[no_mangle]
pub unsafe extern "C" fn bollard_solve_into(
    ptr: *mut ModelBuilder<i64>,
    out_berths: *mut usize, // Julia owns this
    out_starts: *mut i64,   // Julia owns this
    len: usize,
) -> i64 {
    let builder = &*ptr;

    if len != builder.num_vessels() {
        return -1;
    }
    let model_result = builder.clone().build();

    let model = match model_result {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Model Invalid: {}", e);
            return -1;
        }
    };

    let sol = dummy_solver(&model);

    match sol {
        Some(solution) => {
            let slice_berths = slice::from_raw_parts_mut(out_berths, len);
            let slice_starts = slice::from_raw_parts_mut(out_starts, len);

            let s_berths = solution.vessel_berths();
            let s_starts = solution.vessel_start_times();

            for i in 0..len {
                slice_berths[i] = s_berths[i].get();
                slice_starts[i] = s_starts[i];
            }
            1 // Success
        }
        None => 0, // Infeasible
    }
}

fn dummy_solver(model: &Model<i64>) -> Option<Solution<i64>> {
    let mut b = Vec::new();
    let mut s = Vec::new();
    for i in 0..model.num_vessels() {
        b.push(BerthIndex::new(0));
        s.push(model.arrival_time(VesselIndex::new(i)));
    }
    Some(Solution::new(b, s))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    unsafe fn make_builder(vessels: usize, berths: usize) -> *mut ModelBuilder<i64> {
        bollard_model_new(vessels, berths)
    }

    unsafe fn free_builder(ptr: *mut ModelBuilder<i64>) {
        bollard_model_free(ptr);
    }

    #[test]
    fn test_builder_new_and_free() {
        unsafe {
            let ptr = make_builder(3, 2);
            assert!(!ptr.is_null());
            free_builder(ptr);
        }
    }

    #[test]
    fn test_set_arrival_and_window() {
        unsafe {
            let ptr = make_builder(2, 1);
            // Set arrival and window for both vessels
            bollard_set_arrival(ptr, 0, 10);
            bollard_set_window(ptr, 1, 20, 200);

            // Build and inspect via normal Rust API
            let builder_ref = &*ptr;
            let model = builder_ref.clone().build().unwrap();

            assert_eq!(model.arrival_time(VesselIndex::new(0)), 10);
            assert_eq!(model.arrival_time(VesselIndex::new(1)), 20);
            assert_eq!(model.latest_departure_time(VesselIndex::new(1)), 200);

            free_builder(ptr);
        }
    }

    #[test]
    fn test_set_processing_available_and_unavailable() {
        unsafe {
            let ptr = make_builder(1, 2);
            // time >= 0 -> available, time < 0 -> unavailable
            bollard_set_processing(ptr, 0, 0, 15);
            bollard_set_processing(ptr, 0, 1, -1);

            let builder_ref = &*ptr;
            let model = builder_ref.clone().build().unwrap();

            assert_eq!(
                model
                    .processing_time(VesselIndex::new(0), BerthIndex::new(0))
                    .unwrap(),
                15
            );
            assert!(model
                .processing_time(VesselIndex::new(0), BerthIndex::new(1))
                .is_none());

            free_builder(ptr);
        }
    }

    #[test]
    fn test_add_opening_interval_and_build() {
        unsafe {
            let ptr = make_builder(1, 1);
            // Replace the default opening with a specific interval
            bollard_add_opening_interval(ptr, 0, 5, 50);

            let builder_ref = &*ptr;
            let model = builder_ref.clone().build().unwrap();
            let intervals = model.opening_time(BerthIndex::new(0));
            assert_eq!(intervals.len(), 1);
            assert_eq!(intervals[0], ClosedOpenInterval::new(5, 50));

            free_builder(ptr);
        }
    }

    #[test]
    fn test_solve_into_success_writes_output() {
        unsafe {
            let ptr = make_builder(3, 1);

            // Configure arrivals; dummy_solver assigns berth 0 and start = arrival
            bollard_set_arrival(ptr, 0, 11);
            bollard_set_arrival(ptr, 1, 22);
            bollard_set_arrival(ptr, 2, 33);

            // Prepare Julia-owned buffers (simulated here with Vecs)
            let mut out_berths: Vec<usize> = vec![usize::MAX; 3];
            let mut out_starts: Vec<i64> = vec![i64::MIN; 3];

            let rc = bollard_solve_into(ptr, out_berths.as_mut_ptr(), out_starts.as_mut_ptr(), 3);
            assert_eq!(rc, 1);

            // Check the outputs
            assert_eq!(out_berths, vec![0, 0, 0]); // dummy_solver assigns berth 0
            assert_eq!(out_starts, vec![11, 22, 33]); // starts = arrivals

            free_builder(ptr);
        }
    }

    #[test]
    fn test_solve_into_len_mismatch_returns_minus_one() {
        unsafe {
            let ptr = make_builder(2, 1);

            let mut out_berths: Vec<usize> = vec![0; 1]; // wrong length
            let mut out_starts: Vec<i64> = vec![0; 1];

            let rc = bollard_solve_into(ptr, out_berths.as_mut_ptr(), out_starts.as_mut_ptr(), 1);
            assert_eq!(rc, -1);

            free_builder(ptr);
        }
    }

    #[test]
    fn test_solve_into_invalid_model_returns_minus_one() {
        unsafe {
            let ptr = make_builder(1, 1);
            // Set invalid interval (empty) to cause build error
            bollard_add_opening_interval(ptr, 0, 10, 10);

            let mut out_berths: Vec<usize> = vec![0; 1];
            let mut out_starts: Vec<i64> = vec![0; 1];

            let rc = bollard_solve_into(ptr, out_berths.as_mut_ptr(), out_starts.as_mut_ptr(), 1);
            assert_eq!(rc, -1); // Build should fail

            free_builder(ptr);
        }
    }

    #[test]
    fn test_dummy_solver_behavior() {
        // Validate that dummy_solver mirrors arrivals to starts and assigns berth 0
        let builder = ModelBuilder::<i64>::new(2, 2);
        let builder = builder.clone(); // immutable clone used in FFI too
        let model = builder.build().unwrap();
        let sol = super::dummy_solver(&model).expect("dummy solver should return Some");

        assert_eq!(sol.vessel_berths().len(), model.num_vessels());
        assert!(sol.vessel_berths().iter().all(|b| b.get() == 0));
        for i in 0..model.num_vessels() {
            assert_eq!(
                sol.vessel_start_times()[i],
                model.arrival_time(VesselIndex::new(i))
            );
        }
    }

    #[test]
    fn test_free_null_pointer_is_noop() {
        unsafe {
            // Ensure calling free on a null pointer is a no-op
            bollard_model_free(ptr::null_mut());
        }
    }
}
