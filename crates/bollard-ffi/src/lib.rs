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
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::ModelBuilder,
    time::ProcessingTime,
};

/// Creates a new Bollard model builder with the specified number of vessels and berths.
///
/// # Examples
///
/// ```rust
/// # use bollard_ffi::{bollard_model_new, bollard_model_free};
///
/// let num_vessels = 10;
/// let num_berths = 5;
/// let model_builder_ptr = bollard_model_new(num_vessels, num_berths);
///
/// // Use the model builder...
/// assert!(!model_builder_ptr.is_null());
///
/// // Remember to free the model builder when done
/// unsafe {
///    bollard_model_free(model_builder_ptr);
/// }
/// ```
#[no_mangle]
pub extern "C" fn bollard_model_new(
    num_vessels: usize,
    num_berths: usize,
) -> *mut ModelBuilder<i64> {
    let builder = ModelBuilder::<i64>::new(num_vessels, num_berths);
    Box::into_raw(Box::new(builder))
}

/// Frees the memory allocated for the Bollard model builder.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_new`.
///
/// # Examples
///
/// ```rust
/// # use bollard_ffi::{bollard_model_new, bollard_model_free};
///
/// let model_builder_ptr = bollard_model_new(10, 5);
/// // Use the model builder...
/// unsafe {
///     bollard_model_free(model_builder_ptr);
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn bollard_model_free(ptr: *mut ModelBuilder<i64>) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// Adds a closing time interval for a specific berth in the Bollard model builder.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_new`.
///
/// # Examples
///
/// ```rust
/// # use bollard_ffi::{bollard_model_new, bollard_model_add_closing_time, bollard_model_free};
///
/// let model_builder_ptr = bollard_model_new(10, 5);
/// // Add closing time for berth at index 0
/// unsafe {
///     bollard_model_add_closing_time(model_builder_ptr, 0, 100, 200);
/// }
///
/// // Remember to free the model builder when done
/// unsafe {
///     bollard_model_free(model_builder_ptr);
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn bollard_model_add_closing_time(
    ptr: *mut ModelBuilder<i64>,
    berth_index: usize,
    time_start_inclusive: i64,
    time_end_exclusive: i64,
) {
    assert!(!ptr.is_null());
    let builder = &mut *ptr;
    assert!(berth_index < builder.num_berths());
    builder.add_berth_closing_time(
        BerthIndex::new(berth_index),
        ClosedOpenInterval::new(time_start_inclusive, time_end_exclusive),
    );
}

/// Adds an opening time interval for a specific berth in the Bollard model builder.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_new`.
///
/// # Examples
///
/// ```rust
/// # use bollard_ffi::{bollard_model_new, bollard_model_add_opening_time, bollard_model_free};
///
/// let model_builder_ptr = bollard_model_new(10, 5);
/// // Add opening time for berth at index 0
/// unsafe {
///     bollard_model_add_opening_time(model_builder_ptr, 0, 300, 400);
/// }
///
/// // Remember to free the model builder when done
/// unsafe {
///     bollard_model_free(model_builder_ptr);
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn bollard_model_add_opening_time(
    ptr: *mut ModelBuilder<i64>,
    berth_index: usize,
    time_start_inclusive: i64,
    time_end_exclusive: i64,
) {
    assert!(!ptr.is_null());
    let builder = &mut *ptr;
    assert!(berth_index < builder.num_berths());
    builder.add_berth_opening_time(
        BerthIndex::new(berth_index),
        ClosedOpenInterval::new(time_start_inclusive, time_end_exclusive),
    );
}

/// Sets the arrival time for a specific vessel in the Bollard model builder.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_new`.
///
/// # Examples
///
/// ```rust
/// # use bollard_ffi::{bollard_model_new, bollard_set_arrival_time};
///
/// let model_builder_ptr = bollard_model_new(10, 5);
/// // Set arrival time for vessel at index 0
/// unsafe {
///     bollard_set_arrival_time(model_builder_ptr, 0, 100);
/// }
/// // Remember to free the model builder when done
/// unsafe {
///     bollard_model_free(model_builder_ptr);
/// }
#[no_mangle]
pub unsafe extern "C" fn bollard_set_arrival_time(
    ptr: *mut ModelBuilder<i64>,
    vessel_index: usize,
    time: i64,
) {
    assert!(!ptr.is_null());
    let builder = &mut *ptr;
    assert!(vessel_index < builder.num_vessels());
    builder.set_vessel_arrival_time(VesselIndex::new(vessel_index), time);
}

/// Sets the latest departure time for a specific vessel in the Bollard model builder.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_new`.
///
/// # Examples
///
/// ```rust
/// # use bollard_ffi::{bollard_model_new, bollard_set_latest_departure_time};
///
/// let model_builder_ptr = bollard_model_new(10, 5);
/// // Set latest departure time for vessel at index 0
/// unsafe {
///     bollard_set_latest_departure_time(model_builder_ptr, 0, 200);
/// }
/// // Remember to free the model builder when done
/// unsafe {
///     bollard_model_free(model_builder_ptr);
/// }
#[no_mangle]
pub unsafe extern "C" fn bollard_set_latest_departure_time(
    ptr: *mut ModelBuilder<i64>,
    vessel_index: usize,
    time: i64,
) {
    assert!(!ptr.is_null());
    let builder = &mut *ptr;
    assert!(vessel_index < builder.num_vessels());
    builder.set_vessel_latest_departure_time(VesselIndex::new(vessel_index), time);
}

/// Sets the weight for a specific vessel in the Bollard model builder.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_new`.
///
/// # Examples
///
/// ```rust
/// # use bollard_ffi::{bollard_model_new, bollard_set_vessel_weight};
///
/// let model_builder_ptr = bollard_model_new(10, 5);
/// // Set weight for vessel at index 0
/// unsafe {
///     bollard_set_vessel_weight(model_builder_ptr, 0, 50);
/// }
/// // Remember to free the model builder when done
/// unsafe {
///     bollard_model_free(model_builder_ptr);
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn bollard_model_set_vessel_weight(
    ptr: *mut ModelBuilder<i64>,
    vessel_index: usize,
    weight: i64,
) {
    assert!(!ptr.is_null());
    let builder = &mut *ptr;
    assert!(vessel_index < builder.num_vessels());
    builder.set_vessel_weight(VesselIndex::new(vessel_index), weight);
}

/// Sets the processing time for a specific vessel-berth pair in the Bollard model builder.
/// The caller must ensure that `processing_time` is non-negative.
///
/// # Panics
///
/// This function will panic if `processing_time` is negative,
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_new`.
///
/// # Examples
///
/// ```rust
/// # use bollard_ffi::{bollard_model_new, bollard_model_set_processing_time};
///
/// let model_builder_ptr = bollard_model_new(10, 5);
/// // Set processing time for vessel at index 0 and berth at index 0
/// unsafe {
///     bollard_model_set_processing_time(model_builder_ptr, 0, 0, 150);
/// }
/// // Remember to free the model builder when done
/// unsafe {
///     bollard_model_free(model_builder_ptr);
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn bollard_model_set_processing_time(
    ptr: *mut ModelBuilder<i64>,
    vessel_index: usize,
    berth_index: usize,
    processing_time: i64,
) {
    assert!(!ptr.is_null());
    assert!(processing_time >= 0);
    let builder = &mut *ptr;
    assert!(vessel_index < builder.num_vessels());
    assert!(berth_index < builder.num_berths());
    builder.set_vessel_processing_time(
        VesselIndex::new(vessel_index),
        BerthIndex::new(berth_index),
        ProcessingTime::some(processing_time),
    );
}

/// Forbids the assignment of a specific vessel to a specific berth in the Bollard model builder.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_new`.
///
/// # Examples
///
/// ```rust
/// # use bollard_ffi::{bollard_model_new, bollard_forbid_vessel_berth_assignment, bollard_model_free};
///
/// let model_builder_ptr = bollard_model_new(10, 5);
/// // Forbid assignment of vessel at index 0 to berth at index 0
/// unsafe {
///    bollard_forbid_vessel_berth_assignment(model_builder_ptr, 0, 0);
/// }
///
/// // Remember to free the model builder when done
/// unsafe {
///    bollard_model_free(model_builder_ptr);
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn bollard_forbid_vessel_berth_assignment(
    ptr: *mut ModelBuilder<i64>,
    vessel_index: usize,
    berth_index: usize,
) {
    assert!(!ptr.is_null());
    let builder = &mut *ptr;
    assert!(vessel_index < builder.num_vessels());
    assert!(berth_index < builder.num_berths());
    builder.set_vessel_processing_time(
        VesselIndex::new(vessel_index),
        BerthIndex::new(berth_index),
        ProcessingTime::none(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_free_basic() {
        let ptr = bollard_model_new(3, 2);
        assert!(!ptr.is_null());
        unsafe { bollard_model_free(ptr) };
    }

    #[test]
    fn test_free_null_pointer_is_noop() {
        // SAFETY: calling free on a null pointer should be safe and do nothing.
        unsafe { bollard_model_free(std::ptr::null_mut()) };
    }

    #[test]
    fn test_double_free_is_undefined_behavior_and_avoided() {
        // Demonstrate correct free; we DO NOT double-free in tests.
        let ptr = bollard_model_new(1, 1);
        unsafe { bollard_model_free(ptr) };
    }
}
