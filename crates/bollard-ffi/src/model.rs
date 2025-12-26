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

//! # Foreign Function Interface (FFI) for Bollard Models
//!
//! This module provides a C-compatible API for defining and inspecting the scheduling
//! problems (models) used by the Bollard solver. It acts as a bridge between
//! external environments (C, C++, Python, etc.) and the Rust core.
//!
//! ## Overview
//!
//! The API follows the **Builder Pattern**:
//!
//! 1.  **Mutable Construction**: Users allocate a `ModelBuilder` to define the problem parameters
//!     (vessels, berths, timings, weights).
//! 2.  **Immutable Finalization**: The builder is transformed into a `Model`. This `Model` is
//!     highly optimized, immutable, and thread-safe, ready to be passed to the solver.
//!
//! ## Usage Lifecycle
//!
//! A typical integration flow involves the following steps:
//!
//! 1.  **Instantiation**: Create a builder using `bollard_model_builder_new`.
//! 2.  **Configuration**: Populate the builder with data:
//!     * **Vessels**: Set arrival, departure, and weights.
//!     * **Processing**: Define processing times for valid Vessel-Berth pairs.
//!     * **Availability**: Define opening/closing intervals for berths (using the `FfiOpenClosedInterval` struct).
//! 3.  **Finalization**: Call `bollard_model_builder_build`.
//!     * **Important**: This step consumes the builder. The builder pointer becomes invalid immediately after this call.
//! 4.  **Solving**: Pass the resulting `Model` pointer to the solver FFI.
//! 5.  **Cleanup**: Explicitly free the `Model` using `bollard_model_free` when it is no longer needed.
//!
//! ## Safety
//!
//! This module uses `unsafe` code to interact with raw pointers. Callers **must** ensure:
//!
//! * **Pointer Validity**: Pointers must be allocated by this library.
//! * **Ownership Transfer**: After calling `bollard_model_builder_build`, the builder pointer must **not** be used or freed; ownership of the underlying data is transferred to the new `Model`.
//! * **Bounds**: Indices for vessels and berths must be within the ranges defined during instantiation.
//! * **Null Pointers**: Passing `NULL` will result in a panic.
//!
//! ## Exported API
//!
//! ### Lifecycle & Finalization
//! * `bollard_model_builder_new`
//! * `bollard_model_builder_free`
//! * `bollard_model_builder_build`
//! * `bollard_model_free`
//!
//! ### Configuration (Builder)
//! * `bollard_model_builder_set_arrival_time`
//! * `bollard_model_builder_set_latest_departure_time`
//! * `bollard_model_builder_set_vessel_weight`
//! * `bollard_model_builder_set_processing_time`
//! * `bollard_model_builder_forbid_vessel_berth_assignment`
//! * `bollard_model_builder_add_opening_time`
//! * `bollard_model_builder_add_closing_time`
//!
//! ### Inspection (Model)
//! * `bollard_model_num_vessels`
//! * `bollard_model_num_berths`
//! * `bollard_model_get_vessel_weight`
//! * `bollard_model_get_vessel_arrival_time`
//! * `bollard_model_get_vessel_latest_departure_time`
//! * `bollard_model_get_processing_time`
//! * `bollard_model_get_num_berth_opening_times`
//! * `bollard_model_get_berth_opening_time`
//! * `bollard_model_get_num_berth_closing_times`
//! * `bollard_model_get_berth_closing_time`
//! * `bollard_model_get_model_log_complexity`
//!
//! ### Data Structures
//! * `FfiOpenClosedInterval`

use bollard_core::math::interval::ClosedOpenInterval;
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::{Model, ModelBuilder},
    time::ProcessingTime,
};

/// A C-compatible representation of a closed-open interval [start_inclusive, end_exclusive).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FfiOpenClosedInterval {
    pub start_inclusive: i64,
    pub end_exclusive: i64,
}

impl FfiOpenClosedInterval {
    /// Creates a new `FfiOpenClosedInterval`.
    #[inline]
    pub fn new(start_inclusive: i64, end_exclusive: i64) -> Self {
        assert!(
            start_inclusive <= end_exclusive,
            "called `FfiOpenClosedInterval::new` with invalid interval: [{}, {})",
            start_inclusive,
            end_exclusive
        );

        Self {
            start_inclusive,
            end_exclusive,
        }
    }

    /// Returns the start of the interval (inclusive).
    #[inline]
    pub fn start(&self) -> i64 {
        self.start_inclusive
    }

    /// Returns the end of the interval (exclusive).
    #[inline]
    pub fn end(&self) -> i64 {
        self.end_exclusive
    }
}

impl From<ClosedOpenInterval<i64>> for FfiOpenClosedInterval {
    fn from(interval: ClosedOpenInterval<i64>) -> Self {
        Self {
            start_inclusive: interval.start(),
            end_exclusive: interval.end(),
        }
    }
}

impl From<FfiOpenClosedInterval> for ClosedOpenInterval<i64> {
    fn from(val: FfiOpenClosedInterval) -> Self {
        ClosedOpenInterval::new(val.start_inclusive, val.end_exclusive)
    }
}

impl std::fmt::Display for FfiOpenClosedInterval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {})", self.start_inclusive, self.end_exclusive)
    }
}

/// Creates a new Bollard model builder with the specified number of vessels and berths.
#[no_mangle]
pub extern "C" fn bollard_model_builder_new(
    num_berths: usize,
    num_vessels: usize,
) -> *mut ModelBuilder<i64> {
    let builder = ModelBuilder::<i64>::new(num_berths, num_vessels);
    Box::into_raw(Box::new(builder))
}

/// Frees the memory allocated for the Bollard model builder.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_builder_new`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_builder_free(ptr: *mut ModelBuilder<i64>) {
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
/// allocated by `bollard_model_builder_new`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_builder_add_closing_time(
    ptr: *mut ModelBuilder<i64>,
    berth_index: usize,
    interval: FfiOpenClosedInterval,
) {
    assert!(
        !ptr.is_null(),
        "called `bollard_model_builder_add_closing_time` with null pointer"
    );
    let builder = &mut *ptr;

    assert!(
        berth_index < builder.num_berths(),
        "called `bollard_outcome_get_berth` with berth index out of bounds: the len is {} but the index is {}",
        berth_index,
        builder.num_berths()
    );

    builder.add_berth_closing_time(BerthIndex::new(berth_index), interval.into());
}

/// Adds an opening time interval for a specific berth in the Bollard model builder.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_builder_new`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_builder_add_opening_time(
    ptr: *mut ModelBuilder<i64>,
    berth_index: usize,
    interval: FfiOpenClosedInterval,
) {
    assert!(
        !ptr.is_null(),
        "called `bollard_model_builder_add_opening_time` with null pointer"
    );
    let builder = &mut *ptr;

    assert!(berth_index < builder.num_berths(),
        "called `bollard_outcome_get_berth` with berth index out of bounds: the len is {} but the index is {}",
        berth_index,
        builder.num_berths()
    );

    builder.add_berth_opening_time(BerthIndex::new(berth_index), interval.into());
}

/// Sets the arrival time for a specific vessel in the Bollard model builder.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_builder_new`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_builder_set_arrival_time(
    ptr: *mut ModelBuilder<i64>,
    vessel_index: usize,
    time: i64,
) {
    assert!(
        !ptr.is_null(),
        "called `bollard_model_builder_set_arrival_time` with null pointer"
    );
    let builder = &mut *ptr;

    assert!(vessel_index < builder.num_vessels(),
        "called `bollard_model_builder_set_arrival_time` with vessel index out of bounds: the len is {} but the index is {}",
        vessel_index,
        builder.num_vessels()
    );

    builder.set_vessel_arrival_time(VesselIndex::new(vessel_index), time);
}

/// Sets the latest departure time for a specific vessel in the Bollard model builder.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_builder_new`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_builder_set_latest_departure_time(
    ptr: *mut ModelBuilder<i64>,
    vessel_index: usize,
    time: i64,
) {
    assert!(
        !ptr.is_null(),
        "called `bollard_model_builder_set_latest_departure_time` with null pointer"
    );

    let builder = &mut *ptr;

    assert!(vessel_index < builder.num_vessels(),
        "called `bollard_model_builder_set_latest_departure_time` with vessel index out of bounds: the len is {} but the index is {}",
        vessel_index,
        builder.num_vessels()
    );

    builder.set_vessel_latest_departure_time(VesselIndex::new(vessel_index), time);
}

/// Sets the weight for a specific vessel in the Bollard model builder.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_builder_new`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_builder_set_vessel_weight(
    ptr: *mut ModelBuilder<i64>,
    vessel_index: usize,
    weight: i64,
) {
    assert!(
        !ptr.is_null(),
        "called `bollard_set_vessel_weight` with null pointer"
    );

    let builder = &mut *ptr;

    assert!(vessel_index < builder.num_vessels(),
        "called `bollard_set_vessel_weight` with vessel index out of bounds: the len is {} but the index is {}",
        vessel_index,
        builder.num_vessels()
    );

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
/// allocated by `bollard_model_builder_new`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_builder_set_processing_time(
    ptr: *mut ModelBuilder<i64>,
    vessel_index: usize,
    berth_index: usize,
    processing_time: i64,
) {
    assert!(
        !ptr.is_null(),
        "called `bollard_model_builder_set_processing_time` with null pointer"
    );

    assert!(
        processing_time >= 0,
        "called `bollard_model_builder_set_processing_time` with negative processing time: {}",
        processing_time
    );

    let builder = &mut *ptr;

    assert!(vessel_index < builder.num_vessels(),
        "called `bollard_model_builder_set_processing_time` with vessel index out of bounds: the len is {} but the index is {}",
        vessel_index,
        builder.num_vessels()
    );
    assert!(berth_index < builder.num_berths(),
        "called `bollard_model_builder_set_processing_time` with berth index out of bounds: the len is {} but the index is {}",
        berth_index,
        builder.num_berths()
    );

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
/// allocated by `bollard_model_builder_new`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_builder_forbid_vessel_berth_assignment(
    ptr: *mut ModelBuilder<i64>,
    vessel_index: usize,
    berth_index: usize,
) {
    assert!(
        !ptr.is_null(),
        "called `bollard_model_builder_forbid_vessel_berth_assignment` with null pointer"
    );

    let builder = &mut *ptr;

    assert!(vessel_index < builder.num_vessels(),
        "called `bollard_model_builder_forbid_vessel_berth_assignment` with vessel index out of bounds: the len is {} but the index is {}",
        vessel_index,
        builder.num_vessels()
    );

    assert!(berth_index < builder.num_berths(),
        "called `bollard_model_builder_forbid_vessel_berth_assignment` with berth index out of bounds: the len is {} but the index is {}",
        berth_index,
        builder.num_berths()
    );

    builder.set_vessel_processing_time(
        VesselIndex::new(vessel_index),
        BerthIndex::new(berth_index),
        ProcessingTime::none(),
    );
}

/// Build a `Model<i64>` from a `ModelBuilder<i64>`.
///
/// Note:
/// This function consumes the builder. After calling it, the builder pointer
/// must not be used again.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_builder_new`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_builder_build(
    ptr: *mut ModelBuilder<i64>,
) -> *mut Model<i64> {
    assert!(
        !ptr.is_null(),
        "called `bollard_model_builder_build` with null pointer"
    );

    let builder = Box::from_raw(ptr);
    let model = builder.build();

    Box::into_raw(Box::new(model))
}

/// Frees the memory allocated for the Bollard model.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_builder_build`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_free(ptr: *mut Model<i64>) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// Returns the number of vessels in the Bollard model.
///
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_builder_build`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_num_vessels(ptr: *const Model<i64>) -> usize {
    assert!(
        !ptr.is_null(),
        "called `bollard_model_num_vessels` with null pointer"
    );

    let model = &*ptr;
    model.num_vessels()
}

/// Returns the number of berths in the Bollard model.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_builder_build`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_num_berths(ptr: *const Model<i64>) -> usize {
    assert!(
        !ptr.is_null(),
        "called `bollard_model_num_berths` with null pointer"
    );

    let model = &*ptr;
    model.num_berths()
}

/// Returns the weight of a specific vessel in the Bollard model.
///
/// # Panics
///
/// This function will panic if `vessel_index` is out of bounds,
/// or if the pointer is null.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_builder_build`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_get_vessel_weight(
    ptr: *const Model<i64>,
    vessel_index: usize,
) -> i64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_model_get_vessel_weight` with null pointer"
    );

    let model = &*ptr;

    assert!(vessel_index < model.num_vessels(),
        "called `bollard_model_get_vessel_weight` with vessel index out of bounds: the len is {} but the index is {}",
        model.num_vessels(),
        vessel_index
    );

    model.vessel_weight(VesselIndex::new(vessel_index))
}

/// Returns the processing time for a specific vessel-berth pair in the Bollard model.
/// If the vessel cannot be processed at the berth, returns -1.
///
/// # Panics
///
/// This function will panic if `vessel_index` or `berth_index` is out of bounds,
/// or if the pointer is null.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_builder_build`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_get_processing_time(
    ptr: *const Model<i64>,
    vessel_index: usize,
    berth_index: usize,
) -> i64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_model_get_processing_time` with null pointer"
    );

    let model = &*ptr;

    assert!(vessel_index < model.num_vessels(),
        "called `bollard_model_get_processing_time` with vessel index out of bounds: the len is {} but the index is {}",
        model.num_vessels(),
        vessel_index
    );

    assert!(berth_index < model.num_berths(),
        "called `bollard_model_get_processing_time` with berth index out of bounds: the len is {} but the index is {}",
        model.num_berths(),
        berth_index
    );

    let processing_time_opt =
        model.vessel_processing_time(VesselIndex::new(vessel_index), BerthIndex::new(berth_index));

    if processing_time_opt.is_none() {
        -1
    } else {
        processing_time_opt.unwrap_unchecked()
    }
}

/// Returns the arrival time of a specific vessel in the Bollard model.
///
/// # Panics
///
/// This function will panic if `vessel_index` is out of bounds,
/// or if the pointer is null.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_builder_build`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_get_vessel_arrival_time(
    ptr: *const Model<i64>,
    vessel_index: usize,
) -> i64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_model_get_vessel_arrival_time` with null pointer"
    );

    let model = &*ptr;

    assert!(vessel_index < model.num_vessels(),
        "called `bollard_model_get_vessel_arrival_time` with vessel index out of bounds: the len is {} but the index is {}",
        model.num_vessels(),
        vessel_index
    );

    model.vessel_arrival_time(VesselIndex::new(vessel_index))
}

/// Returns the latest departure time of a specific vessel in the Bollard model.
///
/// # Panics
///
/// This function will panic if `vessel_index` is out of bounds,
/// or if the pointer is null.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_builder_build`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_get_vessel_latest_departure_time(
    ptr: *const Model<i64>,
    vessel_index: usize,
) -> i64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_model_get_vessel_latest_departure_time` with null pointer"
    );

    let model = &*ptr;

    assert!(vessel_index < model.num_vessels(),
        "called `bollard_model_get_vessel_latest_departure_time` with vessel index out of bounds: the len is {} but the index is {}",
        model.num_vessels(),
        vessel_index
    );

    model.vessel_latest_departure_time(VesselIndex::new(vessel_index))
}

/// Returns the number of closing time intervals for a specific berth in the Bollard model.
///
/// # Panics
///
/// This function will panic if `berth_index` is out of bounds,
/// or if the pointer is null.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_builder_build`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_get_num_berth_closing_times(
    ptr: *const Model<i64>,
    berth_index: usize,
) -> usize {
    assert!(
        !ptr.is_null(),
        "called `bollard_model_get_num_berth_closing_times` with null pointer"
    );

    let model = &*ptr;

    assert!(berth_index < model.num_berths(),
        "called `bollard_model_get_num_berth_closing_times` with berth index out of bounds: the len is {} but the index is {}",
        model.num_berths(),
        berth_index
    );

    model
        .berth_closing_times(BerthIndex::new(berth_index))
        .len()
}

/// Returns the number of opening time intervals for a specific berth in the Bollard model.
///
/// # Panics
///
/// This function will panic if `berth_index` is out of bounds,
/// or if the pointer is null.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_builder_build`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_get_num_berth_opening_times(
    ptr: *const Model<i64>,
    berth_index: usize,
) -> usize {
    assert!(
        !ptr.is_null(),
        "called `bollard_model_get_num_berth_opening_times` with null pointer"
    );

    let model = &*ptr;

    assert!(berth_index < model.num_berths(),
        "called `bollard_model_get_num_berth_opening_times` with berth index out of bounds: the len is {} but the index is {}",
        model.num_berths(),
        berth_index
    );

    model
        .berth_opening_times(BerthIndex::new(berth_index))
        .len()
}

/// Retrieves a specific closing time interval for a berth in the Bollard model.
///
/// # Panics
///
/// This function will panic if `berth_index` or `interval_index` is out of bounds,
/// or if the pointer is null.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_builder_build`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_get_berth_closing_time(
    ptr: *const Model<i64>,
    berth_index: usize,
    interval_index: usize,
) -> FfiOpenClosedInterval {
    assert!(
        !ptr.is_null(),
        "called `bollard_model_get_berth_closing_time` with null pointer"
    );

    let model = &*ptr;

    assert!(berth_index < model.num_berths(),
        "called `bollard_model_get_berth_closing_time` with berth index out of bounds: the len is {} but the index is {}",
        model.num_berths(),
        berth_index
    );

    let intervals = model.berth_closing_times(BerthIndex::new(berth_index));

    assert!(interval_index < intervals.len(),
        "called `bollard_model_get_berth_closing_time` with interval index out of bounds: the len is {} but the index is {}",
        intervals.len(),
        interval_index
    );

    let interval = &intervals[interval_index];
    FfiOpenClosedInterval::from(*interval)
}

/// Retrieves a specific opening time interval for a berth in the Bollard model.
///
/// # Panics
///
/// This function will panic if `berth_index` or `interval_index` is out of bounds,
/// or if the pointer is null.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_builder_build`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_get_berth_opening_time(
    ptr: *const Model<i64>,
    berth_index: usize,
    interval_index: usize,
) -> FfiOpenClosedInterval {
    assert!(
        !ptr.is_null(),
        "called `bollard_model_get_berth_opening_time` with null pointer"
    );

    let model = &*ptr;

    assert!(berth_index < model.num_berths(),
        "called `bollard_model_get_berth_opening_time` with berth index out of bounds: the len is {} but the index is {}",
        model.num_berths(),
        berth_index
    );

    let intervals = model.berth_opening_times(BerthIndex::new(berth_index));

    assert!(interval_index < intervals.len(),
        "called `bollard_model_get_berth_opening_time` with interval index out of bounds: the len is {} but the index is {}",
        intervals.len(),
        interval_index
    );

    let interval = &intervals[interval_index];
    FfiOpenClosedInterval::from(*interval)
}

/// Returns the log complexity of the Bollard model as a floating-point number.
///
/// # Panics
///
/// This function will panic if the pointer is null.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was
/// allocated by `bollard_model_builder_build`.
#[no_mangle]
pub unsafe extern "C" fn bollard_model_get_model_log_complexity(ptr: *const Model<i64>) -> f64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_model_get_model_log_complexity` with null pointer"
    );

    let model = &*ptr;
    let complexity = model.complexity();
    complexity.raw()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::null_mut;

    // Build a small model that exercises most builder functions.
    unsafe fn build_fixture() -> *mut Model<i64> {
        let builder = bollard_model_builder_new(2, 2);

        // Opening and closing intervals
        bollard_model_builder_add_opening_time(builder, 0, FfiOpenClosedInterval::new(0, 100));
        bollard_model_builder_add_opening_time(builder, 1, FfiOpenClosedInterval::new(10, 50));
        bollard_model_builder_add_closing_time(builder, 0, FfiOpenClosedInterval::new(20, 30));

        // Vessel timings and weights
        bollard_model_builder_set_arrival_time(builder, 0, 5);
        bollard_model_builder_set_latest_departure_time(builder, 0, 40);
        bollard_model_builder_set_vessel_weight(builder, 0, 3);

        bollard_model_builder_set_arrival_time(builder, 1, 15);
        bollard_model_builder_set_latest_departure_time(builder, 1, 60);
        bollard_model_builder_set_vessel_weight(builder, 1, 5);

        // Processing times
        bollard_model_builder_set_processing_time(builder, 0, 0, 10);
        bollard_model_builder_set_processing_time(builder, 0, 1, 12);
        bollard_model_builder_set_processing_time(builder, 1, 0, 20);
        bollard_model_builder_set_processing_time(builder, 1, 1, 18);

        // Forbid a vessel on a berth
        bollard_model_builder_forbid_vessel_berth_assignment(builder, 1, 0);

        // Build consumes the builder; do not free the builder after this.
        bollard_model_builder_build(builder)
    }

    // Builder lifecycle tests
    #[test]
    fn test_new_and_free_basic() {
        unsafe {
            let ptr = bollard_model_builder_new(1, 1);
            assert!(!ptr.is_null());
            bollard_model_builder_free(ptr);
        }
    }

    #[test]
    fn test_free_null_pointer_is_noop() {
        unsafe {
            bollard_model_builder_free(null_mut());
        }
    }

    #[test]
    fn test_double_free_is_avoided() {
        unsafe {
            let ptr = bollard_model_builder_new(1, 1);
            bollard_model_builder_free(ptr);
            // Intentionally do not free twice to avoid UB.
        }
    }

    // Model construction and dimensions
    #[test]
    fn test_fixture_builds_and_dimensions_are_correct() {
        unsafe {
            let model = build_fixture();
            assert!(!model.is_null());
            assert_eq!(bollard_model_num_vessels(model), 2);
            assert_eq!(bollard_model_num_berths(model), 2);
            bollard_model_free(model);
        }
    }

    // Vessel scalar attributes
    #[test]
    fn test_vessel_weights() {
        unsafe {
            let model = build_fixture();
            assert_eq!(bollard_model_get_vessel_weight(model, 0), 3);
            assert_eq!(bollard_model_get_vessel_weight(model, 1), 5);
            bollard_model_free(model);
        }
    }

    #[test]
    fn test_vessel_arrival_times() {
        unsafe {
            let model = build_fixture();
            assert_eq!(bollard_model_get_vessel_arrival_time(model, 0), 5);
            assert_eq!(bollard_model_get_vessel_arrival_time(model, 1), 15);
            bollard_model_free(model);
        }
    }

    #[test]
    fn test_vessel_latest_departure_times() {
        unsafe {
            let model = build_fixture();
            assert_eq!(bollard_model_get_vessel_latest_departure_time(model, 0), 40);
            assert_eq!(bollard_model_get_vessel_latest_departure_time(model, 1), 60);
            bollard_model_free(model);
        }
    }

    // Processing times per vessel/berth pair
    #[test]
    fn test_processing_times_allowed_pairs() {
        unsafe {
            let model = build_fixture();
            assert_eq!(bollard_model_get_processing_time(model, 0, 0), 10);
            assert_eq!(bollard_model_get_processing_time(model, 0, 1), 12);
            assert_eq!(bollard_model_get_processing_time(model, 1, 1), 18);
            bollard_model_free(model);
        }
    }

    #[test]
    fn test_processing_time_forbidden_pair_returns_negative_one() {
        unsafe {
            let model = build_fixture();
            // Vessel 1 on berth 0 is forbidden in the fixture.
            assert_eq!(bollard_model_get_processing_time(model, 1, 0), -1);
            bollard_model_free(model);
        }
    }

    // Opening intervals
    #[test]
    fn test_opening_intervals_counts_are_nonzero() {
        unsafe {
            let model = build_fixture();
            let num_open_b0 = bollard_model_get_num_berth_opening_times(model, 0);
            let num_open_b1 = bollard_model_get_num_berth_opening_times(model, 1);
            assert!(num_open_b0 >= 1);
            assert!(num_open_b1 >= 1);
            bollard_model_free(model);
        }
    }

    #[test]
    fn test_opening_intervals_first_interval_is_valid() {
        unsafe {
            let model = build_fixture();

            if bollard_model_get_num_berth_opening_times(model, 0) > 0 {
                let iv = bollard_model_get_berth_opening_time(model, 0, 0);
                assert!(iv.start() >= 0 && iv.end() > iv.start());
            }

            if bollard_model_get_num_berth_opening_times(model, 1) > 0 {
                let iv = bollard_model_get_berth_opening_time(model, 1, 0);
                assert!(iv.start() >= 0 && iv.end() > iv.start());
            }

            bollard_model_free(model);
        }
    }

    // Closing intervals
    #[test]
    fn test_closing_intervals_counts() {
        unsafe {
            let model = build_fixture();
            let num_close_b1 = bollard_model_get_num_berth_closing_times(model, 1);
            assert_eq!(num_close_b1, 0);
            bollard_model_free(model);
        }
    }

    #[test]
    fn test_closing_intervals_first_interval_is_valid_when_present() {
        unsafe {
            let model = build_fixture();
            let num_close_b0 = bollard_model_get_num_berth_closing_times(model, 0);
            if num_close_b0 > 0 {
                let iv = bollard_model_get_berth_closing_time(model, 0, 0);
                assert!(iv.start() >= 0 && iv.end() > iv.start());
            }
            bollard_model_free(model);
        }
    }

    // Minimal model memory management
    #[test]
    fn test_minimal_model_no_leak() {
        unsafe {
            let builder = bollard_model_builder_new(0, 0);
            let model = bollard_model_builder_build(builder);
            assert_eq!(bollard_model_num_vessels(model), 0);
            assert_eq!(bollard_model_num_berths(model), 0);
            bollard_model_free(model);
        }
    }
}
