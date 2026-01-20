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

use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use bollard_search::num::SolverNumeric;

/// Calculates the weighted flow time cost for a vessel assigned to a berth at a given start time.
///
/// # Panics
///
/// This function will panic if `vessel_index` is not within `0..model.num_vessels()`
/// or if `berth_index` is not within `0..model.num_berths()`.
#[inline(always)]
pub fn calculate_weighted_flow_time<T>(
    model: &Model<T>,
    vessel_index: VesselIndex,
    berth_index: BerthIndex,
    start_time: T,
) -> Option<T>
where
    T: SolverNumeric,
{
    debug_assert!(
        vessel_index.get() < model.num_vessels(),
        "called `calculate_weighted_flow_time` with vessel index out of bounds: the len is {} but the index is {}",
        model.num_vessels(),
        vessel_index.get()
    );

    debug_assert!(
        berth_index.get() < model.num_berths(),
        "called `calculate_weighted_flow_time` with berth index out of bounds: the len is {} but the index is {}",
        model.num_berths(),
        berth_index.get()
    );

    let deadline = model.vessel_latest_departure_time(vessel_index);
    let pt_opt = model.vessel_processing_time(vessel_index, berth_index);

    if pt_opt.is_none() {
        return None;
    }

    let pt = pt_opt.unwrap_unchecked();
    let completion_time = start_time.saturating_add_val(pt);

    if completion_time > deadline {
        return None;
    }

    let weight = model.vessel_weight(vessel_index);
    Some(completion_time.saturating_mul_val(weight))
}

/// Calculates the weighted flow time cost for a vessel assigned to a berth at a given start time.
///
/// # Panics
///
/// In debug builds, this function will panic if `vessel_index` is not within `0..model.num_vessels()`
/// or if `berth_index` is not within `0..model.num_berths()`.
///
/// # Safety
///
/// The caller must ensure that `vessel_index` is within `0..model.num_vessels()` and
/// `berth_index` is within `0..model.num_berths()`.
#[inline(always)]
pub unsafe fn calculate_weighted_flow_time_unchecked<T>(
    model: &Model<T>,
    vessel_index: VesselIndex,
    berth_index: BerthIndex,
    start_time: T,
) -> Option<T>
where
    T: SolverNumeric,
{
    debug_assert!(
        vessel_index.get() < model.num_vessels(),
        "called `calculate_weighted_flow_time_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
        model.num_vessels(),
        vessel_index.get()
    );

    debug_assert!(
        berth_index.get() < model.num_berths(),
        "called `calculate_weighted_flow_time_unchecked` with berth index out of bounds: the len is {} but the index is {}",
        model.num_berths(),
        berth_index.get()
    );

    let deadline = unsafe { model.vessel_latest_departure_time_unchecked(vessel_index) };
    let pt_opt = unsafe { model.vessel_processing_time_unchecked(vessel_index, berth_index) };

    if pt_opt.is_none() {
        return None;
    }

    let pt = pt_opt.unwrap_unchecked();
    let completion_time = start_time.saturating_add_val(pt);

    if completion_time > deadline {
        return None;
    }

    let weight = unsafe { model.vessel_weight_unchecked(vessel_index) };
    Some(completion_time.saturating_mul_val(weight))
}
