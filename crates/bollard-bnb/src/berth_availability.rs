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

//! Utilities for representing and querying berth availability over time.
//! The `BerthAvailability<T>` structure tracks sorted, disjoint intervals
//! of available and unavailable times for each berth and provides fast
//! normalization and querying routines used by the Branch-and-Bound solver.
//! Intervals are kept canonical to enable linear-time merging and efficient
//! lower-bound searches. Checked and unchecked accessors are provided for
//! safety in typical use and performance in hot paths. Initialization
//! consolidates fixed assignments and exclusions into consistent interval
//! sets, returning false when constraints are contradictory. Use
//! `earliest_availability` to obtain the next feasible start time for a berth,
//! and the interval accessors to inspect the current availability state.

// This implementation derives from my Bachelor's thesis,
// "Efficient Data Structures and Algorithms for a High-Performance
// Berth Allocation Solver Framework" (University of Hamburg, 2025).
// The thesis implementation was called `IntervalSet`. For this module,
// I've tuned it for cache locality and pared it down to essentials:
// a precomputed, sorted flat `Vec` of disjoint unavailable intervals.

use crate::fixed::FixedAssignment;
use bollard_core::{math::interval::ClosedOpenInterval, num::constants::MinusOne};
use bollard_model::{index::BerthIndex, model::Model};
use num_traits::{PrimInt, Signed};

/// Merges a list of closed-open intervals in place, coalescing overlaps and adjacency.
///
/// This function sorts intervals by start time, then performs a linear, in-place
/// compaction to merge any overlapping or adjacent intervals. The output is
/// guaranteed to be sorted by start and disjoint.
///
/// Complexity:
/// - O(N log N) for sorting + O(N) for compaction.
fn merge_intervals_in_place<T>(intervals: &mut Vec<ClosedOpenInterval<T>>)
where
    T: PrimInt,
{
    if intervals.is_empty() {
        return;
    }

    intervals.sort_unstable_by_key(|a| a.start());

    let mut write_index = 0;
    for read_index in 1..intervals.len() {
        let current = unsafe { *intervals.get_unchecked(write_index) };
        let next = unsafe { *intervals.get_unchecked(read_index) };

        if let Some(merged) = current.union(next) {
            unsafe { *intervals.get_unchecked_mut(write_index) = merged };
        } else {
            write_index += 1;
            if write_index != read_index {
                unsafe { *intervals.get_unchecked_mut(write_index) = next };
            }
        }
    }
    intervals.truncate(write_index + 1);

    debug_assert!(
        bollard_core::algorithm::are_disjoint_and_sorted(intervals),
        "`merge_intervals_in_place` output is not disjoint and sorted"
    );
}

/// Detects whether two disjoint, sorted interval sets overlap at any position.
///
/// Scans `right_intervals` while iterating `left_intervals` to determine
/// if any pair intersects. Adjacency (right.start == left.end) is not considered
/// overlap for closed-open intervals.
///
/// ## Invariants:
/// - `left_intervals` must be sorted and disjoint.
/// - `right_intervals` must be sorted and disjoint.
///
/// ## Complexity:
/// - O(|left| + |right|) due to linear advancement with peeking.
///
/// # Panics
///
/// In debug builds, this function will panic if either input slice is not sorted
/// by start time or contains overlapping intervals.
fn has_overlaps<T>(
    left_intervals: &[ClosedOpenInterval<T>],
    right_intervals: &[ClosedOpenInterval<T>],
) -> bool
where
    T: PrimInt,
{
    debug_assert!(
        bollard_core::algorithm::are_disjoint_and_sorted(left_intervals),
        "called `has_overlaps` with `left_intervals` not sorted by start or not disjoint"
    );
    debug_assert!(
        bollard_core::algorithm::are_disjoint_and_sorted(right_intervals),
        "called `has_overlaps` with `right_intervals` not sorted by start or not disjoint"
    );

    if left_intervals.is_empty() || right_intervals.is_empty() {
        return false;
    }

    let mut right_peekable = right_intervals.iter().peekable();

    for left_interval in left_intervals {
        while let Some(&right_interval) = right_peekable.peek() {
            if right_interval.end() <= left_interval.start() {
                right_peekable.next();
            } else {
                break;
            }
        }

        if let Some(&right_interval) = right_peekable.peek()
            && left_interval.intersects(*right_interval)
        {
            return true;
        }
    }
    false
}

/// Compact representation of berth unavailability across a set of berths.
///
/// `BerthAvailability` stores all unavailable time intervals (both fixed and temporary closures)
/// for every berth in a single, contiguous buffer, paired with an offsets array to indicate
/// the slice belonging to each berth. This layout minimizes per-berth allocation overhead and
/// enables cache-friendly iteration when scanning unavailability.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BerthAvailability<T>
where
    T: PrimInt,
{
    // Flat storage of all unavailability intervals across every berth.
    // Intervals are represented as closed-open ranges [start, end) and laid out back-to-back
    // for cache-friendly scans and minimal per-berth allocations.
    unavailable_times_data: Vec<ClosedOpenInterval<T>>,

    // Per-berth index table into `unavailable_times_data`.
    // For berth j, its intervals live in the slice
    // `unavailable_times_data[offsets[j] .. offsets[j + 1]]`.
    // Invariant: `offsets.len() == num_berths + 1`.
    unavailable_times_offsets: Vec<usize>,

    // Total number of berths represented by this structure.
    num_berths: usize,
}

impl<T> BerthAvailability<T>
where
    T: PrimInt,
{
    /// Creates a new empty `BerthAvailability`.
    #[inline]
    pub fn new() -> Self {
        Self {
            unavailable_times_data: Vec::new(),
            unavailable_times_offsets: Vec::new(),
            num_berths: 0,
        }
    }

    /// Creates a new `BerthAvailability` with preallocated capacity for `num_berths` berths.
    #[inline]
    pub fn preallocated(num_berths: usize) -> Self {
        let mut offsets = Vec::with_capacity(num_berths + 1);
        offsets.push(0);
        Self {
            unavailable_times_data: Vec::with_capacity(num_berths * 2),
            unavailable_times_offsets: offsets,
            num_berths,
        }
    }

    /// Ensures internal vectors have capacity for at least `num_berths` berths.
    #[inline]
    pub fn ensure_capacity(&mut self, num_berths: usize) {
        if self.unavailable_times_offsets.len() <= num_berths {
            self.unavailable_times_offsets.resize(num_berths + 1, 0);
        }
    }

    /// Resets all availability data, clearing intervals for all berths.
    #[inline]
    pub fn reset(&mut self) {
        self.unavailable_times_data.clear();
        self.unavailable_times_offsets.clear();
        self.unavailable_times_offsets.push(0);
        self.num_berths = 0;
    }

    /// Initializes availability based on the model and fixed assignments.
    ///
    /// Returns `true` if initialization succeeded (structurally feasible),
    /// or `false` if constraints were violated (overlaps, invalid indices, etc.).
    pub fn initialize(&mut self, model: &Model<T>, fixed: &[FixedAssignment<T>]) -> bool
    where
        T: PrimInt + Signed + MinusOne,
    {
        let num_berths = model.num_berths();
        let num_vessels = model.num_vessels();

        self.num_berths = num_berths;

        // Workspace to perform merging before flattening
        let mut workspace: Vec<Vec<ClosedOpenInterval<T>>> = vec![Vec::new(); num_berths];

        for assignment in fixed {
            let berth_index = assignment.berth_index.get();
            let vessel_index = assignment.vessel_index.get();

            debug_assert!(
                berth_index < num_berths,
                "called `BerthAvailability::initialize` with fixed assignment having invalid berth index: the len is {} but the index is {}",
                num_berths,
                berth_index
            );

            debug_assert!(
                vessel_index < num_vessels,
                "called `BerthAvailability::initialize` with fixed assignment having invalid vessel index: the len is {} but the index is {}",
                num_vessels,
                vessel_index
            );

            if berth_index >= num_berths || vessel_index >= num_vessels {
                return false;
            }

            let processing_time_option =
                model.vessel_processing_time(assignment.vessel_index, assignment.berth_index);

            if processing_time_option.is_none() {
                return false;
            }

            let duration = processing_time_option.unwrap_unchecked();
            let start = assignment.start_time;
            let end = start + duration;

            workspace[berth_index].push(ClosedOpenInterval::new(start, end));
        }

        for (i, fixed_intervals) in workspace.iter_mut().enumerate().take(num_berths) {
            let berth_index = BerthIndex::new(i);
            fixed_intervals.sort_unstable_by_key(|a| a.start());

            if !fixed_intervals.is_empty() {
                for w in fixed_intervals.windows(2) {
                    if w[1].start() < w[0].end() {
                        return false;
                    }
                }
            }

            let closing_times = model.berth_closing_times(berth_index);
            if has_overlaps(fixed_intervals, closing_times) {
                return false;
            }

            fixed_intervals.extend_from_slice(closing_times);
            merge_intervals_in_place(fixed_intervals);
        }

        // Bake into flat structure
        self.unavailable_times_data.clear();
        self.unavailable_times_offsets.clear();
        self.unavailable_times_offsets.push(0);

        for berth_vec in workspace {
            self.unavailable_times_data.extend(berth_vec);
            self.unavailable_times_offsets
                .push(self.unavailable_times_data.len());
        }

        true
    }

    /// Returns the number of berths tracked.
    #[inline]
    pub fn num_berths(&self) -> usize {
        self.num_berths
    }

    /// Returns the unavailable intervals for the given berth.
    ///
    /// # Panics
    ///
    /// This function will panic if `berth_index` is out of bounds.
    #[inline]
    pub fn unavailable_intervals(&self, berth_index: BerthIndex) -> &[ClosedOpenInterval<T>] {
        let index = berth_index.get();

        debug_assert!(
            index < self.num_berths(),
            "called `BerthAvailability::unavailable_intervals` with berth index out of bounds: the len is {} but the index is {}",
            self.num_berths(),
            index
        );

        unsafe {
            let start = *self.unavailable_times_offsets.get_unchecked(index);
            let end = *self.unavailable_times_offsets.get_unchecked(index + 1);
            self.unavailable_times_data.get_unchecked(start..end)
        }
    }

    /// Unsafe version of `unavailable_intervals` that skips bounds checks.
    ///
    /// # Panics
    ///
    /// In debug builds, this function will panic if `berth_index` is out of bounds.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `berth_index` is within bounds.
    #[inline]
    pub unsafe fn unavailable_intervals_unchecked(
        &self,
        berth_index: BerthIndex,
    ) -> &[ClosedOpenInterval<T>] {
        let index = berth_index.get();
        debug_assert!(
            index < self.num_berths(),
            "called `BerthAvailability::unavailable_intervals_unchecked` with berth index out of bounds: the len is {} but the index is {}",
            self.num_berths(),
            index
        );

        unsafe {
            let start = *self.unavailable_times_offsets.get_unchecked(index);
            let end = *self.unavailable_times_offsets.get_unchecked(index + 1);
            self.unavailable_times_data.get_unchecked(start..end)
        }
    }

    /// Finds the earliest availability on the given berth starting at or after `start_time`
    ///
    /// # Panics
    ///
    /// This function will panic if `berth_index` is out of bounds.
    pub fn earliest_availability(
        &self,
        berth_index: BerthIndex,
        start_time: T,
        duration: T,
    ) -> Option<T> {
        let index = berth_index.get();

        debug_assert!(
            index < self.num_berths(),
            "called `BerthAvailability::earliest_availability` with berth index out of bounds: the len is {} but the index is {}",
            self.num_berths(),
            index
        );

        let occupied = self.unavailable_intervals(berth_index);
        if occupied.is_empty() {
            return Some(start_time);
        }

        let mut interval_index = bollard_core::algorithm::lower_bound_start(occupied, start_time);
        let mut cursor_start = start_time;

        if interval_index > 0 {
            // SAFETY: `interval_index` will be in `1..occupied.len()` here.
            // `occupied` is non-empty, so `interval_index - 1` is valid.
            let prev = unsafe { occupied.get_unchecked(interval_index - 1) };
            if cursor_start < prev.end() {
                cursor_start = prev.end();
            }
        }

        while interval_index < occupied.len() {
            // SAFETY: `interval_index` is in `0..occupied.len()` here.
            // So this is safe.
            let block = unsafe { occupied.get_unchecked(interval_index) };
            if block.start() >= cursor_start + duration {
                return Some(cursor_start);
            }
            cursor_start = cursor_start.max(block.end());
            interval_index += 1;
        }

        Some(cursor_start)
    }

    /// Unsafe version of `earliest_availability` that skips bounds checks.
    ///
    /// # Panics
    ///
    /// In debug builds, this function will panic if `berth_index` is out of bounds.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `berth_index` is within bounds.
    pub unsafe fn earliest_availability_unchecked(
        &self,
        berth_index: BerthIndex,
        start_time: T,
        duration: T,
    ) -> Option<T> {
        let index = berth_index.get();

        debug_assert!(
            index < self.num_berths(),
            "called `BerthAvailability::earliest_availability_unchecked` with berth index out of bounds: the len is {} but the index is {}",
            self.num_berths(),
            index
        );

        let occupied = unsafe { self.unavailable_intervals_unchecked(berth_index) };
        if occupied.is_empty() {
            return Some(start_time);
        }

        let mut interval_index = bollard_core::algorithm::lower_bound_start(occupied, start_time);
        let mut cursor_start = start_time;

        if interval_index > 0 {
            // SAFETY: `interval_index` will be in `1..occupied.len()` here.
            // `occupied` is non-empty, so `interval_index - 1` is valid.
            let prev = unsafe { occupied.get_unchecked(interval_index - 1) };
            if cursor_start < prev.end() {
                cursor_start = prev.end();
            }
        }

        while interval_index < occupied.len() {
            // SAFETY: `interval_index` is in `0..occupied.len()` here.
            // So this is safe.
            let block = unsafe { occupied.get_unchecked(interval_index) };
            if block.start() >= cursor_start + duration {
                return Some(cursor_start);
            }
            cursor_start = cursor_start.max(block.end());
            interval_index += 1;
        }

        Some(cursor_start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bollard_model::index::{BerthIndex, VesselIndex};
    use bollard_model::model::ModelBuilder;
    use bollard_model::time::ProcessingTime;

    type IntegerType = i64;

    fn iv(s: IntegerType, e: IntegerType) -> ClosedOpenInterval<IntegerType> {
        ClosedOpenInterval::new(s, e)
    }

    fn build_model_basic() -> Model<IntegerType> {
        let mut builder = ModelBuilder::<IntegerType>::new(2, 2);
        builder.add_berth_closing_time(BerthIndex::new(0), iv(50, 100));
        builder.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(0),
            ProcessingTime::some(20),
        );
        builder.set_vessel_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(0),
            ProcessingTime::some(40),
        );
        builder.build()
    }

    #[test]
    fn test_initialize_layout() {
        let model = build_model_basic();
        let mut ba = BerthAvailability::<IntegerType>::new();
        let fixed = vec![FixedAssignment::new(
            10,
            BerthIndex::new(0),
            VesselIndex::new(0),
        )];
        assert!(ba.initialize(&model, &fixed));

        let unavail = ba.unavailable_intervals(BerthIndex::new(0));
        assert_eq!(unavail, &[iv(10, 30), iv(50, 100)]);
    }

    #[test]
    fn test_earliest_availability_logic() {
        let model = build_model_basic();
        let mut ba = BerthAvailability::<IntegerType>::new();
        let fixed = vec![FixedAssignment::new(
            10,
            BerthIndex::new(0),
            VesselIndex::new(0),
        )]; // [10, 30)
        ba.initialize(&model, &fixed);

        let bi = BerthIndex::new(0);
        // Fits at 0 before the first block starts
        assert_eq!(ba.earliest_availability(bi, 0, 5), Some(0));
        // Needs 15 at start 0; [0, 10) is too small, jumps to gap [30, 50)
        assert_eq!(ba.earliest_availability(bi, 0, 15), Some(30));
    }
}
