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

use bollard_core::{math::interval::ClosedOpenInterval, num::constants::MinusOne};
use bollard_model::{index::BerthIndex, model::Model};
use num_traits::{PrimInt, Signed};

use crate::fixed::FixedAssignment;

/// Checks whether the given intervals are disjoint and sorted by start time.
///
/// Returns `true` if the intervals are disjoint and sorted, `false` otherwise.
#[inline(always)]
fn are_disjoint_and_sorted<T>(intervals: &[ClosedOpenInterval<T>]) -> bool
where
    T: PrimInt,
{
    intervals.windows(2).all(|w| w[0].end() <= w[1].start())
}

/// Highly optimized lower bound search for the first interval
/// whose start time is >= key.
///
/// # Panics
///
/// In debug builds, this function will panic if `intervals` is not sorted
/// by start time in ascending order.
///
/// # Invariants
///
/// - `intervals` must be sorted by start time in ascending order.
#[inline(always)]
fn lower_bound_start<T>(intervals: &[ClosedOpenInterval<T>], key: T) -> usize
where
    T: PrimInt,
{
    debug_assert!(
        are_disjoint_and_sorted(intervals),
        "called `lower_bound_start` with intervals that are not disjoint and sorted"
    );

    let mut lo: usize = 0;
    let mut hi: usize = intervals.len();

    while lo < hi {
        let mid = lo + ((hi - lo) >> 1);
        debug_assert!(
            mid < intervals.len(),
            "`lower_bound_start` computed mid index out of bounds"
        );
        // SAFETY: mid is always in bounds because lo < hi <= intervals.len(),
        // therefore mid < intervals.len()
        if unsafe { intervals.get_unchecked(mid).start() } < key {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

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
        are_disjoint_and_sorted(intervals),
        "`merge_intervals_in_place` output is not disjoint and sorted"
    );
}

/// Computes the set difference of disjoint, sorted intervals: base \ exclusions.
///
/// Given two slices of disjoint, sorted closed-open intervals, this function
/// subtracts all `exclusions` from `base` and writes the resulting disjoint
/// segments into `output`, reusing its capacity.
///
/// Invariants:
/// - `base` must be sorted and disjoint.
/// - `exclusions` must be sorted and disjoint.
/// - Adjacency is non-overlapping; subtraction preserves closed-open semantics.
///
/// Complexity:
/// - O(|base| + |exclusions|) in the common case due to linear scans.
/// - No intermediate heap allocations per base interval; only writes to `output`.
fn subtract_intervals_into<T>(
    base: &[ClosedOpenInterval<T>],
    exclusions: &[ClosedOpenInterval<T>],
    output: &mut Vec<ClosedOpenInterval<T>>,
) where
    T: PrimInt,
{
    debug_assert!(
        are_disjoint_and_sorted(base),
        "called `subtract_intervals_into` with `base` not sorted by start or not disjoint"
    );
    debug_assert!(
        are_disjoint_and_sorted(exclusions),
        "called `subtract_intervals_into` with `exclusions` not sorted by start or not disjoint"
    );

    output.clear();

    if exclusions.is_empty() {
        output.extend_from_slice(base);
        return;
    }

    let mut blocked_index = 0usize;

    for &source_interval in base {
        let mut cursor_start = source_interval.start();
        let cursor_end = source_interval.end();

        while blocked_index < exclusions.len() && exclusions[blocked_index].end() <= cursor_start {
            blocked_index += 1;
        }

        // If the current base interval ends before the next exclusion starts,
        // we are strictly "to the left" of any problems. Just push it and continue.
        if blocked_index < exclusions.len() && cursor_end <= exclusions[blocked_index].start() {
            output.push(source_interval);
            continue;
        }

        let mut scan_blocked_index = blocked_index;
        while scan_blocked_index < exclusions.len() {
            let blocked = exclusions[scan_blocked_index];

            if blocked.start() >= cursor_end {
                break;
            }

            if cursor_start < blocked.start() {
                output.push(ClosedOpenInterval::new(cursor_start, blocked.start()));
            }

            if blocked.end() > cursor_start {
                cursor_start = blocked.end();
            }

            if cursor_start >= cursor_end {
                break;
            }

            if blocked.end() < cursor_end {
                scan_blocked_index += 1;
            } else {
                break;
            }
        }

        if cursor_start < cursor_end {
            output.push(ClosedOpenInterval::new(cursor_start, cursor_end));
        }
    }

    if !output.is_empty() {
        merge_intervals_in_place(output);
    }
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
        are_disjoint_and_sorted(left_intervals),
        "called `has_overlaps` with `left_intervals` not sorted by start or not disjoint"
    );
    debug_assert!(
        are_disjoint_and_sorted(right_intervals),
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BerthAvailability<T>
where
    T: PrimInt,
{
    unavailable_times: Vec<Vec<ClosedOpenInterval<T>>>, // per berth, sorted, non-overlapping disjoint intervals
    available_times: Vec<Vec<ClosedOpenInterval<T>>>, // per berth, sorted, non-overlapping disjoint intervals
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
            unavailable_times: Vec::new(),
            available_times: Vec::new(),
            num_berths: 0,
        }
    }

    /// Creates a new `BerthAvailability` with preallocated capacity for `num_berths` berths.
    #[inline]
    pub fn preallocated(num_berths: usize) -> Self {
        Self {
            unavailable_times: Vec::with_capacity(num_berths),
            available_times: Vec::with_capacity(num_berths),
            num_berths,
        }
    }

    /// Ensures internal vectors have capacity for at least `num_berths` berths.
    /// If current capacity is less, resizes with empty vectors.
    #[inline]
    pub fn ensure_capacity(&mut self, num_berths: usize) {
        if self.unavailable_times.len() < num_berths {
            self.unavailable_times.resize(num_berths, Vec::new());
        }
        if self.available_times.len() < num_berths {
            self.available_times.resize(num_berths, Vec::new());
        }
    }

    /// Resets all availability data, clearing intervals for all berths.
    #[inline]
    pub fn reset(&mut self) {
        for vec in &mut self.unavailable_times {
            vec.clear();
        }
        for vec in &mut self.available_times {
            vec.clear();
        }

        self.num_berths = 0;
        debug_assert_eq!(self.unavailable_times.len(), self.available_times.len());
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
        self.ensure_capacity(num_berths);

        // Clear existing intervals
        // Same as `self.reset()` but does not reset num_berths to `0`.
        for vec in &mut self.unavailable_times {
            vec.clear();
        }
        for vec in &mut self.available_times {
            vec.clear();
        }

        for assignment in fixed {
            let berth_index = assignment.berth_index.get();
            let vessel_index = assignment.vessel_index.get();

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

            unsafe {
                self.unavailable_times
                    .get_unchecked_mut(berth_index)
                    .push(ClosedOpenInterval::new(start, end));
            }
        }

        for i in 0..num_berths {
            let berth_index = BerthIndex::new(i);
            let fixed_intervals = &mut self.unavailable_times[i];
            fixed_intervals.sort_unstable_by_key(|a| a.start());

            if !fixed_intervals.is_empty() {
                for w in 0..fixed_intervals.len() - 1 {
                    let current = unsafe { *fixed_intervals.get_unchecked(w) };
                    let next = unsafe { *fixed_intervals.get_unchecked(w + 1) };
                    if next.start() < current.end() {
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

            let unavailable_ref = &self.unavailable_times[i];
            let available_vec = &mut self.available_times[i];
            let opening_ref = model.berth_opening_times(berth_index);

            subtract_intervals_into(opening_ref, unavailable_ref, available_vec);
        }

        true
    }

    /// Returns the number of berths tracked.
    #[inline]
    pub fn num_berths(&self) -> usize {
        self.num_berths
    }

    /// Returns the available intervals for the given berth.
    ///
    /// # Panics
    ///
    /// This function will panic if `berth_index` is out of bounds.
    #[inline]
    pub fn available_intervals(&self, berth_index: BerthIndex) -> &[ClosedOpenInterval<T>] {
        let index = berth_index.get();

        debug_assert!(
            index < self.num_berths(),
            "called `BerthAvailability::available_intervals` with berth index out of bounds: the len is {} but the index is {}",
            self.num_berths(),
            index
        );

        &self.available_times[index]
    }

    /// Unsafe version of `available_intervals` that skips bounds checks.
    ///
    /// # Panics
    ///
    /// In debug builds, this function will panic if `berth_index` is out of bounds.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `berth_index` is within bounds.
    #[inline]
    pub unsafe fn available_intervals_unchecked(
        &self,
        berth_index: BerthIndex,
    ) -> &[ClosedOpenInterval<T>] {
        let index = berth_index.get();

        debug_assert!(
            index < self.num_berths(),
            "called `BerthAvailability::available_intervals_unchecked` with berth index out of bounds: the len is {} but the index is {}",
            self.num_berths(),
            index
        );

        unsafe { self.available_times.get_unchecked(index) }
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

        &self.unavailable_times[index]
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

        unsafe { self.unavailable_times.get_unchecked(index) }
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

        let intervals = &self.available_times[index];
        if intervals.is_empty() {
            return None;
        }

        let lower_bound = lower_bound_start(intervals, start_time);

        if lower_bound > 0 {
            let interval = &intervals[lower_bound - 1];
            if start_time >= interval.start() && start_time < interval.end() {
                let remaining = interval.end() - start_time;
                if duration <= remaining {
                    return Some(start_time);
                }
            }
        }

        for iv in &intervals[lower_bound..] {
            let end = iv.end();
            let candidate_start = iv.start().max(start_time);
            if candidate_start >= end {
                continue;
            }
            let remaining = end - candidate_start;
            if duration <= remaining {
                return Some(candidate_start);
            }
        }

        None
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
            "called `BerthAvailability::earliest_availability` with berth index out of bounds: the len is {} but the index is {}",
            self.num_berths(),
            index
        );

        let intervals = unsafe { self.available_times.get_unchecked(index) };
        if intervals.is_empty() {
            return None;
        }

        let lower_bound = lower_bound_start(intervals, start_time);
        if lower_bound > 0 {
            let interval = unsafe { intervals.get_unchecked(lower_bound - 1) };
            if start_time >= interval.start() && start_time < interval.end() {
                let finish = start_time + duration;
                if finish <= interval.end() {
                    return Some(start_time);
                }
            }
        }

        for interval in &intervals[lower_bound..] {
            let candidate_start = interval.start().max(start_time);
            if candidate_start >= interval.end() {
                continue;
            }
            let finish = candidate_start + duration;
            if finish <= interval.end() {
                return Some(candidate_start);
            }
        }

        None
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

    #[test]
    fn test_are_disjoint_and_sorted_true_empty() {
        let v: Vec<ClosedOpenInterval<IntegerType>> = vec![];
        assert!(are_disjoint_and_sorted(&v));
    }

    #[test]
    fn test_are_disjoint_and_sorted_true_single() {
        let v = vec![iv(0, 10)];
        assert!(are_disjoint_and_sorted(&v));
    }

    #[test]
    fn test_are_disjoint_and_sorted_true_multiple_disjoint_sorted() {
        let v = vec![iv(0, 5), iv(5, 10), iv(10, 20)];
        assert!(are_disjoint_and_sorted(&v));
    }

    #[test]
    fn test_are_disjoint_and_sorted_false_overlap() {
        let v = vec![iv(0, 10), iv(9, 15)];
        assert!(!are_disjoint_and_sorted(&v));
    }

    #[test]
    fn test_are_disjoint_and_sorted_false_unsorted() {
        let v = vec![iv(10, 20), iv(0, 5)];
        // Even though disjoint, unsorted by start should fail
        assert!(!are_disjoint_and_sorted(&v));
    }

    #[test]
    fn test_lower_bound_start_basic() {
        let v = vec![iv(0, 5), iv(5, 10), iv(10, 20)];
        assert_eq!(lower_bound_start(&v, 0), 0);
        assert_eq!(lower_bound_start(&v, 4), 1); // first start >= 4 is index 1 (5-10)
        assert_eq!(lower_bound_start(&v, 5), 1);
        assert_eq!(lower_bound_start(&v, 6), 2); // first start >= 6 is index 2 (10-20)
        assert_eq!(lower_bound_start(&v, 10), 2);
        assert_eq!(lower_bound_start(&v, 21), 3);
    }

    #[test]
    fn test_merge_intervals_in_place_no_change_disjoint_sorted() {
        let mut v = vec![iv(0, 5), iv(5, 10), iv(15, 20)];
        merge_intervals_in_place(&mut v);
        // Adjacent (0,5) and (5,10) should merge; (15,20) remains
        assert_eq!(v, vec![iv(0, 10), iv(15, 20)]);
        assert!(are_disjoint_and_sorted(&v));
    }

    #[test]
    fn test_merge_intervals_in_place_overlap_merge() {
        let mut v = vec![iv(0, 7), iv(5, 10), iv(11, 12)];
        merge_intervals_in_place(&mut v);
        // (0,7) and (5,10) overlap -> merge to (0,10)
        assert_eq!(v, vec![iv(0, 10), iv(11, 12)]);
        assert!(are_disjoint_and_sorted(&v));
    }

    #[test]
    fn test_merge_intervals_in_place_multiple_merges() {
        let mut v = vec![iv(0, 3), iv(3, 6), iv(6, 6), iv(7, 8), iv(8, 10)];
        merge_intervals_in_place(&mut v);
        // Adjacent chain merges into (0,6) and (7,10)
        assert_eq!(v, vec![iv(0, 6), iv(7, 10)]);
        assert!(are_disjoint_and_sorted(&v));
    }

    #[test]
    fn test_subtract_intervals_into_no_exclusions() {
        let base = vec![iv(0, 10)];
        let excl: Vec<ClosedOpenInterval<IntegerType>> = vec![];
        let mut out = Vec::new();
        subtract_intervals_into(&base, &excl, &mut out);
        assert_eq!(out, base);
        assert!(are_disjoint_and_sorted(&out));
    }

    #[test]
    fn test_subtract_intervals_into_full_cover() {
        let base = vec![iv(0, 10)];
        let excl = vec![iv(0, 10)];
        let mut out = Vec::new();
        subtract_intervals_into(&base, &excl, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_subtract_intervals_into_middle_hole() {
        let base = vec![iv(0, 10)];
        let excl = vec![iv(3, 7)];
        let mut out = Vec::new();
        subtract_intervals_into(&base, &excl, &mut out);
        assert_eq!(out, vec![iv(0, 3), iv(7, 10)]);
        assert!(are_disjoint_and_sorted(&out));
    }

    #[test]
    fn test_subtract_intervals_into_multiple_exclusions() {
        let base = vec![iv(0, 20)];
        let excl = vec![iv(2, 5), iv(5, 7), iv(10, 12), iv(15, 20)];
        let mut out = Vec::new();
        subtract_intervals_into(&base, &excl, &mut out);
        // Expect segments: [0,2), [7,10), [12,15)
        assert_eq!(out, vec![iv(0, 2), iv(7, 10), iv(12, 15)]);
        assert!(are_disjoint_and_sorted(&out));
    }

    #[test]
    fn test_subtract_intervals_into_exclusions_outside() {
        let base = vec![iv(10, 20)];
        let excl = vec![iv(0, 5), iv(5, 9), iv(21, 30)];
        let mut out = Vec::new();
        subtract_intervals_into(&base, &excl, &mut out);
        assert_eq!(out, vec![iv(10, 20)]);
    }

    #[test]
    fn test_has_overlaps_false_when_empty() {
        let a: Vec<ClosedOpenInterval<IntegerType>> = vec![];
        let b: Vec<ClosedOpenInterval<IntegerType>> = vec![iv(0, 1)];
        assert!(!has_overlaps(&a, &b));
        assert!(!has_overlaps(&b, &a));
    }

    #[test]
    fn test_has_overlaps_true_overlap() {
        let a = vec![iv(0, 10), iv(20, 30)];
        let b = vec![iv(5, 15)];
        assert!(has_overlaps(&a, &b));
    }

    #[test]
    fn test_has_overlaps_false_adjacent_only() {
        let a = vec![iv(0, 10)];
        let b = vec![iv(10, 20)];
        // Closed-open adjacency should NOT be considered overlap
        assert!(!has_overlaps(&a, &b));
    }

    #[test]
    fn test_has_overlaps_multiple_scans() {
        let a = vec![iv(0, 5), iv(10, 15), iv(20, 25)];
        let b = vec![iv(6, 9), iv(12, 13), iv(30, 40)];
        assert!(has_overlaps(&a, &b)); // (10,15) overlaps (12,13)
        // inversely also should detect overlap
        assert!(has_overlaps(&b, &a));
    }

    fn build_model_basic() -> Model<IntegerType> {
        // 2 berths, 2 vessels
        let mut builder = ModelBuilder::<IntegerType>::new(2, 2);

        // Berth 0 closes [50, 100), so it opens [0,50) and [100, i64::MAX)
        builder.add_berth_closing_time(BerthIndex::new(0), iv(50, 100));

        // Berth 1 stays open [0, i64::MAX); no closing times

        // Vessel 0 allowed on both berths with processing times
        builder.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(0),
            ProcessingTime::from_option(Some(20)),
        );
        builder.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(1),
            ProcessingTime::from_option(Some(30)),
        );

        // Vessel 1 allowed only on berth 0 with processing time 40; forbidden on berth 1
        builder.set_vessel_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(0),
            ProcessingTime::from_option(Some(40)),
        );
        builder.set_vessel_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(1),
            ProcessingTime::none(),
        );

        builder.build()
    }

    #[test]
    fn test_initialize_no_fixed_assignments() {
        let model = build_model_basic();
        let mut ba = BerthAvailability::<IntegerType>::new();

        let fixed: Vec<FixedAssignment<IntegerType>> = vec![];
        assert!(ba.initialize(&model, &fixed));

        // Berth 0: unavailable is only closing [50, 100)
        assert_eq!(ba.unavailable_intervals(BerthIndex::new(0)), &[iv(50, 100)]);
        // Berth 0: availability mirrors model opening
        assert_eq!(
            ba.available_intervals(BerthIndex::new(0)),
            &[iv(0, 50), iv(100, i64::MAX),]
        );

        // Berth 1: no closing, no fixed -> unavailable empty, availability full
        assert!(ba.unavailable_intervals(BerthIndex::new(1)).is_empty());
        assert_eq!(
            ba.available_intervals(BerthIndex::new(1)),
            &[iv(0, i64::MAX)]
        );
    }

    #[test]
    fn test_initialize_with_fixed_assignments_merged() {
        // Use a clean model without the [50,100) closure to simplify testing merged assignments
        let mut builder = ModelBuilder::<IntegerType>::new(1, 2);
        // Closure further out at [200, 300)
        builder.add_berth_closing_time(BerthIndex::new(0), iv(200, 300));
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
        let model = builder.build();

        let mut ba = BerthAvailability::<IntegerType>::new();

        // Fixed assignments on berth 0:
        // - Vessel 0 at t=10, duration 20 -> [10, 30)
        // - Vessel 1 at t=30, duration 40 -> [30, 70)
        // These are ADJACENT (not overlapping) and will merge into [10, 70)
        let fixed = vec![
            FixedAssignment::new(10, BerthIndex::new(0), VesselIndex::new(0)),
            FixedAssignment::new(30, BerthIndex::new(0), VesselIndex::new(1)),
        ];

        assert!(ba.initialize(&model, &fixed));

        // Unavailable: [10, 70) merged with closure [200, 300)
        assert_eq!(
            ba.unavailable_intervals(BerthIndex::new(0)),
            &[iv(10, 70), iv(200, 300)]
        );

        // Availability: [0, 10), [70, 200), [300, MAX)
        assert_eq!(
            ba.available_intervals(BerthIndex::new(0)),
            &[iv(0, 10), iv(70, 200), iv(300, i64::MAX),]
        );
    }

    #[test]
    fn test_initialize_forbidden_fixed_returns_false() {
        let model = build_model_basic();
        let mut ba = BerthAvailability::<IntegerType>::new();

        // Vessel 1 is forbidden on berth 1 -> initialize should return false
        let bad_fixed = vec![FixedAssignment::new(
            0,
            BerthIndex::new(1),
            VesselIndex::new(1),
        )];
        assert!(!ba.initialize(&model, &bad_fixed));
    }

    #[test]
    fn test_initialize_overlapping_fixed_returns_false() {
        let model = build_model_basic();
        let mut ba = BerthAvailability::<IntegerType>::new();

        // Two assignments on berth 0 that overlap:
        // V0: [10, 30)
        // V1: [20, 60) (Overlap 20-30)
        let bad_fixed = vec![
            FixedAssignment::new(10, BerthIndex::new(0), VesselIndex::new(0)),
            FixedAssignment::new(20, BerthIndex::new(0), VesselIndex::new(1)),
        ];
        assert!(!ba.initialize(&model, &bad_fixed));
    }

    #[test]
    fn test_earliest_availability_basic() {
        // Model with closure at [50, 100)
        let model = build_model_basic();
        let mut ba = BerthAvailability::<IntegerType>::new();

        // Fixed on berth 0: [10, 30)
        // Unavailable: [10, 30) and [50, 100)
        // Available: [0, 10), [30, 50), [100, MAX)
        let fixed = vec![FixedAssignment::new(
            10,
            BerthIndex::new(0),
            VesselIndex::new(0),
        )];
        assert!(ba.initialize(&model, &fixed));

        // 1. Request duration 5 at 0 -> Fits in [0, 10)
        assert_eq!(ba.earliest_availability(BerthIndex::new(0), 0, 5), Some(0));

        // 2. Request duration 15 at 0 -> [0, 10) too short.
        //    Next slot is [30, 50). Length 20. 15 fits starting at 30.
        assert_eq!(
            ba.earliest_availability(BerthIndex::new(0), 0, 15),
            Some(30)
        );

        // 3. Request duration 25 at 0.
        //    [0, 10) too short.
        //    [30, 50) len 20. Too short.
        //    [100, MAX) fits.
        assert_eq!(
            ba.earliest_availability(BerthIndex::new(0), 0, 25),
            Some(100)
        );

        // 4. Start search inside unavailable block [10, 30).
        //    Should jump to next available [30, 50).
        assert_eq!(
            ba.earliest_availability(BerthIndex::new(0), 20, 5),
            Some(30)
        );
    }

    #[test]
    fn test_available_unavailable_unchecked_match_checked() {
        let mut builder = ModelBuilder::<IntegerType>::new(2, 2);
        // Simple model, no closures
        builder.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(0),
            ProcessingTime::some(10),
        );
        builder.set_vessel_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(0),
            ProcessingTime::some(10),
        );
        builder.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(1),
            ProcessingTime::some(10),
        );
        let model = builder.build();

        let mut ba = BerthAvailability::<IntegerType>::new();

        // Valid disjoint/adjacent assignments
        let fixed = vec![
            FixedAssignment::new(10, BerthIndex::new(0), VesselIndex::new(0)), // [10, 20)
            FixedAssignment::new(20, BerthIndex::new(0), VesselIndex::new(1)), // [20, 30)
            FixedAssignment::new(200, BerthIndex::new(1), VesselIndex::new(0)), // [200, 210)
        ];

        assert!(ba.initialize(&model, &fixed));

        for b in 0..model.num_berths() {
            let bi = BerthIndex::new(b);
            let checked_avail = ba.available_intervals(bi);
            let checked_unavail = ba.unavailable_intervals(bi);
            let unchecked_avail = unsafe { ba.available_intervals_unchecked(bi) };
            let unchecked_unavail = unsafe { ba.unavailable_intervals_unchecked(bi) };
            assert_eq!(checked_avail, unchecked_avail);
            assert_eq!(checked_unavail, unchecked_unavail);
        }
    }
}
