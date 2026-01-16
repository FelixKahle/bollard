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

//! Interval utilities for closed-open ranges [start, end).
//!
//! This module provides helpers to validate and search collections of
//! disjoint, start-sorted `ClosedOpenInterval<T>` values where `T: PrimInt`.
//!
//! # Provided functions:
//!
//! - `are_disjoint_and_sorted` — checks that intervals are pairwise disjoint and sorted by start.
//! - `lower_bound_start_linear` — O(n) lower bound: first index with `start() >= key`.
//! - `lower_bound_start_binary` — O(log n) lower bound using binary search.
//! - `lower_bound_start` — default lower bound (currently delegates to binary search).
//! - `lower_bound_start_with_threshold` — lets you pick a size threshold to switch between linear/binary.
//!
//! # Semantics
//!
//! Given a slice of intervals sorted by `start()` and pairwise disjoint
//! (i.e., `intervals[i].end() <= intervals[i + 1].start()`), the lower-bound
//! functions return the smallest index `i` such that `intervals[i].start() >= key`,
//! or `intervals.len()` if no such interval exists.
//!
//! # Complexity and performance
//!
//! - Linear search: `O(n)`
//! - Binary search: `O(log n)`
//!
//! Benchmarks (noted in code comments) indicate the binary variant outperforms
//! linear even for very small `n` on modern CPUs, so `lower_bound_start` uses
//! the binary implementation by default.

use crate::math::interval::ClosedOpenInterval;
use num_traits::PrimInt;

/// Checks whether the given intervals are disjoint and sorted by start time.
///
/// Returns `true` if the intervals are disjoint and sorted, `false` otherwise.
#[inline(always)]
pub fn are_disjoint_and_sorted<T>(intervals: &[ClosedOpenInterval<T>]) -> bool
where
    T: PrimInt,
{
    intervals.windows(2).all(|w| w[0].end() <= w[1].start())
}

/// Simple linear search for the first interval
/// whose start time is >= key.
///
/// # Panics
///
/// In debug builds, this function will panic if `intervals` is not sorted
/// by start time in ascending order.
///
/// # Invariants
///
/// `intervals` must be sorted by start time in ascending order.
#[inline(always)]
pub fn lower_bound_start_linear<T>(intervals: &[ClosedOpenInterval<T>], key: T) -> usize
where
    T: PrimInt,
{
    debug_assert!(
        are_disjoint_and_sorted(intervals),
        "called `lower_bound_start_linear` with intervals that are not disjoint and sorted"
    );

    for i in 0..intervals.len() {
        // SAFETY: i is always in bounds because i < intervals.len()
        if unsafe { intervals.get_unchecked(i).start() } >= key {
            return i;
        }
    }
    intervals.len()
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
/// `intervals` must be sorted by start time in ascending order.
#[inline(always)]
pub fn lower_bound_start_binary<T>(intervals: &[ClosedOpenInterval<T>], key: T) -> usize
where
    T: PrimInt,
{
    debug_assert!(
        are_disjoint_and_sorted(intervals),
        "called `lower_bound_start_binary` with intervals that are not disjoint and sorted"
    );

    let mut lo: usize = 0;
    let mut hi: usize = intervals.len();

    while lo < hi {
        let mid = lo + ((hi - lo) >> 1);
        debug_assert!(
            mid < intervals.len(),
            "`lower_bound_start_binary` computed mid index out of bounds"
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

/// Chooses between linear and binary search for lower bound
/// based on the length of the intervals slice.
///
/// The idea is that for small slices, linear search can be faster due to
/// better cache locality and lower overhead, while for larger slices,
/// binary search is more efficient.
///
/// # Note
///
/// For now we always use binary search as benchmarks have shown
/// that it is strictly faster than linear search for all tested sizes
/// (N=1 to N=256).
///
/// # Panics
///
/// In debug builds, this function will panic if `intervals` is not sorted
/// by start time in ascending order.
///
/// # Invariants
///
/// `intervals` must be sorted by start time in ascending order.
#[inline(always)]
pub fn lower_bound_start<T>(intervals: &[ClosedOpenInterval<T>], key: T) -> usize
where
    T: PrimInt,
{
    // Benchmarks (Criterion, M3 Max) show that binary search is strictly faster
    // than linear search for all tested sizes (N=1 to N=256).
    //
    // Reason:
    // 1. L1 Cache Residency: The interval structs are small, so the entire slice
    //    often fits in L1 cache. This negates the "random access penalty" that
    //    usually hurts binary search on small arrays.
    // 2. Instruction Count: With memory latency effectively zero, performance
    //    is dominated by instruction count. Binary search performs O(log N)
    //    comparisons versus O(N), winning immediately.
    // 3. Iterator Overhead: Rust's iterator state machine in linear search
    //    proved slightly more expensive than raw index math even at N=1.
    lower_bound_start_binary(intervals, key)
}

/// Chooses between linear and binary search for lower bound
/// based on the length of the intervals slice and a custom threshold.
///
/// The idea is that for small slices, linear search can be faster due to
/// better cache locality and lower overhead, while for larger slices,
/// binary search is more efficient.
///
/// # Panics
///
/// In debug builds, this function will panic if `intervals` is not sorted
/// by start time in ascending order.
///
/// # Invariants
///
/// `intervals` must be sorted by start time in ascending order.
#[inline(always)]
pub fn lower_bound_start_with_threshold<T>(
    intervals: &[ClosedOpenInterval<T>],
    key: T,
    threshold: usize,
) -> usize
where
    T: PrimInt,
{
    debug_assert!(
        are_disjoint_and_sorted(intervals),
        "called `lower_bound_start` with intervals that are not disjoint and sorted"
    );

    if intervals.len() <= threshold {
        lower_bound_start_linear(intervals, key)
    } else {
        lower_bound_start_binary(intervals, key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
