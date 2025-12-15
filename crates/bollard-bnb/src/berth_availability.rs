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

// TODO: Remove
#![allow(dead_code)]

use bollard_core::math::interval::ClosedOpenInterval;
use bollard_model::index::BerthIndex;
use num_traits::PrimInt;

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

#[derive(Clone, Debug, Default)]
pub struct BerthAvailability<T>
where
    T: PrimInt,
{
    unavailable_times: Vec<Vec<ClosedOpenInterval<T>>>, // per berth, sorted, non-overlapping disjoint intervals
    available_times: Vec<Vec<ClosedOpenInterval<T>>>, // per berth, sorted, non-overlapping disjoint intervals
}

impl<T> PartialEq for BerthAvailability<T>
where
    T: PrimInt,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.unavailable_times == other.unavailable_times
            && self.available_times == other.available_times
    }
}

impl<T> Eq for BerthAvailability<T> where T: PrimInt {}

impl<T> BerthAvailability<T>
where
    T: PrimInt,
{
    #[inline]
    pub fn empty(num_berths: usize) -> Self {
        Self {
            unavailable_times: vec![Vec::new(); num_berths],
            available_times: vec![Vec::new(); num_berths],
        }
    }

    #[inline]
    pub fn num_berths(&self) -> usize {
        self.unavailable_times.len()
    }

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
            let iv = &intervals[lower_bound - 1];
            if start_time >= iv.start() && start_time < iv.end() {
                let remaining = iv.end() - start_time;
                if duration <= remaining {
                    return Some(start_time);
                }
            }
        }

        for iv in &intervals[lower_bound..] {
            let s = iv.start();
            let e = iv.end();
            let candidate_start = if s > start_time { s } else { start_time };
            if candidate_start >= e {
                continue;
            }
            let remaining = e - candidate_start;
            if duration <= remaining {
                return Some(candidate_start);
            }
        }

        None
    }

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
            let candidate_start = if interval.start() > start_time {
                interval.start()
            } else {
                start_time
            };
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
    use bollard_model::index::BerthIndex;

    type IntegerType = i64;

    #[inline]
    fn iv(start: IntegerType, end: IntegerType) -> ClosedOpenInterval<IntegerType> {
        ClosedOpenInterval::new(start, end)
    }
    #[inline]
    fn b(i: usize) -> BerthIndex {
        BerthIndex::new(i)
    }

    #[test]
    fn test_empty_availability_returns_none() {
        let mut ba = BerthAvailability::<IntegerType>::empty(1);
        ba.available_times[0] = vec![];
        assert_eq!(ba.earliest_availability(b(0), 0, 1), None);
        assert_eq!(
            unsafe { ba.earliest_availability_unchecked(b(0), 0, 1) },
            None
        );
    }

    #[test]
    fn test_snap_to_interval_start_when_before() {
        let mut ba = BerthAvailability::<IntegerType>::empty(1);
        ba.available_times[0] = vec![iv(10, 20)];
        // Start before the interval, duration fits fully in [10,20)
        assert_eq!(ba.earliest_availability(b(0), 0, 5), Some(10));
        assert_eq!(
            unsafe { ba.earliest_availability_unchecked(b(0), 0, 5) },
            Some(10)
        );
    }

    #[test]
    fn test_start_inside_interval_stays_at_start_time() {
        let mut ba = BerthAvailability::<IntegerType>::empty(1);
        ba.available_times[0] = vec![iv(10, 20)];
        // Start inside interval, duration fits
        assert_eq!(ba.earliest_availability(b(0), 12, 3), Some(12));
        assert_eq!(
            unsafe { ba.earliest_availability_unchecked(b(0), 12, 3) },
            Some(12)
        );
    }

    #[test]
    fn test_exact_fit_at_end_is_allowed() {
        let mut ba = BerthAvailability::<IntegerType>::empty(1);
        ba.available_times[0] = vec![iv(10, 20)];
        // Start inside at 15 with duration 5 -> finish == 20 (end-exclusive boundary)
        assert_eq!(ba.earliest_availability(b(0), 15, 5), Some(15));
        assert_eq!(
            unsafe { ba.earliest_availability_unchecked(b(0), 15, 5) },
            Some(15)
        );
    }

    #[test]
    fn test_crossing_interval_end_is_rejected_and_uses_next_interval() {
        let mut ba = BerthAvailability::<IntegerType>::empty(1);
        ba.available_times[0] = vec![iv(10, 20), iv(25, 40)];
        // Start inside first interval but duration crosses end -> should snap to next interval start
        assert_eq!(ba.earliest_availability(b(0), 18, 5), Some(25));
        assert_eq!(
            unsafe { ba.earliest_availability_unchecked(b(0), 18, 5) },
            Some(25)
        );
    }

    #[test]
    fn test_start_at_interval_end_is_not_inside_uses_next() {
        let mut ba = BerthAvailability::<IntegerType>::empty(1);
        ba.available_times[0] = vec![iv(10, 20), iv(30, 50)];
        // start_time == end (20) is outside closed-open interval; next is 30
        assert_eq!(ba.earliest_availability(b(0), 20, 1), Some(30));
        assert_eq!(
            unsafe { ba.earliest_availability_unchecked(b(0), 20, 1) },
            Some(30)
        );
    }

    #[test]
    fn test_no_future_interval_large_start_time_returns_none() {
        let mut ba = BerthAvailability::<IntegerType>::empty(1);
        ba.available_times[0] = vec![iv(10, 20)];
        assert_eq!(ba.earliest_availability(b(0), 100, 1), None);
        assert_eq!(
            unsafe { ba.earliest_availability_unchecked(b(0), 100, 1) },
            None
        );
    }

    #[test]
    fn test_multiple_berths_independent_schedules() {
        let mut ba = BerthAvailability::<IntegerType>::empty(2);
        ba.available_times[0] = vec![iv(0, 5), iv(10, 20)];
        ba.available_times[1] = vec![iv(3, 8)];

        // Berth 0
        assert_eq!(ba.earliest_availability(b(0), 1, 3), Some(1)); // fits in [0,5)
        assert_eq!(ba.earliest_availability(b(0), 6, 4), Some(10)); // snaps to [10,20)

        // Berth 1
        assert_eq!(ba.earliest_availability(b(1), 0, 2), Some(3)); // snaps to 3
        assert_eq!(ba.earliest_availability(b(1), 7, 2), None); // cannot fit in [3,8)
    }

    #[test]
    fn test_lower_bound_edges() {
        let mut ba = BerthAvailability::<IntegerType>::empty(1);
        ba.available_times[0] = vec![iv(5, 10), iv(15, 20), iv(25, 30)];

        // Exactly on an interval start
        assert_eq!(ba.earliest_availability(b(0), 15, 1), Some(15));

        // Between intervals: 12 should pick 15
        assert_eq!(ba.earliest_availability(b(0), 12, 2), Some(15));

        // Inside last interval, long duration that doesn't fit -> None
        assert_eq!(ba.earliest_availability(b(0), 26, 10), None);
    }

    #[test]
    fn test_checked_and_unchecked_match_over_various_cases() {
        let mut ba = BerthAvailability::<IntegerType>::empty(1);
        ba.available_times[0] = vec![iv(0, 5), iv(10, 12), iv(20, 25), iv(30, 35), iv(40, 50)];

        let queries: &[(IntegerType, IntegerType)] = &[
            (0, 1),
            (1, 4),
            (4, 1),
            (5, 1),  // boundary: end of [0,5)
            (6, 2),  // between [5,10) -> 10
            (10, 2), // exactly at 10
            (11, 2), // crosses 12 -> next at 20
            (21, 3),
            (34, 2),
            (49, 1), // end-fitting at 50-1
            (49, 2), // cannot fit
            (100, 1),
        ];

        for &(start, dur) in queries {
            let a = ba.earliest_availability(b(0), start, dur);
            let u = unsafe { ba.earliest_availability_unchecked(b(0), start, dur) };
            assert_eq!(a, u, "mismatch for start={}, duration={}", start, dur);
        }
    }
}
