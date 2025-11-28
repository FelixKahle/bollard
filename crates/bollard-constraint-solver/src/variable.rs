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

use num_traits::PrimInt;
use smallvec::SmallVec;
use std::{
    iter::FusedIterator,
    ops::{Bound, RangeBounds},
};

/// A closed interval `[lower, upper]` over a primitive integer type `T`.
///
/// The interval includes both `lower` and `upper` bounds.
/// It maintains the invariant that `lower <= upper`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClosedInterval<T> {
    lower: T,
    upper: T,
}

impl<T> ClosedInterval<T> {
    /// Creates a new `ClosedInterval` without checking the `lower <= upper` invariant.
    ///
    /// # Safety
    /// This function is safe to call, but creating an interval where `lower > upper`
    /// violates the logical invariants of this struct and may lead to incorrect results
    /// in subsequent operations.
    ///
    /// In debug builds, this will trigger a `debug_assert!`.
    #[inline]
    pub fn new_unchecked(lower: T, upper: T) -> Self
    where
        T: Ord,
    {
        debug_assert!(lower <= upper, "lower bound must be <= upper bound");

        ClosedInterval { lower, upper }
    }

    /// Creates a new `ClosedInterval`.
    ///
    /// # Panics
    /// Panics if `lower > upper`.
    #[inline]
    pub fn new(lower: T, upper: T) -> Self
    where
        T: Ord,
    {
        assert!(lower <= upper, "lower bound must be <= upper bound");

        ClosedInterval { lower, upper }
    }

    /// Tries to create a new `ClosedInterval`.
    ///
    /// Returns `None` if `lower > upper`.
    #[inline]
    pub fn try_new(lower: T, upper: T) -> Option<Self>
    where
        T: Ord,
    {
        if lower <= upper {
            Some(ClosedInterval { lower, upper })
        } else {
            None
        }
    }

    /// Creates a singleton interval `[val, val]`.
    #[inline]
    pub fn singleton(val: T) -> Self
    where
        T: Ord + Copy,
    {
        Self::new_unchecked(val, val)
    }

    /// Returns the inclusive lower bound of the interval.
    #[inline]
    pub fn lower(&self) -> T
    where
        T: Copy,
    {
        self.lower
    }

    /// Returns the inclusive upper bound of the interval.
    #[inline]
    pub fn upper(&self) -> T
    where
        T: Copy,
    {
        self.upper
    }

    /// Returns the number of elements in the interval.
    ///
    /// # Saturating Arithmetic
    /// This method uses saturating arithmetic to handle overflows.
    /// If the length of the interval exceeds the maximum value representable by `T`
    /// (e.g., `[-128, 127]` for `i8`, which has length 256), this returns `T::max_value()`.
    ///
    /// This behavior ensures the method never panics and is consistent with constraint
    /// solving philosophies (e.g., OR-Tools) where "very large" is treated effectively as infinite.
    #[inline(always)]
    pub fn len(&self) -> T
    where
        T: PrimInt,
    {
        self.upper
            .saturating_sub(self.lower)
            .saturating_add(T::one())
    }

    /// Returns `true` if the interval contains the given `value`.
    #[inline]
    pub fn contains(&self, value: T) -> bool
    where
        T: Ord,
    {
        self.lower <= value && value <= self.upper
    }

    /// Returns `true` if this interval fully contains `other`.
    #[inline]
    pub fn contains_interval(&self, other: ClosedInterval<T>) -> bool
    where
        T: Ord,
    {
        self.lower <= other.lower && other.upper <= self.upper
    }

    /// Returns `true` if this interval intersects with `other`.
    ///
    /// Intersection is defined as sharing at least one common value.
    #[inline]
    pub fn intersects(&self, other: ClosedInterval<T>) -> bool
    where
        T: Copy + Ord,
    {
        std::cmp::max(self.lower, other.lower) <= std::cmp::min(self.upper, other.upper)
    }

    /// Returns `true` if the intervals are strictly adjacent.
    ///
    /// Two intervals are adjacent if there is no gap between them, but they do not overlap.
    /// For example, `[1, 5]` and `[6, 10]` are adjacent.
    #[inline(always)]
    pub fn adjacent(&self, other: ClosedInterval<T>) -> bool
    where
        T: PrimInt,
    {
        let self_touches_other = self.upper.checked_add(&T::one()) == Some(other.lower);
        let other_touches_self = other.upper.checked_add(&T::one()) == Some(self.lower);
        self_touches_other || other_touches_self
    }

    /// Returns `true` if the intervals intersect or are adjacent.
    ///
    /// This is an optimized check equivalent to `intersects(other) || adjacent(other)`.
    /// If true, the union of the two intervals forms a single contiguous interval.
    #[inline(always)]
    pub fn intersects_or_adjacent(&self, other: ClosedInterval<T>) -> bool
    where
        T: PrimInt,
    {
        let max_start = std::cmp::max(self.lower, other.lower);
        let min_end = std::cmp::min(self.upper, other.upper);
        max_start <= min_end.saturating_add(T::one())
    }

    /// Merges two intervals if they intersect or are adjacent.
    ///
    /// Returns `Some(merged_interval)` if the intervals can be combined into a single
    /// contiguous block. Returns `None` if they are disjoint and separated by a gap.
    #[inline]
    pub fn merge(&self, other: ClosedInterval<T>) -> Option<ClosedInterval<T>>
    where
        T: PrimInt,
    {
        if self.intersects_or_adjacent(other) {
            Some(ClosedInterval {
                lower: std::cmp::min(self.lower, other.lower),
                upper: std::cmp::max(self.upper, other.upper),
            })
        } else {
            None
        }
    }

    /// Calculates the set difference `self - other`.
    ///
    /// Returns a `SmallVec` containing 0, 1, or 2 intervals.
    /// * **0**: If `other` fully covers `self`.
    /// * **1**: If `other` is disjoint or overlaps only one side.
    /// * **2**: If `other` is strictly contained within `self` (punching a hole).
    #[inline]
    pub fn subtract(&self, other: ClosedInterval<T>) -> SmallVec<[ClosedInterval<T>; 2]>
    where
        T: PrimInt,
    {
        if !self.intersects(other) {
            return smallvec::smallvec![*self];
        }

        let mut result = SmallVec::new();
        let one = T::one();

        if self.lower < other.lower {
            result.push(ClosedInterval::new_unchecked(self.lower, other.lower - one));
        }

        if self.upper > other.upper {
            result.push(ClosedInterval::new_unchecked(other.upper + one, self.upper));
        }

        result
    }

    /// Returns the set complement of this interval.
    ///
    /// The result represents the set of all values in `T` that are *not* in this interval.
    /// Returns a `SmallVec` with 0, 1, or 2 intervals.
    #[inline]
    pub fn complement(&self) -> SmallVec<[ClosedInterval<T>; 2]>
    where
        T: PrimInt,
    {
        let mut result = SmallVec::new();
        let min_val = T::min_value();
        let max_val = T::max_value();
        let one = T::one();

        if self.lower > min_val {
            result.push(ClosedInterval::new_unchecked(min_val, self.lower - one));
        }

        if self.upper < max_val {
            result.push(ClosedInterval::new_unchecked(self.upper + one, max_val));
        }

        result
    }

    /// Creates an iterator over all integer values contained in the interval.
    ///
    /// # Note
    /// The iterator handles full-range intervals (e.g., `[0, 255u8]`) correctly without infinite loops.
    #[inline]
    pub fn iter(&self) -> ClosedIntervalIterator<T>
    where
        T: PrimInt,
    {
        ClosedIntervalIterator {
            current: self.lower,
            upper: self.upper,
            finished: false,
        }
    }
}

impl<T> PartialOrd for ClosedInterval<T>
where
    T: PrimInt,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for ClosedInterval<T>
where
    T: PrimInt,
{
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.lower.cmp(&other.lower) {
            std::cmp::Ordering::Equal => self.upper.cmp(&other.upper),
            ord => ord,
        }
    }
}

impl<T> RangeBounds<T> for ClosedInterval<T>
where
    T: PrimInt,
{
    fn start_bound(&self) -> Bound<&T> {
        Bound::Included(&self.lower)
    }

    fn end_bound(&self) -> Bound<&T> {
        Bound::Included(&self.upper)
    }
}

/// An iterator over the values of a `ClosedInterval`.
#[derive(Copy, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClosedIntervalIterator<T>
where
    T: PrimInt,
{
    current: T,
    upper: T,
    finished: bool,
}

impl<T> Iterator for ClosedIntervalIterator<T>
where
    T: PrimInt,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let result = self.current;

        if self.current >= self.upper {
            self.finished = true;
        } else {
            self.current = self.current + T::one();
        }

        Some(result)
    }
}

impl<T> FusedIterator for ClosedIntervalIterator<T> where T: PrimInt {}

impl<T> std::fmt::Debug for ClosedInterval<T>
where
    T: PrimInt + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}, {:?}]", self.lower, self.upper)
    }
}

impl<T> std::fmt::Display for ClosedInterval<T>
where
    T: PrimInt + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {}]", self.lower, self.upper)
    }
}

impl<T> IntoIterator for ClosedInterval<T>
where
    T: PrimInt,
{
    type Item = T;
    type IntoIter = ClosedIntervalIterator<T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> From<std::ops::RangeInclusive<T>> for ClosedInterval<T>
where
    T: PrimInt,
{
    #[inline]
    fn from(range: std::ops::RangeInclusive<T>) -> Self {
        ClosedInterval::new(*range.start(), *range.end())
    }
}

#[cfg(test)]
mod tests {
    macro_rules! generate_interval_tests {
        ($mod_name:ident, $t:ty) => {
            mod $mod_name {
                use super::super::*;

                // Helpers for succinct tests
                fn i(l: $t, u: $t) -> ClosedInterval<$t> {
                    ClosedInterval::new(l, u)
                }

                #[test]
                fn test_construction_and_accessors() {
                    let interval = ClosedInterval::new(1, 10);
                    assert_eq!(interval.lower(), 1);
                    assert_eq!(interval.upper(), 10);

                    // try_new valid
                    assert!(ClosedInterval::try_new(5, 5).is_some());
                    // try_new invalid
                    assert!(ClosedInterval::try_new(10, 5).is_none());
                }

                #[test]
                #[should_panic]
                fn test_new_panic_on_invalid() {
                    let _ = ClosedInterval::new(10, 9);
                }

                #[test]
                fn test_len_standard() {
                    // Standard length
                    assert_eq!(i(1, 5).len(), 5); // 1,2,3,4,5
                    assert_eq!(i(1, 1).len(), 1);
                }

                #[test]
                fn test_len_saturation() {
                    // This is the CRITICAL OR-Tools behavior check.
                    // If the interval is the full width of the type, len() should return T::MAX.

                    let min = <$t>::min_value();
                    let max = <$t>::max_value();
                    let full = i(min, max);

                    // The "True" math length is max - min + 1.
                    // But that doesn't fit in T. It should saturate to T::MAX.
                    assert_eq!(full.len(), max);

                    // Test just below max
                    let almost_full = i(min + 1, max);
                    // This length might fit or might saturate depending on signed/unsigned.
                    // If T is u8: [1, 255] len is 255. Fits!
                    // If T is i8: [-127, 127] len is 255. Cannot fit in i8 (127). Saturates.

                    if <$t>::min_value() == 0 {
                        // Unsigned: min+1 to max has length max. Fits exactly.
                        assert_eq!(almost_full.len(), max);
                    } else {
                        // Signed: Length is still > T::MAX. Should saturate.
                        assert_eq!(almost_full.len(), max);
                    }
                }

                #[test]
                fn test_contains() {
                    let x = i(10, 20);
                    assert!(x.contains(10));
                    assert!(x.contains(20));
                    assert!(x.contains(15));
                    assert!(!x.contains(9));
                    assert!(!x.contains(21));

                    // Interval containment
                    assert!(x.contains_interval(i(12, 18)));
                    assert!(x.contains_interval(i(10, 20)));
                    assert!(!x.contains_interval(i(9, 20)));
                    assert!(!x.contains_interval(i(10, 21)));
                }

                #[test]
                fn test_intersects() {
                    let base = i(10, 20);

                    // Inside
                    assert!(base.intersects(i(12, 18)));
                    // Overlap Left
                    assert!(base.intersects(i(5, 10)));
                    // Overlap Right
                    assert!(base.intersects(i(20, 25)));
                    // Envelops
                    assert!(base.intersects(i(0, 30)));

                    // No intersection
                    assert!(!base.intersects(i(0, 9)));
                    assert!(!base.intersects(i(21, 30)));
                }

                #[test]
                fn test_adjacency() {
                    let base = i(10, 20);
                    // Touching Left
                    assert!(base.adjacent(i(5, 9)));
                    // Touching Right
                    assert!(base.adjacent(i(21, 30)));

                    // Intersecting is NOT adjacent
                    assert!(!base.adjacent(i(20, 25)));
                    // Gap is NOT adjacent
                    assert!(!base.adjacent(i(22, 30)));

                    // Edge case: MAX
                    let max_interval = i(<$t>::max_value(), <$t>::max_value());
                    let almost_max = i(<$t>::max_value() - 1, <$t>::max_value() - 1);
                    assert!(max_interval.adjacent(almost_max));
                    assert!(almost_max.adjacent(max_interval));

                    // Overflow check: MAX adjacent to nothing above it
                    // The code handles this via checked_add -> map_or(false).
                    // We can't construct an interval above MAX, so this is implicitly tested by the logic not crashing.
                }

                #[test]
                fn test_merge_logic() {
                    let base = i(10, 20);

                    // Intersecting -> Merge
                    let m1 = base.merge(i(15, 25));
                    assert_eq!(m1, Some(i(10, 25)));

                    // Adjacent -> Merge
                    let m2 = base.merge(i(21, 30));
                    assert_eq!(m2, Some(i(10, 30)));

                    // Adjacent Left -> Merge
                    let m3 = base.merge(i(0, 9));
                    assert_eq!(m3, Some(i(0, 20)));

                    // Disjoint -> None
                    let m4 = base.merge(i(22, 30));
                    assert_eq!(m4, None);
                }

                #[test]
                fn test_subtract() {
                    let base = i(10, 20);

                    // 1. Disjoint (No change)
                    let r1 = base.subtract(i(25, 30));
                    assert_eq!(r1.len(), 1);
                    assert_eq!(r1[0], base);

                    // 2. Complete Cover (Empty)
                    let r2 = base.subtract(i(0, 30));
                    assert!(r2.is_empty());

                    // 3. Punch Hole (Split)
                    let r3 = base.subtract(i(12, 18));
                    assert_eq!(r3.len(), 2);
                    assert_eq!(r3[0], i(10, 11));
                    assert_eq!(r3[1], i(19, 20));

                    // 4. Trim Right
                    let r4 = base.subtract(i(15, 30));
                    assert_eq!(r4.len(), 1);
                    assert_eq!(r4[0], i(10, 14));

                    // 5. Trim Left
                    let r5 = base.subtract(i(0, 15));
                    assert_eq!(r5.len(), 1);
                    assert_eq!(r5[0], i(16, 20));

                    // 6. Identity (Exact Match)
                    let r6 = base.subtract(base);
                    assert!(r6.is_empty());
                }

                #[test]
                fn test_subtract_edge_cases() {
                    let min = <$t>::min_value();
                    let max = <$t>::max_value();

                    // Punch hole at min
                    let full = i(min, max);
                    let sub_min = full.subtract(i(min, min));
                    // Should be [min+1, max]
                    assert_eq!(sub_min.len(), 1);
                    assert_eq!(sub_min[0], i(min + 1, max));

                    // Punch hole at max
                    let sub_max = full.subtract(i(max, max));
                    // Should be [min, max-1]
                    assert_eq!(sub_max.len(), 1);
                    assert_eq!(sub_max[0], i(min, max - 1));
                }

                #[test]
                fn test_complement() {
                    let min = <$t>::min_value();
                    let max = <$t>::max_value();

                    // Complement of something in the middle
                    let mid = i(10, 20);
                    let comp = mid.complement();
                    assert_eq!(comp.len(), 2);
                    // Check bounds carefully regarding min/max
                    if min < 10 {
                        assert_eq!(comp[0], i(min, 9));
                    }
                    if max > 20 {
                        assert_eq!(comp[1], i(21, max));
                    }

                    // Complement of Full Universe is Empty
                    let full = i(min, max);
                    assert!(full.complement().is_empty());
                }

                #[test]
                fn test_iterator_basic() {
                    let interval = i(1, 3);
                    let vec: Vec<$t> = interval.iter().collect();
                    assert_eq!(vec, vec![1, 2, 3]);
                }

                #[test]
                fn test_iterator_single() {
                    let interval = i(10, 10);
                    let vec: Vec<$t> = interval.iter().collect();
                    assert_eq!(vec, vec![10]);
                }

                #[test]
                fn test_iterator_boundary_max() {
                    // This tests the Infinite Loop bug fix.
                    let max = <$t>::max_value();
                    let interval = i(max - 1, max);
                    let vec: Vec<$t> = interval.iter().collect();
                    assert_eq!(vec, vec![max - 1, max]);
                }
            }
        };
    }

    generate_interval_tests!(tests_u8, u8);
    generate_interval_tests!(tests_i8, i8);
    generate_interval_tests!(tests_u16, u16);
    generate_interval_tests!(tests_i16, i16);
    generate_interval_tests!(tests_u32, u32);
    generate_interval_tests!(tests_i32, i32);
    generate_interval_tests!(tests_u64, u64);
    generate_interval_tests!(tests_i64, i64);
    generate_interval_tests!(tests_u128, u128);
    generate_interval_tests!(tests_i128, i128);
    generate_interval_tests!(tests_usize, usize);
    generate_interval_tests!(tests_isize, isize);
}
