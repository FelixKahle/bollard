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

use bollard_num::ops::{checked_arithmetic, saturating_arithmetic};
use num_traits::{Bounded, One, PrimInt};
use smallvec::SmallVec;
use std::{
    iter::FusedIterator,
    ops::{Add, Bound, RangeBounds, Sub},
};

/// A closed interval `[lower, upper]` over a primitive integer type.
///
/// The interval represents a contiguous set of integers including both the `lower`
/// and `upper` bounds. It maintains the invariant that `lower <= upper`.
///
/// # Examples
///
/// ```
/// use bollard_constraint_solver::variable::ClosedInterval;
///
/// let interval = ClosedInterval::new(1, 10);
/// assert!(interval.contains(1));
/// assert!(interval.contains(10));
/// assert!(!interval.contains(11));
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClosedInterval<T> {
    lower: T,
    upper: T,
}

impl<T> ClosedInterval<T> {
    /// Creates a new `ClosedInterval` without checking the `lower <= upper` invariant.
    ///
    /// # Safety
    ///
    /// While this function is technically safe (it will not cause undefined behavior in the
    /// memory safety sense), creating an interval where `lower > upper` violates the
    /// logical invariants of this type. Subsequent operations like `len()`, `contains()`,
    /// or `iter()` may produce incorrect results or wrap unexpectedly.
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
    ///
    /// Panics if `lower > upper`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::ClosedInterval;
    ///
    /// let interval = ClosedInterval::new(5, 10); // OK
    /// ```
    ///
    /// The following causes a panic:
    ///
    /// ```should_panic
    /// use bollard_constraint_solver::variable::ClosedInterval;
    ///
    /// let interval = ClosedInterval::new(10, 5); // Panics!
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::ClosedInterval;
    ///
    /// assert!(ClosedInterval::try_new(1, 5).is_some());
    /// assert!(ClosedInterval::try_new(5, 1).is_none());
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::ClosedInterval;
    ///
    /// let single = ClosedInterval::singleton(5);
    /// assert_eq!(single.len(), 1);
    /// assert!(single.contains(5));
    /// ```
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
    /// # Saturating Behavior
    ///
    /// This method uses saturating arithmetic. If the mathematical length of the
    /// interval exceeds the maximum value representable by `T` (e.g., `[-128, 127]`
    /// for `i8` has length 256, which fits in `u8` but not `i8`), this returns
    /// `T::max_value()`.
    ///
    /// This ensures the method never panics and treats "oversized" intervals as
    /// effectively infinite for the purposes of constraint solving.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::ClosedInterval;
    ///
    /// let interval = ClosedInterval::new(1u8, 5u8);
    /// assert_eq!(interval.len(), 5);
    ///
    /// // Saturation example
    /// let full = ClosedInterval::new(0u8, 255u8);
    /// assert_eq!(full.len(), 255); // 256 saturates to 255
    /// ```
    #[inline(always)]
    pub fn len(&self) -> T
    where
        T: Copy + saturating_arithmetic::SaturatingAdd + saturating_arithmetic::SaturatingSub + One,
    {
        self.upper
            .saturating_sub(self.lower)
            .saturating_add(T::one())
    }

    /// Returns `true` if the interval contains the given `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::ClosedInterval;
    ///
    /// let i = ClosedInterval::new(10, 20);
    /// assert!(i.contains(10));
    /// assert!(i.contains(15));
    /// assert!(!i.contains(30));
    /// ```
    #[inline]
    pub fn contains(&self, value: T) -> bool
    where
        T: Ord,
    {
        self.lower <= value && value <= self.upper
    }

    /// Returns `true` if this interval fully contains `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::ClosedInterval;
    ///
    /// let outer = ClosedInterval::new(0, 100);
    /// let inner = ClosedInterval::new(10, 20);
    ///
    /// assert!(outer.contains_interval(inner));
    /// assert!(!inner.contains_interval(outer));
    /// ```
    #[inline]
    pub fn contains_interval(&self, other: ClosedInterval<T>) -> bool
    where
        T: Ord,
    {
        self.lower <= other.lower && other.upper <= self.upper
    }

    /// Returns `true` if this interval intersects with `other`.
    ///
    /// Intersection implies the two intervals share at least one common value.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::ClosedInterval;
    ///
    /// let a = ClosedInterval::new(1, 5);
    /// let b = ClosedInterval::new(5, 10);
    /// let c = ClosedInterval::new(6, 10);
    ///
    /// assert!(a.intersects(b)); // Share '5'
    /// assert!(!a.intersects(c)); // Disjoint
    /// ```
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
    ///
    ///  and [b+1, c]]
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::ClosedInterval;
    ///
    /// let a = ClosedInterval::new(1, 5);
    /// let b = ClosedInterval::new(6, 10);
    ///
    /// assert!(a.adjacent(b));
    /// assert!(b.adjacent(a));
    /// ```
    #[inline(always)]
    pub fn adjacent(&self, other: ClosedInterval<T>) -> bool
    where
        T: Copy + PartialEq + checked_arithmetic::CheckedAdd + One,
    {
        let self_touches_other = self.upper.checked_add(T::one()) == Some(other.lower);
        let other_touches_self = other.upper.checked_add(T::one()) == Some(self.lower);
        self_touches_other || other_touches_self
    }

    /// Returns `true` if the intervals intersect or are adjacent.
    ///
    /// This is an optimized check equivalent to `intersects(other) || adjacent(other)`.
    /// If true, the union of the two intervals forms a single contiguous interval.
    #[inline(always)]
    pub fn intersects_or_adjacent(&self, other: ClosedInterval<T>) -> bool
    where
        T: Copy + saturating_arithmetic::SaturatingAdd + Ord + One,
    {
        let max_start = std::cmp::max(self.lower, other.lower);
        let min_end = std::cmp::min(self.upper, other.upper);
        max_start <= min_end.saturating_add(T::one())
    }

    /// Merges two intervals if they intersect or are adjacent.
    ///
    /// Returns `Some(merged)` if the intervals can be combined into a single
    /// contiguous block. Returns `None` if they are disjoint and separated by a gap.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::ClosedInterval;
    ///
    /// let a = ClosedInterval::new(1, 5);
    /// let b = ClosedInterval::new(6, 10);
    /// let disjoint = ClosedInterval::new(12, 15);
    ///
    /// assert_eq!(a.merge(b), Some(ClosedInterval::new(1, 10)));
    /// assert_eq!(a.merge(disjoint), None);
    /// ```
    #[inline]
    pub fn merge(&self, other: ClosedInterval<T>) -> Option<ClosedInterval<T>>
    where
        T: Copy + saturating_arithmetic::SaturatingAdd + Ord + One,
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
    ///
    /// * **0 elements**: If `other` fully covers `self`.
    /// * **1 element**: If `other` is disjoint or overlaps only one side.
    /// * **2 elements**: If `other` is strictly contained within `self`, punching a hole
    ///   in the middle and splitting `self` into two parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::ClosedInterval;
    ///
    /// let base = ClosedInterval::new(0, 10);
    /// let hole = ClosedInterval::new(4, 6);
    ///
    /// let result = base.subtract(hole);
    /// assert_eq!(result.len(), 2);
    /// assert_eq!(result[0], ClosedInterval::new(0, 3));
    /// assert_eq!(result[1], ClosedInterval::new(7, 10));
    /// ```
    #[inline]
    pub fn subtract(&self, other: ClosedInterval<T>) -> SmallVec<ClosedInterval<T>, 2>
    where
        T: Copy + Ord + One + Sub<Output = T> + Add<Output = T>,
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

    /// Returns the set complement of this interval within the domain of `T`.
    ///
    /// The result represents the set of all values in `T` that are *not* in this interval.
    /// Returns a `SmallVec` with 0, 1, or 2 intervals.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::ClosedInterval;
    ///
    /// // Assuming T is i8 (range -128 to 127)
    /// let i = ClosedInterval::new(0i8, 10i8);
    /// let comp = i.complement();
    ///
    /// assert_eq!(comp.len(), 2);
    /// assert_eq!(comp[0], ClosedInterval::new(-128, -1));
    /// assert_eq!(comp[1], ClosedInterval::new(11, 127));
    /// ```
    #[inline]
    pub fn complement(&self) -> SmallVec<ClosedInterval<T>, 2>
    where
        T: Copy + Ord + Bounded + One + Sub<Output = T> + Add<Output = T>,
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
    /// The iterator yields values from `lower` to `upper` (inclusive).
    ///
    /// # Edge Cases
    ///
    /// The iterator handles full-range intervals (e.g., `[0, 255u8]`) correctly
    /// without entering an infinite loop when the counter overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::ClosedInterval;
    ///
    /// let i = ClosedInterval::new(1, 3);
    /// let collected: Vec<_> = i.iter().collect();
    /// assert_eq!(collected, vec![1, 2, 3]);
    /// ```
    #[inline]
    pub fn iter(&self) -> ClosedIntervalIterator<T>
    where
        T: Copy,
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
    T: Copy + PartialEq + Ord,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for ClosedInterval<T>
where
    T: Copy + Ord,
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

/// An iterator over the values of a `ClosedInterval`
///
/// This `struct` is created by the `iter` method on `ClosedInterval`.
/// It yields integer values of type `T` in ascending order, including both the
/// lower and upper bounds.
///
/// # Logic and Safety
///
/// This iterator is designed to correctly handle intervals that end at the maximum
/// value of the integer type (e.g., `u8::MAX`). Standard iterators or naive `for` loops
/// often cause infinite loops or panics due to overflow when incrementing past the
/// maximum value. This implementation uses an internal flag to track completion safely.
#[derive(Copy, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClosedIntervalIterator<T> {
    current: T,
    upper: T,
    finished: bool,
}

impl<T> Iterator for ClosedIntervalIterator<T>
where
    T: Copy + Ord + One + Add<Output = T>,
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

impl<T> DoubleEndedIterator for ClosedIntervalIterator<T>
where
    T: Copy + Ord + One + Sub<Output = T> + Add<Output = T>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let result = self.upper;

        if self.current >= self.upper {
            self.finished = true;
        } else {
            self.upper = self.upper - T::one();
        }

        Some(result)
    }
}

impl<T> FusedIterator for ClosedIntervalIterator<T> where T: PrimInt {}

impl<T> std::fmt::Debug for ClosedInterval<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}, {:?}]", self.lower, self.upper)
    }
}

impl<T> std::fmt::Display for ClosedInterval<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {}]", self.lower, self.upper)
    }
}

impl<T> IntoIterator for ClosedInterval<T>
where
    T: Copy + Ord + One + Add<Output = T>,
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
    T: Copy + Ord,
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

                fn i(l: $t, u: $t) -> ClosedInterval<$t> {
                    ClosedInterval::new(l, u)
                }

                #[test]
                fn test_construction_and_accessors() {
                    let interval = ClosedInterval::new(1, 10);
                    assert_eq!(interval.lower(), 1);
                    assert_eq!(interval.upper(), 10);
                    assert!(ClosedInterval::try_new(5, 5).is_some());
                    assert!(ClosedInterval::try_new(10, 5).is_none());
                }

                #[test]
                #[should_panic]
                fn test_new_panic_on_invalid() {
                    let _ = ClosedInterval::new(10, 9);
                }

                #[test]
                fn test_len_standard() {
                    assert_eq!(i(1, 5).len(), 5);
                    assert_eq!(i(1, 1).len(), 1);
                }

                #[test]
                fn test_len_saturation() {
                    let min = <$t>::min_value();
                    let max = <$t>::max_value();
                    let full = i(min, max);

                    assert_eq!(full.len(), max);

                    let almost_full = i(min + 1, max);

                    if <$t>::min_value() == 0 {
                        assert_eq!(almost_full.len(), max);
                    } else {
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
                    assert!(x.contains_interval(i(12, 18)));
                    assert!(x.contains_interval(i(10, 20)));
                    assert!(!x.contains_interval(i(9, 20)));
                    assert!(!x.contains_interval(i(10, 21)));
                }

                #[test]
                fn test_intersects() {
                    let base = i(10, 20);

                    assert!(base.intersects(i(12, 18)));
                    assert!(base.intersects(i(5, 10)));
                    assert!(base.intersects(i(20, 25)));
                    assert!(base.intersects(i(0, 30)));
                    assert!(!base.intersects(i(0, 9)));
                    assert!(!base.intersects(i(21, 30)));
                }

                #[test]
                fn test_adjacency() {
                    let base = i(10, 20);

                    assert!(base.adjacent(i(5, 9)));
                    assert!(base.adjacent(i(21, 30)));
                    assert!(!base.adjacent(i(20, 25)));
                    assert!(!base.adjacent(i(22, 30)));

                    let max_interval = i(<$t>::max_value(), <$t>::max_value());
                    let almost_max = i(<$t>::max_value() - 1, <$t>::max_value() - 1);
                    assert!(max_interval.adjacent(almost_max));
                    assert!(almost_max.adjacent(max_interval));
                }

                #[test]
                fn test_merge_logic() {
                    let base = i(10, 20);

                    let m1 = base.merge(i(15, 25));
                    assert_eq!(m1, Some(i(10, 25)));

                    let m2 = base.merge(i(21, 30));
                    assert_eq!(m2, Some(i(10, 30)));

                    let m3 = base.merge(i(0, 9));
                    assert_eq!(m3, Some(i(0, 20)));

                    let m4 = base.merge(i(22, 30));
                    assert_eq!(m4, None);
                }

                #[test]
                fn test_subtract() {
                    let base = i(10, 20);

                    let r1 = base.subtract(i(25, 30));
                    assert_eq!(r1.len(), 1);
                    assert_eq!(r1[0], base);

                    let r2 = base.subtract(i(0, 30));
                    assert!(r2.is_empty());

                    let r3 = base.subtract(i(12, 18));
                    assert_eq!(r3.len(), 2);
                    assert_eq!(r3[0], i(10, 11));
                    assert_eq!(r3[1], i(19, 20));

                    let r4 = base.subtract(i(15, 30));
                    assert_eq!(r4.len(), 1);
                    assert_eq!(r4[0], i(10, 14));

                    let r5 = base.subtract(i(0, 15));
                    assert_eq!(r5.len(), 1);
                    assert_eq!(r5[0], i(16, 20));

                    let r6 = base.subtract(base);
                    assert!(r6.is_empty());
                }

                #[test]
                fn test_subtract_edge_cases() {
                    let min = <$t>::min_value();
                    let max = <$t>::max_value();

                    let full = i(min, max);
                    let sub_min = full.subtract(i(min, min));
                    assert_eq!(sub_min.len(), 1);
                    assert_eq!(sub_min[0], i(min + 1, max));

                    let sub_max = full.subtract(i(max, max));
                    assert_eq!(sub_max.len(), 1);
                    assert_eq!(sub_max[0], i(min, max - 1));
                }

                #[test]
                fn test_complement() {
                    let min = <$t>::min_value();
                    let max = <$t>::max_value();

                    let mid = i(10, 20);
                    let comp = mid.complement();
                    assert_eq!(comp.len(), 2);

                    if min < 10 {
                        assert_eq!(comp[0], i(min, 9));
                    }
                    if max > 20 {
                        assert_eq!(comp[1], i(21, max));
                    }

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
                    let max = <$t>::max_value();
                    let interval = i(max - 1, max);
                    let vec: Vec<$t> = interval.iter().collect();
                    assert_eq!(vec, vec![max - 1, max]);
                }

                #[test]
                fn test_from_range_inclusive() {
                    let range = 5 as $t..=15 as $t;
                    let interval: ClosedInterval<$t> = range.into();
                    assert_eq!(interval, i(5, 15));
                }

                #[test]
                fn test_double_ended() {
                    let interval = ClosedInterval::new(1, 5);
                    let mut iter = interval.iter();

                    assert_eq!(iter.next(), Some(1));
                    assert_eq!(iter.next_back(), Some(5));

                    assert_eq!(iter.next(), Some(2));
                    assert_eq!(iter.next_back(), Some(4));
                    assert_eq!(iter.next(), Some(3));

                    assert_eq!(iter.next(), None);
                    assert_eq!(iter.next_back(), None);
                }

                #[test]
                fn test_double_ended_exhaustion_forward_then_back() {
                    let interval = ClosedInterval::new(1, 3);
                    let mut iter = interval.iter();

                    assert_eq!(iter.next(), Some(1));
                    assert_eq!(iter.next(), Some(2));
                    assert_eq!(iter.next(), Some(3));

                    assert_eq!(iter.next(), None);
                    assert_eq!(iter.next_back(), None);
                }

                #[test]
                fn test_double_ended_exhaustion_back_then_forward() {
                    let interval = ClosedInterval::new(1, 3);
                    let mut iter = interval.iter();

                    assert_eq!(iter.next_back(), Some(3));
                    assert_eq!(iter.next_back(), Some(2));
                    assert_eq!(iter.next_back(), Some(1));

                    assert_eq!(iter.next(), None);
                    assert_eq!(iter.next_back(), None);
                }

                #[test]
                fn test_double_ended_singleton() {
                    let interval = ClosedInterval::new(7, 7);
                    let mut iter = interval.iter();

                    assert_eq!(iter.next(), Some(7));
                    assert_eq!(iter.next_back(), None);

                    let mut iter2 = interval.iter();
                    assert_eq!(iter2.next_back(), Some(7));
                    assert_eq!(iter2.next(), None);
                }

                #[test]
                fn test_double_ended_full_boundary_type_max() {
                    let max = <$t>::max_value();
                    let min = <$t>::min_value();
                    let interval = ClosedInterval::new(min, max);
                    let mut iter = interval.iter();

                    let first = iter.next();
                    let last = iter.next_back();
                    assert_eq!(first, Some(min));
                    assert_eq!(last, Some(max));

                    let f2 = iter.next();
                    let b2 = iter.next_back();
                    if min < max {
                        assert_eq!(f2, Some(min + 1));
                        assert_eq!(b2, Some(max - 1));
                    }
                }

                #[test]
                fn test_into_iter_collect() {
                    let interval = ClosedInterval::new(2, 5);
                    let v: Vec<$t> = interval.into_iter().collect();
                    assert_eq!(v, vec![2, 3, 4, 5]);
                }

                #[test]
                fn test_display_and_debug() {
                    let interval = ClosedInterval::new(4, 9);
                    assert_eq!(format!("{}", interval), "[4, 9]");
                    assert_eq!(format!("{:?}", interval), "[4, 9]");
                }

                #[test]
                fn test_iter_alternate_until_finish() {
                    let interval = ClosedInterval::new(10, 14);
                    let mut iter = interval.iter();

                    assert_eq!(iter.next(), Some(10));
                    assert_eq!(iter.next_back(), Some(14));
                    assert_eq!(iter.next(), Some(11));
                    assert_eq!(iter.next_back(), Some(13));
                    assert_eq!(iter.next(), Some(12));

                    assert_eq!(iter.next(), None);
                    assert_eq!(iter.next_back(), None);
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
