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
    cmp::{max, min},
    iter::FusedIterator,
    ops::{BitAnd, BitOr},
};

/// A half-open interval `[start, end)` defined by a start (inclusive) and end (exclusive).
///
/// This struct represents a contiguous set of integers. It supports standard
/// set-theoretic operations such as intersection, union, and difference, as well as
/// geometric queries like overlap and adjacency checks.
///
/// # Invariants
/// `start_inclusive` must always be less than or equal to `end_exclusive`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClosedOpenInterval<T>
where
    T: PrimInt,
{
    start_inclusive: T,
    end_exclusive: T,
}

/// An iterator over the integer points contained within a `ClosedOpenInterval`.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::math::interval::ClosedOpenInterval;
///
/// let iv = ClosedOpenInterval::new(1, 5);
/// let points: Vec<_> = iv.iter().collect();
/// assert_eq!(points, vec![1, 2, 3, 4]);
/// ```
pub struct ClosedOpenIntervalIterator<T>
where
    T: PrimInt,
{
    end_exclusive: T,
    current: T,
}

impl<T> Iterator for ClosedOpenIntervalIterator<T>
where
    T: PrimInt,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end_exclusive {
            let result = self.current;
            self.current = self.current + T::one();
            Some(result)
        } else {
            None
        }
    }
}

impl<T> DoubleEndedIterator for ClosedOpenIntervalIterator<T>
where
    T: PrimInt,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.current < self.end_exclusive {
            self.end_exclusive = self.end_exclusive - T::one();
            Some(self.end_exclusive)
        } else {
            None
        }
    }
}

impl<T> ExactSizeIterator for ClosedOpenIntervalIterator<T>
where
    T: PrimInt,
{
    fn len(&self) -> usize {
        let dist = self.end_exclusive - self.current;
        if dist <= T::zero() {
            return 0;
        }
        dist.to_usize()
            .expect("ClosedOpenIntervalIterator: remaining length exceeds usize::MAX")
    }
}

impl<T> FusedIterator for ClosedOpenIntervalIterator<T> where T: PrimInt {}

impl<T> ClosedOpenInterval<T>
where
    T: PrimInt,
{
    /// Creates a new `ClosedOpenInterval`.
    ///
    /// # Panics
    ///
    /// Panics if `start_inclusive > end_exclusive`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// let iv = ClosedOpenInterval::new(0, 10);
    /// assert_eq!(iv.len(), 10);
    /// ```
    #[inline]
    pub fn new(start_inclusive: T, end_exclusive: T) -> Self {
        assert!(
            start_inclusive <= end_exclusive,
            "Invalid interval: start_inclusive must be less than or equal to end_exclusive"
        );
        Self {
            start_inclusive,
            end_exclusive,
        }
    }

    /// Creates a new `ClosedOpenInterval` if the inputs are valid.
    ///
    /// Returns `None` if `start_inclusive > end_exclusive`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// assert!(ClosedOpenInterval::try_new(0, 10).is_some());
    /// assert!(ClosedOpenInterval::try_new(10, 0).is_none());
    /// ```
    #[inline]
    pub fn try_new(start_inclusive: T, end_exclusive: T) -> Option<Self> {
        if start_inclusive <= end_exclusive {
            Some(Self {
                start_inclusive,
                end_exclusive,
            })
        } else {
            None
        }
    }

    /// Creates a new `ClosedOpenInterval` without checking invariants in release builds.
    ///
    /// # Safety
    ///
    /// The caller must ensure `start_inclusive <= end_exclusive`.
    /// This function contains a `debug_assert!` to catch errors during development.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// let iv = ClosedOpenInterval::new_unchecked(0, 10);
    /// ```
    #[inline]
    pub fn new_unchecked(start_inclusive: T, end_exclusive: T) -> Self {
        debug_assert!(
            start_inclusive <= end_exclusive,
            "Invalid interval: start_inclusive must be less than or equal to end_exclusive"
        );
        Self {
            start_inclusive,
            end_exclusive,
        }
    }

    /// Returns the inclusive start bound of the interval.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// let iv = ClosedOpenInterval::new(5, 10);
    /// assert_eq!(iv.start(), 5);
    /// ```
    #[inline]
    pub const fn start(&self) -> T {
        self.start_inclusive
    }

    /// Returns the exclusive end bound of the interval.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// let iv = ClosedOpenInterval::new(5, 10);
    /// assert_eq!(iv.end(), 10);
    /// ```
    #[inline]
    pub const fn end(&self) -> T {
        self.end_exclusive
    }

    /// Returns `true` if this interval overlaps with `other`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// let a = ClosedOpenInterval::new(0, 10);
    /// let b = ClosedOpenInterval::new(5, 15);
    /// assert!(a.intersects(b));
    ///
    /// let c = ClosedOpenInterval::new(10, 20); // Adjacent
    /// assert!(!a.intersects(c));
    /// ```
    #[inline]
    pub fn intersects(&self, other: Self) -> bool {
        self.start_inclusive < other.end_exclusive && other.start_inclusive < self.end_exclusive
    }

    /// Returns `true` if the intervals share a boundary but do not overlap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// let a = ClosedOpenInterval::new(0, 10);
    /// let b = ClosedOpenInterval::new(10, 20);
    /// assert!(a.adjacent(b));
    /// ```
    #[inline]
    pub fn adjacent(&self, other: Self) -> bool {
        self.end_exclusive == other.start_inclusive || other.end_exclusive == self.start_inclusive
    }

    /// Returns `true` if the intervals are disjoint (neither intersecting nor adjacent).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// let a = ClosedOpenInterval::new(0, 10);
    /// assert!(a.disjoint(ClosedOpenInterval::new(15, 20))); // Disjoint
    /// assert!(!a.disjoint(ClosedOpenInterval::new(5, 15))); // Intersects
    /// assert!(!a.disjoint(ClosedOpenInterval::new(10, 15))); // Adjacent
    /// ```
    #[inline]
    pub fn disjoint(&self, other: Self) -> bool {
        !self.intersects_or_adjacent(other)
    }

    /// Returns `true` if the intervals either intersect or are adjacent.
    ///
    /// This is useful for determining if two intervals can be merged into a single contiguous interval.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// let a = ClosedOpenInterval::new(0, 10);
    /// assert!(a.intersects_or_adjacent(ClosedOpenInterval::new(10, 20))); // Adjacent
    /// assert!(a.intersects_or_adjacent(ClosedOpenInterval::new(5, 15)));  // Intersects
    /// assert!(!a.intersects_or_adjacent(ClosedOpenInterval::new(12, 20))); // Gap
    /// ```
    #[inline]
    pub fn intersects_or_adjacent(&self, other: Self) -> bool {
        self.start_inclusive <= other.end_exclusive && other.start_inclusive <= self.end_exclusive
    }

    /// Returns `true` if `value` is contained in the interval `[start, end)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// let iv = ClosedOpenInterval::new(0, 10);
    /// assert!(iv.contains_point(0));
    /// assert!(iv.contains_point(9));
    /// assert!(!iv.contains_point(10));
    /// ```
    #[inline]
    pub fn contains_point(&self, value: T) -> bool {
        self.start_inclusive <= value && value < self.end_exclusive
    }

    /// Returns `true` if `other` is strictly contained within `self`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// let a = ClosedOpenInterval::new(0, 10);
    /// let b = ClosedOpenInterval::new(2, 8);
    /// assert!(a.contains_interval(b));
    /// ```
    #[inline]
    pub fn contains_interval(&self, other: Self) -> bool {
        self.start_inclusive <= other.start_inclusive && other.end_exclusive <= self.end_exclusive
    }

    /// Returns the length of the interval (`end - start`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// assert_eq!(ClosedOpenInterval::new(10, 20).len(), 10);
    /// ```
    #[inline]
    pub fn len(&self) -> T {
        self.end_exclusive - self.start_inclusive
    }

    /// Returns `true` if the interval is empty (`start == end`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// assert!(ClosedOpenInterval::new(10, 10).is_empty());
    /// assert!(!ClosedOpenInterval::new(10, 11).is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start_inclusive == self.end_exclusive
    }

    /// Calculates the intersection of two intervals.
    ///
    /// Returns `None` if the intervals are disjoint or adjacent (resulting in an empty set,
    /// though empty intersection of overlapping boundaries is handled gracefully).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// let a = ClosedOpenInterval::new(0, 10);
    /// let b = ClosedOpenInterval::new(5, 15);
    /// assert_eq!(a.intersection(b), Some(ClosedOpenInterval::new(5, 10)));
    /// ```
    #[inline]
    pub fn intersection(&self, other: Self) -> Option<Self> {
        let new_start = max(self.start_inclusive, other.start_inclusive);
        let new_end = min(self.end_exclusive, other.end_exclusive);

        if new_start < new_end {
            Some(Self::new_unchecked(new_start, new_end))
        } else {
            None
        }
    }

    /// Calculates the union of two intervals.
    ///
    /// Returns `Some(union)` if the intervals overlap or are adjacent.
    /// Returns `None` if the intervals are disjoint (separated by a gap).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// let a = ClosedOpenInterval::new(0, 10);
    /// let b = ClosedOpenInterval::new(10, 20);
    /// assert_eq!(a.union(b), Some(ClosedOpenInterval::new(0, 20)));
    /// ```
    #[inline]
    pub fn union(&self, other: Self) -> Option<Self> {
        if self.intersects_or_adjacent(other) {
            Some(Self {
                start_inclusive: min(self.start_inclusive, other.start_inclusive),
                end_exclusive: max(self.end_exclusive, other.end_exclusive),
            })
        } else {
            None
        }
    }

    /// Calculates the set difference `self - other`.
    ///
    /// This removes the portion of `self` that overlaps with `other`.
    ///
    /// # Returns
    ///
    /// A `Vec` containing:
    /// * 0 intervals: If `other` fully covers `self`.
    /// * 1 interval: If `other` clips one side of `self` or is disjoint.
    /// * 2 intervals: If `other` is strictly contained within `self`, splitting it into two.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// let a = ClosedOpenInterval::new(0, 10);
    /// let split = ClosedOpenInterval::new(4, 6);
    ///
    /// let diff = a.difference(split);
    /// assert_eq!(diff.len(), 2);
    /// assert_eq!(diff[0], ClosedOpenInterval::new(0, 4));
    /// assert_eq!(diff[1], ClosedOpenInterval::new(6, 10));
    /// ```
    pub fn difference(&self, other: Self) -> SmallVec<Self, 2> {
        if !self.intersects(other) {
            return smallvec::smallvec![*self];
        }

        if other.is_empty() || !self.intersects(other) {
            return smallvec::smallvec![*self];
        }

        let mut result = SmallVec::new();
        if self.start_inclusive < other.start_inclusive {
            result.push(Self::new_unchecked(
                self.start_inclusive,
                other.start_inclusive,
            ));
        }
        if self.end_exclusive > other.end_exclusive {
            result.push(Self::new_unchecked(other.end_exclusive, self.end_exclusive));
        }
        result
    }

    /// Calculates the midpoint of the interval.
    ///
    /// The calculation is robust against integer overflow.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// let iv = ClosedOpenInterval::new(0, 10);
    /// assert_eq!(iv.midpoint(), 5);
    /// ```
    #[inline]
    pub fn midpoint(&self) -> T {
        let len = self.end_exclusive - self.start_inclusive;
        self.start_inclusive + (len >> 1)
    }

    /// Returns the gap between two strictly disjoint intervals.
    ///
    /// Returns `None` if the intervals intersect or are adjacent.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// let a = ClosedOpenInterval::new(0, 5);
    /// let b = ClosedOpenInterval::new(10, 15);
    /// assert_eq!(a.gap(b), Some(ClosedOpenInterval::new(5, 10)));
    /// ```
    #[inline]
    pub fn gap(&self, other: Self) -> Option<Self> {
        if self.end_exclusive < other.start_inclusive {
            Some(Self::new_unchecked(
                self.end_exclusive,
                other.start_inclusive,
            ))
        } else if other.end_exclusive < self.start_inclusive {
            Some(Self::new_unchecked(
                other.end_exclusive,
                self.start_inclusive,
            ))
        } else {
            None
        }
    }

    /// Splits the interval into two at the given `value`.
    ///
    /// Returns `Some((left, right))` if `value` is strictly inside the interval (`start < value < end`).
    /// Returns `None` if `value` is outside or at the boundaries.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// let iv = ClosedOpenInterval::new(0, 10);
    /// let (left, right) = iv.split_at(5).unwrap();
    /// assert_eq!(left, ClosedOpenInterval::new(0, 5));
    /// assert_eq!(right, ClosedOpenInterval::new(5, 10));
    /// ```
    #[inline]
    pub fn split_at(&self, value: T) -> Option<(Self, Self)> {
        if self.start_inclusive < value && value < self.end_exclusive {
            Some((
                Self::new_unchecked(self.start_inclusive, value),
                Self::new_unchecked(value, self.end_exclusive),
            ))
        } else {
            None
        }
    }

    /// Creates an iterator over the points in the interval.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    ///
    /// let iv = ClosedOpenInterval::new(1, 4);
    /// let points: Vec<_> = iv.iter().collect();
    /// assert_eq!(points, vec![1, 2, 3]);
    /// ```
    #[inline]
    pub fn iter(&self) -> ClosedOpenIntervalIterator<T> {
        ClosedOpenIntervalIterator {
            end_exclusive: self.end_exclusive,
            current: self.start_inclusive,
        }
    }
}

impl<T> BitAnd for ClosedOpenInterval<T>
where
    T: PrimInt,
{
    type Output = Option<Self>;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        self.intersection(rhs)
    }
}

impl<T> BitOr for ClosedOpenInterval<T>
where
    T: PrimInt,
{
    type Output = Option<Self>;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl<T> Default for ClosedOpenInterval<T>
where
    T: PrimInt,
{
    #[inline]
    fn default() -> Self {
        Self {
            start_inclusive: T::zero(),
            end_exclusive: T::zero(),
        }
    }
}

impl<T> std::fmt::Debug for ClosedOpenInterval<T>
where
    T: PrimInt + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClosedOpenInterval")
            .field("start_inclusive", &self.start_inclusive)
            .field("end_exclusive", &self.end_exclusive)
            .finish()
    }
}

impl<T> std::fmt::Display for ClosedOpenInterval<T>
where
    T: PrimInt + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {})", self.start_inclusive, self.end_exclusive)
    }
}

impl<T> std::ops::RangeBounds<T> for ClosedOpenInterval<T>
where
    T: PrimInt,
{
    fn start_bound(&self) -> std::ops::Bound<&T> {
        std::ops::Bound::Included(&self.start_inclusive)
    }

    fn end_bound(&self) -> std::ops::Bound<&T> {
        std::ops::Bound::Excluded(&self.end_exclusive)
    }
}

impl<T> IntoIterator for ClosedOpenInterval<T>
where
    T: PrimInt,
{
    type Item = T;
    type IntoIter = ClosedOpenIntervalIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> IntoIterator for &ClosedOpenInterval<T>
where
    T: PrimInt,
{
    type Item = T;
    type IntoIter = ClosedOpenIntervalIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> From<std::ops::Range<T>> for ClosedOpenInterval<T>
where
    T: PrimInt,
{
    #[inline]
    fn from(range: std::ops::Range<T>) -> Self {
        Self::new(range.start, range.end)
    }
}

impl<T> From<ClosedOpenInterval<T>> for std::ops::Range<T>
where
    T: PrimInt,
{
    #[inline]
    fn from(iv: ClosedOpenInterval<T>) -> Self {
        std::ops::Range {
            start: iv.start_inclusive,
            end: iv.end_exclusive,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::{Bound, RangeBounds};

    #[test]
    fn test_construction_valid() {
        let iv = ClosedOpenInterval::new(10, 20);
        assert_eq!(iv.start(), 10);
        assert_eq!(iv.end(), 20);
        assert_eq!(iv.len(), 10);
        assert!(!iv.is_empty());
    }

    #[test]
    fn test_construction_empty() {
        let iv = ClosedOpenInterval::new(10, 10);
        assert_eq!(iv.start(), 10);
        assert_eq!(iv.end(), 10);
        assert_eq!(iv.len(), 0);
        assert!(iv.is_empty());
    }

    #[test]
    fn test_try_new() {
        assert!(ClosedOpenInterval::try_new(5, 10).is_some());
        assert!(ClosedOpenInterval::try_new(5, 5).is_some());
        // Invalid: start > end
        assert!(ClosedOpenInterval::try_new(10, 5).is_none());
    }

    #[test]
    #[should_panic(expected = "Invalid interval")]
    fn test_new_panic() {
        ClosedOpenInterval::new(10, 5);
    }

    #[test]
    fn test_default() {
        let iv: ClosedOpenInterval<i32> = Default::default();
        assert!(iv.is_empty());
        assert_eq!(iv.start(), 0);
        assert_eq!(iv.end(), 0);
    }

    #[test]
    fn test_intersects() {
        let a = ClosedOpenInterval::new(0, 10);

        // Disjoint left
        assert!(!a.intersects(ClosedOpenInterval::new(-5, 0)));
        // Adjacent left (touching) - strictly NO intersection
        assert!(!a.intersects(ClosedOpenInterval::new(-5, 0)));
        // Overlap left
        assert!(a.intersects(ClosedOpenInterval::new(-5, 5)));
        // Contained
        assert!(a.intersects(ClosedOpenInterval::new(2, 8)));
        // Identity
        assert!(a.intersects(a));
        // Overlap right
        assert!(a.intersects(ClosedOpenInterval::new(5, 15)));
        // Adjacent right
        assert!(!a.intersects(ClosedOpenInterval::new(10, 15)));
        // Disjoint right
        assert!(!a.intersects(ClosedOpenInterval::new(11, 15)));
    }

    #[test]
    fn test_adjacent() {
        let a = ClosedOpenInterval::new(0, 10);

        // Touching start
        assert!(a.adjacent(ClosedOpenInterval::new(-5, 0)));
        // Touching end
        assert!(a.adjacent(ClosedOpenInterval::new(10, 15)));
        // Overlapping (not adjacent)
        assert!(!a.adjacent(ClosedOpenInterval::new(9, 11)));
        // Disjoint (not adjacent)
        assert!(!a.adjacent(ClosedOpenInterval::new(12, 15)));
    }

    #[test]
    fn test_intersects_or_adjacent() {
        let a = ClosedOpenInterval::new(0, 10);
        // Intersection
        assert!(a.intersects_or_adjacent(ClosedOpenInterval::new(5, 15)));
        // Adjacency
        assert!(a.intersects_or_adjacent(ClosedOpenInterval::new(10, 20)));
        // Gap
        assert!(!a.intersects_or_adjacent(ClosedOpenInterval::new(11, 20)));
    }

    #[test]
    fn test_contains_point() {
        let a = ClosedOpenInterval::new(0, 10);
        assert!(a.contains_point(0)); // Inclusive start
        assert!(a.contains_point(5));
        assert!(a.contains_point(9));
        assert!(!a.contains_point(10)); // Exclusive end
        assert!(!a.contains_point(-1));
    }

    #[test]
    fn test_contains_interval() {
        let main = ClosedOpenInterval::new(0, 10);

        // Exact match
        assert!(main.contains_interval(ClosedOpenInterval::new(0, 10)));
        // Strict subset
        assert!(main.contains_interval(ClosedOpenInterval::new(2, 8)));
        // Touching bounds
        assert!(main.contains_interval(ClosedOpenInterval::new(0, 5)));
        assert!(main.contains_interval(ClosedOpenInterval::new(5, 10)));

        // Overflow bounds
        assert!(!main.contains_interval(ClosedOpenInterval::new(-1, 5)));
        assert!(!main.contains_interval(ClosedOpenInterval::new(5, 11)));

        // Disjoint
        assert!(!main.contains_interval(ClosedOpenInterval::new(20, 30)));
    }

    #[test]
    fn test_intersection() {
        let a = ClosedOpenInterval::new(0, 10);
        let b = ClosedOpenInterval::new(5, 15);

        // Standard overlap
        assert_eq!(a.intersection(b), Some(ClosedOpenInterval::new(5, 10)));

        // Subset
        let c = ClosedOpenInterval::new(2, 8);
        assert_eq!(a.intersection(c), Some(c));

        let d = ClosedOpenInterval::new(10, 20);
        assert_eq!(a.intersection(d), None);

        // Disjoint
        let e = ClosedOpenInterval::new(12, 20);
        assert_eq!(a.intersection(e), None);
    }

    #[test]
    fn test_union() {
        let a = ClosedOpenInterval::new(0, 10);

        // Overlapping
        let b = ClosedOpenInterval::new(5, 15);
        assert_eq!(a.union(b), Some(ClosedOpenInterval::new(0, 15)));

        // Adjacent
        let c = ClosedOpenInterval::new(10, 20);
        assert_eq!(a.union(c), Some(ClosedOpenInterval::new(0, 20)));

        // Contained
        let d = ClosedOpenInterval::new(2, 8);
        assert_eq!(a.union(d), Some(a));

        // Disjoint (cannot union into single interval)
        let e = ClosedOpenInterval::new(12, 20);
        assert_eq!(a.union(e), None);
    }

    #[test]
    fn test_difference() {
        let base = ClosedOpenInterval::new(0, 10);

        // 1. Disjoint (No effect)
        let diff = base.difference(ClosedOpenInterval::new(12, 15));
        assert_eq!(diff.len(), 1);
        assert_eq!(diff[0], base);

        // 2. Full Cover (Empty result)
        let diff = base.difference(ClosedOpenInterval::new(-5, 15));
        assert!(diff.is_empty());

        // 3. Clip Right
        let diff = base.difference(ClosedOpenInterval::new(8, 15));
        assert_eq!(diff.len(), 1);
        assert_eq!(diff[0], ClosedOpenInterval::new(0, 8));

        // 4. Clip Left
        let diff = base.difference(ClosedOpenInterval::new(-5, 2));
        assert_eq!(diff.len(), 1);
        assert_eq!(diff[0], ClosedOpenInterval::new(2, 10));

        // 5. Split (The "Hole" case)
        let diff = base.difference(ClosedOpenInterval::new(4, 6));
        assert_eq!(diff.len(), 2);
        assert_eq!(diff[0], ClosedOpenInterval::new(0, 4));
        assert_eq!(diff[1], ClosedOpenInterval::new(6, 10));
    }

    #[test]
    fn test_gap() {
        let a = ClosedOpenInterval::new(0, 5);
        let b = ClosedOpenInterval::new(10, 15);

        // A ... B
        let g = a.gap(b).unwrap();
        assert_eq!(g, ClosedOpenInterval::new(5, 10));

        // B ... A (Commutative check)
        let g = b.gap(a).unwrap();
        assert_eq!(g, ClosedOpenInterval::new(5, 10));

        // Adjacent (No gap)
        let c = ClosedOpenInterval::new(5, 10);
        assert!(a.gap(c).is_none());

        // Intersecting
        let d = ClosedOpenInterval::new(4, 6);
        assert!(a.gap(d).is_none());
    }

    #[test]
    fn test_midpoint() {
        // Even length
        let a = ClosedOpenInterval::new(0, 10);
        assert_eq!(a.midpoint(), 5);

        // Odd length (truncation)
        let b = ClosedOpenInterval::new(0, 3);
        // len 3, 3>>1 = 1. start+1 = 1.
        assert_eq!(b.midpoint(), 1);

        // Negative numbers
        let c = ClosedOpenInterval::new(-10, -4);
        // len 6. half 3. -10 + 3 = -7.
        assert_eq!(c.midpoint(), -7);

        // Overflow safety (u8)
        let d: ClosedOpenInterval<u8> = ClosedOpenInterval::new(250, 254);
        // (250+254)/2 = 252. Naive add would panic.
        assert_eq!(d.midpoint(), 252);
    }

    #[test]
    fn test_split_at() {
        let a = ClosedOpenInterval::new(0, 10);
        assert!(a.split_at(0).is_none());
    }

    #[test]
    fn test_iterator() {
        let a = ClosedOpenInterval::new(1, 4);
        let collected: Vec<i32> = a.iter().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn test_iterator_empty() {
        let a = ClosedOpenInterval::new(5, 5);
        let mut iter = a.iter();
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_double_ended_iterator() {
        let a = ClosedOpenInterval::new(1, 4);
        let mut iter = a.iter();

        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next_back(), Some(3));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iterator_trait() {
        let a = ClosedOpenInterval::new(0, 3);
        let mut count = 0;
        for i in a {
            // Consumes a
            assert_eq!(i, count);
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_into_iterator_ref_trait() {
        let a = ClosedOpenInterval::new(0, 3);
        for (count, i) in (&a).into_iter().enumerate() {
            // Borrows a
            assert_eq!(i, count);
        }
        // a is still valid here
        assert_eq!(a.len(), 3);
    }

    #[test]
    fn test_fused_iterator() {
        let a = ClosedOpenInterval::new(0, 1);
        let mut iter = a.iter();
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None); // Should continue returning None
    }

    #[test]
    fn test_traits_display_debug() {
        let a = ClosedOpenInterval::new(10, 20);
        assert_eq!(format!("{}", a), "[10, 20)");
        assert_eq!(
            format!("{:?}", a),
            "ClosedOpenInterval { start_inclusive: 10, end_exclusive: 20 }"
        );
    }

    #[test]
    fn test_from_range() {
        let range = 0..10;
        let iv = ClosedOpenInterval::from(range);
        assert_eq!(iv.start(), 0);
        assert_eq!(iv.end(), 10);
    }

    #[test]
    fn test_range_bounds() {
        let iv = ClosedOpenInterval::new(5, 10);

        match iv.start_bound() {
            Bound::Included(&x) => assert_eq!(x, 5),
            _ => panic!("Wrong start bound"),
        }

        match iv.end_bound() {
            Bound::Excluded(&x) => assert_eq!(x, 10),
            _ => panic!("Wrong end bound"),
        }
    }
}
