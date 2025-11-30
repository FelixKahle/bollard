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
use num_traits::{Bounded, One, Zero};
use smallvec::SmallVec;
use std::{
    iter::FusedIterator,
    ops::{Add, Bound, RangeBounds, Rem, Sub},
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
    #[inline]
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
    #[inline]
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
    #[inline]
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

    /// Creates an iterator over the interval that yields values spaced by `step`.
    ///
    /// The iterator starts at `self.lower` and advances by `step` each time,
    /// yielding values up to and including `self.upper` when possible. If the next
    /// step would exceed `self.upper`, the iterator stops yielding from the front.
    /// When used as a `DoubleEndedIterator`, it yields from `self.upper` backwards
    /// by `step` until doing so would cross below `self.lower`.
    ///
    /// # Known Limitations (Signed Integers)
    ///
    /// For signed integer types (e.g., `i8`, `i32`), if the interval spans the entire representable
    /// range (e.g., `i8::MIN` to `i8::MAX`), the mathematical distance `upper - lower` exceeds
    /// `T::MAX`. Calculating the alignment for reverse iteration requires this distance.
    ///
    /// In this specific edge case, the iterator defaults to starting reverse iteration at `self.upper`,
    /// even if `self.upper` does not mathematically align with the grid starting from `self.lower`.
    /// This avoids the need for casting to wider types, maintaining generic simplicity at the cost
    /// of strict alignment in this rare overflow scenario.
    ///
    /// # Debug Assertions
    ///
    /// - In debug builds, passing a non-positive `step` (e.g., zero or a value that
    ///   does not strictly move the iterator forward/backward for the type) will
    ///   cause a debug assertion to fail.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::ClosedInterval;
    ///
    /// // Forward iteration with step 3
    /// let i = ClosedInterval::new(1, 10);
    /// let v: Vec<_> = i.iter_step(3).collect();
    /// assert_eq!(v, vec![1, 4, 7, 10]);
    ///
    /// // Double-ended alternating
    /// let mut it = i.iter_step(3);
    /// assert_eq!(it.next(), Some(1));
    /// assert_eq!(it.next_back(), Some(10));
    /// ```
    pub fn iter_step(&self, step: T) -> ClosedIntervalStepIterator<T>
    where
        T: Copy + Zero + Ord + Rem<Output = T> + checked_arithmetic::CheckedSub,
    {
        debug_assert!(step > T::zero(), "step must be positive");

        // We attempt to calculate the largest value K <= upper such that K = lower + n * step.
        // This is done via: aligned_upper = upper - (distance % step).
        //
        // If (upper - lower) overflows (e.g. i8::MIN to i8::MAX), checked_sub returns None.
        // In that case, we fallback to just using upper. This is the documented limitation.
        let aligned_upper = match self.upper.checked_sub(self.lower) {
            Some(diff) => self.upper - (diff % step),
            None => self.upper,
        };

        ClosedIntervalStepIterator {
            current: self.lower,
            upper: aligned_upper,
            step,
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

impl<T> RangeBounds<T> for ClosedInterval<T> {
    fn start_bound(&self) -> Bound<&T> {
        Bound::Included(&self.lower)
    }

    fn end_bound(&self) -> Bound<&T> {
        Bound::Included(&self.upper)
    }
}

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
    #[inline]
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

impl<T> FusedIterator for ClosedIntervalIterator<T> where
    T: Copy + Ord + One + Add<Output = T> + Sub<Output = T>
{
}

impl<T> std::fmt::Display for ClosedIntervalIterator<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ClosedIntervalIterator(current: {}, upper: {}, finished: {})",
            self.current, self.upper, self.finished
        )
    }
}

#[derive(Copy, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClosedIntervalStepIterator<T> {
    current: T,
    upper: T,
    step: T,
    finished: bool,
}

impl<T> Iterator for ClosedIntervalStepIterator<T>
where
    T: Copy + Ord + One + checked_arithmetic::CheckedAdd,
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
            match self.current.checked_add(self.step) {
                Some(next_val) => {
                    if next_val > self.upper {
                        self.finished = true;
                    } else {
                        self.current = next_val;
                    }
                }
                None => {
                    self.finished = true;
                }
            }
        }

        Some(result)
    }
}

impl<T> DoubleEndedIterator for ClosedIntervalStepIterator<T>
where
    T: Copy + Ord + One + checked_arithmetic::CheckedSub + checked_arithmetic::CheckedAdd,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let result = self.upper;

        if self.current >= self.upper {
            self.finished = true;
        } else {
            match self.upper.checked_sub(self.step) {
                Some(prev_val) => {
                    if prev_val < self.current {
                        self.finished = true;
                    } else {
                        self.upper = prev_val;
                    }
                }
                None => {
                    self.finished = true;
                }
            }
        }

        Some(result)
    }
}

impl<T> FusedIterator for ClosedIntervalStepIterator<T> where
    T: Copy + Ord + One + checked_arithmetic::CheckedAdd
{
}

impl<T> std::fmt::Display for ClosedIntervalStepIterator<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ClosedIntervalStepIterator(current: {}, upper: {}, step: {}, finished: {})",
            self.current, self.upper, self.step, self.finished
        )
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

/// Error indicating that an `IntegerDomain` is empty.
///
/// This error is returned by methods that require the domain to contain at least one value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EmptyIntegerDomainError;

impl std::fmt::Display for EmptyIntegerDomainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IntegerDomain is empty")
    }
}

impl std::error::Error for EmptyIntegerDomainError {}

/// A set of integers represented by a list of disjoint, sorted `ClosedInterval`s.
///
/// `IntegerDomain` is efficient for representing sets of integers that may have gaps.
/// It maintains the invariant that the stored intervals are:
/// 1. **Sorted**: Ordered strictly by their lower bounds.
/// 2. **Disjoint**: No two intervals overlap or touch (adjacent intervals should ideally be merged,
///    though strict separation `upper < lower` is the hard requirement).
///
/// # Examples
///
/// ```
/// use bollard_constraint_solver::variable::{IntegerDomain, ClosedInterval};
///
/// // Represents the set {1, 2, 3, 4, 5, 10, 11, 12}
/// let intervals = vec![
///     ClosedInterval::new(1, 5),
///     ClosedInterval::new(10, 12)
/// ];
/// let domain: IntegerDomain<i32, 1> = IntegerDomain::from_sorted_vec(intervals);
///
/// assert!(domain.contains_value(3));
/// assert!(!domain.contains_value(7));
/// assert!(domain.contains_value(10));
/// ```
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct IntegerDomain<T, const N: usize = 1> {
    // Invariant: intervals are non-overlapping and sorted by lower bound.
    intervals: SmallVec<ClosedInterval<T>, N>,
}

impl<T, const N: usize> IntegerDomain<T, N> {
    /// Creates a new `IntegerDomain` containing a single interval `[min, max]`.
    ///
    /// # Panics
    ///
    /// Panics if `min > max`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::IntegerDomain;
    ///
    /// let d: IntegerDomain<i32, 1> = IntegerDomain::new(1, 10);
    /// assert_eq!(d.len(), 1);
    /// ```
    #[inline]
    pub fn new(min: T, max: T) -> Self
    where
        T: Ord,
    {
        assert!(min <= max, "min must be <= max");

        IntegerDomain {
            intervals: smallvec::smallvec![ClosedInterval::new(min, max)],
        }
    }

    /// Creates a new `IntegerDomain` without checking `min <= max`.
    ///
    /// # Safety
    ///
    /// The caller must ensure `min <= max`. If this invariant is violated,
    /// the behavior of subsequent operations is undefined.
    ///
    /// # Debug Assertions
    ///
    /// In debug builds, this function asserts that `min <= max`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::IntegerDomain;
    ///
    /// let d: IntegerDomain<i32, 1> = unsafe { IntegerDomain::new_unchecked(1, 10) };
    /// assert_eq!(d.len(), 1);
    /// ```
    #[inline]
    pub fn new_unchecked(min: T, max: T) -> Self
    where
        T: Ord,
    {
        debug_assert!(min <= max, "min must be <= max");

        IntegerDomain {
            intervals: smallvec::smallvec![ClosedInterval::new_unchecked(min, max)],
        }
    }

    /// Creates a domain containing exactly one value `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::IntegerDomain;
    ///
    /// let d: IntegerDomain<i32, 1> = IntegerDomain::from_singleton(42);
    /// assert_eq!(d.len(), 1);
    /// ```
    #[inline]
    pub fn from_singleton(value: T) -> Self
    where
        T: Copy + Ord,
    {
        IntegerDomain {
            intervals: smallvec::smallvec![ClosedInterval::singleton(value)],
        }
    }

    /// Creates an empty integer domain.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::IntegerDomain;
    ///
    /// let d: IntegerDomain<i32, 1> = IntegerDomain::empty();
    /// assert!(d.is_empty());
    /// ```
    #[inline]
    pub fn empty() -> Self {
        IntegerDomain {
            intervals: SmallVec::new(),
        }
    }

    /// Creates a domain covering the entire range of type `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::IntegerDomain;
    ///
    /// let d: IntegerDomain<u8, 1> = IntegerDomain::all();
    /// assert_eq!(d.size(), 255); // Saturates
    /// ```
    #[inline]
    pub fn all() -> Self
    where
        T: Bounded + Ord,
    {
        IntegerDomain {
            // Save to use new_unchecked here because min_value() <= max_value() is always true.
            intervals: smallvec::smallvec![ClosedInterval::new_unchecked(
                T::min_value(),
                T::max_value()
            )],
        }
    }

    /// Creates a domain from a vector of intervals.
    ///
    /// # Preconditions
    ///
    /// The input `intervals` must be:
    /// 1. **Sorted** by lower bound.
    /// 2. **Strictly Disjoint**: `intervals[i].upper < intervals[i+1].lower`.
    ///
    /// If these conditions are not met, the domain invariants are broken.
    ///
    /// # Debug Assertions
    ///
    /// In debug builds, this function asserts that the intervals are sorted and disjoint.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::{IntegerDomain, ClosedInterval};
    /// use smallvec::smallvec;
    ///
    /// let intervals = vec![
    ///     ClosedInterval::new(1, 5),
    ///     ClosedInterval::new(10, 15)
    /// ];
    ///
    /// let domain: IntegerDomain<i32, 2> = IntegerDomain::from_sorted_vec(intervals);
    /// assert_eq!(domain.len(), 2);
    /// ```
    #[inline]
    pub fn from_sorted_vec(intervals: Vec<ClosedInterval<T>>) -> Self
    where
        T: Ord,
    {
        let smallvec: SmallVec<ClosedInterval<T>, N> = intervals.into_iter().collect();

        debug_assert!(
            smallvec.windows(2).all(|w| w[0].upper < w[1].lower),
            "IntegerDomain intervals must be sorted and strictly disjoint (no overlaps)"
        );

        IntegerDomain {
            intervals: smallvec,
        }
    }

    /// Creates a domain from a `SmallVec` of intervals.
    ///
    /// # Preconditions
    ///
    /// The input `intervals` must be:
    /// 1. **Sorted** by lower bound.
    /// 2. **Strictly Disjoint**: `intervals[i].upper < intervals[i+1].lower`.
    ///
    /// If these conditions are not met, the domain invariants are broken.
    ///
    /// # Debug Assertions
    ///
    /// In debug builds, this function asserts that the intervals are sorted and disjoint.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::{IntegerDomain, ClosedInterval};
    /// use smallvec::smallvec;
    ///
    /// let intervals = smallvec![
    ///     ClosedInterval::new(1, 5),
    ///     ClosedInterval::new(10, 15)
    /// ];
    ///
    /// let domain: IntegerDomain<i32, 2> = IntegerDomain::from_sorted_smallvec(intervals);
    /// assert_eq!(domain.len(), 2);
    /// ```
    #[inline]
    pub fn from_sorted_smallvec(intervals: SmallVec<ClosedInterval<T>, N>) -> Self
    where
        T: Ord,
    {
        debug_assert!(
            intervals.windows(2).all(|w| w[0].upper < w[1].lower),
            "IntegerDomain intervals must be sorted and strictly disjoint (no overlaps)"
        );

        IntegerDomain { intervals }
    }

    /// Creates a domain from an iterator of intervals.
    ///
    /// # Preconditions
    ///
    /// The input `intervals` must be:
    /// 1. **Sorted** by lower bound.
    /// 2. **Strictly Disjoint**: `intervals[i].upper < intervals[i+1].lower`.
    ///
    #[inline]
    pub fn from_sorted_intervals<I>(iter: I) -> Self
    where
        T: Ord,
        I: IntoIterator<Item = ClosedInterval<T>>,
    {
        let intervals: SmallVec<ClosedInterval<T>, N> = iter.into_iter().collect();

        debug_assert!(
            intervals.windows(2).all(|w| w[0].upper < w[1].lower),
            "IntegerDomain intervals must be sorted and strictly disjoint (no overlaps)"
        );

        IntegerDomain { intervals }
    }

    /// Returns the number of disjoint intervals in the domain.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::{IntegerDomain, ClosedInterval};
    ///
    /// let domain: IntegerDomain<i32, 2> = IntegerDomain::from_sorted_vec(vec![
    ///     ClosedInterval::new(1, 5),
    ///     ClosedInterval::new(10, 15)
    /// ]);
    ///
    /// assert_eq!(domain.len(), 2);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.intervals.len()
    }

    /// Returns `true` if the domain contains no intervals (is empty).
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::IntegerDomain;
    ///
    /// let empty_domain: IntegerDomain<i32, 1> = IntegerDomain::empty();
    /// assert!(empty_domain.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    /// Returns the minimum value in the domain.
    ///
    /// # Errors
    ///
    /// Returns `EmptyIntegerDomainError` if the domain is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::{IntegerDomain, ClosedInterval};
    ///
    /// let domain: IntegerDomain<i32, 1> = IntegerDomain::from_sorted_vec(vec![
    ///     ClosedInterval::new(5, 10),
    ///     ClosedInterval::new(20, 25)
    /// ]);
    ///
    /// assert_eq!(domain.min().unwrap(), 5);
    /// ```
    #[inline]
    pub fn min(&self) -> Result<T, EmptyIntegerDomainError>
    where
        T: Copy,
    {
        self.intervals
            .first()
            .map(|interval| interval.lower())
            .ok_or(EmptyIntegerDomainError)
    }

    /// Returns the minimum value in the domain.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the domain is non-empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::{IntegerDomain, ClosedInterval};
    ///
    /// let domain: IntegerDomain<i32, 1> = IntegerDomain::from_sorted_vec(vec![
    ///    ClosedInterval::new(5, 10),
    ///    ClosedInterval::new(20, 25)
    /// ]);
    ///
    /// assert!(!domain.is_empty());
    /// assert_eq!(unsafe { domain.min_unchecked() }, 5);
    /// ```
    #[inline]
    pub unsafe fn min_unchecked(&self) -> T
    where
        T: Copy,
    {
        debug_assert!(
            !self.intervals.is_empty(),
            "IntegerDomain must be non-empty"
        );

        unsafe { self.intervals.get_unchecked(0).lower() }
    }

    /// Returns the maximum value in the domain.
    ///
    /// # Errors
    ///
    /// Returns `EmptyIntegerDomainError` if the domain is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::{IntegerDomain, ClosedInterval};
    ///
    /// let domain: IntegerDomain<i32, 1> = IntegerDomain::from_sorted_vec(vec![
    ///     ClosedInterval::new(5, 10),
    ///     ClosedInterval::new(20, 25)
    /// ]);
    ///
    /// assert_eq!(domain.max().unwrap(), 25);
    /// ```
    #[inline]
    pub fn max(&self) -> Result<T, EmptyIntegerDomainError>
    where
        T: Copy,
    {
        self.intervals
            .last()
            .map(|interval| interval.upper())
            .ok_or(EmptyIntegerDomainError)
    }

    /// Returns the maximum value in the domain.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the domain is non-empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::{IntegerDomain, ClosedInterval};
    ///
    /// let domain: IntegerDomain<i32, 1> = IntegerDomain::from_sorted_vec(vec![
    ///     ClosedInterval::new(5, 10),
    ///     ClosedInterval::new(20, 25)
    /// ]);
    ///
    /// assert!(!domain.is_empty());
    /// assert_eq!(unsafe { domain.max_unchecked() }, 25);
    /// ```
    #[inline]
    pub unsafe fn max_unchecked(&self) -> T
    where
        T: Copy,
    {
        debug_assert!(
            !self.intervals.is_empty(),
            "IntegerDomain must be non-empty"
        );

        unsafe { self.intervals.get_unchecked(self.intervals.len() - 1).upper }
    }

    /// Returns `true` if the domain represents a single fixed value.
    ///
    /// # Errors
    ///
    /// Returns `EmptyIntegerDomainError` if the domain is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::{IntegerDomain, ClosedInterval};
    ///
    /// let fixed_domain: IntegerDomain<i32, 1> = IntegerDomain::from_singleton(42);
    /// assert!(fixed_domain.is_fixed().unwrap());
    /// ```
    #[inline]
    pub fn is_fixed(&self) -> Result<bool, EmptyIntegerDomainError>
    where
        T: Copy + PartialEq,
    {
        if self.is_empty() {
            Err(EmptyIntegerDomainError)
        } else {
            // Safe because we just checked is_empty()
            unsafe { Ok(self.min_unchecked() == self.max_unchecked()) }
        }
    }

    /// Returns `true` if the domain represents a single fixed value.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the domain is non-empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::{IntegerDomain, ClosedInterval};
    ///
    /// let fixed_domain: IntegerDomain<i32, 1> = IntegerDomain::from_singleton(42);
    /// assert!(!fixed_domain.is_empty());
    /// assert!(unsafe { fixed_domain.is_fixed_unchecked() });
    /// ```
    #[inline]
    pub unsafe fn is_fixed_unchecked(&self) -> bool
    where
        T: Copy + PartialEq,
    {
        debug_assert!(
            !self.intervals.is_empty(),
            "IntegerDomain must be non-empty"
        );

        unsafe { self.min_unchecked() == self.max_unchecked() }
    }

    /// Returns the total number of integer values in the domain.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::{IntegerDomain, ClosedInterval};
    ///
    /// let domain: IntegerDomain<i32, 1> = IntegerDomain::from_sorted_vec(vec![
    ///     ClosedInterval::new(1, 5),
    ///     ClosedInterval::new(10, 15)
    /// ]);
    ///
    /// assert_eq!(domain.size(), 11); // (5 - 1 + 1) + (15 - 10 + 1) = 5 + 6 = 11
    /// ```
    #[inline]
    pub fn size(&self) -> T
    where
        T: Copy
            + Zero
            + One
            + saturating_arithmetic::SaturatingAdd
            + saturating_arithmetic::SaturatingSub,
    {
        self.intervals.iter().fold(T::zero(), |acc, interval| {
            acc.saturating_add(interval.len())
        })
    }

    /// Checks if this domain intersects with another domain.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::{IntegerDomain, ClosedInterval};
    ///
    /// let domain_a: IntegerDomain<i32, 1> = IntegerDomain::from_sorted_vec(vec![
    ///     ClosedInterval::new(1, 5),
    ///     ClosedInterval::new(10, 15)
    /// ]);
    ///
    /// let domain_b: IntegerDomain<i32, 1> = IntegerDomain::from_sorted_vec(vec![
    ///     ClosedInterval::new(4, 8)
    /// ]);
    ///
    /// let domain_c: IntegerDomain<i32, 1> = IntegerDomain::from_sorted_vec(vec![
    ///     ClosedInterval::new(6, 9)
    /// ]);
    ///
    /// assert!(domain_a.intersects(&domain_b));
    /// assert!(!domain_a.intersects(&domain_c));
    /// ```
    pub fn intersects(&self, other: &Self) -> bool
    where
        T: Copy + PartialOrd,
    {
        let a = &self.intervals;
        let b = &other.intervals;

        let mut i = 0usize;
        let mut j = 0usize;

        while i < a.len() && j < b.len() {
            let a_i = *unsafe { a.get_unchecked(i) };
            let b_j = *unsafe { b.get_unchecked(j) };

            if a_i.upper < b_j.lower {
                i += 1;
            } else if b_j.upper < a_i.lower {
                j += 1;
            } else {
                return true;
            }
        }

        false
    }

    /// Checks if the domain contains the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::{IntegerDomain, ClosedInterval};
    ///
    /// let domain: IntegerDomain<i32, 1> = IntegerDomain::from_sorted_vec(vec![
    ///     ClosedInterval::new(1, 5),
    ///     ClosedInterval::new(10, 15)
    /// ]);
    ///
    /// assert!(domain.contains_value(3));
    /// assert!(!domain.contains_value(7));
    /// assert!(domain.contains_value(10));
    /// ```
    pub fn contains_value(&self, value: T) -> bool
    where
        T: Copy + Ord,
    {
        let intervals = &self.intervals;
        let mut left = 0usize;
        let mut right = intervals.len(); // exclusive

        while left < right {
            let mid = left + ((right - left) >> 1);
            let interval = *unsafe { intervals.get_unchecked(mid) };

            if value < interval.lower {
                right = mid;
            } else if value > interval.upper {
                left = mid + 1;
            } else {
                return true;
            }
        }
        false
    }

    /// Checks if the domain fully contains the given interval.
    ///
    /// # Examples
    ///
    /// ```
    /// use bollard_constraint_solver::variable::{IntegerDomain, ClosedInterval};
    ///
    /// let domain: IntegerDomain<i32, 1> = IntegerDomain::from_sorted_vec(vec![
    ///     ClosedInterval::new(1, 5),
    ///     ClosedInterval::new(10, 15)
    /// ]);
    ///
    /// assert!(domain.contains_interval(ClosedInterval::new(2, 4)));
    /// assert!(!domain.contains_interval(ClosedInterval::new(5, 10)));
    /// ```
    pub fn contains_interval(&self, interval: ClosedInterval<T>) -> bool
    where
        T: Copy + Ord,
    {
        let a = &self.intervals;
        if a.is_empty() {
            return false;
        }

        let low = interval.lower;
        let up = interval.upper;

        let mut left = 0usize;
        let mut right = a.len(); // exclusive

        while left < right {
            let mid = left + ((right - left) >> 1);
            let a_i = *unsafe { a.get_unchecked(mid) };

            if a_i.lower <= low {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        if left == 0 {
            return false;
        }

        let idx = left - 1;
        let container = *unsafe { a.get_unchecked(idx) };
        container.lower <= low && up <= container.upper
    }
}

impl<T> std::fmt::Debug for IntegerDomain<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut tuple = f.debug_tuple("IntegerDomain");
        for interval in &self.intervals {
            tuple.field(interval);
        }
        tuple.finish()
    }
}

impl<T> std::fmt::Display for IntegerDomain<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IntegerDomain(")?;
        let mut iter = self.intervals.iter();
        if let Some(first) = iter.next() {
            write!(f, "{}", first)?;
            for interval in iter {
                write!(f, ", {}", interval)?;
            }
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_interval<T: Ord + Copy>(l: T, u: T) -> ClosedInterval<T> {
        ClosedInterval::new(l, u)
    }

    fn new_closed_interval<T: Ord + Copy>(l: T, u: T) -> ClosedInterval<T> {
        ClosedInterval::new(l, u)
    }

    macro_rules! generate_interval_tests {
        ($mod_name:ident, $t:ty) => {
            mod $mod_name {
                use super::super::*;
                use super::*;

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
                    assert_eq!(new_interval(1, 5).len(), 5);
                    assert_eq!(new_interval(1, 1).len(), 1);
                }

                #[test]
                fn test_len_saturation() {
                    let min = <$t>::min_value();
                    let max = <$t>::max_value();
                    let full = new_interval(min, max);

                    assert_eq!(full.len(), max);

                    let almost_full = new_interval(min + 1, max);

                    if <$t>::min_value() == 0 {
                        assert_eq!(almost_full.len(), max);
                    } else {
                        assert_eq!(almost_full.len(), max);
                    }
                }

                #[test]
                fn test_contains() {
                    let x = new_interval(10, 20);
                    assert!(x.contains(10));
                    assert!(x.contains(20));
                    assert!(x.contains(15));
                    assert!(!x.contains(9));
                    assert!(!x.contains(21));
                    assert!(x.contains_interval(new_interval(12, 18)));
                    assert!(x.contains_interval(new_interval(10, 20)));
                    assert!(!x.contains_interval(new_interval(9, 20)));
                    assert!(!x.contains_interval(new_interval(10, 21)));
                }

                #[test]
                fn test_intersects() {
                    let base = new_interval(10, 20);

                    assert!(base.intersects(new_interval(12, 18)));
                    assert!(base.intersects(new_interval(5, 10)));
                    assert!(base.intersects(new_interval(20, 25)));
                    assert!(base.intersects(new_interval(0, 30)));
                    assert!(!base.intersects(new_interval(0, 9)));
                    assert!(!base.intersects(new_interval(21, 30)));
                }

                #[test]
                fn test_adjacency() {
                    let base = new_interval(10, 20);

                    assert!(base.adjacent(new_interval(5, 9)));
                    assert!(base.adjacent(new_interval(21, 30)));
                    assert!(!base.adjacent(new_interval(20, 25)));
                    assert!(!base.adjacent(new_interval(22, 30)));

                    let max_interval = new_interval(<$t>::max_value(), <$t>::max_value());
                    let almost_max = new_interval(<$t>::max_value() - 1, <$t>::max_value() - 1);
                    assert!(max_interval.adjacent(almost_max));
                    assert!(almost_max.adjacent(max_interval));
                }

                #[test]
                fn test_merge_logic() {
                    let base = new_interval(10, 20);

                    let m1 = base.merge(new_interval(15, 25));
                    assert_eq!(m1, Some(new_interval(10, 25)));

                    let m2 = base.merge(new_interval(21, 30));
                    assert_eq!(m2, Some(new_interval(10, 30)));

                    let m3 = base.merge(new_interval(0, 9));
                    assert_eq!(m3, Some(new_interval(0, 20)));

                    let m4 = base.merge(new_interval(22, 30));
                    assert_eq!(m4, None);
                }

                #[test]
                fn test_subtract() {
                    let base = new_interval(10, 20);

                    let r1 = base.subtract(new_interval(25, 30));
                    assert_eq!(r1.len(), 1);
                    assert_eq!(r1[0], base);

                    let r2 = base.subtract(new_interval(0, 30));
                    assert!(r2.is_empty());

                    let r3 = base.subtract(new_interval(12, 18));
                    assert_eq!(r3.len(), 2);
                    assert_eq!(r3[0], new_interval(10, 11));
                    assert_eq!(r3[1], new_interval(19, 20));

                    let r4 = base.subtract(new_interval(15, 30));
                    assert_eq!(r4.len(), 1);
                    assert_eq!(r4[0], new_interval(10, 14));

                    let r5 = base.subtract(new_interval(0, 15));
                    assert_eq!(r5.len(), 1);
                    assert_eq!(r5[0], new_interval(16, 20));

                    let r6 = base.subtract(base);
                    assert!(r6.is_empty());
                }

                #[test]
                fn test_subtract_edge_cases() {
                    let min = <$t>::min_value();
                    let max = <$t>::max_value();

                    let full = new_interval(min, max);
                    let sub_min = full.subtract(new_interval(min, min));
                    assert_eq!(sub_min.len(), 1);
                    assert_eq!(sub_min[0], new_interval(min + 1, max));

                    let sub_max = full.subtract(new_interval(max, max));
                    assert_eq!(sub_max.len(), 1);
                    assert_eq!(sub_max[0], new_interval(min, max - 1));
                }

                #[test]
                fn test_complement() {
                    let min = <$t>::min_value();
                    let max = <$t>::max_value();

                    let mid = new_interval(10, 20);
                    let comp = mid.complement();
                    assert_eq!(comp.len(), 2);

                    if min < 10 {
                        assert_eq!(comp[0], new_interval(min, 9));
                    }
                    if max > 20 {
                        assert_eq!(comp[1], new_interval(21, max));
                    }

                    let full = new_interval(min, max);
                    assert!(full.complement().is_empty());
                }

                #[test]
                fn test_iterator_basic() {
                    let interval = new_interval(1, 3);
                    let vec: Vec<$t> = interval.iter().collect();
                    assert_eq!(vec, vec![1, 2, 3]);
                }

                #[test]
                fn test_iterator_single() {
                    let interval = new_interval(10, 10);
                    let vec: Vec<$t> = interval.iter().collect();
                    assert_eq!(vec, vec![10]);
                }

                #[test]
                fn test_iterator_boundary_max() {
                    let max = <$t>::max_value();
                    let interval = new_interval(max - 1, max);
                    let vec: Vec<$t> = interval.iter().collect();
                    assert_eq!(vec, vec![max - 1, max]);
                }

                #[test]
                fn test_from_range_inclusive() {
                    let range = 5 as $t..=15 as $t;
                    let interval: ClosedInterval<$t> = range.into();
                    assert_eq!(interval, new_interval(5, 15));
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

                #[test]
                fn test_iter_step_basic() {
                    let interval = new_interval(1, 10);
                    let vec: Vec<$t> = interval.iter_step(3 as $t).collect();
                    assert_eq!(vec, vec![1, 4, 7, 10]);
                }

                #[test]
                fn test_iter_step_singleton() {
                    let interval = new_interval(7, 7);
                    let vec: Vec<$t> = interval.iter_step(3 as $t).collect();
                    assert_eq!(vec, vec![7]);
                }

                #[test]
                fn test_iter_step_uneven_alignment() {
                    let interval = new_interval(1, 8);
                    let vec: Vec<$t> = interval.iter_step(3 as $t).collect();
                    assert_eq!(vec, vec![1, 4, 7]);

                    let interval2 = new_interval(2, 9);
                    let vec2: Vec<$t> = interval2.iter_step(4 as $t).collect();
                    assert_eq!(vec2, vec![2, 6]);
                }

                #[test]
                fn test_iter_step_exact_alignment() {
                    let interval = new_interval(1, 9);
                    let vec: Vec<$t> = interval.iter_step(4 as $t).collect();
                    assert_eq!(vec, vec![1, 5, 9]);
                }

                #[test]
                fn test_iter_step_double_ended_basic() {
                    let interval = new_interval(1, 10);
                    let mut iter = interval.iter_step(3 as $t);

                    assert_eq!(iter.next(), Some(1));
                    assert_eq!(iter.next_back(), Some(10));

                    assert_eq!(iter.next(), Some(4));
                    assert_eq!(iter.next_back(), Some(7));

                    assert_eq!(iter.next(), None);
                    assert_eq!(iter.next_back(), None);
                }

                #[test]
                fn test_iter_step_double_ended_exhaustion_forward_then_back() {
                    let interval = new_interval(1, 10);
                    let mut iter = interval.iter_step(3 as $t);

                    assert_eq!(iter.next(), Some(1));
                    assert_eq!(iter.next(), Some(4));
                    assert_eq!(iter.next(), Some(7));
                    assert_eq!(iter.next(), Some(10));

                    assert_eq!(iter.next(), None);
                    assert_eq!(iter.next_back(), None);
                }

                #[test]
                fn test_iter_step_double_ended_exhaustion_back_then_forward() {
                    let interval = new_interval(1, 10);
                    let mut iter = interval.iter_step(3 as $t);

                    assert_eq!(iter.next_back(), Some(10));
                    assert_eq!(iter.next_back(), Some(7));
                    assert_eq!(iter.next_back(), Some(4));
                    assert_eq!(iter.next_back(), Some(1));

                    assert_eq!(iter.next(), None);
                    assert_eq!(iter.next_back(), None);
                }

                #[test]
                fn test_iter_step_double_ended_singleton() {
                    let interval = new_interval(7, 7);
                    let mut iter = interval.iter_step(5 as $t);

                    assert_eq!(iter.next(), Some(7));
                    assert_eq!(iter.next_back(), None);

                    let mut iter2 = interval.iter_step(5 as $t);
                    assert_eq!(iter2.next_back(), Some(7));
                    assert_eq!(iter2.next(), None);
                }

                #[test]
                fn test_iter_step_collect() {
                    let interval = new_interval(2, 14);
                    let v: Vec<$t> = interval.iter_step(5 as $t).collect();
                    assert_eq!(v, vec![2, 7, 12]);
                }

                #[test]
                fn test_iter_step_boundary_max_min() {
                    let min = <$t>::min_value();
                    let max = <$t>::max_value();
                    if min < max {
                        let step = 2 as $t;
                        let mut iter = new_interval(min, max).iter_step(step);

                        let first = iter.next();
                        let last = iter.next_back();
                        assert_eq!(first, Some(min));
                        assert!(last.is_some());
                        let l = last.unwrap();
                        assert!(l <= max);
                        assert!(l >= min);

                        let f2 = iter.next();
                        let b2 = iter.next_back();
                        if min + step <= max {
                            assert!(f2.is_some());
                            assert!(b2.is_some());
                            assert!(f2.unwrap() >= min);
                            assert!(b2.unwrap() <= max);
                        }
                    }
                }

                #[test]
                fn test_iter_step_misalignment() {
                    let interval = new_interval(1, 4);
                    let mut iter = interval.iter_step(2 as $t);

                    assert_eq!(iter.next_back(), Some(3));
                    assert_eq!(iter.next_back(), Some(1));
                    assert_eq!(iter.next_back(), None);
                }

                #[test]
                fn test_iter_step_signed_overflow_limitation() {
                    // Range: -128 to 127
                    let interval = ClosedInterval::new(i8::MIN, i8::MAX);
                    let step = 10;

                    let mut forward_iter = interval.iter_step(step);
                    assert_eq!(forward_iter.next(), Some(-128));
                    assert_eq!(forward_iter.last(), Some(122));

                    // Backward Behavior (The Limitation):
                    // Mathematically, this SHOULD start at 122 to match the forward grid.
                    // However, because 127 - (-128) overflows i8, we fallback to upper.
                    let mut backward_iter = interval.iter_step(step);
                    assert_eq!(backward_iter.next_back(), Some(127));
                    assert_eq!(backward_iter.next_back(), Some(117));
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

    #[test]
    fn test_contains_value_binary_search() {
        let intervals = vec![
            new_interval(10, 20),
            new_interval(30, 40),
            new_interval(50, 60),
        ];
        let domain: IntegerDomain<i32, 1> = IntegerDomain::from_sorted_vec(intervals);

        assert!(domain.contains_value(10));
        assert!(domain.contains_value(15));
        assert!(domain.contains_value(20));
        assert!(domain.contains_value(35));
        assert!(domain.contains_value(60));

        assert!(!domain.contains_value(0));
        assert!(!domain.contains_value(25));
        assert!(!domain.contains_value(45));
        assert!(!domain.contains_value(70));
    }

    #[test]
    fn test_domain_new_and_singleton() {
        let d: IntegerDomain<i32, 1> = IntegerDomain::new(1i32, 5);
        assert_eq!(d.len(), 1);
        assert!(!d.is_empty());
        assert_eq!(d.min().unwrap(), 1);
        assert_eq!(d.max().unwrap(), 5);

        let s: IntegerDomain<i32, 1> = IntegerDomain::from_singleton(7i32);
        assert_eq!(s.len(), 1);
        assert_eq!(s.min().unwrap(), 7);
        assert_eq!(s.max().unwrap(), 7);
        assert_eq!(s.is_fixed().unwrap(), true);
    }

    #[test]
    fn test_domain_empty_and_all() {
        let e: IntegerDomain<i32> = IntegerDomain::empty();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
        assert!(e.min().is_err());
        assert!(e.max().is_err());
        assert!(e.is_fixed().is_err());

        let a: IntegerDomain<i32> = IntegerDomain::all();
        assert!(!a.is_empty());
        assert_eq!(a.len(), 1);
        assert_eq!(a.min().unwrap(), i32::MIN);
        assert_eq!(a.max().unwrap(), i32::MAX);
        assert_eq!(a.is_fixed().unwrap(), false);
    }

    #[test]
    fn test_domain_from_sorted_builders() {
        let v = vec![
            new_closed_interval(1i32, 3),
            new_closed_interval(10, 12),
            new_closed_interval(20, 25),
        ];
        let d_vec: IntegerDomain<i32, 1> = IntegerDomain::from_sorted_vec(v.clone());
        assert_eq!(d_vec.len(), 3);
        assert_eq!(d_vec.min().unwrap(), 1);
        assert_eq!(d_vec.max().unwrap(), 25);

        let sv = smallvec::smallvec![
            new_closed_interval(1i32, 3),
            new_closed_interval(10, 12),
            new_closed_interval(20, 25)
        ];
        let d_sv: IntegerDomain<i32, 1> = IntegerDomain::from_sorted_smallvec(sv);
        assert_eq!(d_sv.len(), 3);
        assert_eq!(d_sv.min().unwrap(), 1);
        assert_eq!(d_sv.max().unwrap(), 25);

        let d_iter: IntegerDomain<i32, 1> = IntegerDomain::from_sorted_intervals([
            new_closed_interval(1i32, 3),
            new_closed_interval(10, 12),
            new_closed_interval(20, 25),
        ]);
        assert_eq!(d_iter.len(), 3);
        assert_eq!(d_iter.min().unwrap(), 1);
        assert_eq!(d_iter.max().unwrap(), 25);
    }

    #[test]
    fn test_domain_min_max_unchecked_and_is_fixed_unchecked() {
        let d: IntegerDomain<i32, 1> = IntegerDomain::from_sorted_intervals([
            new_closed_interval(5i32, 5),
            new_closed_interval(10, 20),
        ]);
        unsafe {
            assert_eq!(d.min_unchecked(), 5);
            assert_eq!(d.max_unchecked(), 20);
            assert_eq!(d.is_fixed_unchecked(), false);
        }

        let s: IntegerDomain<i32, 1> = IntegerDomain::from_singleton(42i32);
        unsafe {
            assert_eq!(s.min_unchecked(), 42);
            assert_eq!(s.max_unchecked(), 42);
            assert_eq!(s.is_fixed_unchecked(), true);
        }
    }

    #[test]
    fn test_domain_size_and_saturation() {
        let all_u8: IntegerDomain<u8> = IntegerDomain::all();
        assert_eq!(all_u8.size(), u8::MAX);

        let d: IntegerDomain<u8, 1> = IntegerDomain::from_sorted_intervals([
            new_closed_interval(0u8, 10),
            new_closed_interval(20, 30),
        ]);
        // [0..=10] => 11, [20..=30] => 11, sum = 22
        assert_eq!(d.size(), 22);
    }

    #[test]
    fn test_domain_intersects_cases() {
        // A: [1,5], [10,12]
        let a = IntegerDomain::from_sorted_intervals([
            new_closed_interval(1i32, 5),
            new_closed_interval(10, 12),
        ]);

        // B1: [6,9] -> disjoint (gap)
        let b1 = IntegerDomain::from_sorted_intervals([new_closed_interval(6, 9)]);
        assert_eq!(a.intersects(&b1), false);

        // B2: [5,7] -> intersects at 5 with [1,5]
        let b2 = IntegerDomain::from_sorted_intervals([new_closed_interval(5, 7)]);
        assert_eq!(a.intersects(&b2), true);

        // B3: [12,15] -> intersects at 12 with [10,12]
        let b3 = IntegerDomain::from_sorted_intervals([new_closed_interval(12, 15)]);
        assert_eq!(a.intersects(&b3), true);

        // B4: [13,14] -> disjoint and not adjacent to any interval
        let b4 = IntegerDomain::from_sorted_intervals([new_closed_interval(13, 14)]);
        assert_eq!(a.intersects(&b4), false);

        // C: empty
        let c: IntegerDomain<i32> = IntegerDomain::empty();
        assert_eq!(a.intersects(&c), false);
        assert_eq!(c.intersects(&a), false);
    }

    #[test]
    fn test_domain_contains_value_binary_search_examples() {
        // [10, 20], [30, 40], [50, 60]
        let d: IntegerDomain<i32, 1> = IntegerDomain::from_sorted_intervals([
            new_closed_interval(10i32, 20),
            new_closed_interval(30, 40),
            new_closed_interval(50, 60),
        ]);

        // Hits
        assert!(d.contains_value(10));
        assert!(d.contains_value(15));
        assert!(d.contains_value(20));
        assert!(d.contains_value(35));
        assert!(d.contains_value(60));

        // Misses
        assert!(!d.contains_value(0));
        assert!(!d.contains_value(25));
        assert!(!d.contains_value(45));
        assert!(!d.contains_value(70));
    }

    #[test]
    fn test_domain_contains_interval_cases() {
        // Domain: [1,5], [10,20]
        let d: IntegerDomain<i32, 1> = IntegerDomain::from_sorted_intervals([
            new_closed_interval(1i32, 5),
            new_closed_interval(10, 20),
        ]);

        // Fully contained within first
        assert!(d.contains_interval(new_closed_interval(2, 4)));
        // Exact match of second
        assert!(d.contains_interval(new_closed_interval(10, 20)));
        // In the gap
        assert!(!d.contains_interval(new_closed_interval(6, 9)));
        // Spanning across intervals is not contained
        assert!(!d.contains_interval(new_closed_interval(4, 11)));
        // Outside entirely
        assert!(!d.contains_interval(new_closed_interval(21, 30)));
    }

    #[test]
    fn test_domain_display_and_debug() {
        // Show: IntegerDomain([1, 5], [10, 12])
        let d = IntegerDomain::from_sorted_intervals([
            new_closed_interval(1i32, 5),
            new_closed_interval(10, 12),
        ]);
        let disp = format!("{}", d);
        let dbg = format!("{:?}", d);

        assert_eq!(disp, "IntegerDomain([1, 5], [10, 12])");
        assert_eq!(dbg, "IntegerDomain([1, 5], [10, 12])");
    }

    #[test]
    fn test_domain_len_and_is_empty() {
        let d0: IntegerDomain<i64> = IntegerDomain::empty();
        assert_eq!(d0.len(), 0);
        assert!(d0.is_empty());

        let d1: IntegerDomain<i64, 1> = IntegerDomain::from_singleton(42i64);
        assert_eq!(d1.len(), 1);
        assert!(!d1.is_empty());

        let d2: IntegerDomain<i64, 1> = IntegerDomain::from_sorted_intervals([
            new_closed_interval(0i64, 0),
            new_closed_interval(10, 20),
        ]);
        assert_eq!(d2.len(), 2);
        assert!(!d2.is_empty());
    }

    #[test]
    fn test_domain_is_fixed() {
        let empty: IntegerDomain<i32> = IntegerDomain::empty();
        assert!(empty.is_fixed().is_err());

        let single: IntegerDomain<i32, 1> = IntegerDomain::from_singleton(5i32);
        assert_eq!(single.is_fixed().unwrap(), true);

        let multi: IntegerDomain<i32, 1> = IntegerDomain::from_sorted_intervals([
            new_closed_interval(5i32, 5),
            new_closed_interval(10, 10),
        ]);
        assert_eq!(multi.is_fixed().unwrap(), false);

        let range: IntegerDomain<i32, 1> = IntegerDomain::new(1i32, 2);
        assert_eq!(range.is_fixed().unwrap(), false);
    }
}
