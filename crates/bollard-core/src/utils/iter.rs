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

//! # Optional Iterator Wrapper
//!
//! Utilities for composing iterators that may or may not exist. `MaybeIter<I>`
//! wraps an `Option<I>` and exposes a standard iterator interface, avoiding the
//! need to sprinkle `Option<Iterator>` branching throughout your code.
//!
//! ## Motivation
//!
//! In scheduling and data-processing pipelines, sources and views can be
//! conditionally present. `MaybeIter` lets you unify iteration logic:
//! when the inner iterator is `None`, it behaves as an empty iterator;
//! when it is `Some(I)`, it forwards all iterator behavior transparently.
//!
//! ## Highlights
//!
//! - Implements `Iterator` with correct `size_hint` for `None` (`(0, Some(0))`).
//! - Supports `DoubleEndedIterator`, `ExactSizeIterator`, and `FusedIterator`
//!   when the wrapped iterator implements them.
//! - Clone-friendly when the inner iterator is `Clone`.
//!
//! ## Usage
//!
//! ```rust
//! use bollard_core::utils::iter::MaybeIter;
//!
//! let present = MaybeIter::new(Some(vec![1, 2, 3].into_iter()));
//! let absent: MaybeIter<std::vec::IntoIter<i32>> = MaybeIter::new(None);
//!
//! assert_eq!(present.collect::<Vec<_>>(), vec![1, 2, 3]);
//! assert_eq!(absent.collect::<Vec<_>>(), Vec::<i32>::new());
//! ```

use std::iter::FusedIterator;

/// An iterator that may or may not be present.
///
/// This is useful for cases where an iterator is optional,
/// and you want to avoid dealing with `Option<Iterator>` directly.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::utils::iter::MaybeIter;
///
/// let some_iter = MaybeIter::new(Some(vec![1, 2, 3].into_iter()));
/// let none_iter: MaybeIter<std::vec::IntoIter<i32>> = MaybeIter::new(None);
///
/// assert_eq!(some_iter.collect::<Vec<_>>(), vec![1, 2, 3]);
/// assert_eq!(none_iter.collect::<Vec<_>>(), vec![]);
/// ```
#[derive(Debug, Clone)]
pub struct MaybeIter<T> {
    inner: Option<T>,
}

impl<T> MaybeIter<T> {
    /// Creates a new `MaybeIter` from an optional iterator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::utils::iter::MaybeIter;
    ///
    /// let some_iter = MaybeIter::new(Some(vec![1, 2, 3].into_iter()));
    /// let none_iter: MaybeIter<std::vec::IntoIter<i32>> = MaybeIter::new(None);
    ///
    /// assert_eq!(some_iter.collect::<Vec<_>>(), vec![1, 2, 3]);
    /// assert_eq!(none_iter.collect::<Vec<_>>(), vec![]);
    /// ```
    #[inline]
    pub fn new(inner: Option<T>) -> Self {
        Self { inner }
    }
}

impl<I: Iterator> Iterator for MaybeIter<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.as_mut()?.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.as_ref().map_or((0, Some(0)), |i| i.size_hint())
    }
}

impl<I> FusedIterator for MaybeIter<I> where I: FusedIterator {}

impl<I> DoubleEndedIterator for MaybeIter<I>
where
    I: DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.as_mut()?.next_back()
    }
}

impl<I> ExactSizeIterator for MaybeIter<I>
where
    I: ExactSizeIterator,
{
    #[inline]
    fn len(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter::FusedIterator;

    #[test]
    fn test_maybe_iter_with_some() {
        let data = vec![10, 20, 30];
        // Create a MaybeIter wrapping a standard Vec iterator
        let mut iter = MaybeIter::new(Some(data.into_iter()));

        assert_eq!(iter.next(), Some(10));
        assert_eq!(iter.next(), Some(20));
        assert_eq!(iter.next(), Some(30));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_maybe_iter_with_none() {
        // Explicitly define the type as MaybeIter wrapping a Vec iterator
        let mut iter: MaybeIter<std::vec::IntoIter<i32>> = MaybeIter::new(None);

        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_size_hint() {
        // Case 1: Inner exists
        let data = vec![1, 2, 3];
        let iter = MaybeIter::new(Some(data.into_iter()));
        assert_eq!(iter.size_hint(), (3, Some(3)));

        // Case 2: Inner is None
        let empty_iter: MaybeIter<std::vec::IntoIter<i32>> = MaybeIter::new(None);
        assert_eq!(empty_iter.size_hint(), (0, Some(0)));
    }

    #[test]
    fn test_exact_size_iterator() {
        // Case 1: Inner exists
        let data = vec![1, 2, 3, 4, 5];
        let mut iter = MaybeIter::new(Some(data.into_iter()));

        assert_eq!(iter.len(), 5);
        iter.next();
        assert_eq!(iter.len(), 4);

        // Case 2: Inner is None
        let empty_iter: MaybeIter<std::vec::IntoIter<i32>> = MaybeIter::new(None);
        assert_eq!(empty_iter.len(), 0);
    }

    #[test]
    fn test_fused_iterator() {
        // FusedIterator guarantees that once None is returned, it returns None forever.
        // While standard Iterators usually do this, FusedIterator enforces it in the type system.
        let data = vec![1];
        let mut iter = MaybeIter::new(Some(data.into_iter()));

        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);

        // Ensure the compiler accepts it where a FusedIterator is required
        fn assert_fused<I: FusedIterator>(_: I) {}
        assert_fused(iter);
    }

    #[test]
    fn test_clone() {
        // Note: The inner iterator must be Clone for MaybeIter to be Clone.
        // vec::IntoIter is not Clone, but slice::Iter is.
        let data = [1, 2, 3];
        let iter = MaybeIter::new(Some(data.iter()));
        let mut cloned_iter = iter.clone();

        assert_eq!(cloned_iter.next(), Some(&1));
        assert_eq!(cloned_iter.next(), Some(&2));
        assert_eq!(cloned_iter.next(), Some(&3));
        assert_eq!(cloned_iter.next(), None);
    }

    #[test]
    fn test_composition() {
        // Ensure it works nicely with standard iterator adaptors
        let data = vec![1, 2, 3];
        let iter = MaybeIter::new(Some(data.into_iter()));

        let result: Vec<i32> = iter.map(|x| x * 2).collect();
        assert_eq!(result, [2, 4, 6]);
    }
}
