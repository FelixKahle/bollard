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

use bollard_core::num::constants;
use num_traits::Signed;

/// A processing time that may be absent.
///
/// Instead of using `Option<T>`, this type uses a sentinel encoding to avoid
/// the additional discriminant that `Option` typically introduces for integer
/// types. In hot loops and dense collections, keeping the value to a single
/// machine word can improve cache locality and reduce memory traffic.
///
/// Encoding:
/// - Non-negative values (>= 0) represent a concrete processing time.
/// - Negative values (<= -1) are reserved to indicate absence.
///
/// This convention assumes valid processing times are non-negative. If negative
/// values are meaningful in your domain, use `Option<T>` instead.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProcessingTime<T>(T)
where
    T: Signed;

impl<T> ProcessingTime<T>
where
    T: Copy + Signed + constants::MinusOne,
{
    const NONE_SENTINEL: T = T::MINUS_ONE;

    /// Creates a `ProcessingTime` from an `Option<T>`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::time::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(5i32));
    /// assert!(some_time.is_some());
    /// assert_eq!(some_time.raw(), 5);
    /// ```
    #[inline]
    pub fn from_option(value: Option<T>) -> Self {
        match value {
            Some(v) => ProcessingTime(v),
            None => ProcessingTime(Self::NONE_SENTINEL),
        }
    }

    /// Creates a `ProcessingTime` from a raw value without checking for sentinel.
    /// If you pass a negative value, it will be treated as `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::time::ProcessingTime;
    ///
    /// let time = ProcessingTime::from_raw(10i32);
    /// assert!(time.is_some());
    /// assert_eq!(time.raw(), 10);
    /// ```
    #[inline]
    pub const fn from_raw(value: T) -> Self {
        ProcessingTime(value)
    }

    /// Creates a `ProcessingTime` representing `Some`.
    ///
    /// # Panics
    ///
    /// This function will panic if the provided value is negative.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::time::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::some(5i32);
    /// assert!(some_time.is_some());
    /// assert_eq!(some_time.raw(), 5);
    /// ```
    pub fn some(value: T) -> Self
    where
        T: PartialOrd + std::fmt::Display,
    {
        assert!(
            value > Self::NONE_SENTINEL,
            "called `ProcessingTime::some` with a negative value: {}",
            value
        );

        ProcessingTime(value)
    }

    /// Creates a `ProcessingTime` representing `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::time::ProcessingTime;
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// assert!(none_time.is_none());
    /// ```
    #[inline]
    pub fn none() -> Self {
        ProcessingTime(Self::NONE_SENTINEL)
    }

    /// Checks if the `ProcessingTime` represents `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::time::ProcessingTime;
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// assert!(none_time.is_none());
    /// ```
    #[inline]
    pub fn is_none(&self) -> bool
    where
        T: PartialOrd,
    {
        self.0 <= Self::NONE_SENTINEL
    }

    /// Checks if the `ProcessingTime` represents `Some`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::time::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(3i32));
    /// assert!(some_time.is_some());
    /// ```
    #[inline]
    pub fn is_some(&self) -> bool
    where
        T: PartialOrd,
    {
        !self.is_none()
    }

    /// Returns the raw value, including sentinel if present.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::time::ProcessingTime;
    ///
    /// let time = ProcessingTime::from_option(Some(7i32));
    /// assert_eq!(time.raw(), 7);
    /// ```
    #[inline]
    pub fn raw(&self) -> T {
        self.0
    }

    /// Converts the `ProcessingTime` back into an `Option<T>`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::time::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(4i32));
    /// assert_eq!(some_time.into_option(), Some(4));
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// assert_eq!(none_time.into_option(), None);
    /// ```
    #[inline]
    pub fn into_option(&self) -> Option<T>
    where
        T: PartialOrd,
    {
        if self.is_none() { None } else { Some(self.0) }
    }

    /// Unwraps the `ProcessingTime`, panicking if it is `None`.
    ///
    /// # Panics
    ///
    /// This function will panic if called on a `ProcessingTime` that represents `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::time::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(6i32));
    /// assert_eq!(some_time.unwrap(), 6);
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// // The following line would panic:
    /// // none_time.unwrap();
    /// ```
    pub fn unwrap(&self) -> T
    where
        T: PartialOrd,
    {
        if self.is_none() {
            panic!("called `ProcessingTime::unwrap()` on a `None` value")
        }
        self.0
    }

    /// Unwraps the `ProcessingTime` without checking for `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::time::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(6i32));
    /// assert_eq!(some_time.unwrap_unchecked(), 6);
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// // The following line will NOT panic, but yields an invalid value:
    /// // none_time.unwrap_unchecked();
    /// ```
    pub fn unwrap_unchecked(&self) -> T {
        self.0
    }

    /// Unwraps the `ProcessingTime`, returning a default value if it is `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::time::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(8i32));
    /// assert_eq!(some_time.unwrap_or(0), 8);
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// assert_eq!(none_time.unwrap_or(0), 0);
    /// ```
    #[inline]
    pub fn unwrap_or(&self, default: T) -> T
    where
        T: PartialOrd,
    {
        if self.is_none() { default } else { self.0 }
    }

    /// Unwraps the `ProcessingTime`, computing a default value if it is `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::time::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(9i32));
    /// assert_eq!(some_time.unwrap_or_else(|| 1 + 1), 9);
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// assert_eq!(none_time.unwrap_or_else(|| 1 + 1), 2);
    /// ```
    #[inline]
    pub fn unwrap_or_else<F>(&self, f: F) -> T
    where
        T: PartialOrd,
        F: FnOnce() -> T,
    {
        if self.is_none() { f() } else { self.0 }
    }
}

impl<T> std::fmt::Debug for ProcessingTime<T>
where
    T: Copy + Signed + PartialOrd + constants::MinusOne + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_none() {
            write!(f, "ProcessingTime(None)")
        } else {
            write!(f, "ProcessingTime(Some({:?}))", self.0)
        }
    }
}

impl<T> std::fmt::Display for ProcessingTime<T>
where
    T: Copy + Signed + PartialOrd + constants::MinusOne + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_none() {
            write!(f, "ProcessingTime(None)")
        } else {
            write!(f, "ProcessingTime({})", self.0)
        }
    }
}

impl<T> From<Option<T>> for ProcessingTime<T>
where
    T: Copy + Signed + constants::MinusOne,
{
    #[inline]
    fn from(value: Option<T>) -> Self {
        ProcessingTime::from_option(value)
    }
}

impl<T> From<ProcessingTime<T>> for Option<T>
where
    T: Copy + Signed + PartialOrd + constants::MinusOne,
{
    #[inline]
    fn from(val: ProcessingTime<T>) -> Self {
        val.into_option()
    }
}
