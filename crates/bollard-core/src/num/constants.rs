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

//! # Integer Constant Traits
//!
//! Compile-time constants for common numeric sentinel values across integer types.
//! These traits provide a uniform way to access `-1`, `0`, and `+1` as associated
//! constants on types that support them, enabling concise and generic code without
//! littering call sites with type-specific literals.
//!
//! ## Motivation
//!
//! In numeric and indexing-heavy code, sentinel values like `-1`, `0`, and `+1` are
//! frequently used for offsets, markers, and normalization. Rather than hard-coding
//! them for each integer type, these traits expose self-describing, type-checked
//! constants that improve readability and reduce mistakes.
//!
//! ## Provided Traits
//!
//! - `MinusOne` — exposes `MINUS_ONE` for signed integers.
//! - `Zero` — exposes `ZERO` for all integer primitives.
//! - `PlusOne` — exposes `PLUS_ONE` for all integer primitives.
//!
//! All core integer primitives implement the applicable traits.
//!
//! ## Usage
//!
//! ```rust
//! use bollard_core::num::constants::{MinusOne, Zero, PlusOne};
//!
//! fn step_forward<T: PlusOne>(x: T) -> T where T: core::ops::Add<Output = T> {
//!     x + T::PLUS_ONE
//! }
//!
//! fn is_sentinel<T: MinusOne + PartialEq>(x: T) -> bool {
//!     x == T::MINUS_ONE
//! }
//!
//! fn reset<T: Zero>() -> T { T::ZERO }
//! ```

/// A trait for integer types that have a constant representing -1.
pub trait MinusOne {
    /// The constant representing -1 for the implementing type.
    const MINUS_ONE: Self;
}

/// A trait for integer types that have a constant representing +1.
pub trait PlusOne {
    /// The constant representing +1 for the implementing type.
    const PLUS_ONE: Self;
}

/// A trait for integer types that have a constant representing 0.
pub trait Zero {
    /// The constant representing 0 for the implementing type.
    const ZERO: Self;
}

macro_rules! impl_const_for {
    ($trait_name:ident, $const_name:ident, $value:expr, $t:ty) => {
        impl $trait_name for $t {
            const $const_name: Self = $value;
        }
    };
}

macro_rules! impl_minus_one_for {
    ($t:ty) => {
        impl_const_for!(MinusOne, MINUS_ONE, -1, $t);
    };
}

macro_rules! impl_plus_one_for {
    ($t:ty) => {
        impl_const_for!(PlusOne, PLUS_ONE, 1, $t);
    };
}

macro_rules! impl_zero_for {
    ($t:ty) => {
        impl_const_for!(Zero, ZERO, 0, $t);
    };
}

impl_minus_one_for!(i8);
impl_minus_one_for!(i16);
impl_minus_one_for!(i32);
impl_minus_one_for!(i64);
impl_minus_one_for!(i128);
impl_minus_one_for!(isize);

impl_plus_one_for!(i8);
impl_plus_one_for!(u8);
impl_plus_one_for!(i16);
impl_plus_one_for!(u16);
impl_plus_one_for!(i32);
impl_plus_one_for!(u32);
impl_plus_one_for!(i64);
impl_plus_one_for!(u64);
impl_plus_one_for!(i128);
impl_plus_one_for!(u128);
impl_plus_one_for!(isize);
impl_plus_one_for!(usize);

impl_zero_for!(i8);
impl_zero_for!(u8);
impl_zero_for!(i16);
impl_zero_for!(u16);
impl_zero_for!(i32);
impl_zero_for!(u32);
impl_zero_for!(i64);
impl_zero_for!(u64);
impl_zero_for!(i128);
impl_zero_for!(u128);
impl_zero_for!(isize);
impl_zero_for!(usize);
