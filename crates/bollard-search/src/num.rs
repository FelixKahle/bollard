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

use bollard_core::num::{
    constants::{MinusOne, PlusOne, Zero},
    ops::{checked_arithmetic, saturating_arithmetic},
};
use num_traits::{FromPrimitive, PrimInt, Signed};

/// A trait alias for numeric types that can be used in the solver.
/// This includes integer types that support various arithmetic operations
/// with both saturating and checked semantics.
/// These are usually all signed integer types `i8`, `i16`, `i32`, `i64` and `isize`.
///
/// # Note
///
/// `i128` are intentionally excluded due to performance reasons, as
/// they are significantly slower on many platforms.
pub trait SolverNumeric:
    PrimInt
    + Signed
    + FromPrimitive
    + From<i64>
    + std::fmt::Debug
    + std::fmt::Display
    + MinusOne
    + PlusOne
    + Zero
    + saturating_arithmetic::SaturatingAddVal
    + saturating_arithmetic::SaturatingSubVal
    + saturating_arithmetic::SaturatingMulVal
    + saturating_arithmetic::SaturatingNegVal
    + checked_arithmetic::CheckedAddVal
    + checked_arithmetic::CheckedSubVal
    + checked_arithmetic::CheckedMulVal
    + checked_arithmetic::CheckedNegVal
    + checked_arithmetic::CheckedDivVal
    + checked_arithmetic::CheckedRemVal
    + checked_arithmetic::CheckedShlVal
    + checked_arithmetic::CheckedShrVal
    + Send
    + Sync
{
}
