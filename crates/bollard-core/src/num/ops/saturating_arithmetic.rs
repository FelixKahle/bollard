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

use core::ops::{Add, Mul, Neg, Sub};

macro_rules! saturating_impl_binary_val {
    ($trait_name:ident, $method:ident, $t:ty, $src_method:ident) => {
        impl $trait_name for $t {
            #[inline(always)]
            fn $method(self, v: Self) -> Self {
                <$t>::$src_method(self, v)
            }
        }
    };
}

macro_rules! saturating_impl_unary_val {
    ($trait_name:ident, $method:ident, $t:ty, $src_method:ident) => {
        impl $trait_name for $t {
            #[inline(always)]
            fn $method(self) -> Self {
                <$t>::$src_method(self)
            }
        }
    };
}

/// Saturating addition by value (no references).
///
/// This trait provides a by-value API for saturating addition, clamping the
/// result to the numeric bounds of the type instead of overflowing. It
/// mirrors the inherent `saturating_add` on primitive integers but avoids
/// any ambiguity with reference-based trait APIs.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::saturating_arithmetic::SaturatingAddVal;
///
/// let a: u8 = 250;
/// let b: u8 = 10;
/// assert_eq!(a.saturating_add_val(b), 255); // Clamps at u8::MAX
///
/// let x: i8 = 120;
/// let y: i8 = 10;
/// assert_eq!(x.saturating_add_val(y), 127); // Clamps at i8::MAX
///
/// let m: i8 = -120;
/// let n: i8 = -20;
/// assert_eq!(m.saturating_add_val(n), -128); // Clamps at i8::MIN
/// ```
pub trait SaturatingAddVal: Sized + Add<Self, Output = Self> {
    /// Performs saturating addition by value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::num::ops::saturating_arithmetic::SaturatingAddVal;
    ///
    /// let a: u8 = 250;
    /// let b: u8 = 10;
    /// assert_eq!(a.saturating_add_val(b), 255); // Clamps at u8::MAX
    /// ```
    fn saturating_add_val(self, v: Self) -> Self;
}

saturating_impl_binary_val!(SaturatingAddVal, saturating_add_val, u8, saturating_add);
saturating_impl_binary_val!(SaturatingAddVal, saturating_add_val, u16, saturating_add);
saturating_impl_binary_val!(SaturatingAddVal, saturating_add_val, u32, saturating_add);
saturating_impl_binary_val!(SaturatingAddVal, saturating_add_val, u64, saturating_add);
saturating_impl_binary_val!(SaturatingAddVal, saturating_add_val, usize, saturating_add);
saturating_impl_binary_val!(SaturatingAddVal, saturating_add_val, u128, saturating_add);

saturating_impl_binary_val!(SaturatingAddVal, saturating_add_val, i8, saturating_add);
saturating_impl_binary_val!(SaturatingAddVal, saturating_add_val, i16, saturating_add);
saturating_impl_binary_val!(SaturatingAddVal, saturating_add_val, i32, saturating_add);
saturating_impl_binary_val!(SaturatingAddVal, saturating_add_val, i64, saturating_add);
saturating_impl_binary_val!(SaturatingAddVal, saturating_add_val, isize, saturating_add);
saturating_impl_binary_val!(SaturatingAddVal, saturating_add_val, i128, saturating_add);

/// Saturating subtraction by value (no references).
///
/// This trait provides a by-value API for saturating subtraction, clamping the
/// result to the numeric bounds of the type instead of under/overflowing.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::saturating_arithmetic::SaturatingSubVal;
///
/// let a: u8 = 5;
/// let b: u8 = 10;
/// assert_eq!(a.saturating_sub_val(b), 0); // Clamps at u8::MIN
///
/// let x: i8 = -120;
/// let y: i8 = 20;
/// assert_eq!(x.saturating_sub_val(y), -128); // Clamps at i8::MIN
///
/// let m: i8 = 120;
/// let n: i8 = -20;
/// assert_eq!(m.saturating_sub_val(n), 127); // Clamps at i8::MAX
/// ```
pub trait SaturatingSubVal: Sized + Sub<Self, Output = Self> {
    /// Performs saturating subtraction by value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::num::ops::saturating_arithmetic::SaturatingSubVal;
    ///
    /// let a: u8 = 5;
    /// let b: u8 = 10;
    /// assert_eq!(a.saturating_sub_val(b), 0); // Clamps at u8::MIN
    /// ```
    fn saturating_sub_val(self, v: Self) -> Self;
}

saturating_impl_binary_val!(SaturatingSubVal, saturating_sub_val, u8, saturating_sub);
saturating_impl_binary_val!(SaturatingSubVal, saturating_sub_val, u16, saturating_sub);
saturating_impl_binary_val!(SaturatingSubVal, saturating_sub_val, u32, saturating_sub);
saturating_impl_binary_val!(SaturatingSubVal, saturating_sub_val, u64, saturating_sub);
saturating_impl_binary_val!(SaturatingSubVal, saturating_sub_val, usize, saturating_sub);
saturating_impl_binary_val!(SaturatingSubVal, saturating_sub_val, u128, saturating_sub);

saturating_impl_binary_val!(SaturatingSubVal, saturating_sub_val, i8, saturating_sub);
saturating_impl_binary_val!(SaturatingSubVal, saturating_sub_val, i16, saturating_sub);
saturating_impl_binary_val!(SaturatingSubVal, saturating_sub_val, i32, saturating_sub);
saturating_impl_binary_val!(SaturatingSubVal, saturating_sub_val, i64, saturating_sub);
saturating_impl_binary_val!(SaturatingSubVal, saturating_sub_val, isize, saturating_sub);
saturating_impl_binary_val!(SaturatingSubVal, saturating_sub_val, i128, saturating_sub);

/// Saturating multiplication by value (no references).
///
/// This trait provides a by-value API for saturating multiplication, clamping
/// the result to the numeric bounds of the type instead of overflowing.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::saturating_arithmetic::SaturatingMulVal;
///
/// let a: u8 = 64;
/// let b: u8 = 10;
/// assert_eq!(a.saturating_mul_val(b), 255); // 640 -> clamps at u8::MAX
///
/// let x: i8 = 30;
/// let y: i8 = 10;
/// assert_eq!(x.saturating_mul_val(y), 127); // 300 -> clamps at i8::MAX
///
/// let m: i8 = -30;
/// let n: i8 = 10;
/// assert_eq!(m.saturating_mul_val(n), -128); // -300 -> clamps at i8::MIN
/// ```
pub trait SaturatingMulVal: Sized + Mul<Self, Output = Self> {
    /// Performs saturating multiplication by value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::num::ops::saturating_arithmetic::SaturatingMulVal;
    ///
    /// let a: u8 = 64;
    /// let b: u8 = 10;
    /// assert_eq!(a.saturating_mul_val(b), 255); // 640 -> clamps at u8::MAX
    /// ```
    fn saturating_mul_val(self, v: Self) -> Self;
}

saturating_impl_binary_val!(SaturatingMulVal, saturating_mul_val, u8, saturating_mul);
saturating_impl_binary_val!(SaturatingMulVal, saturating_mul_val, u16, saturating_mul);
saturating_impl_binary_val!(SaturatingMulVal, saturating_mul_val, u32, saturating_mul);
saturating_impl_binary_val!(SaturatingMulVal, saturating_mul_val, u64, saturating_mul);
saturating_impl_binary_val!(SaturatingMulVal, saturating_mul_val, usize, saturating_mul);
saturating_impl_binary_val!(SaturatingMulVal, saturating_mul_val, u128, saturating_mul);

saturating_impl_binary_val!(SaturatingMulVal, saturating_mul_val, i8, saturating_mul);
saturating_impl_binary_val!(SaturatingMulVal, saturating_mul_val, i16, saturating_mul);
saturating_impl_binary_val!(SaturatingMulVal, saturating_mul_val, i32, saturating_mul);
saturating_impl_binary_val!(SaturatingMulVal, saturating_mul_val, i64, saturating_mul);
saturating_impl_binary_val!(SaturatingMulVal, saturating_mul_val, isize, saturating_mul);
saturating_impl_binary_val!(SaturatingMulVal, saturating_mul_val, i128, saturating_mul);

/// Saturating negation by value (no references).
///
/// This trait provides a by-value API for saturating negation. For signed
/// integers, negating the minimum value would overflow; saturating negation
/// clamps to the maximum instead.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::saturating_arithmetic::SaturatingNegVal;
///
/// let a: i8 = 100;
/// assert_eq!(a.saturating_neg_val(), -100);
///
/// let b: i8 = -128;
/// assert_eq!(b.saturating_neg_val(), 127); // Clamps to i8::MAX
/// ```
pub trait SaturatingNegVal: Sized + Neg<Output = Self> {
    /// Performs saturating negation by value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::num::ops::saturating_arithmetic::SaturatingNegVal;
    /// let a: i8 = 100;
    /// assert_eq!(a.saturating_neg_val(), -100);
    /// ```
    fn saturating_neg_val(self) -> Self;
}

saturating_impl_unary_val!(SaturatingNegVal, saturating_neg_val, i8, saturating_neg);
saturating_impl_unary_val!(SaturatingNegVal, saturating_neg_val, i16, saturating_neg);
saturating_impl_unary_val!(SaturatingNegVal, saturating_neg_val, i32, saturating_neg);
saturating_impl_unary_val!(SaturatingNegVal, saturating_neg_val, i64, saturating_neg);
saturating_impl_unary_val!(SaturatingNegVal, saturating_neg_val, isize, saturating_neg);
saturating_impl_unary_val!(SaturatingNegVal, saturating_neg_val, i128, saturating_neg);

#[cfg(test)]
mod tests {
    use super::*;

    fn saturating_add_val<T: SaturatingAddVal>(a: T, b: T) -> T {
        a.saturating_add_val(b)
    }
    fn saturating_sub_val<T: SaturatingSubVal>(a: T, b: T) -> T {
        a.saturating_sub_val(b)
    }
    fn saturating_mul_val<T: SaturatingMulVal>(a: T, b: T) -> T {
        a.saturating_mul_val(b)
    }
    fn saturating_neg_val<T: SaturatingNegVal>(a: T) -> T {
        a.saturating_neg_val()
    }

    #[test]
    fn test_saturating_add_val() {
        assert_eq!(saturating_add_val(255u8, 1u8), 255u8);
        assert_eq!(saturating_add_val(127i8, 1i8), 127i8);
        assert_eq!(saturating_add_val(-128i8, -1i8), -128i8);
    }

    #[test]
    fn test_saturating_sub_val() {
        assert_eq!(saturating_sub_val(0u8, 1u8), 0u8);
        assert_eq!(saturating_sub_val(-128i8, 1i8), -128i8);
        assert_eq!(saturating_sub_val(127i8, -1i8), 127i8);
    }

    #[test]
    fn test_saturating_mul_val() {
        assert_eq!(saturating_mul_val(255u8, 2u8), 255u8);
        assert_eq!(saturating_mul_val(127i8, 2i8), 127i8);
        assert_eq!(saturating_mul_val(-128i8, 2i8), -128i8);
    }

    #[test]
    fn test_saturating_neg_val() {
        assert_eq!(saturating_neg_val(127i8), -127i8);
        assert_eq!(saturating_neg_val(-128i8), 127i8);
    }
}
