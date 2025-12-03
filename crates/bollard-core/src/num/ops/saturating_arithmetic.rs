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

macro_rules! saturating_impl_binary {
    ($trait_name:ident, $method:ident, $t:ty) => {
        impl $trait_name for $t {
            #[inline(always)]
            fn $method(self, v: Self) -> Self {
                <$t>::$method(self, v)
            }
        }
    };
}

macro_rules! saturating_impl_unary {
    ($trait_name:ident, $method:ident, $t:ty) => {
        impl $trait_name for $t {
            #[inline(always)]
            fn $method(self) -> Self {
                <$t>::$method(self)
            }
        }
    };
}

/// A trait for types that support saturating addition.
///
/// Saturating addition clamps the result at the numeric bounds of the type
/// instead of overflowing.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::saturating_arithmetic::SaturatingAdd;
///
/// let a: u8 = 250;
/// let b: u8 = 10;
/// assert_eq!(a.saturating_add(b), 255); // Clamps at u8::MAX
///
/// let x: i8 = 120;
/// let y: i8 = 10;
/// assert_eq!(x.saturating_add(y), 127); // Clamps at i8::MAX
///
/// let m: i8 = -120;
/// let n: i8 = -20;
/// assert_eq!(m.saturating_add(n), -128); // Clamps at i8::MIN
/// ```
pub trait SaturatingAdd: Sized + Add<Self, Output = Self> {
    /// Performs saturating addition.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::num::ops::saturating_arithmetic::SaturatingAdd;
    ///
    /// let a: u8 = 250;
    /// let b: u8 = 10;
    /// assert_eq!(a.saturating_add(b), 255); // Clamps at u8::MAX
    ///
    /// let x: i8 = 120;
    /// let y: i8 = 10;
    /// assert_eq!(x.saturating_add(y), 127); // Clamps at i8::MAX
    ///
    /// let m: i8 = -120;
    /// let n: i8 = -20;
    /// assert_eq!(m.saturating_add(n), -128); // Clamps at i8::MIN
    /// ```
    fn saturating_add(self, v: Self) -> Self;
}

saturating_impl_binary!(SaturatingAdd, saturating_add, u8);
saturating_impl_binary!(SaturatingAdd, saturating_add, u16);
saturating_impl_binary!(SaturatingAdd, saturating_add, u32);
saturating_impl_binary!(SaturatingAdd, saturating_add, u64);
saturating_impl_binary!(SaturatingAdd, saturating_add, usize);
saturating_impl_binary!(SaturatingAdd, saturating_add, u128);

saturating_impl_binary!(SaturatingAdd, saturating_add, i8);
saturating_impl_binary!(SaturatingAdd, saturating_add, i16);
saturating_impl_binary!(SaturatingAdd, saturating_add, i32);
saturating_impl_binary!(SaturatingAdd, saturating_add, i64);
saturating_impl_binary!(SaturatingAdd, saturating_add, isize);
saturating_impl_binary!(SaturatingAdd, saturating_add, i128);

/// A trait for types that support saturating subtraction.
///
/// Saturating subtraction clamps the result at the numeric bounds of the type
/// instead of underflowing/overflowing.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::saturating_arithmetic::SaturatingSub;
///
/// let a: u8 = 5;
/// let b: u8 = 10;
/// assert_eq!(a.saturating_sub(b), 0); // Clamps at u8::MIN
///
/// let x: i8 = -120;
/// let y: i8 = 20;
/// assert_eq!(x.saturating_sub(y), -128); // Clamps at i8::MIN
///
/// let m: i8 = 120;
/// let n: i8 = -20;
/// assert_eq!(m.saturating_sub(n), 127); // Clamps at i8::MAX
/// ```
pub trait SaturatingSub: Sized + Sub<Self, Output = Self> {
    /// Performs saturating subtraction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::num::ops::saturating_arithmetic::SaturatingSub;
    ///
    /// let a: u8 = 5;
    /// let b: u8 = 10;
    /// assert_eq!(a.saturating_sub(b), 0); // Clamps at u8::MIN
    ///
    /// let x: i8 = -120;
    /// let y: i8 = 20;
    /// assert_eq!(x.saturating_sub(y), -128); // Clamps at i8::MIN
    ///
    /// let m: i8 = 120;
    /// let n: i8 = -20;
    /// assert_eq!(m.saturating_sub(n), 127); // Clamps at i8::MAX
    /// ```
    fn saturating_sub(self, v: Self) -> Self;
}

saturating_impl_binary!(SaturatingSub, saturating_sub, u8);
saturating_impl_binary!(SaturatingSub, saturating_sub, u16);
saturating_impl_binary!(SaturatingSub, saturating_sub, u32);
saturating_impl_binary!(SaturatingSub, saturating_sub, u64);
saturating_impl_binary!(SaturatingSub, saturating_sub, usize);
saturating_impl_binary!(SaturatingSub, saturating_sub, u128);

saturating_impl_binary!(SaturatingSub, saturating_sub, i8);
saturating_impl_binary!(SaturatingSub, saturating_sub, i16);
saturating_impl_binary!(SaturatingSub, saturating_sub, i32);
saturating_impl_binary!(SaturatingSub, saturating_sub, i64);
saturating_impl_binary!(SaturatingSub, saturating_sub, isize);
saturating_impl_binary!(SaturatingSub, saturating_sub, i128);

/// A trait for types that support saturating multiplication.
///
/// Saturating multiplication clamps the result at the numeric bounds of the type
/// instead of overflowing.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::saturating_arithmetic::SaturatingMul;
///
/// let a: u8 = 64;
/// let b: u8 = 10;
/// assert_eq!(a.saturating_mul(b), 255); // 640 -> clamps at u8::MAX
///
/// let x: i8 = 30;
/// let y: i8 = 10;
/// assert_eq!(x.saturating_mul(y), 127); // 300 -> clamps at i8::MAX
///
/// let m: i8 = -30;
/// let n: i8 = 10;
/// assert_eq!(m.saturating_mul(n), -128); // -300 -> clamps at i8::MIN
/// ```
pub trait SaturatingMul: Sized + Mul<Self, Output = Self> {
    /// Performs saturating multiplication.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::num::ops::saturating_arithmetic::SaturatingMul;
    ///
    /// let a: u8 = 64;
    /// let b: u8 = 10;
    /// assert_eq!(a.saturating_mul(b), 255); // 640 -> clamps at u8::MAX
    ///
    /// let x: i8 = 30;
    /// let y: i8 = 10;
    /// assert_eq!(x.saturating_mul(y), 127); // 300 -> clamps at i8::MAX
    ///
    /// let m: i8 = -30;
    /// let n: i8 = 10;
    /// assert_eq!(m.saturating_mul(n), -128); // -300 -> clamps at i8::MIN
    /// ```
    fn saturating_mul(self, v: Self) -> Self;
}

saturating_impl_binary!(SaturatingMul, saturating_mul, u8);
saturating_impl_binary!(SaturatingMul, saturating_mul, u16);
saturating_impl_binary!(SaturatingMul, saturating_mul, u32);
saturating_impl_binary!(SaturatingMul, saturating_mul, u64);
saturating_impl_binary!(SaturatingMul, saturating_mul, usize);
saturating_impl_binary!(SaturatingMul, saturating_mul, u128);

saturating_impl_binary!(SaturatingMul, saturating_mul, i8);
saturating_impl_binary!(SaturatingMul, saturating_mul, i16);
saturating_impl_binary!(SaturatingMul, saturating_mul, i32);
saturating_impl_binary!(SaturatingMul, saturating_mul, i64);
saturating_impl_binary!(SaturatingMul, saturating_mul, isize);
saturating_impl_binary!(SaturatingMul, saturating_mul, i128);

/// A trait for types that support saturating negation.
///
/// Saturating negation clamps the result at the numeric bounds of the type.
/// For signed integers, negating the minimum value would overflow; saturating
/// negation clamps to the maximum instead.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::saturating_arithmetic::SaturatingNeg;
///
/// let a: i8 = 100;
/// assert_eq!(a.saturating_neg(), -100);
///
/// let b: i8 = -128;
/// assert_eq!(b.saturating_neg(), 127); // Clamps to i8::MAX
/// ```
pub trait SaturatingNeg: Sized + Neg<Output = Self> {
    /// Performs saturating negation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::num::ops::saturating_arithmetic::SaturatingNeg;
    /// let a: i8 = 100;
    /// assert_eq!(a.saturating_neg(), -100);
    ///
    /// let b: i8 = -128;
    /// assert_eq!(b.saturating_neg(), 127); // Clamps to i8::MAX
    /// ```
    fn saturating_neg(self) -> Self;
}

saturating_impl_unary!(SaturatingNeg, saturating_neg, i8);
saturating_impl_unary!(SaturatingNeg, saturating_neg, i16);
saturating_impl_unary!(SaturatingNeg, saturating_neg, i32);
saturating_impl_unary!(SaturatingNeg, saturating_neg, i64);
saturating_impl_unary!(SaturatingNeg, saturating_neg, isize);
saturating_impl_unary!(SaturatingNeg, saturating_neg, i128);

#[test]
fn test_saturating_traits() {
    fn saturating_add<T: SaturatingAdd>(a: T, b: T) -> T {
        a.saturating_add(b)
    }
    fn saturating_sub<T: SaturatingSub>(a: T, b: T) -> T {
        a.saturating_sub(b)
    }
    fn saturating_mul<T: SaturatingMul>(a: T, b: T) -> T {
        a.saturating_mul(b)
    }
    fn saturating_neg<T: SaturatingNeg>(a: T) -> T {
        a.saturating_neg()
    }
    assert_eq!(saturating_add(255, 1), 255u8);
    assert_eq!(saturating_add(127, 1), 127i8);
    assert_eq!(saturating_add(-128, -1), -128i8);
    assert_eq!(saturating_sub(0, 1), 0u8);
    assert_eq!(saturating_sub(-128, 1), -128i8);
    assert_eq!(saturating_sub(127, -1), 127i8);
    assert_eq!(saturating_mul(255, 2), 255u8);
    assert_eq!(saturating_mul(127, 2), 127i8);
    assert_eq!(saturating_mul(-128, 2), -128i8);
    assert_eq!(saturating_neg(127i8), -127i8);
}
