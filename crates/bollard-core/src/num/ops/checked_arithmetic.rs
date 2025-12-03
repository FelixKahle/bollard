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

use core::ops::{Add, Div, Mul, Rem, Shl, Shr, Sub};

/// A trait for types that support checked addition.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::checked_arithmetic::CheckedAdd;
/// let a: u8 = 200;
/// let b: u8 = 100;
/// assert_eq!(a.checked_add(b), None); // Overflow occurs
/// let c: u8 = 50;
/// assert_eq!(a.checked_add(c), Some(250)); // No overflow
/// ```
pub trait CheckedAdd: Sized + Add<Self, Output = Self> {
    /// Performs checked addition, returning `None` if overflow occurs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::num::ops::checked_arithmetic::CheckedAdd;
    /// let a: u8 = 200;
    /// let b: u8 = 100;
    /// assert_eq!(a.checked_add(b), None); // Overflow occurs
    /// let c: u8 = 50;
    /// assert_eq!(a.checked_add(c), Some(250)); // No overflow
    /// ```
    fn checked_add(self, v: Self) -> Option<Self>;
}

macro_rules! checked_impl {
    ($trait_name:ident, $method:ident, $t:ty) => {
        impl $trait_name for $t {
            #[inline(always)]
            fn $method(self, v: $t) -> Option<$t> {
                <$t>::$method(self, v)
            }
        }
    };
}

checked_impl!(CheckedAdd, checked_add, u8);
checked_impl!(CheckedAdd, checked_add, u16);
checked_impl!(CheckedAdd, checked_add, u32);
checked_impl!(CheckedAdd, checked_add, u64);
checked_impl!(CheckedAdd, checked_add, usize);
checked_impl!(CheckedAdd, checked_add, u128);

checked_impl!(CheckedAdd, checked_add, i8);
checked_impl!(CheckedAdd, checked_add, i16);
checked_impl!(CheckedAdd, checked_add, i32);
checked_impl!(CheckedAdd, checked_add, i64);
checked_impl!(CheckedAdd, checked_add, isize);
checked_impl!(CheckedAdd, checked_add, i128);

/// A trait for types that support checked subtraction.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::checked_arithmetic::CheckedSub;
///
/// let a: u8 = 50;
/// let b: u8 = 100;
/// assert_eq!(a.checked_sub(b), None); // Underflow occurs
/// let c: u8 = 20;
/// assert_eq!(a.checked_sub(c), Some(30)); // No underflow
/// ```
pub trait CheckedSub: Sized + Sub<Self, Output = Self> {
    /// Performs checked subtraction, returning `None` if underflow occurs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::num::ops::checked_arithmetic::CheckedSub;
    ///
    /// let a: u8 = 50;
    /// let b: u8 = 100;
    /// assert_eq!(a.checked_sub(b), None); // Underflow occurs
    /// let c: u8 = 20;
    /// assert_eq!(a.checked_sub(c), Some(30)); // No underflow
    /// ```
    fn checked_sub(self, v: Self) -> Option<Self>;
}

checked_impl!(CheckedSub, checked_sub, u8);
checked_impl!(CheckedSub, checked_sub, u16);
checked_impl!(CheckedSub, checked_sub, u32);
checked_impl!(CheckedSub, checked_sub, u64);
checked_impl!(CheckedSub, checked_sub, usize);
checked_impl!(CheckedSub, checked_sub, u128);

checked_impl!(CheckedSub, checked_sub, i8);
checked_impl!(CheckedSub, checked_sub, i16);
checked_impl!(CheckedSub, checked_sub, i32);
checked_impl!(CheckedSub, checked_sub, i64);
checked_impl!(CheckedSub, checked_sub, isize);
checked_impl!(CheckedSub, checked_sub, i128);

/// A trait for types that support checked multiplication.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::checked_arithmetic::CheckedMul;
///
/// let a: u8 = 20;
/// let b: u8 = 10;
/// assert_eq!(a.checked_mul(b), Some(200)); // No overflow
/// let c: u8 = 20;
/// assert_eq!(a.checked_mul(c), None); // Overflow occurs (20*20 = 400 > 255)
/// ```
pub trait CheckedMul: Sized + Mul<Self, Output = Self> {
    /// Performs checked multiplication, returning `None` if overflow occurs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::num::ops::checked_arithmetic::CheckedMul;
    ///
    /// let a: u8 = 20;
    /// let b: u8 = 10;
    /// assert_eq!(a.checked_mul(b), Some(200)); // No overflow
    /// let c: u8 = 20;
    /// assert_eq!(a.checked_mul(c), None); // Overflow occurs (20*20 = 400 > 255)
    /// ```
    fn checked_mul(self, v: Self) -> Option<Self>;
}

checked_impl!(CheckedMul, checked_mul, u8);
checked_impl!(CheckedMul, checked_mul, u16);
checked_impl!(CheckedMul, checked_mul, u32);
checked_impl!(CheckedMul, checked_mul, u64);
checked_impl!(CheckedMul, checked_mul, usize);
checked_impl!(CheckedMul, checked_mul, u128);

checked_impl!(CheckedMul, checked_mul, i8);
checked_impl!(CheckedMul, checked_mul, i16);
checked_impl!(CheckedMul, checked_mul, i32);
checked_impl!(CheckedMul, checked_mul, i64);
checked_impl!(CheckedMul, checked_mul, isize);
checked_impl!(CheckedMul, checked_mul, i128);

/// A trait for types that support checked division.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::checked_arithmetic::CheckedDiv;
///
/// let a: u8 = 100;
/// let b: u8 = 0;
/// assert_eq!(a.checked_div(b), None); // Division by zero
/// let c: u8 = 4;
/// assert_eq!(a.checked_div(c), Some(25)); // No division by zero
/// ```
pub trait CheckedDiv: Sized + Div<Self, Output = Self> {
    /// Performs checked division, returning `None` if division by zero occurs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::num::ops::checked_arithmetic::CheckedDiv;
    ///
    /// let a: u8 = 100;
    /// let b: u8 = 0;
    /// assert_eq!(a.checked_div(b), None); // Division by zero
    /// let c: u8 = 4;
    /// assert_eq!(a.checked_div(c), Some(25)); // No division by zero
    /// ```
    fn checked_div(self, v: Self) -> Option<Self>;
}

checked_impl!(CheckedDiv, checked_div, u8);
checked_impl!(CheckedDiv, checked_div, u16);
checked_impl!(CheckedDiv, checked_div, u32);
checked_impl!(CheckedDiv, checked_div, u64);
checked_impl!(CheckedDiv, checked_div, usize);
checked_impl!(CheckedDiv, checked_div, u128);

checked_impl!(CheckedDiv, checked_div, i8);
checked_impl!(CheckedDiv, checked_div, i16);
checked_impl!(CheckedDiv, checked_div, i32);
checked_impl!(CheckedDiv, checked_div, i64);
checked_impl!(CheckedDiv, checked_div, isize);
checked_impl!(CheckedDiv, checked_div, i128);

/// A trait for types that support checked remainder.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::checked_arithmetic::CheckedRem;
///
/// let a: u8 = 10;
/// let b: u8 = 0;
/// assert_eq!(a.checked_rem(b), None); // Division by zero
/// let c: u8 = 3;
/// assert_eq!(a.checked_rem(c), Some(1)); // No division by zero
/// ```
pub trait CheckedRem: Sized + Rem<Self, Output = Self> {
    /// Performs checked remainder, returning `None` if division by zero occurs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::num::ops::checked_arithmetic::CheckedRem;
    ///
    /// let a: u8 = 10;
    /// let b: u8 = 0;
    /// assert_eq!(a.checked_rem(b), None); // Division by zero
    /// let c: u8 = 3;
    /// assert_eq!(a.checked_rem(c), Some(1)); // No division by zero
    /// ```
    fn checked_rem(self, v: Self) -> Option<Self>;
}

checked_impl!(CheckedRem, checked_rem, u8);
checked_impl!(CheckedRem, checked_rem, u16);
checked_impl!(CheckedRem, checked_rem, u32);
checked_impl!(CheckedRem, checked_rem, u64);
checked_impl!(CheckedRem, checked_rem, usize);
checked_impl!(CheckedRem, checked_rem, u128);

checked_impl!(CheckedRem, checked_rem, i8);
checked_impl!(CheckedRem, checked_rem, i16);
checked_impl!(CheckedRem, checked_rem, i32);
checked_impl!(CheckedRem, checked_rem, i64);
checked_impl!(CheckedRem, checked_rem, isize);
checked_impl!(CheckedRem, checked_rem, i128);

macro_rules! checked_impl_unary {
    ($trait_name:ident, $method:ident, $t:ty) => {
        impl $trait_name for $t {
            #[inline(always)]
            fn $method(self) -> Option<$t> {
                <$t>::$method(self)
            }
        }
    };
}

/// A trait for types that support checked negation.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::checked_arithmetic::CheckedNeg;
///
/// let a: i8 = -128;
/// assert_eq!(a.checked_neg(), None); // Overflow occurs
/// let b: i8 = 100;
/// assert_eq!(b.checked_neg(), Some(-100)); // No overflow
/// ```
pub trait CheckedNeg: Sized {
    /// Performs checked negation, returning `None` if overflow occurs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::num::ops::checked_arithmetic::CheckedNeg;
    ///
    /// let a: i8 = -128;
    /// assert_eq!(a.checked_neg(), None); // Overflow occurs
    /// let b: i8 = 100;
    /// assert_eq!(b.checked_neg(), Some(-100)); // No overflow
    /// ```
    fn checked_neg(self) -> Option<Self>;
}

checked_impl_unary!(CheckedNeg, checked_neg, u8);
checked_impl_unary!(CheckedNeg, checked_neg, u16);
checked_impl_unary!(CheckedNeg, checked_neg, u32);
checked_impl_unary!(CheckedNeg, checked_neg, u64);
checked_impl_unary!(CheckedNeg, checked_neg, usize);
checked_impl_unary!(CheckedNeg, checked_neg, u128);

checked_impl_unary!(CheckedNeg, checked_neg, i8);
checked_impl_unary!(CheckedNeg, checked_neg, i16);
checked_impl_unary!(CheckedNeg, checked_neg, i32);
checked_impl_unary!(CheckedNeg, checked_neg, i64);
checked_impl_unary!(CheckedNeg, checked_neg, isize);
checked_impl_unary!(CheckedNeg, checked_neg, i128);

/// A trait for types that support checked left shift.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::checked_arithmetic::CheckedShl;
///
/// let a: u8 = 1;
/// let b: u32 = 8;
/// assert_eq!(a.checked_shl(b), None); // Overflow occurs
/// let c: u32 = 3;
/// assert_eq!(a.checked_shl(c), Some(8)); // No overflow
/// ```
pub trait CheckedShl: Sized + Shl<u32, Output = Self> {
    /// Performs checked left shift, returning `None` if overflow occurs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::num::ops::checked_arithmetic::CheckedShl;
    ///
    /// let a: u8 = 1;
    /// let b: u32 = 8;
    /// assert_eq!(a.checked_shl(b), None); // Overflow occurs
    /// let c: u32 = 3;
    /// assert_eq!(a.checked_shl(c), Some(8)); // No overflow
    /// ```
    fn checked_shl(self, rhs: u32) -> Option<Self>;
}

macro_rules! checked_shift_impl {
    ($trait_name:ident, $method:ident, $t:ty) => {
        impl $trait_name for $t {
            #[inline(always)]
            fn $method(self, rhs: u32) -> Option<$t> {
                <$t>::$method(self, rhs)
            }
        }
    };
}

checked_shift_impl!(CheckedShl, checked_shl, u8);
checked_shift_impl!(CheckedShl, checked_shl, u16);
checked_shift_impl!(CheckedShl, checked_shl, u32);
checked_shift_impl!(CheckedShl, checked_shl, u64);
checked_shift_impl!(CheckedShl, checked_shl, usize);
checked_shift_impl!(CheckedShl, checked_shl, u128);

checked_shift_impl!(CheckedShl, checked_shl, i8);
checked_shift_impl!(CheckedShl, checked_shl, i16);
checked_shift_impl!(CheckedShl, checked_shl, i32);
checked_shift_impl!(CheckedShl, checked_shl, i64);
checked_shift_impl!(CheckedShl, checked_shl, isize);
checked_shift_impl!(CheckedShl, checked_shl, i128);

/// A trait for types that support checked right shift.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::checked_arithmetic::CheckedShr;
///
/// let a: u8 = 1;
/// let b: u32 = 8;
/// assert_eq!(a.checked_shr(b), None); // Shift amount >= bit width returns None
/// let c: u32 = 3;
/// assert_eq!(a.checked_shr(c), Some(0)); // 1 >> 3 = 0
/// ```
pub trait CheckedShr: Sized + Shr<u32, Output = Self> {
    /// Performs checked right shift, returning `None` if the shift amount is
    /// greater than or equal to the number of bits in the type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::num::ops::checked_arithmetic::CheckedShr;
    ///
    /// let a: u8 = 1;
    /// let b: u32 = 8;
    /// assert_eq!(a.checked_shr(b), None); // Shift amount >= bit width returns None
    /// let c: u32 = 3;
    /// assert_eq!(a.checked_shr(c), Some(0)); // 1 >> 3 = 0
    /// ```
    fn checked_shr(self, rhs: u32) -> Option<Self>;
}

checked_shift_impl!(CheckedShr, checked_shr, u8);
checked_shift_impl!(CheckedShr, checked_shr, u16);
checked_shift_impl!(CheckedShr, checked_shr, u32);
checked_shift_impl!(CheckedShr, checked_shr, u64);
checked_shift_impl!(CheckedShr, checked_shr, usize);
checked_shift_impl!(CheckedShr, checked_shr, u128);

checked_shift_impl!(CheckedShr, checked_shr, i8);
checked_shift_impl!(CheckedShr, checked_shr, i16);
checked_shift_impl!(CheckedShr, checked_shr, i32);
checked_shift_impl!(CheckedShr, checked_shr, i64);
checked_shift_impl!(CheckedShr, checked_shr, isize);
checked_shift_impl!(CheckedShr, checked_shr, i128);
