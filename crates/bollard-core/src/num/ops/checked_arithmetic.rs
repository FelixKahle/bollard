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

/// A trait for types that support checked addition by value (no references).
///
/// This mirrors the semantics of primitive integer `checked_add`, but provides
/// a trait-based API that does not take references (unlike some num_traits APIs).
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::checked_arithmetic::CheckedAddVal;
/// let a: u8 = 200;
/// let b: u8 = 100;
/// assert_eq!(a.checked_add_val(b), None); // Overflow occurs
/// let c: u8 = 50;
/// assert_eq!(a.checked_add_val(c), Some(250)); // No overflow
/// ```
pub trait CheckedAddVal: Sized + Add<Self, Output = Self> {
    /// Performs checked addition by value, returning `None` if overflow occurs.
    fn checked_add_val(self, v: Self) -> Option<Self>;
}

macro_rules! checked_impl_val {
    ($trait_name:ident, $method:ident, $t:ty, $src_method:ident) => {
        impl $trait_name for $t {
            #[inline(always)]
            fn $method(self, v: $t) -> Option<$t> {
                <$t>::$src_method(self, v)
            }
        }
    };
}

checked_impl_val!(CheckedAddVal, checked_add_val, u8, checked_add);
checked_impl_val!(CheckedAddVal, checked_add_val, u16, checked_add);
checked_impl_val!(CheckedAddVal, checked_add_val, u32, checked_add);
checked_impl_val!(CheckedAddVal, checked_add_val, u64, checked_add);
checked_impl_val!(CheckedAddVal, checked_add_val, usize, checked_add);
checked_impl_val!(CheckedAddVal, checked_add_val, u128, checked_add);

checked_impl_val!(CheckedAddVal, checked_add_val, i8, checked_add);
checked_impl_val!(CheckedAddVal, checked_add_val, i16, checked_add);
checked_impl_val!(CheckedAddVal, checked_add_val, i32, checked_add);
checked_impl_val!(CheckedAddVal, checked_add_val, i64, checked_add);
checked_impl_val!(CheckedAddVal, checked_add_val, isize, checked_add);
checked_impl_val!(CheckedAddVal, checked_add_val, i128, checked_add);

/// A trait for types that support checked subtraction by value (no references).
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::checked_arithmetic::CheckedSubVal;
///
/// let a: u8 = 50;
/// let b: u8 = 100;
/// assert_eq!(a.checked_sub_val(b), None); // Underflow occurs
/// let c: u8 = 20;
/// assert_eq!(a.checked_sub_val(c), Some(30)); // No underflow
/// ```
pub trait CheckedSubVal: Sized + Sub<Self, Output = Self> {
    /// Performs checked subtraction by value, returning `None` if underflow occurs.
    fn checked_sub_val(self, v: Self) -> Option<Self>;
}

checked_impl_val!(CheckedSubVal, checked_sub_val, u8, checked_sub);
checked_impl_val!(CheckedSubVal, checked_sub_val, u16, checked_sub);
checked_impl_val!(CheckedSubVal, checked_sub_val, u32, checked_sub);
checked_impl_val!(CheckedSubVal, checked_sub_val, u64, checked_sub);
checked_impl_val!(CheckedSubVal, checked_sub_val, usize, checked_sub);
checked_impl_val!(CheckedSubVal, checked_sub_val, u128, checked_sub);

checked_impl_val!(CheckedSubVal, checked_sub_val, i8, checked_sub);
checked_impl_val!(CheckedSubVal, checked_sub_val, i16, checked_sub);
checked_impl_val!(CheckedSubVal, checked_sub_val, i32, checked_sub);
checked_impl_val!(CheckedSubVal, checked_sub_val, i64, checked_sub);
checked_impl_val!(CheckedSubVal, checked_sub_val, isize, checked_sub);
checked_impl_val!(CheckedSubVal, checked_sub_val, i128, checked_sub);

/// A trait for types that support checked multiplication by value (no references).
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::checked_arithmetic::CheckedMulVal;
///
/// let a: u8 = 20;
/// let b: u8 = 10;
/// assert_eq!(a.checked_mul_val(b), Some(200)); // No overflow
/// let c: u8 = 20;
/// assert_eq!(a.checked_mul_val(c), None); // Overflow occurs (20*20 = 400 > 255)
/// ```
pub trait CheckedMulVal: Sized + Mul<Self, Output = Self> {
    /// Performs checked multiplication by value, returning `None` if overflow occurs.
    fn checked_mul_val(self, v: Self) -> Option<Self>;
}

checked_impl_val!(CheckedMulVal, checked_mul_val, u8, checked_mul);
checked_impl_val!(CheckedMulVal, checked_mul_val, u16, checked_mul);
checked_impl_val!(CheckedMulVal, checked_mul_val, u32, checked_mul);
checked_impl_val!(CheckedMulVal, checked_mul_val, u64, checked_mul);
checked_impl_val!(CheckedMulVal, checked_mul_val, usize, checked_mul);
checked_impl_val!(CheckedMulVal, checked_mul_val, u128, checked_mul);

checked_impl_val!(CheckedMulVal, checked_mul_val, i8, checked_mul);
checked_impl_val!(CheckedMulVal, checked_mul_val, i16, checked_mul);
checked_impl_val!(CheckedMulVal, checked_mul_val, i32, checked_mul);
checked_impl_val!(CheckedMulVal, checked_mul_val, i64, checked_mul);
checked_impl_val!(CheckedMulVal, checked_mul_val, isize, checked_mul);
checked_impl_val!(CheckedMulVal, checked_mul_val, i128, checked_mul);

/// A trait for types that support checked division by value (no references).
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::checked_arithmetic::CheckedDivVal;
///
/// let a: u8 = 100;
/// let b: u8 = 0;
/// assert_eq!(a.checked_div_val(b), None); // Division by zero
/// let c: u8 = 4;
/// assert_eq!(a.checked_div_val(c), Some(25)); // No division by zero
/// ```
pub trait CheckedDivVal: Sized + Div<Self, Output = Self> {
    /// Performs checked division by value, returning `None` if division by zero occurs.
    fn checked_div_val(self, v: Self) -> Option<Self>;
}

checked_impl_val!(CheckedDivVal, checked_div_val, u8, checked_div);
checked_impl_val!(CheckedDivVal, checked_div_val, u16, checked_div);
checked_impl_val!(CheckedDivVal, checked_div_val, u32, checked_div);
checked_impl_val!(CheckedDivVal, checked_div_val, u64, checked_div);
checked_impl_val!(CheckedDivVal, checked_div_val, usize, checked_div);
checked_impl_val!(CheckedDivVal, checked_div_val, u128, checked_div);

checked_impl_val!(CheckedDivVal, checked_div_val, i8, checked_div);
checked_impl_val!(CheckedDivVal, checked_div_val, i16, checked_div);
checked_impl_val!(CheckedDivVal, checked_div_val, i32, checked_div);
checked_impl_val!(CheckedDivVal, checked_div_val, i64, checked_div);
checked_impl_val!(CheckedDivVal, checked_div_val, isize, checked_div);
checked_impl_val!(CheckedDivVal, checked_div_val, i128, checked_div);

/// A trait for types that support checked remainder by value (no references).
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::checked_arithmetic::CheckedRemVal;
///
/// let a: u8 = 10;
/// let b: u8 = 0;
/// assert_eq!(a.checked_rem_val(b), None); // Division by zero
/// let c: u8 = 3;
/// assert_eq!(a.checked_rem_val(c), Some(1)); // No division by zero
/// ```
pub trait CheckedRemVal: Sized + Rem<Self, Output = Self> {
    /// Performs checked remainder by value, returning `None` if division by zero occurs.
    fn checked_rem_val(self, v: Self) -> Option<Self>;
}

checked_impl_val!(CheckedRemVal, checked_rem_val, u8, checked_rem);
checked_impl_val!(CheckedRemVal, checked_rem_val, u16, checked_rem);
checked_impl_val!(CheckedRemVal, checked_rem_val, u32, checked_rem);
checked_impl_val!(CheckedRemVal, checked_rem_val, u64, checked_rem);
checked_impl_val!(CheckedRemVal, checked_rem_val, usize, checked_rem);
checked_impl_val!(CheckedRemVal, checked_rem_val, u128, checked_rem);

checked_impl_val!(CheckedRemVal, checked_rem_val, i8, checked_rem);
checked_impl_val!(CheckedRemVal, checked_rem_val, i16, checked_rem);
checked_impl_val!(CheckedRemVal, checked_rem_val, i32, checked_rem);
checked_impl_val!(CheckedRemVal, checked_rem_val, i64, checked_rem);
checked_impl_val!(CheckedRemVal, checked_rem_val, isize, checked_rem);
checked_impl_val!(CheckedRemVal, checked_rem_val, i128, checked_rem);

macro_rules! checked_impl_unary_val {
    ($trait_name:ident, $method:ident, $t:ty, $src_method:ident) => {
        impl $trait_name for $t {
            #[inline(always)]
            fn $method(self) -> Option<$t> {
                <$t>::$src_method(self)
            }
        }
    };
}

/// A trait for types that support checked negation by value (no references).
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::checked_arithmetic::CheckedNegVal;
///
/// let a: i8 = -128;
/// assert_eq!(a.checked_neg_val(), None); // Overflow occurs
/// let b: i8 = 100;
/// assert_eq!(b.checked_neg_val(), Some(-100)); // No overflow
/// ```
pub trait CheckedNegVal: Sized {
    /// Performs checked negation by value, returning `None` if overflow occurs.
    fn checked_neg_val(self) -> Option<Self>;
}

checked_impl_unary_val!(CheckedNegVal, checked_neg_val, u8, checked_neg);
checked_impl_unary_val!(CheckedNegVal, checked_neg_val, u16, checked_neg);
checked_impl_unary_val!(CheckedNegVal, checked_neg_val, u32, checked_neg);
checked_impl_unary_val!(CheckedNegVal, checked_neg_val, u64, checked_neg);
checked_impl_unary_val!(CheckedNegVal, checked_neg_val, usize, checked_neg);
checked_impl_unary_val!(CheckedNegVal, checked_neg_val, u128, checked_neg);

checked_impl_unary_val!(CheckedNegVal, checked_neg_val, i8, checked_neg);
checked_impl_unary_val!(CheckedNegVal, checked_neg_val, i16, checked_neg);
checked_impl_unary_val!(CheckedNegVal, checked_neg_val, i32, checked_neg);
checked_impl_unary_val!(CheckedNegVal, checked_neg_val, i64, checked_neg);
checked_impl_unary_val!(CheckedNegVal, checked_neg_val, isize, checked_neg);
checked_impl_unary_val!(CheckedNegVal, checked_neg_val, i128, checked_neg);

/// A trait for types that support checked left shift by value (no references).
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::checked_arithmetic::CheckedShlVal;
///
/// let a: u8 = 1;
/// let b: u32 = 8;
/// assert_eq!(a.checked_shl_val(b), None); // Overflow occurs
/// let c: u32 = 3;
/// assert_eq!(a.checked_shl_val(c), Some(8)); // No overflow
/// ```
pub trait CheckedShlVal: Sized + Shl<u32, Output = Self> {
    /// Performs checked left shift by value, returning `None` if overflow occurs.
    fn checked_shl_val(self, rhs: u32) -> Option<Self>;
}

macro_rules! checked_shift_impl_val {
    ($trait_name:ident, $method:ident, $t:ty, $src_method:ident) => {
        impl $trait_name for $t {
            #[inline(always)]
            fn $method(self, rhs: u32) -> Option<$t> {
                <$t>::$src_method(self, rhs)
            }
        }
    };
}

checked_shift_impl_val!(CheckedShlVal, checked_shl_val, u8, checked_shl);
checked_shift_impl_val!(CheckedShlVal, checked_shl_val, u16, checked_shl);
checked_shift_impl_val!(CheckedShlVal, checked_shl_val, u32, checked_shl);
checked_shift_impl_val!(CheckedShlVal, checked_shl_val, u64, checked_shl);
checked_shift_impl_val!(CheckedShlVal, checked_shl_val, usize, checked_shl);
checked_shift_impl_val!(CheckedShlVal, checked_shl_val, u128, checked_shl);

checked_shift_impl_val!(CheckedShlVal, checked_shl_val, i8, checked_shl);
checked_shift_impl_val!(CheckedShlVal, checked_shl_val, i16, checked_shl);
checked_shift_impl_val!(CheckedShlVal, checked_shl_val, i32, checked_shl);
checked_shift_impl_val!(CheckedShlVal, checked_shl_val, i64, checked_shl);
checked_shift_impl_val!(CheckedShlVal, checked_shl_val, isize, checked_shl);
checked_shift_impl_val!(CheckedShlVal, checked_shl_val, i128, checked_shl);

/// A trait for types that support checked right shift by value (no references).
///
/// # Examples
///
/// ```rust
/// # use bollard_core::num::ops::checked_arithmetic::CheckedShrVal;
///
/// let a: u8 = 1;
/// let b: u32 = 8;
/// assert_eq!(a.checked_shr_val(b), None); // Shift amount >= bit width returns None
/// let c: u32 = 3;
/// assert_eq!(a.checked_shr_val(c), Some(0)); // 1 >> 3 = 0
/// ```
pub trait CheckedShrVal: Sized + Shr<u32, Output = Self> {
    /// Performs checked right shift by value, returning `None` if the shift amount is
    /// greater than or equal to the number of bits in the type.
    fn checked_shr_val(self, rhs: u32) -> Option<Self>;
}

checked_shift_impl_val!(CheckedShrVal, checked_shr_val, u8, checked_shr);
checked_shift_impl_val!(CheckedShrVal, checked_shr_val, u16, checked_shr);
checked_shift_impl_val!(CheckedShrVal, checked_shr_val, u32, checked_shr);
checked_shift_impl_val!(CheckedShrVal, checked_shr_val, u64, checked_shr);
checked_shift_impl_val!(CheckedShrVal, checked_shr_val, usize, checked_shr);
checked_shift_impl_val!(CheckedShrVal, checked_shr_val, u128, checked_shr);

checked_shift_impl_val!(CheckedShrVal, checked_shr_val, i8, checked_shr);
checked_shift_impl_val!(CheckedShrVal, checked_shr_val, i16, checked_shr);
checked_shift_impl_val!(CheckedShrVal, checked_shr_val, i32, checked_shr);
checked_shift_impl_val!(CheckedShrVal, checked_shr_val, i64, checked_shr);
checked_shift_impl_val!(CheckedShrVal, checked_shr_val, isize, checked_shr);
checked_shift_impl_val!(CheckedShrVal, checked_shr_val, i128, checked_shr);
