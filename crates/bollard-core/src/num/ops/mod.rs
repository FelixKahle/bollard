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

//! # Numeric Operations Traits
//!
//! Unified, by-value numeric operation traits for integer primitives.
//! This module groups checked and saturating arithmetic abstractions that
//! mirror Rust’s intrinsic methods, but expose consistent trait-based APIs
//! suitable for generic code without references.
//!
//! ## Submodules
//!
//! - `checked_arithmetic`: Traits like `CheckedAddVal`, `CheckedSubVal`,
//!   `CheckedMulVal`, `CheckedDivVal`, `CheckedRemVal`, `CheckedNegVal`,
//!   `CheckedShlVal`, `CheckedShrVal` returning `Option<T>` on error
//!   (overflow/underflow/div-by-zero/invalid shift).
//! - `saturating_arithmetic`: Traits like `SaturatingAddVal`, `SaturatingSubVal`,
//!   `SaturatingMulVal`, `SaturatingNegVal` that clamp results to type bounds.
//!
//! ## Motivation
//!
//! Generic optimization/scheduling code benefits from predictable numeric
//! semantics. These traits enable:
//! - By-value operations that match primitive behavior.
//! - Composable bounds for algorithms without ad hoc per-type calls.
//! - Clear error handling (`Option<T>` for checked ops) or clamped outcomes
//!   (saturating ops) for critical paths.
//!
//! Refer to each submodule for examples and trait lists.

pub mod checked_arithmetic;
pub mod saturating_arithmetic;
