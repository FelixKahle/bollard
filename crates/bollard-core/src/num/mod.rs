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

//! # Numeric Foundations
//!
//! Traits and utilities for integer-centric numeric programming. This module
//! consolidates compile-time constants and by-value arithmetic traits that
//! mirror Rust’s intrinsic behaviors while providing uniform, generic APIs.
//!
//! ## Submodules
//!
//! - `constants`: Associated-constant traits (`MinusOne`, `Zero`, `PlusOne`)
//!   implemented for all core integer types to access sentinel values in
//!   a type-safe, self-describing way.
//! - `ops`: Checked and saturating arithmetic traits (by value) for addition,
//!   subtraction, multiplication, division, remainder, negation, and shifts,
//!   enabling predictable error handling (`Option<T>`) or clamped outcomes.
//!
//! ## Motivation
//!
//! Optimization and scheduling pipelines demand robust numeric semantics.
//! These modules provide concise, generic building blocks that reduce ad hoc
//! per-type code, improve readability, and help avoid overflow/underflow bugs.
//!
//! Refer to each submodule for detailed APIs and examples.

pub mod constants;
pub mod ops;
