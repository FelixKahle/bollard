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

//! # Bollard Core
//!
//! Foundational utilities, numerics, and math primitives for the Bollard
//! scheduling ecosystem. This crate consolidates reusable building blocks
//! focused on performance, correctness, and ergonomic APIs that underpin
//! higher-level model and solver crates.
//!
//! ## Modules
//!
//! - `math`: Closed-open interval `[start, end)` primitives with validation,
//!   set operations (intersection/union/difference/gap/split), measurements,
//!   iteration (`Iterator`, `DoubleEndedIterator`, `ExactSizeIterator`,
//!   `FusedIterator`), and conversions to/from `std::ops::Range`.
//! - `num`: Integer-centric utilities including associated constant traits
//!   (`MinusOne`, `Zero`, `PlusOne`) and by-value arithmetic traits for
//!   checked (`Option<T>`) and saturating operations.
//! - `utils`: Core helpers such as phantom-tagged, strongly typed indices
//!   (`TypedIndex<T>`), optional iterator wrapper (`MaybeIter<I>`), and
//!   nominal typing (`Brand<'x>`).
//!
//! ## Purpose
//!
//! These primitives enable robust, generic code in optimization and scheduling
//! pipelines, reducing accidental bugs (e.g., index mixing, overflow) while
//! keeping runtime overhead minimal.
//!
//! Refer to each module for detailed APIs and examples.

pub mod math;
pub mod num;
pub mod utils;
