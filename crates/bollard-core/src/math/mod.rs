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

//! # Math Primitives
//!
//! Foundational mathematical structures for scheduling and time-window logic.
//! This module currently focuses on closed-open interval math, designed to
//! integrate cleanly with Rust’s range and iterator ecosystem.
//!
//! ## Submodules
//!
//! - `interval`: A generic `[start, end)` interval type with validation,
//!   predicates (intersection, adjacency, containment), set operations
//!   (intersection/union/difference/gap/split), measurements, and
//!   iteration support (`Iterator`, `DoubleEndedIterator`,
//!   `ExactSizeIterator`, `FusedIterator`). Includes conversions to/from
//!   `std::ops::Range` and `RangeBounds`.
//!
//! ## Motivation
//!
//! Scheduling and constraint-solving routinely manipulate windows of time and
//! resource availability. Closed-open intervals are robust against off-by-one
//! errors and compose well with standard ranges and iterators.
//!
//! Refer to the `interval` module for detailed APIs and examples.

pub mod interval;
