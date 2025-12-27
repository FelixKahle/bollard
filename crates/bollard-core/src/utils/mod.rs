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

//! # Core Utilities
//!
//! Common utility primitives used across the Bollard ecosystem. These modules
//! provide zero-cost, type-safe wrappers and iterator composition helpers that
//! improve correctness and ergonomics in indexing-heavy and iterator-centric code.
//!
//! ## Submodules
//!
//! - `index`: Phantom-tagged, strongly typed indices (`TypedIndex<T>`) with
//!   human-readable tags (`TypedIndexTag`) to prevent mixing domains.
//! - `iter`: Optional iterator wrapper (`MaybeIter<I>`) that unifies iteration
//!   over `Option<I>` with support for `DoubleEndedIterator`, `ExactSizeIterator`,
//!   and `FusedIterator` when available.
//! - `marker`: Nominal typing via a zero-sized `Brand<'x>` marker to separate
//!   logically distinct types without runtime cost.
//!
//! ## Motivation
//!
//! Scheduling and optimization code often manipulates multiple index spaces and
//! conditional iterator sources. These utilities provide compile-time guarantees,
//! predictable iteration behavior, and clean composition patterns.
//!
//! Refer to each submodule for detailed APIs and examples.

pub mod index;
pub mod iter;
pub mod marker;
