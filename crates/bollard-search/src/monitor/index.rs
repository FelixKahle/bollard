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

//! # Monitor Indices
//!
//! Strongly typed index wrapper for addressing monitors within composite or
//! indexed collections. Built on `bollard_core::utils::index::TypedIndex`,
//! `MonitorIndex` prevents accidental mixing with other index spaces while
//! remaining zero-cost at runtime.
//!
//! ## Motivation
//!
//! Search monitoring can involve multiple monitors managed in arrays or
//! composite structures. Using raw `usize` invites bugs where a different
//! domain’s index is passed by mistake. `MonitorIndex` encodes intent at the
//! type level for safer, clearer code.
//!
//! ## Highlights
//!
//! - `MonitorIndex` is a type alias over `TypedIndex<MonitorIndexTag>`.
//! - Human-readable formatting via the tag’s `NAME` ("MonitorIndex").
//! - Zero runtime overhead (`#[repr(transparent)]` under the hood).
//! - Works with arithmetic ops against `usize` and conversions to/from `usize`
//!   provided by `TypedIndex`.
//!
//! ## Usage
//!
//! ```rust
//! use bollard_search::monitor::index::MonitorIndex;
//!
//! let m = MonitorIndex::new(0);
//! assert_eq!(format!("{}", m), "MonitorIndex(0)");
//! ```

use bollard_core::utils::index::{TypedIndex, TypedIndexTag};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct MonitorIndexTag;

impl TypedIndexTag for MonitorIndexTag {
    const NAME: &'static str = "MonitorIndex";
}

/// A typed index for monitors.
pub type MonitorIndex = TypedIndex<MonitorIndexTag>;
