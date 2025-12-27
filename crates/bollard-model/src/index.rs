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

//! # Strongly Typed Indices
//!
//! This module provides zero-cost wrappers around `usize` to strictly distinguish between
//! different dimensions of the scheduling problem.
//!
//! ## Motivation
//!
//! In complex scheduling algorithms, multiple dimensions (Vessels, Berths, Time Slots) are often
//! iterated simultaneously. Using raw `usize` for all of them invites "swapped index" bugs,
//! where a vessel index is accidentally used to look up a berth property.
//!
//! ## Usage
//!
//! These types implement the **Newtype Pattern**. They compile down to a simple `usize`
//! (zero runtime overhead) but enforce type correctness at compile time.
//!
//! ```rust
//! use bollard_model::index::{VesselIndex, BerthIndex};
//!
//! let v = VesselIndex::new(5);
//! let b = BerthIndex::new(2);
//!
//! // process(v, v); // Compile Error: Expected BerthIndex, found VesselIndex
//! // process(v, b); // OK
//!
//! fn process(v: VesselIndex, b: BerthIndex) {}
//! ```

use bollard_core::utils::index::{TypedIndex, TypedIndexTag};

/// A tag type for vessel indices.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct VesselIndexTag;

impl TypedIndexTag for VesselIndexTag {
    const NAME: &'static str = "VesselIndex";
}

/// A typed index for vessels.
pub type VesselIndex = TypedIndex<VesselIndexTag>;

/// A tag type for berth indices.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BerthIndexTag;

impl TypedIndexTag for BerthIndexTag {
    const NAME: &'static str = "BerthIndex";
}

/// A typed index for berths.
pub type BerthIndex = TypedIndex<BerthIndexTag>;
