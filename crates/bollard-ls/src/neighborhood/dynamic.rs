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

//! Dynamic neighborhood wrapper using trait objects.
//!
//! This module provides `DynamicNeighborhoods`, a thin wrapper around any type that
//! implements the `Neighborhoods` trait, enabling dynamic dispatch at runtime.

use crate::neighborhood::neighborhoods::Neighborhoods;
use bollard_model::index::VesselIndex;

/// A type-erasing wrapper around any `Neighborhoods` implementation, enabling runtime selection
/// and substitution of neighborhood strategies without exposing their concrete types. The wrapper
/// owns a boxed trait object and forwards all trait calls to the inner implementation, making it
/// suitable for configuration-driven setups and for crossing crate or plugin boundaries. The
/// lifetime `'a` ties the wrapper to the lifetime of the inner value. Safety notes: methods marked
/// `unsafe` delegate directly to the inner implementation; callers must uphold the safety
/// requirements defined by the `Neighborhoods` trait.
#[derive(Debug)]
pub struct DynamicNeighborhoods<'a> {
    inner: Box<dyn Neighborhoods + 'a>,
}

impl<'a> DynamicNeighborhoods<'a> {
    /// Creates a new `DynamicNeighborhoods` instance from a boxed `Neighborhoods` implementation.
    #[inline]
    pub fn new(inner: Box<dyn Neighborhoods + 'a>) -> Self {
        Self { inner }
    }

    /// Creates a new `DynamicNeighborhoods` instance from any type that implements the `Neighborhoods` trait.
    #[inline]
    pub fn from_neighborhood<N>(neighborhood: N) -> Self
    where
        N: Neighborhoods + 'a,
    {
        Self {
            inner: Box::new(neighborhood),
        }
    }

    /// Returns a reference to the inner `Neighborhoods` implementation.
    #[inline]
    pub fn inner(&self) -> &dyn Neighborhoods {
        self.inner.as_ref()
    }
}

impl<'a> From<Box<dyn Neighborhoods + 'a>> for DynamicNeighborhoods<'a> {
    #[inline]
    fn from(inner: Box<dyn Neighborhoods + 'a>) -> Self {
        Self::new(inner)
    }
}

impl<'a> std::fmt::Display for DynamicNeighborhoods<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DynamicNeighborhoods")
    }
}

impl Neighborhoods for DynamicNeighborhoods<'_> {
    fn num_vessels(&self) -> usize {
        self.inner.num_vessels()
    }

    unsafe fn are_neighbors_unchecked(&self, a: VesselIndex, b: VesselIndex) -> bool {
        unsafe { self.inner.are_neighbors_unchecked(a, b) }
    }

    unsafe fn neighbors_of_unchecked(&self, v: VesselIndex) -> &[VesselIndex] {
        unsafe { self.inner.neighbors_of_unchecked(v) }
    }
}
