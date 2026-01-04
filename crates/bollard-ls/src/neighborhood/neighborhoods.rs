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

//! Neighborhood abstractions for vessel interaction in local search.
//!
//! This module defines the `Neighborhoods` trait, a high-performance interface for expressing
//! connectivity between vessels in the solver’s search space. Neighborhood relations act as a
//! filter to focus move generation on pairs that can meaningfully interact, reducing wasted work
//! on disjoint or impossible configurations.
//!
//! Implementations expose contiguous slices of neighbors to enable efficient iteration with
//! minimal overhead. Checked and unchecked variants are provided to support hot-loop usage while
//! maintaining clear contracts about bounds safety. The design emphasizes cache locality and
//! predictable access patterns, making it suitable for rejection sampling and other inner-loop
//! heuristics where latency and memory layout are critical.

use bollard_model::index::VesselIndex;

/// A trait for defining search space connectivity (neighborhoods) between vessels.
///
/// In local search algorithms, the "neighborhood" of a vessel defines which other vessels
/// it can interact with (e.g., via swaps, shifts, or crossovers). Restricting these
/// interactions is a powerful optimization to prune moves that are statically known
/// to be unproductive, such as swapping vessels with completely disjoint time windows.
///
/// This trait is designed for **maximum performance** by providing direct slice access
/// to pre-allocated neighborhood data, ensuring $O(1)$ lookup times and high cache locality.
pub trait Neighborhoods: std::fmt::Debug + Send + Sync {
    /// Returns the total number of vessels in the problem instance.
    fn num_vessels(&self) -> usize;

    /// Returns `true` if vessels `a` and `b` are considered neighbors.
    ///
    /// This method is typically used as a **Rejection Sampling** filter in the hot loop
    /// of a solver. If it returns `false`, the solver should skip the evaluation of
    /// any move involving these two vessels.
    fn are_neighbors(&self, a: VesselIndex, b: VesselIndex) -> bool {
        let a_index = a.get();
        let b_index = b.get();

        assert!(
            a_index < self.num_vessels(),
            "called `Neighborhoods::are_neighbors` with vessel index out of bounds: the len is {} but the index is {}",
            a_index,
            self.num_vessels(),
        );

        assert!(
            b_index < self.num_vessels(),
            "called `Neighborhoods::are_neighbors` with vessel index out of bounds: the len is {} but the index is {}",
            b_index,
            self.num_vessels(),
        );

        unsafe { self.are_neighbors_unchecked(a, b) }
    }

    /// Returns `true` if vessels `a` and `b` are considered neighbors.
    ///
    /// This method is typically used as a **Rejection Sampling** filter in the hot loop
    /// of a solver. If it returns `false`, the solver should skip the evaluation of
    /// any move involving these two vessels.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `a` is within `0..self.num_vessels()` and
    /// `b` is within `0..self.num_vessels()`.
    unsafe fn are_neighbors_unchecked(&self, a: VesselIndex, b: VesselIndex) -> bool;

    /// Returns a slice containing all neighbors of the specified vessel.
    ///
    /// This method provides the primary way to iterate over valid moves for a vessel.
    /// By returning a slice (`&[VesselIndex]`), the trait guarantees that the data
    /// is contiguous in memory, allowing the CPU to iterate with zero-cost overhead.
    fn neighbors_of(&self, v: VesselIndex) -> &[VesselIndex] {
        let index = v.get();
        assert!(
            index < self.num_vessels(),
            "called `Neighborhoods::neighbors_of` with vessel index out of bounds: the len is {} but the index is {}",
            self.num_vessels(),
            index,
        );

        unsafe { self.neighbors_of_unchecked(v) }
    }

    /// Returns a slice containing all neighbors of the specified vessel.
    ///
    /// This method provides the primary way to iterate over valid moves for a vessel.
    /// By returning a slice (`&[VesselIndex]`), the trait guarantees that the data
    /// is contiguous in memory, allowing the CPU to iterate with zero-cost overhead.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `v` is within `0..self.num_vessels()`.
    unsafe fn neighbors_of_unchecked(&self, v: VesselIndex) -> &[VesselIndex];
}
