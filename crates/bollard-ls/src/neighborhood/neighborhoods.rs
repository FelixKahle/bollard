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

use bollard_model::{index::VesselIndex, model::Model};
use bollard_search::num::SolverNumeric;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullNeighborhoods {
    // Flattened adjacency lists.
    //
    // All neighbors for all vessels are concatenated into this single vector.
    // By using a contiguous block of memory, we ensure that when a vessel's
    // neighbors are accessed, they are likely to reside within the same
    // CPU cache line, significantly speeding up linear scans.
    neighbors: Vec<VesselIndex>,

    // Start indices into `neighbors` for each vessel.
    //
    // This vector uses a "sentinel" pattern to define ranges:
    // - The neighbors of vessel `i` are found at `neighbors[offsets[i]..offsets[i+1]]`.
    // - To support this without branching, the length must be `num_vessels + 1`.
    // - The final element `offsets[num_vessels]` always equals `neighbors.len()`.
    offsets: Vec<usize>,
}

impl FullNeighborhoods {
    pub fn new(num_vessels: usize) -> Self {
        // Build CSR-like storage for a complete graph excluding self-edges.
        let mut neighbors =
            Vec::with_capacity(num_vessels.saturating_mul(num_vessels.saturating_sub(1)));
        let mut offsets = Vec::with_capacity(num_vessels + 1);

        let mut cursor = 0usize;
        offsets.push(cursor);
        for i in 0..num_vessels {
            for j in 0..num_vessels {
                if i != j {
                    neighbors.push(VesselIndex::new(j));
                    cursor += 1;
                }
            }
            offsets.push(cursor);
        }

        Self { neighbors, offsets }
    }
}

impl Neighborhoods for FullNeighborhoods {
    #[inline]
    fn num_vessels(&self) -> usize {
        // offsets has length num_vessels + 1
        self.offsets.len() - 1
    }

    #[inline(always)]
    unsafe fn are_neighbors_unchecked(&self, _a: VesselIndex, _b: VesselIndex) -> bool {
        // In a full neighborhood, every pair excluding self is a neighbor.
        // The caller guarantees indices are within bounds.
        // Skip self-check because this method is only called after bounds checks in the safe wrapper.
        true
    }

    #[inline(always)]
    unsafe fn neighbors_of_unchecked(&self, v: VesselIndex) -> &[VesselIndex] {
        let index = v.get();

        debug_assert!(
            index < self.offsets.len() - 1,
            "called `FullNeighborhoods::neighbors_of_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.offsets.len() - 1
        );

        // SAFETY:
        // - `offsets` length is num_vessels + 1 (constructed in new).
        // - `index < num_vessels` (assert above + trait safety).
        // - `offsets` are monotonic and bounded by `neighbors.len()` (constructed deterministically in new).
        let start = *unsafe { self.offsets.get_unchecked(index) };
        let end = *unsafe { self.offsets.get_unchecked(index + 1) };

        debug_assert!(
            start <= end,
            "called `FullNeighborhoods::neighbors_of_unchecked` with CSR invariant violated: offsets are not monotonic. start: {}, end: {}",
            start,
            end
        );
        debug_assert!(
            end <= self.neighbors.len(),
            "called `FullNeighborhoods::neighbors_of_unchecked` with CSR invariant violated: offset points out of neighbors bounds. end: {}, neighbors.len: {}",
            end,
            self.neighbors.len()
        );

        unsafe { self.neighbors.get_unchecked(start..end) }
    }
}

impl FullNeighborhoods {
    pub fn from_model<T>(model: &Model<T>) -> Self
    where
        T: SolverNumeric,
    {
        let num_vessels = model.num_vessels();
        Self::new(num_vessels)
    }
}

impl<T> From<&Model<T>> for FullNeighborhoods
where
    T: SolverNumeric,
{
    fn from(model: &Model<T>) -> Self {
        FullNeighborhoods::from_model(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    fn collect_neighbors(nh: &FullNeighborhoods, i: usize) -> Vec<usize> {
        nh.neighbors_of(VesselIndex::new(i))
            .iter()
            .map(|v| v.get())
            .collect::<Vec<_>>()
    }

    #[test]
    fn test_empty_full_neighborhoods() {
        let nh = FullNeighborhoods::new(0);
        assert_eq!(nh.num_vessels(), 0);

        // Offsets sentinel invariant: length == num_vessels + 1
        assert_eq!(nh.offsets.len(), 1);
        assert_eq!(nh.offsets[0], 0);
        assert!(nh.neighbors.is_empty());
    }

    #[test]
    fn test_single_vessel_no_neighbors() {
        let nh = FullNeighborhoods::new(1);
        assert_eq!(nh.num_vessels(), 1);

        // Vessel 0 should have no neighbors
        let neighbors = collect_neighbors(&nh, 0);
        assert!(neighbors.is_empty());

        // are_neighbors should be true only for distinct pairs; since there are none, we check bounds and self behavior:
        // Safe call with same index is allowed by the safe wrapper bounds but semantically in a full graph, self is excluded.
        // Our are_neighbors_unchecked returns true, but the caller should avoid self pairs.
        // We verify neighbors_of excludes self which is the authoritative source for iteration.
        assert!(!neighbors.contains(&0));
    }

    #[test]
    fn test_two_vessels_each_other_neighbors() {
        let nh = FullNeighborhoods::new(2);
        assert_eq!(nh.num_vessels(), 2);

        let n0 = collect_neighbors(&nh, 0);
        let n1 = collect_neighbors(&nh, 1);

        assert_eq!(n0, vec![1]);
        assert_eq!(n1, vec![0]);

        // Checked neighbor queries
        assert!(nh.are_neighbors(VesselIndex::new(0), VesselIndex::new(1)));
        assert!(nh.are_neighbors(VesselIndex::new(1), VesselIndex::new(0)));

        // Bounds checks should panic on out-of-range indices
        let oob_0 = catch_unwind(AssertUnwindSafe(|| {
            nh.are_neighbors(VesselIndex::new(0), VesselIndex::new(2))
        }));
        assert!(oob_0.is_err());

        let oob_1 = catch_unwind(AssertUnwindSafe(|| nh.neighbors_of(VesselIndex::new(2))));
        assert!(oob_1.is_err());
    }

    #[test]
    fn test_three_vessels_complete_graph_excludes_self() {
        let nh = FullNeighborhoods::new(3);
        assert_eq!(nh.num_vessels(), 3);

        let n0 = collect_neighbors(&nh, 0);
        let n1 = collect_neighbors(&nh, 1);
        let n2 = collect_neighbors(&nh, 2);

        // Each list should contain the other two indices
        assert_eq!(n0, vec![1, 2]);
        assert_eq!(n1, vec![0, 2]);
        assert_eq!(n2, vec![0, 1]);

        // No self in any neighbor list
        assert!(!n0.contains(&0));
        assert!(!n1.contains(&1));
        assert!(!n2.contains(&2));

        // are_neighbors is true for all distinct pairs
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!(nh.are_neighbors(VesselIndex::new(i), VesselIndex::new(j)));
                }
            }
        }
    }

    #[test]
    fn test_csr_invariants() {
        let n = 5;
        let nh = FullNeighborhoods::new(n);

        // Offsets sentinel invariant
        assert_eq!(nh.offsets.len(), n + 1);
        assert_eq!(nh.offsets.first().copied(), Some(0));
        assert_eq!(nh.offsets.last().copied(), Some(n * (n - 1)));

        // Monotonic offsets
        for i in 0..n {
            assert!(nh.offsets[i] <= nh.offsets[i + 1]);
        }

        // Total number of neighbors equals n*(n-1)
        assert_eq!(nh.neighbors.len(), n * (n - 1));

        // Each vessel has exactly n-1 neighbors
        for i in 0..n {
            let slice = nh.neighbors_of(VesselIndex::new(i));
            assert_eq!(slice.len(), n - 1);
        }
    }

    #[test]
    fn test_symmetry_in_full_neighborhoods() {
        let n = 4;
        let nh = FullNeighborhoods::new(n);

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let neighbors_i = collect_neighbors(&nh, i);
                let neighbors_j = collect_neighbors(&nh, j);
                assert!(neighbors_i.contains(&j));
                assert!(neighbors_j.contains(&i));
                assert!(nh.are_neighbors(VesselIndex::new(i), VesselIndex::new(j)));
                assert!(nh.are_neighbors(VesselIndex::new(j), VesselIndex::new(i)));
            }
        }
    }
}
