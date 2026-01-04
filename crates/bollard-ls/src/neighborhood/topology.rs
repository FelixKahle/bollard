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

//! Static neighborhood topology for vessel interactions.
//!
//! This module provides a cache-friendly representation of vessel neighborhoods used by the solver.
//! It derives connectivity from physical resource contention in the `Model`, identifying vessels as
//! neighbors when they share at least one berth where both are allowed to dock. The resulting
//! structure is immutable and optimized for fast iteration during the solve phase.
//!
//! Neighborhoods are stored in a compressed sparse row layout. All neighbor lists are flattened
//! into a single contiguous vector, and a sentinel offsets array maps each vessel to its slice of
//! neighbors. This design minimizes pointer chasing and leverages CPU cache locality for linear
//! scans, making rejection sampling and local search moves efficient.
//!
//! The construction performs a one-time analysis over vessels and berths to detect shared
//! feasibility. Accessors expose slices directly, enabling zero-cost iteration while preserving
//! memory safety under the trait’s checked and unchecked APIs.

use crate::neighborhood::neighborhoods::Neighborhoods;
use bollard_model::{index::VesselIndex, model::Model};
use bollard_search::num::SolverNumeric;

/// A static, immutable topology using a Compressed Sparse Row (CSR) layout.
///
/// This implementation is optimized for the "Solving Phase." It stores all
/// neighborhood lists in a single contiguous block of memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticTopology {
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

impl Neighborhoods for StaticTopology {
    /// Returns the total number of vessels in the problem instance.
    #[inline]
    fn num_vessels(&self) -> usize {
        // offsets has length num_vessels + 1
        self.offsets.len() - 1
    }

    #[inline(always)]
    unsafe fn are_neighbors_unchecked(&self, a: VesselIndex, b: VesselIndex) -> bool {
        let a_index = a.get();
        let b_index = b.get();

        debug_assert!(
            a_index < self.offsets.len() - 1,
            "called `Neighborhoods::are_neighbors_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            a_index,
            self.offsets.len() - 1
        );

        debug_assert!(
            b_index < self.offsets.len() - 1,
            "called `Neighborhoods::are_neighbors_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            b_index,
            self.offsets.len() - 1
        );

        // Linear scan over a small contiguous slice is typically faster
        // than a Hash-Set due to L1 cache prefetching.
        // SAFETY: caller upholds bounds; debug asserts above validate invariants.
        unsafe { self.neighbors_of_unchecked(a).contains(&b) }
    }

    #[inline(always)]
    unsafe fn neighbors_of_unchecked(&self, vessel_index: VesselIndex) -> &[VesselIndex] {
        let index = vessel_index.get();
        debug_assert!(
            index < self.offsets.len() - 1,
            "called `Neighborhoods::neighbors_of` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.offsets.len() - 1
        );

        // SAFETY:
        // 1. The constructor guarantees offsets.len() == num_vessels + 1.
        // 2. The debug_assert and trait safety contract guarantee index < num_vessels.
        // 3. Therefore, index and index + 1 are always valid indices for offsets.
        // 4. The values in offsets are monotonically increasing and bound by neighbors.len().
        unsafe {
            let start = *self.offsets.get_unchecked(index);
            let end = *self.offsets.get_unchecked(index + 1);

            debug_assert!(
                start <= end,
                "called `StaticTopology::neighbors_of_unchecked` with CSR invariant violated: offsets are not monotonic. start: {}, end: {}",
                start,
                end
            );
            debug_assert!(
                end <= self.neighbors.len(),
                "called `StaticTopology::neighbors_of_unchecked` with CSR invariant violated: offset points out of neighbors bounds. end: {}, neighbors.len: {}",
                end,
                self.neighbors.len()
            );

            // Constructing a slice from the flattened neighbors array.
            self.neighbors.get_unchecked(start..end)
        }
    }
}

impl StaticTopology {
    /// Constructs a `StaticTopology` by analyzing physical resource contention in a `Model`.
    ///
    /// This constructor identifies vessels that are "neighbors" based on their compatibility
    /// with the available berths. Two vessels are considered neighbors if, and only if, there
    /// exists at least one berth where both vessels are allowed to dock.
    ///
    /// # Logical Rationale
    ///
    /// If two vessels share no common berths, swapping their positions in a priority
    /// queue or attempting to interact with them in a neighborhood move will never
    /// result in a physical conflict or a change in the relative scheduling of those
    /// specific vessels. By pruning these "disjoint" pairs, the search space is
    /// significantly reduced, focusing the solver on vessels that actually contend
    /// for the same resources.
    ///
    /// # Search Space Reduction
    ///
    /// This effectively transforms a complete graph (where any vessel can be swapped
    /// with any other) into a sparse graph of physical dependencies.
    ///
    /// # Performance
    ///
    /// The constructor performs a one-time $O(N^2 \cdot M)$ analysis (where $N$ is
    /// the number of vessels and $M$ is the number of berths). The resulting
    /// topology is backed by a highly optimized CSR (Compressed Sparse Row)
    /// structure for $O(1)$ slice access during the solve phase.
    pub fn from_model<T>(model: &Model<T>) -> Self
    where
        T: SolverNumeric,
    {
        let num_vessels = model.num_vessels();
        let num_berths = model.num_berths();

        let mut neighbors: Vec<VesselIndex> =
            Vec::with_capacity(num_vessels.saturating_mul(num_vessels.saturating_sub(1)));
        let mut offsets: Vec<usize> = Vec::with_capacity(num_vessels + 1);

        // Sentinel start
        offsets.push(0);

        // For each vessel, compute neighbors that share at least one allowed berth.
        for i in 0..num_vessels {
            let vi = VesselIndex::new(i);

            // Track if j shares any allowed berth with i.
            'vessel_j: for j in 0..num_vessels {
                if i == j {
                    continue;
                }
                let vj = VesselIndex::new(j);

                // Check if there exists a berth k where both vessels are allowed.
                for k in 0..num_berths {
                    let bk = bollard_model::index::BerthIndex::new(k);
                    if model.vessel_allowed_on_berth(vi, bk)
                        && model.vessel_allowed_on_berth(vj, bk)
                    {
                        neighbors.push(vj);
                        // Found a common berth; no need to check other berths for this pair.
                        continue 'vessel_j;
                    }
                }
            }

            // Push the end offset for vessel i (start of vessel i+1).
            offsets.push(neighbors.len());
        }

        // The size of the vector is now fixed; shrink to fit.
        neighbors.shrink_to_fit();
        offsets.shrink_to_fit();

        StaticTopology { neighbors, offsets }
    }
}

impl<T> From<&Model<T>> for StaticTopology
where
    T: SolverNumeric,
{
    fn from(model: &Model<T>) -> Self {
        StaticTopology::from_model(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bollard_model::index::{BerthIndex, VesselIndex};
    use bollard_model::model::ModelBuilder;
    use bollard_model::time::ProcessingTime;

    type Num = i64;

    fn build_model(num_berths: usize, num_vessels: usize) -> Model<Num> {
        ModelBuilder::<Num>::new(num_berths, num_vessels).build()
    }

    #[test]
    fn test_empty_model_topology() {
        let model = build_model(0, 0);
        let topo = StaticTopology::from_model(&model);

        assert_eq!(topo.num_vessels(), 0);
        // offsets must be num_vessels + 1 = 1
        assert_eq!(topo.offsets.len(), 1);
        assert_eq!(topo.neighbors.len(), 0);
        assert_eq!(topo.offsets[0], 0);
    }

    #[test]
    fn test_single_vessel_no_neighbors() {
        let mut builder = ModelBuilder::<Num>::new(1, 1);
        // Allow vessel 0 on berth 0
        builder.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(0),
            ProcessingTime::from_option(Some(10)),
        );
        let model = builder.build();

        let topo = StaticTopology::from_model(&model);
        assert_eq!(topo.num_vessels(), 1);

        let v0 = VesselIndex::new(0);
        let neighbors_v0 = topo.neighbors_of(v0);
        assert_eq!(neighbors_v0.len(), 0);
    }

    #[test]
    fn test_two_vessels_disjoint_berths_no_neighbors() {
        // 2 berths, 2 vessels
        let mut builder = ModelBuilder::<Num>::new(2, 2);
        // Vessel 0 allowed only on berth 0
        builder.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(0),
            ProcessingTime::from_option(Some(10)),
        );
        builder.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(1),
            ProcessingTime::none(),
        );

        // Vessel 1 allowed only on berth 1
        builder.set_vessel_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(0),
            ProcessingTime::none(),
        );
        builder.set_vessel_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(1),
            ProcessingTime::from_option(Some(10)),
        );

        let model = builder.build();
        let topo = StaticTopology::from_model(&model);

        let v0 = VesselIndex::new(0);
        let v1 = VesselIndex::new(1);

        assert_eq!(topo.neighbors_of(v0).len(), 0);
        assert_eq!(topo.neighbors_of(v1).len(), 0);
        assert!(!topo.are_neighbors(v0, v1));
        assert!(!topo.are_neighbors(v1, v0));
    }

    #[test]
    fn test_two_vessels_shared_berth_are_neighbors() {
        let mut builder = ModelBuilder::<Num>::new(2, 2);
        // Both vessels allowed on berth 0
        for v in 0..2 {
            builder.set_vessel_processing_time(
                VesselIndex::new(v),
                BerthIndex::new(0),
                ProcessingTime::from_option(Some(10)),
            );
            // Disallow berth 1 (to exercise filtering)
            builder.set_vessel_processing_time(
                VesselIndex::new(v),
                BerthIndex::new(1),
                ProcessingTime::none(),
            );
        }

        let model = builder.build();
        let topo = StaticTopology::from_model(&model);

        let v0 = VesselIndex::new(0);
        let v1 = VesselIndex::new(1);

        let n0 = topo.neighbors_of(v0);
        let n1 = topo.neighbors_of(v1);

        assert_eq!(n0, &[v1]);
        assert_eq!(n1, &[v0]);
        assert!(topo.are_neighbors(v0, v1));
        assert!(topo.are_neighbors(v1, v0));
    }

    #[test]
    fn test_three_vessels_mixed_connectivity() {
        // 2 berths, 3 vessels
        // v0: allowed on berth 0
        // v1: allowed on berth 0
        // v2: allowed on berth 1
        // Neighbors: v0<->v1, v2 isolated relative to v0 and v1
        let mut builder = ModelBuilder::<Num>::new(2, 3);

        builder.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(0),
            ProcessingTime::from_option(Some(5)),
        );
        builder.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(1),
            ProcessingTime::none(),
        );

        builder.set_vessel_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(0),
            ProcessingTime::from_option(Some(7)),
        );
        builder.set_vessel_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(1),
            ProcessingTime::none(),
        );

        builder.set_vessel_processing_time(
            VesselIndex::new(2),
            BerthIndex::new(0),
            ProcessingTime::none(),
        );
        builder.set_vessel_processing_time(
            VesselIndex::new(2),
            BerthIndex::new(1),
            ProcessingTime::from_option(Some(6)),
        );

        let model = builder.build();
        let topo = StaticTopology::from_model(&model);

        let v0 = VesselIndex::new(0);
        let v1 = VesselIndex::new(1);
        let v2 = VesselIndex::new(2);

        // v0 neighbors: v1
        assert_eq!(topo.neighbors_of(v0), &[v1]);
        // v1 neighbors: v0
        assert_eq!(topo.neighbors_of(v1), &[v0]);
        // v2 neighbors: none
        assert_eq!(topo.neighbors_of(v2).len(), 0);

        assert!(topo.are_neighbors(v0, v1));
        assert!(topo.are_neighbors(v1, v0));
        assert!(!topo.are_neighbors(v0, v2));
        assert!(!topo.are_neighbors(v1, v2));
    }

    #[test]
    fn test_complete_graph_excludes_self_and_is_symmetric() {
        let n = 4;
        let mut builder = ModelBuilder::<Num>::new(1, n);
        // All vessels allowed on the single berth
        for v in 0..n {
            builder.set_vessel_processing_time(
                VesselIndex::new(v),
                BerthIndex::new(0),
                ProcessingTime::from_option(Some(1)),
            );
        }
        let model = builder.build();
        let topo = StaticTopology::from_model(&model);

        for i in 0..n {
            let vi = VesselIndex::new(i);
            let neighbors = topo.neighbors_of(vi);

            // Should have all other vessels as neighbors
            assert_eq!(neighbors.len(), n - 1);

            // Should not contain self
            assert!(!neighbors.contains(&vi));

            // Symmetry check and are_neighbors contract
            for j in 0..n {
                let vj = VesselIndex::new(j);
                if i == j {
                    assert!(!topo.are_neighbors(vi, vj));
                } else {
                    assert!(topo.are_neighbors(vi, vj));
                    assert!(topo.are_neighbors(vj, vi));
                }
            }
        }
    }
}
