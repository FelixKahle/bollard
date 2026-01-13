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
//! neighbors when they share at least one berth where both are allowed to dock.
//!
//! # Important: Time Window Independence
//!
//! Unlike standard scheduling problems where disjoint time windows imply independence, this topology
//! **ignores time windows** and defines neighbors solely based on physical compatibility.
//!
//! This is necessary because the solver employs an **append-only greedy decoder** (SSGS). In such a
//! decoder, the processing of a "late" vessel before an "early" vessel on the same berth advances
//! the berth's availability pointer permanently, potentially blocking the early vessel. Therefore,
//! the relative permutation order of *any* two vessels sharing a berth matters, even if their
//! arrival times are disjoint.
//!
//! # Storage Layout
//!
//! Neighborhoods are stored in a compressed sparse row (CSR) layout. All neighbor lists are flattened
//! into a single contiguous vector, and a sentinel offsets array maps each vessel to its slice of
//! neighbors. This design minimizes pointer chasing and leverages CPU cache locality for linear
//! scans, making rejection sampling and local search moves efficient.

use crate::neighborhood::neighborhoods::Neighborhoods;
use crate::num::SolverNumeric;
use bollard_model::{index::VesselIndex, model::Model};

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
            "called `Neighborhoods::neighbors_of_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
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
    /// This constructor identifies vessels that are "neighbors" based strictly on their compatibility
    /// with the available berths. Two vessels are considered neighbors if, and only if, there exists
    /// at least one berth where both vessels are allowed to dock.
    ///
    /// # Logical Rationale
    ///
    /// The topology ignores time window disjointness. This is because the append-only greedy decoder
    /// maintains a monotonically increasing "free time" pointer for each berth. If a late vessel is
    /// scheduled before an early vessel on the same berth, the early vessel is blocked, regardless of
    /// whether their original time windows overlapped. Therefore, any two vessels sharing a compatible
    /// berth must be considered interacting neighbors to allow the local search to correct their ordering.
    ///
    /// # Performance
    ///
    /// The constructor performs a one-time O(N^2 · M) analysis. The resulting topology is backed by
    /// a highly optimized CSR (Compressed Sparse Row) structure for O(1) slice access during the solve phase.
    pub fn from_model<T>(model: &Model<T>) -> Self
    where
        T: SolverNumeric,
    {
        let number_of_vessels = model.num_vessels();
        let number_of_berths = model.num_berths();

        // Heuristic capacity: N * (N-1) is worst case (dense).
        let mut flattened_neighbor_list: Vec<VesselIndex> = Vec::with_capacity(
            number_of_vessels.saturating_mul(number_of_vessels.saturating_sub(1)),
        );
        let mut neighbor_offsets: Vec<usize> = Vec::with_capacity(number_of_vessels + 1);

        // CSR sentinel: start with 0 so that offsets[i]..offsets[i+1] works for i=0.
        neighbor_offsets.push(0);

        // Outer loop over each source vessel i.
        for source_vessel_index in 0..number_of_vessels {
            let source_vessel = VesselIndex::new(source_vessel_index);

            // Inner loop over candidate neighbor vessels j.
            'candidate_neighbor: for candidate_vessel_index in 0..number_of_vessels {
                // Exclude self from adjacency.
                if source_vessel_index == candidate_vessel_index {
                    continue;
                }
                let candidate_vessel = VesselIndex::new(candidate_vessel_index);

                // Note: We do NOT check for time window overlap here.
                // Due to the append-only nature of the decoder, physical resource sharing
                // implies dependency regardless of time.

                // Check if there exists a berth where both vessels are allowed to dock.
                for berth_index in 0..number_of_berths {
                    let berth = bollard_model::index::BerthIndex::new(berth_index);

                    let source_allowed_on_berth =
                        model.vessel_allowed_on_berth(source_vessel, berth);
                    let candidate_allowed_on_berth =
                        model.vessel_allowed_on_berth(candidate_vessel, berth);

                    if source_allowed_on_berth && candidate_allowed_on_berth {
                        // Establish adjacency and move to the next candidate vessel.
                        flattened_neighbor_list.push(candidate_vessel);
                        continue 'candidate_neighbor;
                    }
                }
            }

            // Record the end of the current vessel's neighbor slice.
            neighbor_offsets.push(flattened_neighbor_list.len());
        }

        flattened_neighbor_list.shrink_to_fit();
        neighbor_offsets.shrink_to_fit();

        StaticTopology {
            neighbors: flattened_neighbor_list,
            offsets: neighbor_offsets,
        }
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

    fn build_model(number_of_berths: usize, number_of_vessels: usize) -> Model<Num> {
        ModelBuilder::<Num>::new(number_of_berths, number_of_vessels).build()
    }

    #[test]
    fn test_empty_model_topology() {
        let model = build_model(0, 0);
        let topology = StaticTopology::from_model(&model);

        assert_eq!(topology.num_vessels(), 0);
        // offsets must be num_vessels + 1 = 1
        assert_eq!(topology.offsets.len(), 1);
        assert_eq!(topology.neighbors.len(), 0);
        assert_eq!(topology.offsets[0], 0);
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

        let topology = StaticTopology::from_model(&model);
        assert_eq!(topology.num_vessels(), 1);

        let vessel_0 = VesselIndex::new(0);
        let neighbors_of_vessel_0 = topology.neighbors_of(vessel_0);
        assert_eq!(neighbors_of_vessel_0.len(), 0);
    }

    #[test]
    fn test_two_vessels_disjoint_berths_no_neighbors() {
        // 2 berths, 2 vessels
        // Even with infinite time, if they can't use the same berth, they are independent.
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
        let topology = StaticTopology::from_model(&model);

        let vessel_0 = VesselIndex::new(0);
        let vessel_1 = VesselIndex::new(1);

        assert_eq!(topology.neighbors_of(vessel_0).len(), 0);
        assert_eq!(topology.neighbors_of(vessel_1).len(), 0);
        assert!(!topology.are_neighbors(vessel_0, vessel_1));
        assert!(!topology.are_neighbors(vessel_1, vessel_0));
    }

    #[test]
    fn test_two_vessels_shared_berth_are_neighbors() {
        let mut builder = ModelBuilder::<Num>::new(2, 2);
        // Both vessels allowed on berth 0
        for vessel_id in 0..2 {
            builder.set_vessel_processing_time(
                VesselIndex::new(vessel_id),
                BerthIndex::new(0),
                ProcessingTime::from_option(Some(10)),
            );
            // Disallow berth 1
            builder.set_vessel_processing_time(
                VesselIndex::new(vessel_id),
                BerthIndex::new(1),
                ProcessingTime::none(),
            );
        }

        let model = builder.build();
        let topology = StaticTopology::from_model(&model);

        let vessel_0 = VesselIndex::new(0);
        let vessel_1 = VesselIndex::new(1);

        let neighbors_of_v0 = topology.neighbors_of(vessel_0);
        let neighbors_of_v1 = topology.neighbors_of(vessel_1);

        assert_eq!(neighbors_of_v0, &[vessel_1]);
        assert_eq!(neighbors_of_v1, &[vessel_0]);
        assert!(topology.are_neighbors(vessel_0, vessel_1));
        assert!(topology.are_neighbors(vessel_1, vessel_0));
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
        let topology = StaticTopology::from_model(&model);

        let vessel_0 = VesselIndex::new(0);
        let vessel_1 = VesselIndex::new(1);
        let vessel_2 = VesselIndex::new(2);

        // v0 neighbors: v1
        assert_eq!(topology.neighbors_of(vessel_0), &[vessel_1]);
        // v1 neighbors: v0
        assert_eq!(topology.neighbors_of(vessel_1), &[vessel_0]);
        // v2 neighbors: none
        assert_eq!(topology.neighbors_of(vessel_2).len(), 0);

        assert!(topology.are_neighbors(vessel_0, vessel_1));
        assert!(topology.are_neighbors(vessel_1, vessel_0));
        assert!(!topology.are_neighbors(vessel_0, vessel_2));
        assert!(!topology.are_neighbors(vessel_1, vessel_2));
    }

    #[test]
    fn test_complete_graph_excludes_self_and_is_symmetric() {
        let number_of_vessels = 4;
        let mut builder = ModelBuilder::<Num>::new(1, number_of_vessels);
        // All vessels allowed on the single berth
        for vessel_id in 0..number_of_vessels {
            builder.set_vessel_processing_time(
                VesselIndex::new(vessel_id),
                BerthIndex::new(0),
                ProcessingTime::from_option(Some(1)),
            );
        }
        let model = builder.build();
        let topology = StaticTopology::from_model(&model);

        for vessel_idx in 0..number_of_vessels {
            let vessel_i = VesselIndex::new(vessel_idx);
            let neighbors = topology.neighbors_of(vessel_i);

            // Should have all other vessels as neighbors
            assert_eq!(neighbors.len(), number_of_vessels - 1);

            // Should not contain self
            assert!(!neighbors.contains(&vessel_i));

            // Symmetry check and are_neighbors contract
            for other_idx in 0..number_of_vessels {
                let vessel_j = VesselIndex::new(other_idx);
                if vessel_idx == other_idx {
                    assert!(!topology.are_neighbors(vessel_i, vessel_j));
                } else {
                    assert!(topology.are_neighbors(vessel_i, vessel_j));
                    assert!(topology.are_neighbors(vessel_j, vessel_i));
                }
            }
        }
    }

    #[test]
    fn test_shared_berth_but_disjoint_times_are_neighbors() {
        let mut builder = ModelBuilder::<Num>::new(1, 2);

        // Both vessels allowed on the same berth.
        for vessel_id in 0..2 {
            builder.set_vessel_processing_time(
                VesselIndex::new(vessel_id),
                BerthIndex::new(0),
                ProcessingTime::from_option(Some(10)),
            );
        }

        // Set completely disjoint time windows
        builder.set_vessel_arrival_time(VesselIndex::new(0), 0);
        builder.set_vessel_latest_departure_time(VesselIndex::new(0), 100);
        builder.set_vessel_arrival_time(VesselIndex::new(1), 100);
        builder.set_vessel_latest_departure_time(VesselIndex::new(1), 200);

        let model = builder.build();
        let topology = StaticTopology::from_model(&model);

        let vessel_0 = VesselIndex::new(0);
        let vessel_1 = VesselIndex::new(1);

        // Because of the append-only decoder, these MUST be neighbors to allow
        // the solver to swap them if they come in the wrong order.
        assert!(topology.are_neighbors(vessel_0, vessel_1));
        assert!(topology.are_neighbors(vessel_1, vessel_0));
        assert_eq!(topology.neighbors_of(vessel_0), &[vessel_1]);
    }
}
