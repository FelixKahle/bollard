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

//! The Shift Operator.
//!
//! This module implements the shift operator (also known as insertion), which moves a
//! single vessel from its current position to a new position in the priority queue.
//! Unlike the swap operator, which exchanges two items, the shift operator changes
//! the priority of one vessel while preserving the relative order of the others.
//!
//! # Search Space
//!
//! The operator explores the neighborhood defined by all pairs `(i, j)` where `i != j`.
//! - `i`: The source index (vessel to move).
//! - `j`: The target index (new position).
//!
//! # Topology Pruning
//!
//! To maintain efficiency, the operator consults the `Neighborhoods` topology.
//! A shift of vessel `u` (at `i`) to position `j` (occupied by vessel `v`) is only
//! attempted if `u` and `v` are neighbors. This heuristic assumes that reordering
//! disjoint (non-interacting) vessels is unlikely to yield significant improvements,
//! effectively focusing the search on resolving conflicts.

use crate::{
    memory::Schedule, mutator::Mutator, neighborhood::neighborhoods::Neighborhoods,
    operator::local_search_operator::LocalSearchOperator, queue::VesselPriorityQueue,
};
use bollard_search::num::SolverNumeric;

/// An operator that moves a vessel from one position to another.
///
/// This provides a finer granularity of modification than swapping, allowing a vessel
/// to be "nudged" earlier or later in the schedule without disrupting the relative
/// order of non-adjacent vessels.
#[derive(Debug, Clone, Default)]
pub struct ShiftOperator<T, N> {
    i: usize,                                   // Source index cursor
    j: usize,                                   // Target index cursor
    num_vessels: usize,                         // Total number of vessels in the queue
    _phantom: std::marker::PhantomData<(T, N)>, // Phantom data for generic types
}

impl<T, N> ShiftOperator<T, N> {
    /// Creates a new `ShiftOperator`.
    pub fn new() -> Self {
        Self {
            i: 0,
            j: 0,
            num_vessels: 0,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, N> LocalSearchOperator<T, N> for ShiftOperator<T, N>
where
    T: SolverNumeric,
    N: Neighborhoods,
{
    fn name(&self) -> &str {
        "ShiftOperator"
    }

    fn prepare(&mut self, _schedule: &Schedule<T>, queue: &VesselPriorityQueue, _n: &N) {
        self.num_vessels = queue.len();
        self.i = 0;
        // Start j at 0. The loop logic will increment it immediately to 1 on the first step
        // if i=0, effectively checking (0, 1) first.
        // If we want to check (0,0) (which is invalid), the loop handles it.
        // We set j=0 here to start the scan at the beginning of the row.
        self.j = 0;
    }

    fn next_neighbor(
        &mut self,
        _schedule: &Schedule<T>,
        mutator: &mut Mutator<T>,
        neighborhoods: &N,
    ) -> bool {
        // Need at least 2 vessels to perform a shift.
        if self.num_vessels < 2 {
            return false;
        }

        loop {
            // Advance the cursors.
            // We iterate j from 0 to N-1, then increment i.
            self.j += 1;

            if self.j >= self.num_vessels {
                self.i += 1;
                self.j = 0;
            }

            // Termination: If source index exceeds bounds, we have scanned the whole matrix.
            if self.i >= self.num_vessels {
                return false;
            }

            // Skip "no-op" moves where source and target are the same.
            if self.i == self.j {
                continue;
            }

            debug_assert!(
                self.i < self.num_vessels && self.j < self.num_vessels,
                "ShiftOperator cursors out of bounds: i={}, j={}, len={}",
                self.i,
                self.j,
                self.num_vessels
            );

            // Topology Check:
            // We interpret the move "Shift i to j" as an interaction between
            // the vessel moving (u) and the vessel currently at the target (v).
            // If they are not neighbors (e.g., disjoint time windows), moving u
            // next to v is unlikely to resolve a bottleneck relevant to v.
            let u = mutator.get_unchecked(self.i);
            let v = mutator.get_unchecked(self.j);

            // Safety: Neighborhood bounds checked by implementation.
            let are_neighbors = unsafe { neighborhoods.are_neighbors_unchecked(u, v) };

            if are_neighbors {
                // Apply Mutation: Move element at `i` to `j`.
                mutator.shift(self.i, self.j);
                return true;
            }
        }
    }

    fn reset(&mut self) {
        self.i = 0;
        self.j = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::SearchMemory;
    use crate::neighborhood::topology::StaticTopology;
    use bollard_model::index::{BerthIndex, VesselIndex};
    use bollard_model::model::ModelBuilder;
    use bollard_model::solution::Solution;
    use bollard_model::time::ProcessingTime;

    fn build_model(num_berths: usize, num_vessels: usize) -> bollard_model::model::Model<i64> {
        ModelBuilder::<i64>::new(num_berths, num_vessels).build()
    }

    #[test]
    fn test_next_neighbor_zero_vessels_returns_false() {
        // Empty model and topology
        let model = build_model(0, 0);
        let topology = StaticTopology::from_model(&model);

        // Construct an empty solution
        let solution = Solution::<i64>::new(0, Vec::new(), Vec::new());

        // Initialize search memory and get schedule/mutator
        let mut memory = SearchMemory::<i64>::new();
        memory.initialize(&solution);
        let (schedule, mutator) = memory.prepare_operator();
        let mut mutator = mutator;

        let mut op = ShiftOperator::<i64, StaticTopology>::new();
        // Use the queue reference from the mutator to avoid aliasing memory
        op.prepare(schedule, mutator.queue(), &topology);

        // No moves are available; should immediately return false.
        let moved = op.next_neighbor(schedule, &mut mutator, &topology);
        assert!(
            !moved,
            "next_neighbor should return false with zero vessels"
        );
    }

    #[test]
    fn test_non_neighbors_are_skipped_and_operator_terminates() {
        // Two vessels on disjoint berths (no shared berth) => not neighbors.
        let mut builder = ModelBuilder::<i64>::new(2, 2);

        // Vessel 0 allowed only on berth 0; time window [0, 10)
        builder.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(0),
            ProcessingTime::from_option(Some(10)),
        );
        builder.set_vessel_arrival_time(VesselIndex::new(0), 0);
        builder.set_vessel_latest_departure_time(VesselIndex::new(0), 10);

        // Vessel 1 allowed only on berth 1; time window [0, 10)
        builder.set_vessel_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(1),
            ProcessingTime::from_option(Some(10)),
        );
        builder.set_vessel_arrival_time(VesselIndex::new(1), 0);
        builder.set_vessel_latest_departure_time(VesselIndex::new(1), 10);

        let model = builder.build();
        let topology = StaticTopology::from_model(&model);

        // Build a solution with start times to ensure queue order [0, 1]
        let berths = vec![BerthIndex::new(0), BerthIndex::new(1)];
        let start_times = vec![0, 1]; // strictly increasing, stable order
        let solution = Solution::<i64>::new(0, berths, start_times);

        // Initialize search memory and get schedule/mutator
        let mut memory = SearchMemory::<i64>::new();
        memory.initialize(&solution);
        let (schedule, mutator) = memory.prepare_operator();
        let mut mutator = mutator;

        let mut op = ShiftOperator::<i64, StaticTopology>::new();
        op.prepare(schedule, mutator.queue(), &topology);

        // Operator scans possible (i,j) pairs with i != j. Since the only pair interacts
        // non-neighbors, the operator should return false and eventually terminate.
        let moved_first = op.next_neighbor(schedule, &mut mutator, &topology);
        assert!(
            !moved_first,
            "next_neighbor should return false when all (i,j) pairs are not neighbors"
        );

        // Queue remains unchanged. Inspect via mutator.queue().
        let buf = mutator.queue().buffer();
        assert_eq!(buf.len(), 2);
        assert_eq!(buf[0].get(), 0);
        assert_eq!(buf[1].get(), 1);

        // Subsequent call should still return false (operator exhausted).
        let moved_second = op.next_neighbor(schedule, &mut mutator, &topology);
        assert!(
            !moved_second,
            "operator should terminate after exhausting all non-neighbor moves"
        );
    }

    #[test]
    fn test_neighbors_shift_behavior_two_vessels() {
        // Two vessels sharing the same berth with overlapping windows => neighbors.
        let mut builder = ModelBuilder::<i64>::new(1, 2);

        // Allow both vessels on berth 0
        builder.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(0),
            ProcessingTime::from_option(Some(5)),
        );
        builder.set_vessel_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(0),
            ProcessingTime::from_option(Some(5)),
        );

        // Overlapping windows: vessel 0 [0, 10), vessel 1 [5, 15)
        builder.set_vessel_arrival_time(VesselIndex::new(0), 0);
        builder.set_vessel_latest_departure_time(VesselIndex::new(0), 10);
        builder.set_vessel_arrival_time(VesselIndex::new(1), 5);
        builder.set_vessel_latest_departure_time(VesselIndex::new(1), 15);

        let model = builder.build();
        let topology = StaticTopology::from_model(&model);

        // Build a solution with start times to ensure initial queue order [0, 1]
        let berths = vec![BerthIndex::new(0), BerthIndex::new(0)];
        let start_times = vec![0, 1]; // strictly increasing
        let solution = Solution::<i64>::new(0, berths, start_times);

        // Initialize search memory and get schedule/mutator
        let mut memory = SearchMemory::<i64>::new();
        memory.initialize(&solution);
        let (schedule, mutator) = memory.prepare_operator();
        let mut mutator = mutator;

        let mut op = ShiftOperator::<i64, StaticTopology>::new();
        op.prepare(schedule, mutator.queue(), &topology);

        // First call: cursors start at i=0, j=0, then j increments to 1 => (0,1).
        // They are neighbors, so it should shift index 0 to position 1.
        // Starting buffer [0,1] -> shift(0 -> 1) => [1,0]
        let moved_first = op.next_neighbor(schedule, &mut mutator, &topology);
        assert!(moved_first, "expected a shift for a neighboring pair");

        let buf = mutator.queue().buffer();
        assert_eq!(buf.len(), 2);
        assert_eq!(buf[0].get(), 1, "queue head should now be vessel 1");
        assert_eq!(buf[1].get(), 0, "queue tail should now be vessel 0");

        // Second call: the operator continues scanning; with two vessels,
        // it will reach (i=1, j=0) and perform a valid neighbor shift again.
        // Current buffer [1,0] -> shift(1 -> 0) => [0,1]
        let moved_second = op.next_neighbor(schedule, &mut mutator, &topology);
        assert!(
            moved_second,
            "expected a second valid shift for the remaining (i,j) pair"
        );

        let buf2 = mutator.queue().buffer();
        assert_eq!(buf2[0].get(), 0);
        assert_eq!(buf2[1].get(), 1);

        // Third call: all (i,j) combinations for i!=j are exhausted; operator should terminate.
        let moved_third = op.next_neighbor(schedule, &mut mutator, &topology);
        assert!(
            !moved_third,
            "operator should terminate after exhausting all moves"
        );

        // Reset should allow re-iteration from the beginning; it will again find (0,1)
        // and perform the shift to produce [1,0].
        op.reset();
        op.prepare(schedule, mutator.queue(), &topology);

        let moved_fourth = op.next_neighbor(schedule, &mut mutator, &topology);
        assert!(moved_fourth, "expected a shift after reset and prepare");

        let buf3 = mutator.queue().buffer();
        assert_eq!(buf3[0].get(), 1);
        assert_eq!(buf3[1].get(), 0);
    }

    #[test]
    fn test_neighbors_shift_nontrivial_three_vessels() {
        // Three vessels all on the same berth, overlapping windows => neighbors among all.
        let mut builder = ModelBuilder::<i64>::new(1, 3);

        // All vessels allowed on berth 0
        for v in 0..3 {
            builder.set_vessel_processing_time(
                VesselIndex::new(v),
                BerthIndex::new(0),
                ProcessingTime::from_option(Some(5)),
            );
            builder.set_vessel_arrival_time(VesselIndex::new(v), v as i64); // staggered arrivals
            builder.set_vessel_latest_departure_time(VesselIndex::new(v), 20);
        }

        let model = builder.build();
        let topology = StaticTopology::from_model(&model);

        // Initial queue order [0, 1, 2]
        let berths = vec![BerthIndex::new(0), BerthIndex::new(0), BerthIndex::new(0)];
        let start_times = vec![0, 1, 2];
        let solution = Solution::<i64>::new(0, berths, start_times);

        // Initialize search memory and get schedule/mutator
        let mut memory = SearchMemory::<i64>::new();
        memory.initialize(&solution);
        let (schedule, mutator) = memory.prepare_operator();
        let mut mutator = mutator;

        let mut op = ShiftOperator::<i64, StaticTopology>::new();
        op.prepare(schedule, mutator.queue(), &topology);

        // First valid neighbor move encountered by the operator:
        // i=0, j=1 => shift(0 -> 1): [0,1,2] -> [1,0,2]
        let moved_first = op.next_neighbor(schedule, &mut mutator, &topology);
        assert!(moved_first, "expected first neighbor shift");

        let buf = mutator.queue().buffer();
        assert_eq!(
            buf,
            &[
                VesselIndex::new(1),
                VesselIndex::new(0),
                VesselIndex::new(2)
            ]
        );

        // Continue: next few calls should traverse additional (i,j) pairs.
        // Next valid is typically i=0, j=2 (after scanning mechanics), which would
        // now refer to the current indices. Based on the operator's cursors,
        // it will progress and apply another valid shift when encountering neighbors.

        // Second move: expect another neighbor shift. The exact pair depends on the cursor,
        // but the queue should change accordingly. We just assert that it moves something.
        let moved_second = op.next_neighbor(schedule, &mut mutator, &topology);
        assert!(moved_second, "expected second neighbor shift");

        // Sanity: queue still has the same elements but reordered.
        let buf2 = mutator.queue().buffer();
        assert_eq!(buf2.len(), 3);
        let set_sorted = {
            let mut v = vec![buf2[0].get(), buf2[1].get(), buf2[2].get()];
            v.sort();
            v
        };
        assert_eq!(
            set_sorted,
            vec![0, 1, 2],
            "all vessels must still be present"
        );

        // Keep advancing until termination to ensure the operator eventually ends.
        // With 3 vessels, there are 6 (i,j) with i!=j. Some may be skipped; eventually it terminates.
        while op.next_neighbor(schedule, &mut mutator, &topology) {}

        // After exhaustion, another call should still return false.
        let moved_final = op.next_neighbor(schedule, &mut mutator, &topology);
        assert!(
            !moved_final,
            "operator should terminate after exhausting all (i,j) neighbor moves"
        );
    }
}
