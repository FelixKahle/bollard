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

//! The 2-Opt Operator.
//!
//! This module implements the 2-Opt operator, a classic local search move typically
//! used in Traveling Salesperson Problems (TSP) to uncross edges. In the context of
//! the Berth Allocation Problem (and permutation sequences in general), it functions
//! by **reversing the order** of a contiguous sub-segment of vessels in the priority queue.
//!
//! # Mechanics
//!
//! Given two indices `i` and `j` (where `i < j`), the operator reverses the slice
//! `queue[i..=j]`.
//!
//! Example:
//! - Queue: `[A, B, C, D, E]`
//! - Move: `2-Opt(1, 3)` (indices of B and D)
//! - Result: `[A, D, C, B, E]`
//!
//! # Search Space
//!
//! The operator explores the neighborhood defined by all pairs `(i, j)` where `i < j`.
//!
//! Similar to the `ShiftOperator`, 2-Opt is treated as an **exhaustive** operator that
//! does not utilize `Neighborhoods` for pruning.
//!
//! - **Rationale**: Reversing a segment affects the relative processing order of
//!   every vessel within that segment. Even if the endpoints `i` and `j` are not
//!   explicit neighbors in the resource graph, the reversal may reorder intermediate
//!   vessels that *do* contend for resources.
//! - **Completeness**: Pruning based on endpoint connectivity would miss beneficial
//!   reorderings of the internal sub-sequence.
//!
//! This operator is particularly effective at escaping local optima where a block of
//! vessels is roughly in the correct position but strictly backwards (e.g., due to
//! arrival time vs. deadline inversions).

use crate::{
    mutator::Mutator, operator::local_search_operator::LocalSearchOperator,
    queue::VesselPriorityQueue,
};
use bollard_model::solution::Solution;
use bollard_search::{neighborhood::neighborhoods::Neighborhoods, num::SolverNumeric};

/// An operator that reverses a contiguous range of vessels in the priority queue.
///
/// It maintains iteration cursors `i` (start) and `j` (end) to enumerate all
/// unique sub-segments of length >= 2.
#[derive(Debug, Clone, Default)]
pub struct TwoOptOperator<T, N> {
    i: usize,           // Start index of the range
    j: usize,           // End index of the range
    num_vessels: usize, // Total vessels
    _phantom: std::marker::PhantomData<(T, N)>,
}

impl<T, N> TwoOptOperator<T, N> {
    /// Creates a new `TwoOptOperator`.
    pub fn new() -> Self {
        Self {
            i: 0,
            j: 0,
            num_vessels: 0,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, N> LocalSearchOperator<T, N> for TwoOptOperator<T, N>
where
    T: SolverNumeric,
    N: Neighborhoods,
{
    fn name(&self) -> &str {
        "TwoOptOperator"
    }

    fn prepare(&mut self, _schedule: &Solution<T>, queue: &VesselPriorityQueue, _n: &N) {
        self.num_vessels = queue.len();
        self.reset();
    }

    fn next_neighbor(
        &mut self,
        _schedule: &Solution<T>,
        mutator: &mut Mutator<T>,
        _neighborhoods: &N,
    ) -> bool {
        // Need at least 2 vessels to reverse a range.
        if self.num_vessels < 2 {
            return false;
        }

        // Advance the cursor `j` (end of range).
        self.j += 1;

        // If j exceeds bounds, advance `i` (start of range) and reset `j`.
        if self.j >= self.num_vessels {
            self.i += 1;
            // The smallest valid 2-opt moves a range of 2 elements, so j starts at i + 1.
            self.j = self.i + 1;
        }

        // Termination: If i reaches the second-to-last element, we have exhausted
        // all pairs (i, j) where i < j.
        // If i == num_vessels - 1, then j would be num_vessels (out of bounds).
        if self.i >= self.num_vessels - 1 {
            return false;
        }

        // Invariant Check:
        // 1. i < j (ensured by initialization and loop logic)
        // 2. j < num_vessels (ensured by loop condition)
        debug_assert!(
            self.i < self.j && self.j < self.num_vessels,
            "TwoOptOperator cursors out of bounds: i={}, j={}, len={}",
            self.i,
            self.j,
            self.num_vessels
        );

        // Apply Mutation
        // We delegate directly to the Mutator. It handles:
        // 1. Backing up the range [i..=j] to the UndoLog.
        // 2. Reversing the slice in place.
        // 3. Ensuring safety constraints (start <= end).
        mutator.reverse(self.i, self.j);

        true
    }

    fn reset(&mut self) {
        self.i = 0;
        // We set j to 0 so that the first execution of next_neighbor:
        // 1. Increments j to 1.
        // 2. Checks bounds (1 < N).
        // 3. Returns pair (0, 1).
        self.j = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::SearchMemory;
    use bollard_model::index::{BerthIndex, VesselIndex};
    use bollard_model::model::ModelBuilder;
    use bollard_model::solution::Solution;
    use bollard_search::neighborhood::topology::StaticTopology;

    fn build_model(num_berths: usize, num_vessels: usize) -> bollard_model::model::Model<i64> {
        ModelBuilder::<i64>::new(num_berths, num_vessels).build()
    }

    #[test]
    fn test_next_neighbor_insufficient_vessels() {
        // 0 vessels
        let model = build_model(0, 0);
        let topology = StaticTopology::from_model(&model);
        let solution = Solution::<i64>::new(0, Vec::new(), Vec::new());
        let mut memory = SearchMemory::<i64>::new();
        memory.initialize(&solution);
        let (schedule, mutator) = memory.prepare_operator();
        let mut mutator = mutator;

        let mut op = TwoOptOperator::<i64, StaticTopology>::new();
        op.prepare(schedule, mutator.queue(), &topology);

        assert!(!op.next_neighbor(schedule, &mut mutator, &topology));

        // 1 vessel (cannot reverse a range of size > 1)
        let mut b2 = ModelBuilder::<i64>::new(1, 1);
        b2.set_vessel_arrival_time(VesselIndex::new(0), 0);
        let m2 = b2.build();
        let t2 = StaticTopology::from_model(&m2);
        let s2 = Solution::<i64>::new(0, vec![BerthIndex::new(0)], vec![0]);
        memory.initialize(&s2);
        let (sched2, mut mut2) = memory.prepare_operator();

        let mut op2 = TwoOptOperator::<i64, StaticTopology>::new();
        op2.prepare(sched2, mut2.queue(), &t2);
        assert!(!op2.next_neighbor(sched2, &mut mut2, &t2));
    }

    #[test]
    fn test_two_opt_behavior_three_vessels() {
        // Setup a model with 3 vessels
        let mut builder = ModelBuilder::<i64>::new(1, 3);
        for v in 0..3 {
            builder.set_vessel_arrival_time(VesselIndex::new(v), v as i64);
        }
        let model = builder.build();
        let topology = StaticTopology::from_model(&model);

        // Initial Queue: [0, 1, 2]
        let solution = Solution::<i64>::new(0, vec![BerthIndex::new(0); 3], vec![0, 1, 2]);

        let mut memory = SearchMemory::<i64>::new();
        memory.initialize(&solution);
        let (schedule, mutator) = memory.prepare_operator();
        let mut mutator = mutator;

        let mut op = TwoOptOperator::<i64, StaticTopology>::new();
        op.prepare(schedule, mutator.queue(), &topology);

        // 1. Move: i=0, j=1. Reverse [0..=1].
        // Queue: [0, 1, 2] -> [1, 0, 2]
        let moved_1 = op.next_neighbor(schedule, &mut mutator, &topology);
        assert!(moved_1);
        let buf = mutator.queue().buffer();
        assert_eq!(buf[0].get(), 1);
        assert_eq!(buf[1].get(), 0);
        assert_eq!(buf[2].get(), 2);

        // 2. Move: i=0, j=2. Reverse [0..=2].
        // Applied to [1, 0, 2] -> [2, 0, 1]
        let moved_2 = op.next_neighbor(schedule, &mut mutator, &topology);
        assert!(moved_2);
        let buf2 = mutator.queue().buffer();
        assert_eq!(buf2[0].get(), 2);
        assert_eq!(buf2[1].get(), 0);
        assert_eq!(buf2[2].get(), 1);

        // 3. Move: loop i=0 exhausted. i=1, j=2.
        // Applied to [2, 0, 1]. Reverse [1..=2] -> [2, 1, 0]
        let moved_3 = op.next_neighbor(schedule, &mut mutator, &topology);
        assert!(moved_3);
        let buf3 = mutator.queue().buffer();
        assert_eq!(buf3[0].get(), 2);
        assert_eq!(buf3[1].get(), 1);
        assert_eq!(buf3[2].get(), 0);

        // 4. Move: Exhausted.
        let moved_4 = op.next_neighbor(schedule, &mut mutator, &topology);
        assert!(!moved_4);
    }

    #[test]
    fn test_reset_functionality() {
        let mut builder = ModelBuilder::<i64>::new(1, 2);
        builder.set_vessel_arrival_time(VesselIndex::new(0), 0);
        builder.set_vessel_arrival_time(VesselIndex::new(1), 1);
        let model = builder.build();
        let topology = StaticTopology::from_model(&model);

        let solution = Solution::<i64>::new(0, vec![BerthIndex::new(0); 2], vec![0, 1]);

        let mut memory = SearchMemory::<i64>::new();
        memory.initialize(&solution);
        let (schedule, mutator) = memory.prepare_operator();
        let mut mutator = mutator;

        let mut op = TwoOptOperator::<i64, StaticTopology>::new();
        op.prepare(schedule, mutator.queue(), &topology);

        // Move 1: (0, 1)
        assert!(op.next_neighbor(schedule, &mut mutator, &topology));

        // Exhaust
        assert!(!op.next_neighbor(schedule, &mut mutator, &topology));

        // Reset
        op.reset();

        // Should find (0, 1) again
        assert!(op.next_neighbor(schedule, &mut mutator, &topology));
    }
}
