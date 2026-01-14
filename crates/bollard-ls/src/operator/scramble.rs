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

//! The Scramble (Shuffle) Operator.
//!
//! This module implements a stochastic local search operator that randomizes the order
//! of vessels within a contiguous sub-segment of the priority queue.
//!
//! # Mechanics
//!
//! Given a window defined by `[start, end)`, the operator applies a random shuffle
//! to the vessels in that range.
//!
//! - **Stochasticity**: Unlike deterministic operators (Swap, Shift, 2-Opt), this operator
//!   generates a *random* neighbor for a given window size. Evaluating the same window
//!   twice may yield different results.
//! - **Usage**: It is typically used to escape basins of attraction where the relative
//!   ordering of a group of vessels is completely wrong, and step-wise moves (swaps)
//!   cannot easily navigate to the correct permutation.
//!
//! # Search Space
//!
//! The operator iterates through all windows `[i, j)` where `j - i >= 2`.
//!
//! - `i`: Start index (inclusive).
//! - `j`: End index (exclusive).
//!
//! For each window, exactly one random permutation is attempted per pass.

use crate::{
    mutator::Mutator, operator::local_search_operator::LocalSearchOperator,
    queue::VesselPriorityQueue,
};
use bollard_model::solution::Solution;
use bollard_search::{neighborhood::neighborhoods::Neighborhoods, num::SolverNumeric};
use rand::Rng;

/// An operator that shuffles a contiguous range of vessels.
///
/// Holds its own Random Number Generator (RNG) to perform the shuffles during
/// neighbor generation.
#[derive(Debug, Clone)]
pub struct ScrambleOperator<T, N, R> {
    i: usize,           // Start index (inclusive)
    j: usize,           // End index (exclusive)
    num_vessels: usize, // Total vessels
    rng: R,             // The RNG used for shuffling
    _phantom: std::marker::PhantomData<(T, N)>,
}

impl<T, N, R> ScrambleOperator<T, N, R>
where
    R: Rng,
{
    /// Creates a new `ScrambleOperator` with the provided RNG.
    ///
    /// # Arguments
    ///
    /// * `rng` - The random number generator to use for shuffling.
    pub fn new(rng: R) -> Self {
        Self {
            i: 0,
            j: 0,
            num_vessels: 0,
            rng,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, N, R> LocalSearchOperator<T, N> for ScrambleOperator<T, N, R>
where
    T: SolverNumeric,
    N: Neighborhoods,
    R: Rng + Clone + Send + Sync,
{
    fn name(&self) -> &str {
        "ScrambleOperator"
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
        // Need at least 2 vessels to shuffle.
        if self.num_vessels < 2 {
            return false;
        }

        // Advance the cursor `j` (exclusive end of range).
        self.j += 1;

        // If j exceeds bounds, advance `i` (start) and reset `j`.
        if self.j > self.num_vessels {
            self.i += 1;
            // Minimum window size is 2.
            // If i + 2 > num_vessels, we can't form any more valid windows.
            if self.i + 2 > self.num_vessels {
                return false;
            }
            self.j = self.i + 2;
        }

        // Safety check (redundant given logic above, but good for invariants)
        if self.i >= self.num_vessels || self.j > self.num_vessels {
            return false;
        }

        debug_assert!(
            self.i < self.j && self.j <= self.num_vessels,
            "ScrambleOperator cursors out of bounds: i={}, j={}, len={}",
            self.i,
            self.j,
            self.num_vessels
        );

        // Apply Mutation
        // mutator.shuffle uses an exclusive end index (start..end).
        // We pass our internal RNG.
        mutator.shuffle(self.i, self.j, &mut self.rng);

        true
    }

    fn reset(&mut self) {
        self.i = 0;
        // Start `j` at 1. The first loop iteration increments it to 2.
        // This sets the first window to [0, 2) -> length 2.
        self.j = 1;
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
    use rand::{SeedableRng, rngs::StdRng};

    fn build_model(num_berths: usize, num_vessels: usize) -> bollard_model::model::Model<i64> {
        ModelBuilder::<i64>::new(num_berths, num_vessels).build()
    }

    #[test]
    fn test_scramble_iterates_bounds_correctly() {
        // Setup a model with 3 vessels
        let mut builder = ModelBuilder::<i64>::new(1, 3);
        for v in 0..3 {
            builder.set_vessel_arrival_time(VesselIndex::new(v), v as i64);
        }
        let model = builder.build();
        let topology = StaticTopology::from_model(&model);
        let solution = Solution::<i64>::new(0, vec![BerthIndex::new(0); 3], vec![0, 1, 2]);

        let mut memory = SearchMemory::<i64>::new();
        memory.initialize(&solution);
        let (schedule, mutator) = memory.prepare_operator();
        let mut mutator = mutator;

        // Use deterministic RNG
        let rng = StdRng::seed_from_u64(42);
        let mut op = ScrambleOperator::<i64, StaticTopology, StdRng>::new(rng);
        op.prepare(schedule, mutator.queue(), &topology);

        // Expected Windows for N=3, min_len=2:
        // 1. [0, 2) -> indices 0, 1
        // 2. [0, 3) -> indices 0, 1, 2
        // 3. [1, 3) -> indices 1, 2

        // Move 1: [0, 2)
        assert!(op.next_neighbor(schedule, &mut mutator, &topology));

        // Move 2: [0, 3)
        assert!(op.next_neighbor(schedule, &mut mutator, &topology));

        // Move 3: [1, 3)
        assert!(op.next_neighbor(schedule, &mut mutator, &topology));

        // Exhausted
        assert!(!op.next_neighbor(schedule, &mut mutator, &topology));
    }

    #[test]
    fn test_scramble_actually_changes_data() {
        // With a fixed seed, we expect *some* change.
        // N=4 to give the shuffle enough room to likely produce a diff.
        let mut builder = ModelBuilder::<i64>::new(1, 4);
        for v in 0..4 {
            builder.set_vessel_arrival_time(VesselIndex::new(v), v as i64);
        }
        let model = builder.build();
        let topology = StaticTopology::from_model(&model);
        let solution = Solution::<i64>::new(0, vec![BerthIndex::new(0); 4], vec![0, 1, 2, 3]);

        let mut memory = SearchMemory::<i64>::new();
        memory.initialize(&solution);
        let (schedule, mutator) = memory.prepare_operator();
        let mut mutator = mutator;

        let rng = StdRng::seed_from_u64(12345);
        let mut op = ScrambleOperator::<i64, StaticTopology, StdRng>::new(rng);
        op.prepare(schedule, mutator.queue(), &topology);

        let initial_state = mutator.queue().buffer().to_vec();

        let mut changed_at_least_once = false;

        // Iterate through all neighbors
        while op.next_neighbor(schedule, &mut mutator, &topology) {
            if mutator.queue().buffer() != initial_state.as_slice() {
                changed_at_least_once = true;
            }
        }

        assert!(
            changed_at_least_once,
            "Scramble operator should modify the queue with sufficient RNG entropy"
        );
    }

    #[test]
    fn test_insufficient_vessels() {
        let model = build_model(1, 1);
        let topology = StaticTopology::from_model(&model);
        let solution = Solution::<i64>::new(0, vec![BerthIndex::new(0)], vec![0]);
        let mut memory = SearchMemory::<i64>::new();
        memory.initialize(&solution);
        let (schedule, mutator) = memory.prepare_operator();
        let mut mutator = mutator;

        let rng = StdRng::seed_from_u64(0);
        let mut op = ScrambleOperator::<i64, StaticTopology, StdRng>::new(rng);
        op.prepare(schedule, mutator.queue(), &topology);

        assert!(!op.next_neighbor(schedule, &mut mutator, &topology));
    }
}
