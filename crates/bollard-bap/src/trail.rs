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

use crate::{constraint::BranchConstraint, pricing::PricingOracle};

/// A high-performance, linear undo log for Branch-and-Price constraints.
///
/// This structure manages the state of the `PricingOracle` by recording applied constraints
/// in a contiguous stack (`entries`) and marking decision levels with `frames`.
///
/// # Architecture
/// * **Linear Log:** Constraints are stored sequentially in `entries`. This improves cache
///   locality compared to linked-list based undo logs.
/// * **Frames:** A secondary stack `frames` records the start index of each decision level.
///   Backtracking simply pops a frame and reverts all entries after that index.
///
/// # Performance
/// * **Static Dispatch:** Methods are generic over `O: PricingOracle`, allowing the compiler
///   to inline `add_constraint` and `remove_constraint` calls.
/// * **Zero-Cost Abstraction:** No vtable lookups during the hot backtracking loop.
/// * **Preallocation:** Supports `ensure_capacity` to prevent resizing during the search.
#[derive(Debug, Clone, Default)]
pub struct BnpTrail {
    /// The linear history of applied constraints.
    /// Acts as the "Undo Stack".
    entries: Vec<BranchConstraint>,

    /// The stack of frame start indices.
    /// `frames[i]` stores the index in `entries` where depth `i` began.
    frames: Vec<usize>,
}

impl BnpTrail {
    /// Creates a new, empty trail.
    #[inline]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            frames: Vec::new(),
        }
    }

    /// Creates a trail with pre-allocated capacity.
    ///
    /// # Arguments
    /// * `capacity`: The estimated maximum depth of the tree (number of constraints).
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            frames: Vec::with_capacity(capacity),
        }
    }

    /// Ensures the trail has capacity for a specified depth to avoid reallocations.
    pub fn ensure_capacity(&mut self, capacity: usize) {
        if self.entries.capacity() < capacity {
            self.entries.reserve(capacity - self.entries.capacity());
        }
        if self.frames.capacity() < capacity {
            self.frames.reserve(capacity - self.frames.capacity());
        }
    }

    /// Pushes a new frame marker onto the stack.
    ///
    /// This must be called **before** applying any constraints for a new tree node.
    /// It marks the "checkpoint" to which `backtrack` will return.
    #[inline]
    pub fn push_frame(&mut self) {
        self.frames.push(self.entries.len());
    }

    /// Applies a constraint to the Oracle and records it in the trail.
    ///
    /// # Arguments
    /// * `constraint`: The branching decision to enforce.
    /// * `oracle`: The solver that needs to respect this constraint.
    #[inline]
    pub fn apply<O: PricingOracle>(&mut self, constraint: BranchConstraint, oracle: &mut O) {
        // 1. Enforce rule in the Subproblem (Static Dispatch)
        oracle.add_constraint(constraint);

        // 2. Log for undo
        self.entries.push(constraint);
    }

    /// Backtracks to the start of the current frame (pops the frame) without
    /// performing any runtime checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `self.frames` is non-empty.
    /// - The last frame index in `self.frames` is a valid index into `self.entries`
    ///   (i.e., `start_index <= self.entries.len()`).
    /// - For every constraint about to be removed, `oracle.remove_constraint` may
    ///   be called safely and maintains all invariants expected by the caller.
    pub unsafe fn backtrack_unchecked<O: PricingOracle>(&mut self, oracle: &mut O) -> usize {
        debug_assert!(
            !self.frames.is_empty(),
            "called `Trail::backtrack_unchecked` with empty frames"
        );

        let start_index = unsafe { self.frames.pop().unwrap_unchecked() };

        debug_assert!(
            start_index <= self.entries.len(),
            "`Trail::backtrack_unchecked`: start_index {} is out of bounds for entries (len = {})",
            start_index,
            self.entries.len(),
        );

        let mut removed_count = 0;

        // Iterate backwards from current top to the frame's start
        while self.entries.len() > start_index {
            debug_assert!(
                !self.entries.is_empty(),
                "`Trail::backtrack_unchecked`: entries must be non-empty when popping"
            );

            // SAFETY:
            // - Loop condition (`self.entries.len() > start_index`) ensures `entries.len() > 0`,
            //   so `pop` returns `Some`.
            // - Caller guarantees `start_index` is a valid frame boundary.
            let constraint = unsafe { self.entries.pop().unwrap_unchecked() };
            oracle.remove_constraint(constraint);
            removed_count += 1;
        }

        removed_count
    }

    /// Backtracks to the start of the current frame (pops the frame).
    ///
    /// This undoes all constraints applied since the last `push_frame()`.
    /// Returns the number of constraints removed.
    pub fn backtrack<O: PricingOracle>(&mut self, oracle: &mut O) -> usize {
        let start_index = match self.frames.pop() {
            Some(idx) => idx,
            None => {
                debug_assert!(
                    self.entries.is_empty(),
                    "`Trail::backtrack`: no frames but entries is non-empty (len = {})",
                    self.entries.len(),
                );
                return 0;
            }
        };

        let mut removed_count = 0;

        debug_assert!(
            start_index <= self.entries.len(),
            "`Trail::backtrack`: start_index {} is out of bounds for entries (len = {})",
            start_index,
            self.entries.len(),
        );

        // Iterate backwards from current top to the frame's start
        while self.entries.len() > start_index {
            debug_assert!(
                !self.entries.is_empty(),
                "`Trail::backtrack`: entries must be non-empty when popping"
            );

            // SAFETY: The loop condition (`self.entries.len() > start_index`) together with the
            // debug_assert! above guarantees that `entries` is non-empty here.
            let constraint = unsafe { self.entries.pop().unwrap_unchecked() };
            oracle.remove_constraint(constraint);
            removed_count += 1;
        }

        removed_count
    }

    /// Backtracks until the trail is at `target_depth` (number of frames).
    ///
    /// This is useful for jumping up multiple levels in the tree.
    pub fn backtrack_to_depth<O: PricingOracle>(&mut self, target_depth: usize, oracle: &mut O) {
        while self.frames.len() > target_depth {
            self.backtrack(oracle);
        }
    }

    /// Clears the trail and resets the Oracle state entirely.
    ///
    /// This is safer/faster than repeated backtracking if restarting the search.
    pub fn reset<O: PricingOracle>(&mut self, oracle: &mut O) {
        // Unwind all constraints
        while let Some(constraint) = self.entries.pop() {
            oracle.remove_constraint(constraint);
        }

        // Reset pointers
        self.entries.clear();
        self.frames.clear();
    }

    /// Returns the current depth (number of active frames).
    #[inline]
    pub fn current_depth(&self) -> usize {
        self.frames.len()
    }

    /// Returns the total number of active constraints.
    #[inline]
    pub fn num_active_constraints(&self) -> usize {
        self.entries.len()
    }

    /// Returns the memory usage in bytes.
    #[inline]
    pub fn allocated_memory_bytes(&self) -> usize {
        (self.entries.capacity() * std::mem::size_of::<BranchConstraint>())
            + (self.frames.capacity() * std::mem::size_of::<usize>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use bollard_model::index::{BerthIndex, VesselIndex};
    use std::collections::HashSet;

    // --- Mock Oracle ---

    struct MockOracle {
        active_constraints: HashSet<BranchConstraint>,
    }

    impl MockOracle {
        fn new() -> Self {
            Self {
                active_constraints: HashSet::new(),
            }
        }
    }

    impl PricingOracle for MockOracle {
        fn solve_pricing(&mut self, _duals: &[f64], _use_real_costs: bool) -> Vec<Column> {
            vec![]
        }

        fn add_constraint(&mut self, constraint: BranchConstraint) {
            let inserted = self.active_constraints.insert(constraint);
            assert!(inserted, "Constraint {:?} was already active!", constraint);
        }

        fn remove_constraint(&mut self, constraint: BranchConstraint) {
            let removed = self.active_constraints.remove(&constraint);
            assert!(removed, "Constraint {:?} was not active!", constraint);
        }
    }

    #[test]
    fn test_push_pop_frame() {
        let mut trail = BnpTrail::new();
        let mut oracle = MockOracle::new();

        let c1 = BranchConstraint::ForceAssignment {
            vessel: VesselIndex::new(0),
            berth: BerthIndex::new(0),
        };

        // Depth 0 -> 1
        trail.push_frame();
        trail.apply(c1, &mut oracle);

        assert_eq!(trail.current_depth(), 1);
        assert_eq!(trail.num_active_constraints(), 1);
        assert!(oracle.active_constraints.contains(&c1));

        // Backtrack
        trail.backtrack(&mut oracle);

        assert_eq!(trail.current_depth(), 0);
        assert_eq!(trail.num_active_constraints(), 0);
        assert!(oracle.active_constraints.is_empty());
    }

    #[test]
    fn test_nested_frames() {
        let mut trail = BnpTrail::new();
        let mut oracle = MockOracle::new();

        let c1 = BranchConstraint::ForceAssignment {
            vessel: VesselIndex::new(1),
            berth: BerthIndex::new(1),
        };
        let c2 = BranchConstraint::ForbidAssignment {
            vessel: VesselIndex::new(2),
            berth: BerthIndex::new(2),
        };

        // Frame 1
        trail.push_frame();
        trail.apply(c1, &mut oracle);

        // Frame 2
        trail.push_frame();
        trail.apply(c2, &mut oracle);

        assert_eq!(trail.current_depth(), 2);
        assert_eq!(trail.num_active_constraints(), 2);

        // Backtrack Frame 2 -> Should remove c2 but keep c1
        trail.backtrack(&mut oracle);
        assert_eq!(trail.current_depth(), 1);
        assert!(!oracle.active_constraints.contains(&c2));
        assert!(oracle.active_constraints.contains(&c1));

        // Backtrack Frame 1 -> Clean
        trail.backtrack(&mut oracle);
        assert_eq!(trail.current_depth(), 0);
        assert!(oracle.active_constraints.is_empty());
    }

    #[test]
    fn test_reset() {
        let mut trail = BnpTrail::new();
        let mut oracle = MockOracle::new();
        let c = BranchConstraint::ForceAssignment {
            vessel: VesselIndex::new(0),
            berth: BerthIndex::new(0),
        };

        trail.push_frame();
        trail.apply(c, &mut oracle);

        trail.reset(&mut oracle);

        assert_eq!(trail.current_depth(), 0);
        assert!(oracle.active_constraints.is_empty());
        assert!(trail.entries.is_empty());
        assert!(trail.frames.is_empty());
    }
}
