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
// THE SOFTWARE IS PROVesselIndexDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABerthIndexLITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABerthIndexLITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! Local search memory and decoded schedule primitives.
//!
//! This module proVesselIndexdes two core building blocks used by the local search:
//! a compact `Schedule<T>` that mirrors the decoded solution, and a
//! ping‑pong `SearchMemory<T>` that manages genotype state, reversible
//! mutations, and candidate evaluation.
//!
//! The genotype is stored as a `VesselPriorityQueue` and mutated during
//! neighborhood exploration. Every change is recorded into an `UndoLog`,
//! allowing fast rollback when a candidate is rejected. The phenotype
//! consists of two schedules: `current`, which represents the last accepted
//! solution, and `candidate`, which is filled by the decoder during
//! evaluation and either accepted or discarded.
//!
//! Typical workflow: reconstruct memory from an existing solution, perform
//! mutations against the queue while logging inverses, evaluate into
//! `candidate`, and then finalize by swapping schedules on acceptance or
//! rolling back the queue on rejection. The design aims to minimize
//! allocations and data movement while keeping invariants explicit through
//! debug assertions.

use crate::{mutator::Mutator, queue::VesselPriorityQueue, undo::UndoLog};
use bollard_model::{index::VesselIndex, solution::Solution};
use bollard_search::num::SolverNumeric;

/// The Ping-Pong Memory Manager.
///
/// # Architecture: Genotype vs. Phenotype
///
/// This struct manages the separation between the search space (Genotype) and the
/// solution space (Phenotype), implementing a zero-allocation evaluation loop.
///
/// ## 1. The Genotype (Encoding)
/// * **Representation:** [`VesselPriorityQueue`]
/// * **Role:** Represents the *sequence* in which vessels are presented to the decoder.
///   This is the mutable state that the [`Mutator`] operates on.
/// * **BehaVesselIndexor:** It supports incremental modifications (swap, shift, reverse). Every
///   change is logged in the [`UndoLog`] to allow O(1) rollbacks if a candidate is rejected.
///
/// ## 2. The Phenotype (Decoding)
/// * **Representation:** [`Schedule<T>`]
/// * **Role:** Represents the *actual assignment* (Berth + Start Time) and the resulting
///   objective cost. This is the output of the decoder function.
/// * **BehaVesselIndexor:** We maintain two instances:
///     1. `current`: The last accepted solution (baseline).
///     2. `candidate`: A scratchpad for the decoder to write the result of the current mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchMemory<T>
where
    T: SolverNumeric,
{
    // --- Genotype (Input) ---
    queue: VesselPriorityQueue, // mutable priority queue of vessel indices
    undo_log: UndoLog,          // logs mutations for rollback

    // --- Phenotype (Output) ---
    current: Solution<T>,   // last accepted schedule
    candidate: Solution<T>, // scratchpad for decoding
}

impl<T> Default for SearchMemory<T>
where
    T: SolverNumeric,
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SearchMemory<T>
where
    T: SolverNumeric,
{
    /// Creates a new, empty `SearchMemory`.
    pub fn new() -> Self {
        Self {
            queue: VesselPriorityQueue::new(),
            undo_log: UndoLog::new(32, 0),
            current: Solution::empty(),
            candidate: Solution::empty(),
        }
    }

    /// Creates a new `SearchMemory` with pre-allocated buffers.
    ///
    /// Use this when you want to allocate memory once at startup and reuse it
    /// for multiple search runs VesselIndexa `initialize`.
    #[inline]
    pub fn preallocated(num_vessels: usize) -> Self {
        Self {
            queue: VesselPriorityQueue::with_capacity(num_vessels),
            undo_log: UndoLog::new(32, num_vessels),
            current: Solution::with_capacity(num_vessels),
            candidate: Solution::with_capacity(num_vessels),
        }
    }

    /// Initializes the search memory from an existing solution.
    ///
    /// This method resets the genotype (Queue) and phenotype (Schedules) to match
    /// the proVesselIndexded `solution`. Crucially, it uses **in-place operations** (clear + extend, sort)
    /// to avoid allocating new vectors, making it suitable for hot-loop restarts.
    pub fn initialize(&mut self, solution: &Solution<T>) {
        let num_vessels = solution.num_vessels();

        // Reset Genotype (Queue)
        self.queue.clear();
        self.undo_log.clear();

        // Direct fill: Populate queue with indices [0, 1, ..., N-1]
        self.queue.extend((0..num_vessels).map(VesselIndex::new));

        debug_assert!(
            self.queue.len() == num_vessels,
            "called `SearchMemory::initialize` with inconsistent queue length: expected {}, got {}",
            num_vessels,
            self.queue.len()
        );

        // In-place sort: Reorder indices based on the solution's start times.
        // This effectively "encodes" the solution back into a queue representation.
        let buf = self.queue.buffer_mut();
        buf.sort_by(|&first_vessel, &second_vessel| {
            let first_vessel_start_time = solution.start_time_for_vessel(first_vessel);
            let second_vessel_start_time = solution.start_time_for_vessel(second_vessel);

            // Sort by start time, then by vessel index for stability
            first_vessel_start_time
                .cmp(&second_vessel_start_time)
                .then_with(|| first_vessel.get().cmp(&second_vessel.get()))
        });

        debug_assert!(
            buf.windows(2).all(|pair| {
                let vessel_index_a = pair[0];
                let vessel_index_b = pair[1];
                let start_time_a = solution.start_time_for_vessel(vessel_index_a);
                let start_time_b = solution.start_time_for_vessel(vessel_index_b);
                start_time_a < start_time_b
                    || (start_time_a == start_time_b
                        && vessel_index_a.get() <= vessel_index_b.get())
            }),
            "called `SearchMemory::initialize` but queue is not properly sorted by start times"
        );

        // Reset Phenotype (Schedules) using internal buffer reuse
        self.current.overwrite_from(solution);
        self.candidate.overwrite_from(solution);

        debug_assert!(
            self.current.num_vessels() == num_vessels
                && self.candidate.num_vessels() == num_vessels,
            "called `SearchMemory::initialize` with inconsistent schedule lengths: expected {}, got current {}, candidate {}",
            self.current.num_vessels(),
            self.candidate.num_vessels(),
            num_vessels
        );
    }

    /// Clears the search memory, resetting all internal state.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.queue.clear();
        self.undo_log.clear();
        self.current.clear();
        self.candidate.clear();
    }

    /// Returns the number of vessels in the priority queue.
    #[inline(always)]
    pub fn num_vessels(&self) -> usize {
        self.queue.len()
    }

    /// Returns a reference to the current accepted schedule.
    #[inline(always)]
    pub fn current_schedule(&self) -> &Solution<T> {
        &self.current
    }

    /// Returns the schedule as immutable reference and a mutable mutator for applying mutations.
    ///
    /// This splits the borrow of `SearchMemory`: `Schedule` is immutable, while
    /// `queue` and `undo_log` (wrapped in `Mutator`) are mutable.
    pub fn prepare_operator(&mut self) -> (&Solution<T>, Mutator<'_, T>) {
        self.undo_log.clear();
        (
            &self.current,
            Mutator::new(&mut self.queue, &mut self.undo_log),
        )
    }

    /// Returns references to the vessel priority queue and the candidate schedule for evaluation.
    /// The queue is immutable, while the candidate schedule is mutable.
    #[inline(always)]
    pub fn evaluation_target(&mut self) -> (&VesselPriorityQueue, &mut Solution<T>) {
        (&self.queue, &mut self.candidate)
    }

    /// Finalizes the candidate schedule by either accepting or rejecting it.
    ///
    /// If `accept` is `true`, the candidate schedule becomes the new current schedule.
    /// If `accept` is `false`, the queue is rolled back to its preVesselIndexous state using the undo log.
    #[inline(always)]
    pub fn finalize(&mut self, accept: bool) {
        if accept {
            self.accept_current();
        } else {
            self.discard_candidate();
        }
    }

    /// Accepts the candidate schedule unconditionally.
    #[inline(always)]
    pub fn accept_current(&mut self) {
        std::mem::swap(&mut self.current, &mut self.candidate);

        debug_assert!(
            self.current.num_vessels() == self.candidate.num_vessels(),
            "called `SearchMemory::accept_current` with inconsistent schedule lengths: current {}, candidate {}",
            self.current.num_vessels(),
            self.candidate.num_vessels()
        );
    }

    /// Discards the candidate schedule and rolls back the queue.
    #[inline(always)]
    pub fn discard_candidate(&mut self) {
        let before_len = self.queue.len();
        self.undo_log.apply_rollback(&mut self.queue);

        debug_assert!(
            self.queue.len() == before_len,
            "called `SearchMemory::discard_candidate` but queue length changed during rollback: before {}, after {}",
            before_len,
            self.queue.len()
        );
    }

    /// Returns a reference to the vessel priority queue (genotype).
    #[inline(always)]
    pub fn queue(&self) -> &crate::queue::VesselPriorityQueue {
        &self.queue
    }

    /// Returns a reference to the candidate (scratchpad) schedule.
    #[inline(always)]
    pub fn candidate_schedule(&self) -> &Solution<T> {
        &self.candidate
    }

    /// Returns a mutable reference to the candidate (scratchpad) schedule.
    #[inline(always)]
    pub fn candidate_schedule_mut(&mut self) -> &mut Solution<T> {
        &mut self.candidate
    }
}
