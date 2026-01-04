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

//! Local search memory and decoded schedule primitives.
//!
//! This module provides two core building blocks used by the local search:
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
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    solution::Solution,
};
use bollard_search::num::SolverNumeric;

/// The Schedule representation used in the Local Search.
///
/// This struct mirrors the `Solution` struct from `bollard-model`.
/// Similarly, it represents a full valid schedule with assigned berths and start times
/// for each vessel, along with the total objective value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schedule<T> {
    berths: Vec<BerthIndex>,
    start_times: Vec<T>,
    objective_value: T,
}

impl<T> Schedule<T>
where
    T: SolverNumeric,
{
    /// Creates a new empty `Schedule`.
    fn new() -> Self {
        Self {
            objective_value: T::zero(),
            berths: Vec::new(),
            start_times: Vec::new(),
        }
    }

    /// Initializes this schedule from an existing solution.
    ///
    /// This method reuses the internal vectors `berths` and `start_times` by clearing
    /// them and extending them with data from `solution`. This avoids reallocating
    /// the underlying heap memory if the capacity is sufficient.
    #[inline]
    fn initialize_from(&mut self, solution: &Solution<T>) {
        self.objective_value = solution.objective_value();

        self.berths.clear();
        self.berths.extend_from_slice(solution.berths());

        self.start_times.clear();
        self.start_times.extend_from_slice(solution.start_times());

        debug_assert!(
            self.berths.len() == self.start_times.len(),
            "called `Schedule::initialize_from` with inconsistent vector lengths: berths.len() = {}, start_times.len() = {}",
            self.berths.len(),
            self.start_times.len()
        );
    }

    /// Creates a new `Schedule` with pre-allocated buffers.
    #[inline]
    fn with_capacity(num_vessels: usize) -> Self {
        Self {
            objective_value: T::zero(),
            berths: Vec::with_capacity(num_vessels),
            start_times: Vec::with_capacity(num_vessels),
        }
    }

    /// Clears the schedule, resetting it to an empty state.
    #[inline]
    fn clear(&mut self) {
        self.objective_value = T::zero();
        self.berths.clear();
        self.start_times.clear();
    }

    /// Returns the assigned berth for a specific vessel.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is out of bounds.
    #[inline]
    pub fn berth_for_vessel(&self, vessel_index: VesselIndex) -> BerthIndex {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels(),
            "called `Schedule::berth_for_vessel` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        self.berths[index]
    }

    /// Returns the assigned start time for a specific vessel.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is out of bounds.
    #[inline]
    pub fn start_time_for_vessel(&self, vessel_index: VesselIndex) -> T {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels(),
            "called `Schedule::start_time_for_vessel` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        self.start_times[index]
    }

    /// Returns the number of vessels in this schedule.
    #[inline]
    pub fn num_vessels(&self) -> usize {
        self.berths.len()
    }

    /// Returns the total objective value of this schedule.
    #[inline]
    pub fn objective_value(&self) -> T {
        self.objective_value
    }

    /// Returns a slice of assigned berths for all vessels.
    #[inline]
    pub fn berths(&self) -> &[BerthIndex] {
        &self.berths
    }

    /// Returns a slice of assigned start times for all vessels.
    #[inline]
    pub fn start_times(&self) -> &[T] {
        &self.start_times
    }

    /// Returns a mutable slice of assigned berths for all vessels.
    #[inline]
    pub fn berth_mut(&mut self) -> &mut [BerthIndex] {
        &mut self.berths
    }

    /// Returns a mutable slice of assigned start times for all vessels.
    #[inline]
    pub fn start_time_mut(&mut self) -> &mut [T] {
        &mut self.start_times
    }

    /// Sets the objective value of this solution.
    #[inline]
    pub fn set_objective_value(&mut self, value: T) {
        self.objective_value = value;
    }

    /// Returns mutable slices of assigned berths and start times for all vessels.
    #[inline(always)]
    pub fn as_mut_slices(&mut self) -> (&mut [BerthIndex], &mut [T]) {
        (&mut self.berths, &mut self.start_times)
    }
}

impl<T> From<Solution<T>> for Schedule<T>
where
    T: SolverNumeric,
{
    #[inline]
    fn from(solution: Solution<T>) -> Self {
        Self {
            berths: solution.berths().to_vec(),
            start_times: solution.start_times().to_vec(),
            objective_value: solution.objective_value(),
        }
    }
}

impl<T> From<Schedule<T>> for Solution<T>
where
    T: SolverNumeric,
{
    #[inline]
    fn from(val: Schedule<T>) -> Self {
        Solution::new(val.objective_value, val.berths, val.start_times)
    }
}

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
/// * **Behavior:** It supports incremental modifications (swap, shift, reverse). Every
///   change is logged in the [`UndoLog`] to allow O(1) rollbacks if a candidate is rejected.
///
/// ## 2. The Phenotype (Decoding)
/// * **Representation:** [`Schedule<T>`]
/// * **Role:** Represents the *actual assignment* (Berth + Start Time) and the resulting
///   objective cost. This is the output of the decoder function.
/// * **Behavior:** We maintain two instances:
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
    current: Schedule<T>,   // last accepted schedule
    candidate: Schedule<T>, // scratchpad for decoding
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
            current: Schedule::new(),
            candidate: Schedule::new(),
        }
    }

    /// Creates a new `SearchMemory` with pre-allocated buffers.
    ///
    /// Use this when you want to allocate memory once at startup and reuse it
    /// for multiple search runs via `initialize`.
    #[inline]
    pub fn preallocated(num_vessels: usize) -> Self {
        Self {
            queue: VesselPriorityQueue::with_capacity(num_vessels),
            undo_log: UndoLog::new(32, num_vessels),
            current: Schedule::with_capacity(num_vessels),
            candidate: Schedule::with_capacity(num_vessels),
        }
    }

    /// Initializes the search memory from an existing solution.
    ///
    /// This method resets the genotype (Queue) and phenotype (Schedules) to match
    /// the provided `solution`. Crucially, it uses **in-place operations** (clear + extend, sort)
    /// to avoid allocating new vectors, making it suitable for hot-loop restarts.
    pub fn initialize(&mut self, solution: &Solution<T>) {
        let num_vessels = solution.num_vessels();

        // Reset Genotype (Queue)
        self.queue.clear();
        self.undo_log.clear();

        // Direct fill: Populate queue with indices [0, 1, ..., N-1]
        self.queue.extend((0..num_vessels).map(VesselIndex::new));

        // In-place sort: Reorder indices based on the solution's start times.
        // This effectively "encodes" the solution back into a queue representation.
        let buf = self.queue.buffer_mut();
        buf.sort_by(|&a, &b| {
            let ta = solution.start_time_for_vessel(a);
            let tb = solution.start_time_for_vessel(b);
            // Sort by Start Time, then by Index for stability
            ta.cmp(&tb).then_with(|| a.get().cmp(&b.get()))
        });

        // Reset Phenotype (Schedules) using internal buffer reuse
        self.current.initialize_from(solution);
        self.candidate.initialize_from(solution);
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
    pub fn current_schedule(&self) -> &Schedule<T> {
        &self.current
    }

    /// Returns the schedule as immutable reference and a mutable mutator for applying mutations.
    ///
    /// This splits the borrow of `SearchMemory`: `Schedule` is immutable, while
    /// `queue` and `undo_log` (wrapped in `Mutator`) are mutable.
    pub fn prepare_operator(&mut self) -> (&Schedule<T>, Mutator<'_, T>) {
        self.undo_log.clear();
        (
            &self.current,
            Mutator::new(&mut self.queue, &mut self.undo_log),
        )
    }

    #[inline(always)]
    pub fn evaluation_target(&mut self) -> (&VesselPriorityQueue, &mut Schedule<T>) {
        (&self.queue, &mut self.candidate)
    }

    /// Finalizes the candidate schedule by either accepting or rejecting it.
    ///
    /// If `accept` is `true`, the candidate schedule becomes the new current schedule.
    /// If `accept` is `false`, the queue is rolled back to its previous state using the undo log.
    #[inline(always)]
    pub fn finalize(&mut self, accept: bool) {
        if accept {
            std::mem::swap(&mut self.current, &mut self.candidate);
        } else {
            self.undo_log.apply_rollback(&mut self.queue);
        }
    }

    /// Accepts the candidate schedule unconditionally.
    #[inline(always)]
    pub fn accept_current(&mut self) {
        std::mem::swap(&mut self.current, &mut self.candidate);
    }

    /// Discards the candidate schedule and rolls back the queue.
    #[inline(always)]
    pub fn discard_candidate(&mut self) {
        self.undo_log.apply_rollback(&mut self.queue);
    }

    /// Returns a reference to the vessel priority queue (genotype).
    #[inline(always)]
    pub fn queue(&self) -> &crate::queue::VesselPriorityQueue {
        &self.queue
    }

    /// Returns a reference to the candidate (scratchpad) schedule.
    #[inline(always)]
    pub fn candidate_schedule(&self) -> &Schedule<T> {
        &self.candidate
    }

    /// Returns a mutable reference to the candidate (scratchpad) schedule.
    #[inline(always)]
    pub fn candidate_schedule_mut(&mut self) -> &mut Schedule<T> {
        &mut self.candidate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline]
    fn bi(n: usize) -> BerthIndex {
        BerthIndex::new(n)
    }
    #[inline]
    fn vi(n: usize) -> VesselIndex {
        VesselIndex::new(n)
    }

    // ---------------------------
    // Schedule<i64> invariants/tests
    // ---------------------------

    #[test]
    fn test_schedule_new_valid_lengths() {
        let berths = vec![bi(0), bi(1), bi(2)];
        let starts = vec![10_i64, 20, 30];
        let mut s = Schedule::new();
        s.initialize_from(&Solution::new(123_i64, berths.clone(), starts.clone()));

        assert_eq!(s.num_vessels(), 3);
        assert_eq!(s.objective_value(), 123);
        assert_eq!(s.berths(), &[bi(0), bi(1), bi(2)]);
        assert_eq!(s.start_times(), &[10, 20, 30]);
    }

    #[test]
    #[should_panic(expected = "inconsistent vector lengths")]
    fn test_schedule_new_mismatched_lengths_panics() {
        let berths = vec![bi(0), bi(1)];
        let starts = vec![10_i64];
        let mut s = Schedule::new();
        s.initialize_from(&Solution::new(123_i64, berths, starts));
    }

    #[test]
    fn test_schedule_accessors_bounds_valid() {
        let berths = vec![bi(0), bi(1)];
        let starts = vec![5_i64, 7];
        let s = Schedule::from(Solution::new(50_i64, berths, starts));

        assert_eq!(s.num_vessels(), 2);
        assert_eq!(s.berth_for_vessel(vi(0)), bi(0));
        assert_eq!(s.berth_for_vessel(vi(1)), bi(1));
        assert_eq!(s.start_time_for_vessel(vi(0)), 5);
        assert_eq!(s.start_time_for_vessel(vi(1)), 7);
    }

    #[test]
    fn test_schedule_from_solution_roundtrip_into() {
        let sol = Solution::new(999_i64, vec![bi(2), bi(3)], vec![11_i64, 22]);
        let sched: Schedule<i64> = Schedule::from(sol);
        assert_eq!(sched.objective_value(), 999);
        assert_eq!(sched.berths(), &[bi(2), bi(3)]);
        assert_eq!(sched.start_times(), &[11, 22]);

        let back: Solution<i64> = sched.into();
        assert_eq!(back.objective_value(), 999);
        assert_eq!(back.berths(), &[bi(2), bi(3)]);
        assert_eq!(back.start_times(), &[11, 22]);
    }

    #[test]
    fn test_schedule_zero_vessels_edge_case() {
        let s = Schedule::<i64>::from(Solution::new(0_i64, vec![], vec![]));
        assert_eq!(s.num_vessels(), 0);
        assert!(s.berths().is_empty());
        assert!(s.start_times().is_empty());
        assert_eq!(s.objective_value(), 0);
    }

    #[test]
    fn test_search_memory_from_solution_orders_queue_by_start_time_with_stable_tie_breaker() {
        // Two vessels with same start time; tie broken by vessel index ascending.
        let berths = vec![bi(0), bi(1), bi(2), bi(3)];
        let starts = vec![10_i64, 5, 5, 20];
        let sol = Solution::new(1234_i64, berths.clone(), starts.clone());

        let mut mem = SearchMemory::preallocated(sol.num_vessels());
        mem.initialize(&sol);

        // queue should be ordered by ascending start time, then vessel index:
        // vessels: 0->10, 1->5, 2->5, 3->20 => order: [1,2,0,3]
        let q = mem.queue;
        let buf = q.buffer();
        assert_eq!(buf, &[vi(1), vi(2), vi(0), vi(3)]);

        // current and candidate should both reflect initial phenotype
        assert_eq!(mem.current.objective_value(), 1234);
        assert_eq!(mem.current.berths(), berths.as_slice());
        assert_eq!(mem.current.start_times(), starts.as_slice());
        assert_eq!(mem.candidate.objective_value(), 1234);
        assert_eq!(mem.candidate.berths(), berths.as_slice());
        assert_eq!(mem.candidate.start_times(), starts.as_slice());
    }

    #[test]
    fn test_search_memory_mutate_clears_undo_and_returns_mutator() {
        let sol = Solution::new(0_i64, vec![bi(0), bi(1)], vec![10_i64, 20]);

        let mut mem = SearchMemory::preallocated(sol.num_vessels());
        mem.initialize(&sol);

        // Pre-fill undo log with something (simulate stale state)
        mem.undo_log.push_set(0, vi(0));
        assert!(!mem.undo_log.is_empty());

        let (_current, _mutator) = mem.prepare_operator();
        // mutate() must clear undo log
        assert!(mem.undo_log.is_empty());
    }

    #[test]
    fn test_search_memory_finalize_accept_swaps_current_and_candidate() {
        let sol = Solution::new(100_i64, vec![bi(0), bi(1)], vec![10_i64, 20]);

        let mut mem = SearchMemory::preallocated(sol.num_vessels());
        mem.initialize(&sol);

        // Make candidate different from current
        let mut cand = Schedule::with_capacity(2);
        cand.initialize_from(&Solution::new(50_i64, vec![bi(1), bi(0)], vec![20_i64, 10]));
        mem.candidate = cand.clone();

        // Accept: current should become candidate; candidate becomes old current
        let old_current = mem.current.clone();
        mem.finalize(true);
        assert_eq!(mem.current, cand);
        assert_eq!(mem.candidate, old_current);
        // Queue must remain unchanged on accept
        assert_eq!(mem.queue.buffer().len(), 2);
    }

    #[test]
    fn test_search_memory_finalize_reject_rolls_back_queue() {
        let sol = Solution::new(0_i64, vec![bi(0), bi(1), bi(2)], vec![3_i64, 2, 1]);

        let mut mem = SearchMemory::preallocated(sol.num_vessels());
        mem.initialize(&sol);

        let original = mem.queue.buffer().to_vec();

        // Record undo operations and mutate in proper order
        mem.undo_log.clear();
        let start = 1;
        let len = 2;

        {
            // mutable borrow to apply forward changes
            let qbuf = mem.queue.buffer_mut();

            // 1) swap positions 0 and 2
            qbuf.swap(0, 2);
            mem.undo_log.push_swap(0, 2);

            // 2) shift inverse: simulate rotate_right on [0..=1], record inverse for undo
            qbuf[0..=1].rotate_right(1);
            mem.undo_log.push_shift_inverse(0, 1);

            // 3) range backup must reflect content AFTER prior mutations but BEFORE overwrite
            let backup_slice = &qbuf[start..start + len];
            mem.undo_log.push_range_backup(start, backup_slice);

            // overwrite the backed-up range
            qbuf[start..start + len].copy_from_slice(&[vi(77), vi(88)]);
        }

        // Reject candidate: rollback should restore original queue
        mem.finalize(false);
        assert_eq!(mem.queue.buffer(), original.as_slice());
    }

    #[test]
    fn test_search_memory_zero_vessels_from_solution() {
        let sol = Solution::new(0_i64, vec![], vec![]);

        let mut mem = SearchMemory::preallocated(sol.num_vessels());
        mem.initialize(&sol);

        // Queue empty
        assert!(mem.queue.is_empty());
        assert_eq!(mem.queue.len(), 0);
        // Undo log allocated but empty
        assert!(mem.undo_log.is_empty());
        // Both schedules empty and equal
        assert_eq!(mem.current.num_vessels(), 0);
        assert_eq!(mem.candidate.num_vessels(), 0);
        assert_eq!(mem.current, mem.candidate);
    }

    #[test]
    fn test_search_memory_reuse_hot_path() {
        // 1. First run: Solution A
        let sol_a = Solution::new(100_i64, vec![bi(0), bi(0)], vec![10, 20]);
        let mut mem = SearchMemory::preallocated(2);
        mem.initialize(&sol_a);

        // Verify state A
        assert_eq!(mem.current.start_time_for_vessel(vi(0)), 10);
        assert_eq!(mem.queue.buffer(), &[vi(0), vi(1)]);

        // 2. Second run: Solution B (different values)
        // Vessel 1 starts first now
        let sol_b = Solution::new(200, vec![bi(1), bi(1)], vec![50, 5]);
        mem.initialize(&sol_b);

        // Verify state B: should be fully overwritten
        assert_eq!(mem.current.objective_value(), 200);
        assert_eq!(mem.current.start_time_for_vessel(vi(0)), 50);
        // Queue should be re-sorted: vi(1) has start 5, so it comes first
        assert_eq!(mem.queue.buffer(), &[vi(1), vi(0)]);
    }

    #[test]
    fn test_search_memory_capacity_growth() {
        // 1. Preallocate small
        let mut mem = SearchMemory::preallocated(1);

        // 2. Initialize with larger solution (size 3)
        let sol = Solution::new(0_i64, vec![bi(0), bi(0), bi(0)], vec![0, 0, 0]);

        // This should safely grow the internal vectors
        mem.initialize(&sol);

        assert_eq!(mem.num_vessels(), 3);
        assert_eq!(mem.current.num_vessels(), 3);
        // Ensure data is correct
        assert_eq!(mem.queue.buffer().len(), 3);
    }

    #[test]
    fn test_schedule_initialize_from_consistency() {
        let sol = Solution::new(55_i64, vec![bi(9)], vec![88]);

        let mut s1 = Schedule::new();
        s1.initialize_from(&sol);

        let s2 = Schedule::from(sol.clone());

        assert_eq!(
            s1, s2,
            "initialize_from should produce identical state to from()"
        );
    }
}
