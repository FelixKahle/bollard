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

//! Mutation helpers for local search with precise rollback.
//!
//! A `Mutator` offers a small, fast API to modify a `VesselPriorityQueue`
//! while recording inverse operations into an `UndoLog`. This enables
//! exploring neighborhoods aggressively and restoring the genotype to
//! its prior state in constant-amortized time per change.
//!
//! The design keeps mutation primitives close to the queue and avoids
//! extra allocations in the hot path. Each method performs domain checks
//! under debug assertions and pushes the minimum information needed to
//! undo the effect later. In typical usage, you start a mutation phase,
//! perform a sequence of changes, evaluate the resulting candidate,
//! and either accept (no rollback) or reject (apply rollback).

#![allow(dead_code)]

use crate::{queue::VesselPriorityQueue, undo::UndoLog};
use bollard_model::index::VesselIndex;
use bollard_search::num::SolverNumeric;
use rand::Rng;

/// A mutator for a vessel priority queue that records changes to an undo log.
///
/// This struct provides methods to mutate the queue while logging
/// the inverse operations to an `UndoLog`, enabling rollback of changes.
#[derive(Debug, PartialEq, Eq)]
pub struct Mutator<'a, T>
where
    T: SolverNumeric,
{
    queue: &'a mut VesselPriorityQueue,
    log: &'a mut UndoLog,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T> Mutator<'a, T>
where
    T: SolverNumeric,
{
    /// Creates a new `Mutator` for the given queue and undo log.
    #[inline(always)]
    pub fn new(queue: &'a mut VesselPriorityQueue, log: &'a mut UndoLog) -> Self {
        Self {
            queue,
            log,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns the number of vessels in the queue.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Returns `true` if the queue is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Read-only access to a vessel index.
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<VesselIndex> {
        self.queue.get(index)
    }

    /// Read-only access to a vessel index without bounds checking.
    ///
    /// # Panics
    ///
    /// In debug builds, this function will panic if `index` is not within `0..len()`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` is within `0..len()`.
    #[inline(always)]
    pub fn get_unchecked(&self, index: usize) -> VesselIndex {
        debug_assert!(
            index < self.queue.len(),
            "called `Mutator::get_unchecked` with index out of bounds: the len is {} but the index is {}",
            self.queue.len(),
            index
        );

        unsafe { self.queue.get_unchecked(index) }
    }

    /// Sets the value at `index`. Records `Set` to undo.
    ///
    /// # Panics
    ///
    /// This function will panic if `index` is not within `0..len()`.
    #[inline(always)]
    pub fn set(&mut self, index: usize, value: VesselIndex) {
        let old_val = self
            .queue
            .get(index)
            .expect("index must be within `0..len()`");
        self.log.push_set(index, old_val);
        self.queue.set(index, value);
    }

    /// Sets the value at `index`. Records `Set` to undo.
    ///
    /// # Panics
    ///
    /// In debug builds, this function will panic if `index` is not within `0..len()`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` is within `0..len()`.
    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, index: usize, value: VesselIndex) {
        debug_assert!(
            index < self.queue.len(),
            "called `Mutator::set_unchecked` with index out of bounds: the len is {} but the index is {}",
            self.queue.len(),
            index
        );

        let old_val = unsafe { self.queue.get_unchecked(index) };
        self.log.push_set(index, old_val);
        self.queue.set(index, value);
    }

    /// Swaps elements at `a` and `b`. Records `Swap` to undo.
    ///
    /// # Panics
    ///
    /// In debug builds, this function will panic if `a` or `b` are not within `0..len()`.
    #[inline(always)]
    pub fn swap(&mut self, a: usize, b: usize) {
        debug_assert!(
            a < self.queue.len(),
            "called `Mutator::swap` with index `a` out of bounds: the len is {} but the index is {}",
            self.queue.len(),
            a
        );
        debug_assert!(
            b < self.queue.len(),
            "called `Mutator::swap` with index `b` out of bounds: the
            len is {} but the index is {}",
            self.queue.len(),
            b
        );

        if a == b {
            return;
        }

        self.log.push_swap(a, b);
        self.queue.buffer_mut().swap(a, b);
    }

    /// Moves element from `from` to `to`. Records `Shift` to undo.
    #[inline(always)]
    pub fn shift(&mut self, from: usize, to: usize) {
        debug_assert!(
            from < self.queue.len(),
            "called `Mutator::shift` with index `from` out of bounds: the len is {} but the index is {}",
            self.queue.len(),
            from
        );
        debug_assert!(
            to < self.queue.len(),
            "called `Mutator::shift` with index `to` out of bounds: the len is {} but the index is {}",
            self.queue.len(),
            to
        );

        if from == to {
            return;
        }

        self.log.push_shift_inverse(to, from);
        let buf = self.queue.buffer_mut();

        if from < to {
            buf[from..=to].rotate_left(1);
        } else {
            buf[to..=from].rotate_right(1);
        }
    }

    /// Reverses the range `start..=end`. Records `Range` to undo.
    ///
    /// # Panics
    ///
    /// In debug builds, this function will panic if `end` is greater than `len() - 1`
    /// or if `start` is greater than `end`.
    #[inline(always)]
    pub fn reverse(&mut self, start: usize, end: usize) {
        debug_assert!(
            end < self.queue.len(),
            "called `Mutator::reverse` with index `end` out of bounds: the len is {} but the index is {}",
            self.queue.len(),
            end
        );
        debug_assert!(
            start <= end,
            "called `Mutator::reverse` with start > end: start is {} but end is {}",
            start,
            end
        );

        if start >= end {
            return;
        }
        let buf = self.queue.buffer_mut();
        self.log.push_range_backup(start, &buf[start..=end]);
        buf[start..=end].reverse();
    }

    /// Shuffles the range `start..end`. Records `Range` to undo.
    /// Note: `end` is exclusive here to match standard RNG APIs.
    ///
    /// # Panics
    ///
    /// In debug builds, this function will panic if `end` is greater than `len()`
    /// or if `start` is greater than `end`.
    #[inline(always)]
    pub fn shuffle<R>(&mut self, start: usize, end: usize, rng: &mut R)
    where
        R: Rng + ?Sized,
    {
        debug_assert!(
            end <= self.queue.len(),
            "called `Mutator::shuffle` with index `end` out of bounds: the len is {} but the index is {}",
            self.queue.len(),
            end
        );
        debug_assert!(
            start <= end,
            "called `Mutator::shuffle` with start > end: start is {} but end is {}",
            start,
            end
        );

        let buf = self.queue.buffer_mut();
        if start >= end || end > buf.len() {
            return;
        }

        let slice = &buf[start..end];
        self.log.push_range_backup(start, slice);

        use rand::seq::SliceRandom;
        buf[start..end].shuffle(rng);
    }

    /// Returns a reference to the underlying vessel priority queue.
    #[inline(always)]
    pub fn queue(&self) -> &VesselPriorityQueue {
        self.queue
    }

    /// Returns a reference to the underlying undo log.
    #[inline(always)]
    pub fn log(&self) -> &UndoLog {
        self.log
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[inline]
    fn vi(n: usize) -> VesselIndex {
        VesselIndex::new(n)
    }

    #[test]
    fn test_mutator_len_and_is_empty_reflect_queue() {
        let mut q = VesselPriorityQueue::new();
        let mut log = UndoLog::new(16, 16);

        {
            let m: Mutator<'_, i64> = Mutator::new(&mut q, &mut log);
            assert!(m.is_empty());
            assert_eq!(m.len(), 0);
        } // m drops here

        // mutate queue directly
        q.extend([vi(1), vi(2), vi(3)]);

        // re-create mutator to observe updated state
        let m: Mutator<'_, i64> = Mutator::new(&mut q, &mut log);
        assert!(!m.is_empty());
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn test_mutator_get_reads_values() {
        let mut q: VesselPriorityQueue = [vi(10), vi(20), vi(30)].into_iter().collect();
        let mut log = UndoLog::new(8, 8);
        let m: Mutator<'_, i64> = Mutator::new(&mut q, &mut log);

        assert_eq!(m.get(0), Some(vi(10)));
        assert_eq!(m.get(1), Some(vi(20)));
        assert_eq!(m.get(2), Some(vi(30)));
        assert_eq!(m.get(3), None);
    }

    #[test]
    fn mutator_set_changes_value_and_can_rollback() {
        let mut q: VesselPriorityQueue = [vi(0), vi(1), vi(2)].into_iter().collect();
        let mut log = UndoLog::new(16, 16);

        {
            let mut m: Mutator<'_, i64> = Mutator::new(&mut q, &mut log);
            // forward set
            m.set(1, vi(99));
            assert_eq!(q.buffer(), &[vi(0), vi(99), vi(2)]);
        } // drop m

        // rollback restores old
        log.apply_rollback(&mut q);
        assert_eq!(q.buffer(), &[vi(0), vi(1), vi(2)]);
        assert!(log.is_empty());
    }

    #[test]
    fn test_mutator_swap_changes_two_positions_and_can_rollback() {
        let mut q: VesselPriorityQueue = [vi(1), vi(2), vi(3), vi(4)].into_iter().collect();
        let mut log = UndoLog::new(16, 16);
        let mut m: Mutator<'_, i64> = Mutator::new(&mut q, &mut log);

        m.swap(0, 3);
        assert_eq!(q.buffer(), &[vi(4), vi(2), vi(3), vi(1)]);

        log.apply_rollback(&mut q);
        assert_eq!(q.buffer(), &[vi(1), vi(2), vi(3), vi(4)]);
        assert!(log.is_empty());
    }

    #[test]
    fn test_mutator_swap_is_noop_when_indices_equal() {
        let mut q: VesselPriorityQueue = [vi(5), vi(6)].into_iter().collect();
        let before = q.buffer().to_vec();

        let mut log = UndoLog::new(16, 16);
        let mut m: Mutator<'_, i64> = Mutator::new(&mut q, &mut log);

        m.swap(1, 1);
        assert_eq!(q.buffer(), before.as_slice());
        assert!(log.is_empty());
    }

    #[test]
    fn test_mutator_shift_forward_and_rollback_from_lt_to() {
        let mut q: VesselPriorityQueue = [vi(10), vi(11), vi(12), vi(13)].into_iter().collect();
        let mut log = UndoLog::new(16, 16);
        let mut m: Mutator<'_, i64> = Mutator::new(&mut q, &mut log);

        // move index 1 -> index 3; forward rotates left on [1..=3]
        m.shift(1, 3);
        assert_eq!(q.buffer(), &[vi(10), vi(12), vi(13), vi(11)]);

        // rollback rotates right on [to..=from] due to inverse recording
        log.apply_rollback(&mut q);
        assert_eq!(q.buffer(), &[vi(10), vi(11), vi(12), vi(13)]);
        assert!(log.is_empty());
    }

    #[test]
    fn test_mutator_shift_backward_and_rollback_from_gt_to() {
        let mut q: VesselPriorityQueue = [vi(0), vi(1), vi(2), vi(3), vi(4)].into_iter().collect();
        let mut log = UndoLog::new(16, 16);
        let mut m: Mutator<'_, i64> = Mutator::new(&mut q, &mut log);

        // move index 4 -> index 2; forward rotates right on [2..=4]
        m.shift(4, 2);
        assert_eq!(q.buffer(), &[vi(0), vi(1), vi(4), vi(2), vi(3)]);

        log.apply_rollback(&mut q);
        assert_eq!(q.buffer(), &[vi(0), vi(1), vi(2), vi(3), vi(4)]);
        assert!(log.is_empty());
    }

    #[test]
    fn test_mutator_shift_is_noop_when_indices_equal() {
        let mut q: VesselPriorityQueue = [vi(1), vi(2), vi(3)].into_iter().collect();
        let before = q.buffer().to_vec();
        let mut log = UndoLog::new(16, 16);
        let mut m: Mutator<'_, i64> = Mutator::new(&mut q, &mut log);

        m.shift(2, 2);
        assert_eq!(q.buffer(), before.as_slice());
        assert!(log.is_empty());
    }

    #[test]
    fn test_mutator_reverse_reverses_range_and_rollback_restores() {
        let mut q: VesselPriorityQueue = [vi(1), vi(2), vi(3), vi(4), vi(5)].into_iter().collect();
        let mut log = UndoLog::new(32, 32);
        let mut m: Mutator<'_, i64> = Mutator::new(&mut q, &mut log);

        m.reverse(1, 3); // reverse [2,3,4] -> [4,3,2]
        assert_eq!(q.buffer(), &[vi(1), vi(4), vi(3), vi(2), vi(5)]);

        log.apply_rollback(&mut q);
        assert_eq!(q.buffer(), &[vi(1), vi(2), vi(3), vi(4), vi(5)]);
        assert!(log.is_empty());
    }

    #[test]
    fn test_mutator_reverse_is_noop_when_start_equal_end() {
        let mut q: VesselPriorityQueue = [vi(1), vi(2), vi(3)].into_iter().collect();
        let before = q.buffer().to_vec();
        let mut log = UndoLog::new(16, 16);

        {
            let mut m: Mutator<'_, i64> = Mutator::new(&mut q, &mut log);
            m.reverse(2, 2); // single element: no-op
        }

        assert_eq!(q.buffer(), before.as_slice());
        assert!(log.is_empty());
    }

    #[test]
    fn test_mutator_shuffle_shuffles_range_and_rollback_restores() {
        let mut q: VesselPriorityQueue = [vi(0), vi(1), vi(2), vi(3), vi(4), vi(5)]
            .into_iter()
            .collect();
        let mut log = UndoLog::new(64, 64);
        let mut m: Mutator<'_, i64> = Mutator::new(&mut q, &mut log);
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);

        m.shuffle(1, 5, &mut rng); // shuffle [1..5) -> indices 1..4
        let after = q.buffer().to_vec();
        assert_ne!(after, vec![vi(0), vi(1), vi(2), vi(3), vi(4), vi(5)]); // likely changed

        log.apply_rollback(&mut q);
        assert_eq!(q.buffer(), &[vi(0), vi(1), vi(2), vi(3), vi(4), vi(5)]);
        assert!(log.is_empty());
    }

    #[test]
    fn test_mutator_sequence_of_ops_lifo_rollback_to_original() {
        let mut q: VesselPriorityQueue = [vi(10), vi(20), vi(30), vi(40), vi(50)]
            .into_iter()
            .collect();
        let original = q.buffer().to_vec();

        let mut log = UndoLog::new(128, 128);
        let mut m: Mutator<'_, i64> = Mutator::new(&mut q, &mut log);

        // Apply several ops:
        // 1) set index 2 to 300
        unsafe {
            m.set_unchecked(2, vi(300));
        }

        // 2) swap indices 1 and 3
        m.swap(1, 3);

        // 3) shift 0 -> 2 (forward left rotate on [0..=2])
        m.shift(0, 2);

        // 4) reverse range [2..=4]
        m.reverse(2, 4);

        // Verify it's different
        assert_ne!(q.buffer(), original.as_slice());

        // Rollback should restore original due to LIFO undo semantics
        log.apply_rollback(&mut q);
        assert_eq!(q.buffer(), original.as_slice());
        assert!(log.is_empty());
    }

    #[test]
    fn test_mutator_large_shuffle_and_reverse_combined() {
        // stress combined range backups
        let mut q: VesselPriorityQueue = (0..20).map(vi).collect();
        let original = q.buffer().to_vec();

        let mut log = UndoLog::new(256, 256);
        let mut m: Mutator<'_, i64> = Mutator::new(&mut q, &mut log);
        let mut rng = StdRng::seed_from_u64(123456);

        // shuffle middle
        m.shuffle(5, 15, &mut rng);
        // reverse tail
        m.reverse(10, 19);
        // set a couple values
        unsafe {
            m.set_unchecked(0, vi(999));
            m.set_unchecked(19, vi(888));
        }
        // swap ends
        m.swap(0, 19);
        // shift an element across a range
        m.shift(3, 12);

        assert_ne!(q.buffer(), original.as_slice());

        log.apply_rollback(&mut q);
        assert_eq!(q.buffer(), original.as_slice());
        assert!(log.is_empty());
    }
}
