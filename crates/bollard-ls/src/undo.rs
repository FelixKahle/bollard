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

//! Undo support for local-search mutations.
//!
//! This module provides a compact, stack-based log that records reversible
//! changes to a `VesselPriorityQueue` and restores them efficiently in
//! reverse order. It is designed for high-throughput neighborhood exploration
//! where many tentative moves must be tried and quickly rolled back.
//!
//! The log stores operation tags, small integer arguments, and any necessary
//! data slices for range-based edits. Recording a change is intentionally
//! lightweight so it can be used per mutation step without noticeable overhead.
//! Rolling back a sequence simply walks the stacks backwards and applies the
//! inverse of each recorded change, restoring the queue to its prior state.
//!
//! Typical usage: start a mutation phase, record every change as you apply it,
//! evaluate the candidate, and then either accept (no rollback) or reject
//! (apply rollback) to return to the original configuration.
//!
//! The design favors zero-cost abstractions and avoids allocations in the hot
//! path by preallocating internal buffers sized to the problem. Debug assertions
//! guard against incorrect recording or consumption of stack entries, helping
//! catch invariants early during development.

#![allow(dead_code)]

use crate::queue::VesselPriorityQueue;
use bollard_model::index::VesselIndex;

/// The types of undo operations supported.
///
/// - Set: Record old value at index before a direct set.
/// - Swap: Record indices of two elements swapped.
/// - Shift: Record inverse shift operation (from, to).
/// - Range: Record start and length of a backed-up range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum UndoOperation {
    Set,   // Sets an element at an index to a new value, recording the old value.
    Swap,  // Swaps two elements at given indices, recording both indices.
    Shift, // Shifts an element from one index to another, recording the inverse operation.
    Range, // Backs up a range of elements, recording start index and length.
}

impl std::fmt::Display for UndoOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UndoOperation::Set => write!(f, "Set"),
            UndoOperation::Swap => write!(f, "Swap"),
            UndoOperation::Shift => write!(f, "Shift"),
            UndoOperation::Range => write!(f, "Range"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UndoLog {
    operations: Vec<UndoOperation>, // LIFO stack of recorded operations
    args: Vec<usize>,               // LIFO stack of op args; counts per op
    values: Vec<VesselIndex>,       // LIFO stack of old values for Set operations
    data_stack: Vec<VesselIndex>,   // LIFO stack of backed-up range data
}

impl UndoLog {
    /// Creates a new `UndoLog` with specified initial capacities.
    #[inline(always)]
    pub fn new(capacity: usize, data_capacity: usize) -> Self {
        Self {
            operations: Vec::with_capacity(capacity),
            args: Vec::with_capacity(capacity * 2),
            values: Vec::with_capacity(capacity),
            data_stack: Vec::with_capacity(data_capacity),
        }
    }

    /// Creates a new `UndoLog` with preallocated capacities based on the number of vessels.
    ///
    /// This uses heuristics to size the internal vectors to reduce reallocations during typical usage.
    #[inline(always)]
    pub fn preallocated(num_vessels: usize) -> Self {
        // Heuristic capacities:
        // - ops/args/values: proportional to the number of vessels (typical LS step touches a few ops per vessel)
        // - data_stack: proportional as well (range backups during reverse/shuffle)
        // Use a minimum floor to avoid tiny allocations that reallocate frequently.
        let min_cap = 16;
        let op_cap = num_vessels.saturating_mul(8).max(min_cap);
        let data_cap = num_vessels.saturating_mul(8).max(min_cap);
        Self::new(op_cap, data_cap)
    }

    /// Clears all recorded operations and resets the log.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.operations.clear();
        self.args.clear();
        self.values.clear();
        self.data_stack.clear();
    }

    /// Checks if the undo log is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Returns the number of recorded operations.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Records a Set operation with the index and old value.
    #[inline(always)]
    pub fn push_set(&mut self, index: usize, old_value: VesselIndex) {
        debug_assert!(
            self.operations.len() <= self.args.len(),
            "called `UndoLog::push_set` with unbalanced stacks: operations.len() = {}, args.len() = {}, values.len() = {}",
            self.operations.len(),
            self.args.len(),
            self.values.len()
        );

        self.operations.push(UndoOperation::Set);
        self.args.push(index);
        self.values.push(old_value);
    }

    /// Records a Swap operation with the two indices.
    #[inline(always)]
    pub fn push_swap(&mut self, a: usize, b: usize) {
        debug_assert!(
            self.operations.len() <= self.args.len(),
            "called `UndoLog::push_swap` with unbalanced stacks: operations.len() = {}, args.len() = {}, values.len() = {}",
            self.operations.len(),
            self.args.len(),
            self.values.len()
        );

        self.operations.push(UndoOperation::Swap);
        self.args.push(a);
        self.args.push(b);
    }

    /// Records a Shift operation with the inverse from and to indices.
    #[inline(always)]
    pub fn push_shift_inverse(&mut self, inverse_from: usize, inverse_to: usize) {
        debug_assert!(
            self.operations.len() <= self.args.len(),
            "called `UndoLog::push_shift_inverse` with unbalanced stacks: operations.len() = {}, args.len() = {}, values.len() = {}",
            self.operations.len(),
            self.args.len(),
            self.values.len()
        );

        self.operations.push(UndoOperation::Shift);
        self.args.push(inverse_from);
        self.args.push(inverse_to);
    }

    /// Records a Range backup operation with the start index and data slice.
    #[inline(always)]
    pub fn push_range_backup(&mut self, start: usize, data: &[VesselIndex]) {
        debug_assert!(
            self.operations.len() <= self.args.len(),
            "called `UndoLog::push_range_backup` with unbalanced stacks: operations.len() = {}, args.len() = {}, values.len() = {}",
            self.operations.len(),
            self.args.len(),
            self.values.len()
        );

        self.operations.push(UndoOperation::Range);
        self.args.push(start);
        self.args.push(data.len());
        self.data_stack.extend_from_slice(data);
    }

    /// Applies the recorded operations in reverse order to rollback changes on the given queue.
    #[inline(always)]
    pub fn apply_rollback(&mut self, queue: &mut VesselPriorityQueue) {
        let buf = queue.buffer_mut();

        while let Some(operation) = self.operations.pop() {
            match operation {
                UndoOperation::Set => {
                    debug_assert!(
                        !self.args.is_empty() && !self.values.is_empty(),
                        "called `UndoLog::apply_rollback` with corrupted Set operation: expected at least 1 arg and 1 value, found {} args and {} values",
                        self.args.len(),
                        self.values.len()
                    );

                    let value = unsafe { self.values.pop().unwrap_unchecked() };
                    let index = unsafe { self.args.pop().unwrap_unchecked() };

                    debug_assert!(
                        index < buf.len(),
                        "called `UndoLog::apply_rollback` with index out of bounds for Set operation: the len is {} but the index is {}",
                        index,
                        buf.len()
                    );

                    unsafe {
                        *buf.get_unchecked_mut(index) = value;
                    }
                }
                UndoOperation::Swap => {
                    debug_assert!(
                        self.args.len() >= 2,
                        "called `UndoLog::apply_rollback` with corrupted Swap operation: expected at least 2 args, found {} args",
                        self.args.len()
                    );

                    let b = unsafe { self.args.pop().unwrap_unchecked() };
                    let a = unsafe { self.args.pop().unwrap_unchecked() };

                    debug_assert!(
                        a < buf.len(),
                        "called `UndoLog::apply_rollback` with index out of bounds for Swap operation: the len is {} but index a is {}",
                        buf.len(),
                        a
                    );
                    debug_assert!(
                        b < buf.len(),
                        "called `UndoLog::apply_rollback` with index out of bounds for Swap operation: the len is {} but index b is {}",
                        buf.len(),
                        b
                    );

                    buf.swap(a, b);
                }
                UndoOperation::Shift => {
                    debug_assert!(
                        self.args.len() >= 2,
                        "called `UndoLog::apply_rollback` with corrupted Shift operation: expected at least 2 args, found {} args",
                        self.args.len()
                    );

                    let to = unsafe { self.args.pop().unwrap_unchecked() };
                    let from = unsafe { self.args.pop().unwrap_unchecked() };

                    debug_assert!(
                        from < buf.len(),
                        "called `UndoLog::apply_rollback` with index out of bounds for Shift operation: the len is {} but from index is {}",
                        buf.len(),
                        from
                    );
                    debug_assert!(
                        to < buf.len(),
                        "called `UndoLog::apply_rollback` with index out of bounds for Shift operation: the len is {} but to index is {}",
                        buf.len(),
                        to
                    );

                    if from < to {
                        buf[from..=to].rotate_left(1);
                    } else {
                        buf[to..=from].rotate_right(1);
                    }
                }
                UndoOperation::Range => {
                    debug_assert!(
                        self.args.len() >= 2,
                        "called `UndoLog::apply_rollback` with corrupted Range operation: expected at least 2 args, found {} args",
                        self.args.len()
                    );

                    let len = unsafe { self.args.pop().unwrap_unchecked() };
                    let start = unsafe { self.args.pop().unwrap_unchecked() };

                    debug_assert!(
                        len > 0,
                        "called `UndoLog::apply_rollback` with invalid Range operation: length must be greater than 0, found length {}",
                        len
                    );

                    debug_assert!(
                        start + len <= buf.len(),
                        "called `UndoLog::apply_rollback` with index out of bounds for Range operation: the len is {} but the range {}..{} is invalid",
                        buf.len(),
                        start,
                        start + len
                    );

                    let stack_len = self.data_stack.len();

                    debug_assert!(
                        stack_len >= len,
                        "called `UndoLog::apply_rollback` with corrupted Range operation: expected data stack to have at least {} elements, found {} elements",
                        len,
                        stack_len
                    );

                    let src_start = stack_len - len;

                    debug_assert!(
                        src_start <= stack_len,
                        "called `UndoLog::apply_rollback` with corrupted Range operation: calculated source start index {} is out of bounds for data stack of length {}",
                        src_start,
                        stack_len
                    );

                    let src = unsafe { self.data_stack.get_unchecked(src_start..stack_len) };
                    let dst = unsafe { buf.get_unchecked_mut(start..(start + len)) };

                    debug_assert!(
                        src.len() == dst.len(),
                        "called `UndoLog::apply_rollback` with corrupted Range operation: source slice length {} does not match destination slice length {}",
                        src.len(),
                        dst.len()
                    );

                    dst.copy_from_slice(src);

                    unsafe { self.data_stack.set_len(src_start) };

                    debug_assert!(
                        self.data_stack.len() == src_start,
                        "called `UndoLog::apply_rollback` with corrupted Range operation: expected data stack length to be {} after restoration, found {}",
                        src_start,
                        self.data_stack.len()
                    );
                }
            }
        }
    }
}

impl<'a> IntoIterator for &'a UndoLog {
    type Item = &'a UndoOperation;
    type IntoIter = std::slice::Iter<'a, UndoOperation>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.operations.iter()
    }
}

impl IntoIterator for UndoLog {
    type Item = UndoOperation;
    type IntoIter = std::vec::IntoIter<UndoOperation>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.operations.into_iter()
    }
}

impl<'a> IntoIterator for &'a mut UndoLog {
    type Item = &'a mut UndoOperation;
    type IntoIter = std::slice::IterMut<'a, UndoOperation>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.operations.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[inline]
    fn vi(n: usize) -> VesselIndex {
        VesselIndex::new(n)
    }

    #[test]
    fn test_new_clear_is_empty() {
        let mut log = UndoLog::new(10, 10);
        assert!(log.is_empty());
        // add a couple ops
        log.push_set(0, vi(1));
        log.push_swap(0, 1);
        assert!(!log.is_empty());
        // clear resets everything
        log.clear();
        assert!(log.is_empty());
    }

    #[test]
    fn test_set_rollback_restores_old_value() {
        let mut q = VesselPriorityQueue::new();
        q.extend([vi(10), vi(20), vi(30)]);

        let mut log = UndoLog::new(4, 0);
        // record old value before mutation
        log.push_set(1, q.get(1).unwrap());
        // forward change
        q.set(1, vi(99));
        assert_eq!(q.buffer(), &[vi(10), vi(99), vi(30)]);
        // rollback
        log.apply_rollback(&mut q);
        // restored
        assert_eq!(q.buffer(), &[vi(10), vi(20), vi(30)]);
        // log emptied
        assert!(log.is_empty());
    }

    #[test]
    fn test_swap_rollback_swaps_back() {
        let mut q = VesselPriorityQueue::new();
        q.extend([vi(1), vi(2), vi(3), vi(4)]);

        let mut log = UndoLog::new(2, 0);
        // perform forward swap
        q.buffer_mut().swap(1, 3);
        assert_eq!(q.buffer(), &[vi(1), vi(4), vi(3), vi(2)]);
        // record swap to undo
        log.push_swap(1, 3);

        log.apply_rollback(&mut q);
        assert_eq!(q.buffer(), &[vi(1), vi(2), vi(3), vi(4)]);
        assert!(log.is_empty());
    }

    #[test]
    fn test_shift_rollback_inverse_forwards_direction_from_lt_to() {
        // Forward operation: move element at index 1 to index 3 via rotate_right(1) on [1..=3].
        let mut q = VesselPriorityQueue::new();
        q.extend([vi(10), vi(11), vi(12), vi(13), vi(14)]);
        // segment [1..=3] = [11,12,13], forward rotate_right(1) -> [13,11,12]
        q.buffer_mut()[1..=3].rotate_right(1);
        assert_eq!(q.buffer(), &[vi(10), vi(13), vi(11), vi(12), vi(14)]);

        let mut log = UndoLog::new(1, 0);
        // record inverse using from < to -> rollback will rotate_left(1) on [from..=to]
        log.push_shift_inverse(1, 3);

        log.apply_rollback(&mut q);
        assert_eq!(q.buffer(), &[vi(10), vi(11), vi(12), vi(13), vi(14)]);
        assert!(log.is_empty());
    }

    #[test]
    fn test_shift_rollback_inverse_backwards_direction_from_gt_to() {
        // Forward operation: move element at index 4 to index 2 via rotate_left(1) on [2..=4].
        let mut q = VesselPriorityQueue::new();
        q.extend([vi(0), vi(1), vi(2), vi(3), vi(4)]);
        // segment [2..=4] = [2,3,4], forward rotate_left(1) -> [3,4,2]
        q.buffer_mut()[2..=4].rotate_left(1);
        assert_eq!(q.buffer(), &[vi(0), vi(1), vi(3), vi(4), vi(2)]);

        let mut log = UndoLog::new(1, 0);
        // record inverse using from > to -> rollback will rotate_right(1) on [to..=from]
        log.push_shift_inverse(4, 2);

        log.apply_rollback(&mut q);
        assert_eq!(q.buffer(), &[vi(0), vi(1), vi(2), vi(3), vi(4)]);
        assert!(log.is_empty());
    }

    #[test]
    fn test_range_backup_and_restore_single_range() {
        let mut q = VesselPriorityQueue::new();
        q.extend([vi(100), vi(200), vi(300), vi(400), vi(500)]);

        let mut log = UndoLog::new(2, 10);
        // Backup a slice [start..start+len)
        let start = 1;
        let data = &q.buffer()[start..start + 3]; // [200,300,400]
        log.push_range_backup(start, data);

        // Mutate the slice forward
        q.buffer_mut()[start..start + 3].copy_from_slice(&[vi(2), vi(3), vi(4)]);
        assert_eq!(q.buffer(), &[vi(100), vi(2), vi(3), vi(4), vi(500)]);

        // Rollback should restore the original [200,300,400]
        log.apply_rollback(&mut q);
        assert_eq!(q.buffer(), &[vi(100), vi(200), vi(300), vi(400), vi(500)]);
        assert!(log.is_empty());
    }

    #[test]
    fn test_range_backup_multiple_segments_lifo_restore_and_stack_shrink() {
        let mut q = VesselPriorityQueue::new();
        q.extend([vi(1), vi(2), vi(3), vi(4), vi(5), vi(6)]);

        let mut log = UndoLog::new(10, 10);

        // backup and modify first segment [1..=3]
        let s1 = 1;
        let l1 = 3;
        log.push_range_backup(s1, &q.buffer()[s1..s1 + l1]); // backs up [2,3,4]
        q.buffer_mut()[s1..s1 + l1].copy_from_slice(&[vi(20), vi(30), vi(40)]);
        assert_eq!(q.buffer(), &[vi(1), vi(20), vi(30), vi(40), vi(5), vi(6)]);

        // backup and modify second segment [3..=5]
        let s2 = 3;
        let l2 = 3;
        log.push_range_backup(s2, &q.buffer()[s2..s2 + l2]); // backs up [40,5,6]
        q.buffer_mut()[s2..s2 + l2].copy_from_slice(&[vi(400), vi(500), vi(600)]);
        assert_eq!(
            q.buffer(),
            &[vi(1), vi(20), vi(30), vi(400), vi(500), vi(600)]
        );

        // Ensure data_stack accumulated 6 entries
        assert_eq!(log.data_stack.len(), 6);

        // Rolling back drains the entire log in LIFO order:
        // - Restores s2 first, then s1. Final result is the original queue.
        log.apply_rollback(&mut q);
        assert_eq!(q.buffer(), &[vi(1), vi(2), vi(3), vi(4), vi(5), vi(6)]);

        // data_stack should be compacted back to 0 and log emptied
        assert_eq!(log.data_stack.len(), 0);
        assert!(log.is_empty());
    }

    #[test]
    fn test_multiple_ops_combined_lifo_order() {
        let mut q = VesselPriorityQueue::new();
        q.extend([vi(10), vi(20), vi(30), vi(40), vi(50)]);

        let mut log = UndoLog::new(20, 20);

        // 1) Set index 2 from 30 to 300
        log.push_set(2, q.get(2).unwrap());
        q.set(2, vi(300));

        // 2) Swap indices 1 and 3
        q.buffer_mut().swap(1, 3);
        log.push_swap(1, 3);

        // 3) Shift inverse (simulate moving idx 0 to idx 2 via forward rotate_right on [0..=2])
        q.buffer_mut()[0..=2].rotate_right(1);
        // record inverse to undo: since from < to, push_shift_inverse(from,to)
        log.push_shift_inverse(0, 2);

        // 4) Range backup and modify [2..5)
        log.push_range_backup(2, &q.buffer()[2..5]);
        q.buffer_mut()[2..5].copy_from_slice(&[vi(1), vi(2), vi(3)]);

        // State after all forwards:
        // Start: [10,20,30,40,50]
        // After set(2,300): [10,20,300,40,50]
        // After swap(1,3):  [10,40,300,20,50]
        // After rotate_right [0..=2]: [300,10,40,20,50]
        // After range set [2..5]=[1,2,3]: [300,10,1,2,3]
        assert_eq!(q.buffer(), &[vi(300), vi(10), vi(1), vi(2), vi(3)]);

        // Apply rollback should undo in reverse order:
        // - Range: restore [40,20,50] into [2..5]
        // - Shift: rotate_left on [0..=2] -> undo rotation
        // - Swap: swap(1,3) again -> undo swap
        // - Set: restore index 2 to 30
        log.apply_rollback(&mut q);
        assert_eq!(q.buffer(), &[vi(10), vi(20), vi(30), vi(40), vi(50)]);
        assert!(log.is_empty());
    }

    #[test]
    fn test_display_of_undoop_variants() {
        // `Display` impl for `UndoOp` should be stable and correct.
        assert_eq!(UndoOperation::Set.to_string(), "Set");
        assert_eq!(UndoOperation::Swap.to_string(), "Swap");
        assert_eq!(UndoOperation::Shift.to_string(), "Shift");
        assert_eq!(UndoOperation::Range.to_string(), "Range");
    }

    #[test]
    fn test_values_and_args_stack_balanced_after_set() {
        let mut q = VesselPriorityQueue::new();
        q.extend([vi(1), vi(2)]);
        let mut log = UndoLog::new(4, 0);

        log.push_set(1, q.get(1).unwrap());
        q.set(1, vi(22));
        // Ensure internal stacks grew
        assert_eq!(log.operations.len(), 1);
        assert_eq!(log.args.len(), 1); // index only
        assert_eq!(log.values.len(), 1); // old value

        log.apply_rollback(&mut q);
        assert_eq!(q.buffer(), &[vi(1), vi(2)]);
        // Ensure stacks consumed
        assert_eq!(log.operations.len(), 0);
        assert_eq!(log.args.len(), 0);
        assert_eq!(log.values.len(), 0);
    }

    #[test]
    fn test_data_stack_compaction_after_range() {
        let mut q = VesselPriorityQueue::new();
        q.extend([vi(10), vi(11), vi(12), vi(13)]);
        let mut log = UndoLog::new(10, 10);

        // push two small ranges
        log.push_range_backup(0, &q.buffer()[0..2]); // [10,11]
        log.push_range_backup(2, &q.buffer()[2..4]); // [12,13]
        assert_eq!(log.data_stack.len(), 4);

        // mutate both ranges
        q.buffer_mut()[0..2].copy_from_slice(&[vi(100), vi(101)]);
        q.buffer_mut()[2..4].copy_from_slice(&[vi(200), vi(201)]);
        assert_eq!(q.buffer(), &[vi(100), vi(101), vi(200), vi(201)]);

        // rollback should restore second range first, then first range
        log.apply_rollback(&mut q);
        assert_eq!(q.buffer(), &[vi(10), vi(11), vi(12), vi(13)]);
        assert_eq!(log.data_stack.len(), 0);
    }
}
