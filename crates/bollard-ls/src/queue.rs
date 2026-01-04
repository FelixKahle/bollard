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

//! Priority queue wrapper for strongly typed vessel indices used in local search.
//!
//! This module defines `VesselPriorityQueue`, a thin container over `Vec<VesselIndex>`
//! that preserves zero-cost semantics while providing a domain-specific API. The queue
//! is optimized for fast push and pop at the tail and straightforward indexed access,
//! which suits iterative improvement and neighborhood exploration phases where vessels
//! are re-ordered and inspected frequently.
//!
//! The type integrates with Rust’s iterator ecosystem to enable idiomatic traversal
//! by value, by shared reference, and by mutable reference, and it supports common
//! collection utilities such as `FromIterator` and `Extend`. Display output is formatted
//! as a readable chain of indices to make logs and diagnostics clear during search runs.
//!
//! Safety-sensitive helpers provide unchecked access for tight inner loops when bounds
//! are proven elsewhere, and debug assertions remain in place to catch accidental misuse
//! during development.

use bollard_model::index::VesselIndex;

/// A priority queue for vessel indices used in local search algorithms.
///
/// This data structure provides efficient storage and retrieval of vessel indices
/// based on their priority. It supports common operations such as insertion,
/// removal, and access to elements by index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VesselPriorityQueue {
    queue: Vec<VesselIndex>,
}

impl VesselPriorityQueue {
    /// Creates a new, empty `VesselPriorityQueue`.
    #[inline]
    pub fn new() -> Self {
        Self { queue: Vec::new() }
    }

    /// Creates a new `VesselPriorityQueue` with the specified capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            queue: Vec::with_capacity(capacity),
        }
    }

    /// Creates a new `VesselPriorityQueue` preallocated for the given number of vessels.
    #[inline]
    pub fn preallocated(num_vessels: usize) -> Self {
        Self {
            queue: Vec::with_capacity(num_vessels),
        }
    }

    /// Returns the number of vessel indices in the priority queue.
    #[inline]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Returns `true` if the priority queue contains no vessel indices,
    /// `false` otherwise.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Clears the priority queue, removing all vessel indices.
    #[inline]
    pub fn clear(&mut self) {
        self.queue.clear();
    }

    /// Returns a copy of the vessel index at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of not within `0..len()`.
    #[inline]
    pub fn get(&self, index: usize) -> Option<VesselIndex> {
        self.queue.get(index).copied()
    }

    /// Returns a copy of the vessel index at the specified index without
    /// performing bounds checking.
    ///
    /// # Panics
    ///
    /// In debug builds, this function will panic if the index is not within `0..len()`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is within `0..len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> VesselIndex {
        debug_assert!(
            index < self.queue.len(),
            "called `VesselPriorityQueue::get_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            self.queue.len(),
            index
        );

        unsafe { *self.queue.get_unchecked(index) }
    }

    /// Returns a mutable reference to the vessel index at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of not within `0..len()`.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut VesselIndex> {
        self.queue.get_mut(index)
    }

    /// Returns a mutable reference to the vessel index at the specified index
    /// without performing bounds checking.
    ///
    /// # Panics
    ///
    /// In debug builds, this function will panic if the index is not within `0..len()`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is within `0..len()`.
    #[inline]
    pub unsafe fn get_mut_unchecked(&mut self, index: usize) -> &mut VesselIndex {
        debug_assert!(
            index < self.queue.len(),
            "called `VesselPriorityQueue::get_mut_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            self.queue.len(),
            index
        );

        unsafe { self.queue.get_unchecked_mut(index) }
    }

    /// Sets the vessel index at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of not within `0..len()`.
    /// In debug builds, this function will also panic if setting the
    /// index results in duplicate vessel indices.
    #[inline]
    pub fn set(&mut self, index: usize, vessel_index: VesselIndex) {
        debug_assert!(
            index < self.queue.len(),
            "called `VesselPriorityQueue::set` with vessel index out of bounds: the len is {} but the index is {}",
            self.queue.len(),
            index
        );

        self.queue[index] = vessel_index;

        debug_assert!(
            self.is_unique(),
            "called `VesselPriorityQueue::set` resulting in duplicate vessel indices after setting index {} to {}: {}",
            index,
            vessel_index,
            self
        );
    }

    /// Sets the vessel index at the specified index without performing
    /// bounds checking.
    ///
    /// # Panics
    ///
    /// In debug builds, this function will panic if the index is not within `0..len()`.
    /// In debug builds, this function will also panic if setting the
    /// index results in duplicate vessel indices.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is within `0..len()`.
    #[inline]
    pub unsafe fn set_unchecked(&mut self, index: usize, vessel_index: VesselIndex) {
        debug_assert!(
            index < self.queue.len(),
            "called `VesselPriorityQueue::set_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            self.queue.len(),
            index
        );

        unsafe {
            *self.queue.get_unchecked_mut(index) = vessel_index;
        }

        debug_assert!(
            self.is_unique(),
            "called `VesselPriorityQueue::set_unchecked` resulting in duplicate vessel indices after setting index {} to {}: {}",
            index,
            vessel_index,
            self
        );
    }

    /// Returns the capacity of the priority queue.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.queue.capacity()
    }

    /// Reserves capacity for at least `additional` more vessel indices to be
    /// inserted in the priority queue.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.queue.reserve(additional);
    }

    /// Pushes a vessel index onto the priority queue.
    ///
    /// # Panics
    ///
    /// In debug builds, this function will panic if pushing the
    /// index results in duplicate vessel indices.
    #[inline]
    pub fn push(&mut self, vessel_index: VesselIndex) {
        self.queue.push(vessel_index);

        debug_assert!(
            self.is_unique(),
            "called `VesselPriorityQueue::push` resulting in duplicate vessel indices after pushing {}: {}",
            vessel_index,
            self
        );
    }

    /// Pops a vessel index off the priority queue.
    #[inline]
    pub fn pop(&mut self) -> Option<VesselIndex> {
        self.queue.pop()
    }

    /// Returns a slice of all vessel indices in the priority queue.
    #[inline]
    pub fn buffer(&self) -> &[VesselIndex] {
        &self.queue
    }

    /// Returns a mutable slice of all vessel indices in the priority queue.
    #[inline]
    pub fn buffer_mut(&mut self) -> &mut [VesselIndex] {
        &mut self.queue
    }

    /// Returns an iterator over the vessel indices in the priority queue.
    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, VesselIndex> {
        self.queue.iter()
    }

    /// Returns a mutable iterator over the vessel indices in the priority queue.
    #[inline]
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, VesselIndex> {
        self.queue.iter_mut()
    }

    /// Checks if all vessel indices in the priority queue are unique.
    ///
    /// This method is only available in debug builds and is intended for
    /// internal consistency checks.
    ///
    /// # Note
    ///
    /// This method allocates a `HashSet` to track seen indices, so it should not be
    /// used in performance-critical paths. While it is fine to use in debug assertions,
    /// it is not recommended for release builds.
    #[cfg(debug_assertions)]
    #[inline(always)]
    fn is_unique(&self) -> bool {
        match self.queue.len() {
            0 | 1 => return true,
            _ => {}
        }

        let mut seen = std::collections::HashSet::with_capacity(self.queue.len());
        for &v in &self.queue {
            if !seen.insert(v) {
                return false;
            }
        }
        true
    }
}

impl IntoIterator for VesselPriorityQueue {
    type Item = VesselIndex;
    type IntoIter = std::vec::IntoIter<VesselIndex>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.queue.into_iter()
    }
}

impl<'a> IntoIterator for &'a VesselPriorityQueue {
    type Item = &'a VesselIndex;
    type IntoIter = core::slice::Iter<'a, VesselIndex>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.queue.iter()
    }
}

impl<'a> IntoIterator for &'a mut VesselPriorityQueue {
    type Item = &'a mut VesselIndex;
    type IntoIter = core::slice::IterMut<'a, VesselIndex>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.queue.iter_mut()
    }
}

impl FromIterator<VesselIndex> for VesselPriorityQueue {
    #[inline]
    fn from_iter<I: IntoIterator<Item = VesselIndex>>(iter: I) -> Self {
        let mut q = VesselPriorityQueue::new();
        q.queue.extend(iter);
        debug_assert!(
            q.is_unique(),
            "constructed `VesselPriorityQueue` via FromIterator with duplicate vessel indices: {}",
            q
        );
        q
    }
}

impl Extend<VesselIndex> for VesselPriorityQueue {
    #[inline]
    fn extend<T: IntoIterator<Item = VesselIndex>>(&mut self, iter: T) {
        self.queue.extend(iter);

        debug_assert!(
            self.is_unique(),
            "called `VesselPriorityQueue::extend` resulting in duplicate vessel indices: {}",
            self
        );
    }
}

impl Default for VesselPriorityQueue {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for VesselPriorityQueue {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut iter = self.queue.iter();
        if let Some(first) = iter.next() {
            write!(f, "{}", first)?;
            for v in iter {
                write!(f, " -> {}", v)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as _;

    #[inline]
    fn vi(n: usize) -> VesselIndex {
        VesselIndex::new(n)
    }

    #[test]
    fn test_new_and_default() {
        let q = VesselPriorityQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);

        let qd: VesselPriorityQueue = Default::default();
        assert!(qd.is_empty());
        assert_eq!(qd.len(), 0);
    }

    #[test]
    fn test_with_capacity_and_preallocated() {
        let q = VesselPriorityQueue::with_capacity(10);
        assert_eq!(q.len(), 0);
        assert!(q.capacity() >= 10);

        let qp = VesselPriorityQueue::preallocated(5);
        assert_eq!(qp.len(), 0);
        assert!(qp.capacity() >= 5);
    }

    #[test]
    fn test_len_is_empty_clear() {
        let mut q = VesselPriorityQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);

        q.push(vi(1));
        q.push(vi(2));
        assert!(!q.is_empty());
        assert_eq!(q.len(), 2);

        q.clear();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn test_capacity_and_reserve() {
        let mut q = VesselPriorityQueue::with_capacity(1);
        let initial_capacity = q.capacity();

        // Reserve ensures capacity >= len + additional
        q.reserve(10);
        assert!(q.capacity() >= q.len() + 10);

        // Capacity should not shrink and usually grows monotonically
        assert!(q.capacity() >= initial_capacity);
    }

    #[test]
    fn test_push_and_pop() {
        let mut q = VesselPriorityQueue::new();
        assert_eq!(q.pop(), None); // pop on empty

        q.push(vi(1));
        q.push(vi(2));
        q.push(vi(3));
        assert_eq!(q.len(), 3);

        // pop follows Vec semantics (LIFO)
        assert_eq!(q.pop(), Some(vi(3)));
        assert_eq!(q.pop(), Some(vi(2)));
        assert_eq!(q.pop(), Some(vi(1)));
        assert_eq!(q.pop(), None);
        assert!(q.is_empty());
    }

    #[test]
    fn test_get_and_get_mut_valid_indices() {
        let mut q = VesselPriorityQueue::new();
        q.extend([vi(10), vi(20), vi(30)]);

        assert_eq!(q.get(0), Some(vi(10)));
        assert_eq!(q.get(1), Some(vi(20)));
        assert_eq!(q.get(2), Some(vi(30)));
        assert_eq!(q.get(3), None); // out of bounds returns None

        if let Some(v) = q.get_mut(1) {
            *v = vi(25);
        }
        assert_eq!(q.get(1), Some(vi(25)));
    }

    #[test]
    fn test_get_unchecked_and_get_mut_unchecked_valid_indices() {
        let mut q = VesselPriorityQueue::new();
        q.extend([vi(7), vi(8), vi(9)]);
        unsafe {
            let v = q.get_unchecked(0);
            assert_eq!(v, vi(7));
            let v = q.get_unchecked(2);
            assert_eq!(v, vi(9));

            let vm = q.get_mut_unchecked(1);
            *vm = vi(88);
        }
        assert_eq!(q.get(1), Some(vi(88)));
    }

    #[test]
    fn test_set_and_set_unchecked_valid_indices() {
        let mut q = VesselPriorityQueue::new();
        q.extend([vi(1), vi(2), vi(3)]);
        q.set(0, vi(10));
        q.set(2, vi(30));
        assert_eq!(q.buffer(), &[vi(10), vi(2), vi(30)]);

        unsafe {
            q.set_unchecked(1, vi(20));
        }
        assert_eq!(q.buffer(), &[vi(10), vi(20), vi(30)]);
    }

    #[test]
    fn test_buffer_and_buffer_mut() {
        let mut q = VesselPriorityQueue::new();
        q.extend([vi(3), vi(4)]);
        let buf = q.buffer();
        assert_eq!(buf, &[vi(3), vi(4)]);

        {
            let buf_mut = q.buffer_mut();
            assert_eq!(buf_mut, &mut [vi(3), vi(4)]);
            buf_mut[0] = vi(33);
            buf_mut[1] = vi(44);
        }
        assert_eq!(q.buffer(), &[vi(33), vi(44)]);
    }

    #[test]
    fn test_iter_and_iter_mut() {
        let mut q = VesselPriorityQueue::new();
        q.extend([vi(1), vi(2), vi(3)]);

        let collected: Vec<_> = q.iter().copied().collect();
        assert_eq!(collected, vec![vi(1), vi(2), vi(3)]);

        for v in q.iter_mut() {
            *v = vi(v.get() * 10);
        }
        assert_eq!(q.buffer(), &[vi(10), vi(20), vi(30)]);
    }

    #[test]
    fn test_intoiterator_owned() {
        let q: VesselPriorityQueue = [vi(5), vi(6), vi(7)].into_iter().collect();
        // consuming into_iter
        let collected: Vec<_> = q.into_iter().collect();
        assert_eq!(collected, vec![vi(5), vi(6), vi(7)]);
    }

    #[test]
    fn test_intoiterator_ref_and_mut_ref() {
        let mut q: VesselPriorityQueue = [vi(1), vi(2)].into_iter().collect();

        // &VesselPriorityQueue
        let collected_ref: Vec<_> = (&q).into_iter().copied().collect();
        assert_eq!(collected_ref, vec![vi(1), vi(2)]);

        // &mut VesselPriorityQueue
        for v in (&mut q).into_iter() {
            *v = vi(v.get() + 10);
        }
        assert_eq!(q.buffer(), &[vi(11), vi(12)]);
    }

    #[test]
    fn test_fromiterator_and_extend() {
        let q: VesselPriorityQueue = (0..3).map(vi).collect();
        assert_eq!(q.buffer(), &[vi(0), vi(1), vi(2)]);

        let mut q2 = VesselPriorityQueue::new();
        q2.extend((5..8).map(vi));
        assert_eq!(q2.buffer(), &[vi(5), vi(6), vi(7)]);
    }

    #[test]
    fn test_display_empty_single_multiple() {
        let q_empty = VesselPriorityQueue::new();
        assert_eq!(format!("{}", q_empty), "");

        let mut q_single = VesselPriorityQueue::new();
        q_single.push(vi(42));
        assert_eq!(format!("{}", q_single), "VesselIndex(42)");

        let mut q_multi = VesselPriorityQueue::new();
        q_multi.extend([vi(1), vi(2), vi(3)]);
        assert_eq!(
            format!("{}", q_multi),
            "VesselIndex(1) -> VesselIndex(2) -> VesselIndex(3)"
        );
    }

    #[test]
    fn test_display_write_into_existing_formatter() {
        // Ensure writing into an existing formatter works without extra writes.
        let mut s = String::new();
        let mut q = VesselPriorityQueue::new();
        q.extend([vi(9), vi(10)]);
        write!(&mut s, "{q}").unwrap();
        assert_eq!(s, "VesselIndex(9) -> VesselIndex(10)");
    }

    #[test]
    fn test_edge_cases_empty_iteration() {
        let q = VesselPriorityQueue::new();

        // owned into_iter on empty
        let collected: Vec<_> = q.clone().into_iter().collect();
        assert!(collected.is_empty());

        // ref iter on empty
        let collected_ref: Vec<_> = q.iter().copied().collect();
        assert!(collected_ref.is_empty());

        // &VesselPriorityQueue IntoIterator on empty
        let collected_into_ref: Vec<_> = (&q).into_iter().copied().collect();
        assert!(collected_into_ref.is_empty());
    }

    #[test]
    fn test_mutation_via_get_mut_and_iter_mut_consistency() {
        let mut q = VesselPriorityQueue::new();
        q.extend([vi(2), vi(4), vi(6)]);

        // mutate middle element via get_mut
        if let Some(m) = q.get_mut(1) {
            *m = vi(40);
        }
        assert_eq!(q.buffer(), &[vi(2), vi(40), vi(6)]);

        // mutate all via iter_mut
        for v in q.iter_mut() {
            *v = vi(v.get() + 1);
        }
        assert_eq!(q.buffer(), &[vi(3), vi(41), vi(7)]);
    }

    #[test]
    fn test_reserve_does_not_shrink() {
        let mut q = VesselPriorityQueue::with_capacity(20);
        let cap_before = q.capacity();
        q.reserve(0);
        let cap_after = q.capacity();
        assert!(cap_after >= cap_before);
    }
}
