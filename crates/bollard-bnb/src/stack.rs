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

#![allow(dead_code)]

use crate::branching::decision::Decision;
use bollard_core::num::ops::saturating_arithmetic::SaturatingAddVal;

/// A frame-structured LIFO stack of pending decisions for search.
///
/// `SearchStack` stores all enqueued `Decision`s linearly and uses
/// a `frames` index stack to mark decision-level boundaries. Popping
/// a frame truncates the `entries` slice back to the recorded start index.
///
/// Performance notes:
/// - `preallocated` help avoid reallocations.
/// - `current_frame_entries()` returns a slice of the decisions in the active frame
///   without copying.
/// - `pop_frame()` is O(1) plus a potential `truncate` on `entries`.
#[derive(Clone, Debug)]
pub struct SearchStack {
    /// The linear stack of pending decisions.
    entries: Vec<Decision>,
    /// A stack of indices pointing to `entries`.
    /// `frames[i]` stores the index in `entries` where depth `i` began.
    frames: Vec<usize>,
}

impl Default for SearchStack {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchStack {
    /// Creates a new, empty `SearchStack`.
    #[inline]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            frames: Vec::new(),
        }
    }

    /// Creates a preallocated `SearchStack` based on problem size.
    #[inline]
    pub fn preallocated(num_berths: usize, num_vessels: usize) -> Self {
        let entry_capacity = num_vessels.saturating_mul(num_berths);
        let frame_capacity = num_vessels.saturating_add_val(1);

        Self {
            entries: Vec::with_capacity(entry_capacity),
            frames: Vec::with_capacity(frame_capacity),
        }
    }

    /// Ensures the stack has capacity for the given problem size.
    #[inline]
    pub fn ensure_capacity(&mut self, num_berths: usize, num_vessels: usize) {
        let entry_capacity = num_vessels.saturating_mul(num_berths);
        let frame_capacity = num_vessels.saturating_add_val(1);

        if self.entries.capacity() < entry_capacity {
            self.entries
                .reserve(entry_capacity - self.entries.capacity());
        }

        if self.frames.capacity() < frame_capacity {
            self.frames.reserve(frame_capacity - self.frames.capacity());
        }
    }

    /// Returns the number of entries (decisions) in the stack.
    #[inline]
    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    /// Returns the number of frames (depth) in the stack.
    #[inline]
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Returns the current search depth (alias for num_frames).
    #[inline]
    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    /// Returns `true` if there are no frames tracked (search exhausted).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Pushes a new frame onto the stack.
    /// This marks the start of a new decision level.
    #[inline]
    pub fn push_frame(&mut self) {
        self.frames.push(self.entries.len());
    }

    /// Pops the current frame, truncating `entries` back to the
    /// start index recorded for this depth.
    #[inline]
    pub fn pop_frame(&mut self) -> Option<()> {
        let start = self.frames.pop()?;
        if self.entries.len() > start {
            self.entries.truncate(start);
        }
        Some(())
    }

    /// Pushes a single decision entry onto the stack.
    #[inline]
    pub fn push(&mut self, decision: Decision) {
        self.entries.push(decision);
    }

    /// Extends the stack with multiple decision entries.
    #[inline]
    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Decision>,
    {
        self.entries.extend(iter);
    }

    /// Pops the next decision (LIFO) from the stack.
    #[inline]
    pub fn pop(&mut self) -> Option<Decision> {
        self.entries.pop()
    }

    /// Clears all entries and frames, but keeps allocated capacity.
    #[inline]
    pub fn reset(&mut self) {
        self.entries.clear();
        self.frames.clear();
    }

    /// Returns the current frame's start index in `entries`, if any.
    #[inline]
    pub fn current_level_start(&self) -> Option<usize> {
        self.frames.last().copied()
    }

    /// Returns `true` if the current level has no remaining decisions.
    #[inline]
    pub fn is_current_level_empty(&self) -> bool {
        match self.current_level_start() {
            Some(start) => self.entries.len() == start,
            None => true,
        }
    }

    /// Returns a slice of all decisions in the current frame.
    #[inline]
    pub fn current_frame_entries(&self) -> &[Decision] {
        match self.frames.last() {
            Some(&start) => &self.entries[start..],
            None => &[],
        }
    }

    /// Returns a slice of all decisions in the stack.
    #[inline]
    pub fn all_entries(&self) -> &[Decision] {
        &self.entries
    }

    /// Returns the total allocated memory in bytes.
    #[inline]
    pub fn allocated_memory_bytes(&self) -> usize {
        let entries_size = self.entries.capacity() * core::mem::size_of::<Decision>();
        let frames_size = self.frames.capacity() * core::mem::size_of::<usize>();
        entries_size + frames_size
    }

    /// Returns an iterator over all decisions in the stack.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, Decision> {
        self.entries.iter()
    }
}

impl std::fmt::Display for SearchStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SearchStack(entries: {}, frames: {})",
            self.entries.len(),
            self.frames.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bollard_model::index::{BerthIndex, VesselIndex};

    fn d(v: usize, b: usize) -> Decision {
        Decision::new(VesselIndex::new(v), BerthIndex::new(b))
    }

    #[test]
    fn test_new_and_preallocated_basic_props() {
        let s = SearchStack::new();
        assert_eq!(s.num_entries(), 0);
        assert_eq!(s.num_frames(), 0);
        assert_eq!(s.depth(), 0);
        assert!(s.is_empty());
        assert!(s.is_current_level_empty());
        assert_eq!(s.current_level_start(), None);
        assert_eq!(s.current_frame_entries(), &[]);
        assert_eq!(s.all_entries(), &[]);

        let s2 = SearchStack::preallocated(3, 5);
        assert_eq!(s2.num_entries(), 0);
        assert_eq!(s2.num_frames(), 0);
        assert!(s2.is_empty());
        assert!(s2.allocated_memory_bytes() > 0);

        // Display sanity
        let disp = format!("{}", s);
        assert!(disp.contains("SearchStack(entries: 0, frames: 0)"));
    }

    #[test]
    fn test_ensure_capacity_grows_but_is_idempotent_when_large_enough() {
        let mut s = SearchStack::preallocated(2, 2);
        let ecap0 = s.entries.capacity();
        let fcap0 = s.frames.capacity();
        let bytes0 = s.allocated_memory_bytes();

        // Request larger capacity
        s.ensure_capacity(5, 7);

        // Capacities should not shrink; they typically increase
        let ecap1 = s.entries.capacity();
        let fcap1 = s.frames.capacity();
        assert!(ecap1 >= ecap0);
        assert!(fcap1 >= fcap0);

        // Allocated bytes are monotonic (non-decreasing)
        let bytes1 = s.allocated_memory_bytes();
        assert!(bytes1 >= bytes0);

        // Request smaller capacity: should be idempotent
        s.ensure_capacity(1, 1);
        assert_eq!(s.entries.capacity(), ecap1);
        assert_eq!(s.frames.capacity(), fcap1);
    }

    #[test]
    fn test_push_frame_and_depth_tracking() {
        let mut s = SearchStack::new();
        assert!(s.is_empty());
        s.push_frame();
        assert_eq!(s.depth(), 1);
        assert!(!s.is_empty());
        assert!(s.is_current_level_empty());
        assert_eq!(s.current_level_start(), Some(0));

        s.push_frame();
        assert_eq!(s.depth(), 2);
        assert!(s.is_current_level_empty());
        assert_eq!(s.current_level_start(), Some(0)); // still 0, no decisions yet
    }

    #[test]
    fn test_push_extend_pop_entries_across_frames() {
        let mut s = SearchStack::new();

        // Root frame
        s.push_frame();
        assert!(s.is_current_level_empty());
        s.push(d(0, 0));
        s.push(d(1, 1));
        assert_eq!(s.num_entries(), 2);
        assert!(!s.is_current_level_empty());

        // Current frame entries slice
        let slice_root = s.current_frame_entries();
        assert_eq!(slice_root.len(), 2);
        assert_eq!(slice_root[0], d(0, 0));
        assert_eq!(slice_root[1], d(1, 1));

        // Second frame
        s.push_frame();
        assert!(s.is_current_level_empty());
        s.extend([d(2, 0), d(3, 1), d(4, 2)]);
        assert_eq!(s.num_entries(), 5);

        // All entries slice covers entire stack
        assert_eq!(s.all_entries().len(), 5);

        // Current frame entries should be last 3
        let slice2 = s.current_frame_entries();
        assert_eq!(slice2, &[d(2, 0), d(3, 1), d(4, 2)]);

        // Pop LIFO
        assert_eq!(s.pop().unwrap(), d(4, 2));
        assert_eq!(s.pop().unwrap(), d(3, 1));
        assert_eq!(s.pop().unwrap(), d(2, 0));
        assert!(s.is_current_level_empty());

        // Pop frame 2
        assert!(s.pop_frame().is_some());
        assert_eq!(s.depth(), 1);

        // Root frame still has two entries
        assert_eq!(s.num_entries(), 2);
        assert_eq!(s.current_frame_entries(), &[d(0, 0), d(1, 1)]);

        // Pop two entries
        assert_eq!(s.pop().unwrap(), d(1, 1));
        assert_eq!(s.pop().unwrap(), d(0, 0));
        assert!(s.is_current_level_empty());

        // Pop frame 1
        assert!(s.pop_frame().is_some());
        assert!(s.is_empty());
        assert_eq!(s.num_entries(), 0);
    }

    #[test]
    fn test_pop_frame_noop_when_empty() {
        let mut s = SearchStack::new();
        assert!(s.is_empty());
        assert_eq!(s.pop_frame(), None);

        // After pushing and popping a frame with no decisions, still empty
        s.push_frame();
        assert_eq!(s.depth(), 1);
        assert!(s.is_current_level_empty());
        assert!(s.pop_frame().is_some());
        assert!(s.is_empty());
        assert_eq!(s.depth(), 0);
    }

    #[test]
    fn test_reset_clears_but_keeps_capacity() {
        let mut s = SearchStack::preallocated(3, 4);
        let ecap = s.entries.capacity();
        let fcap = s.frames.capacity();

        s.push_frame();
        s.extend([d(0, 0), d(1, 1)]);
        assert_eq!(s.num_entries(), 2);
        assert_eq!(s.depth(), 1);

        s.reset();
        assert_eq!(s.num_entries(), 0);
        assert_eq!(s.depth(), 0);
        assert!(s.is_empty());

        assert_eq!(s.entries.capacity(), ecap);
        assert_eq!(s.frames.capacity(), fcap);
    }

    #[test]
    fn test_current_level_empty_and_start_consistency() {
        let mut s = SearchStack::new();

        // No frame -> empty
        assert!(s.is_current_level_empty());

        // Push frame -> empty
        s.push_frame();
        assert!(s.is_current_level_empty());
        assert_eq!(s.current_level_start(), Some(0));

        // Add one decision
        s.push(d(0, 0));
        assert!(!s.is_current_level_empty());
        assert_eq!(s.num_entries(), 1);

        // Remove it
        assert_eq!(s.pop().unwrap(), d(0, 0));
        assert!(s.is_current_level_empty());

        // Extend with multiple, then truncate via pop_frame; invariant holds
        s.extend([d(1, 0), d(2, 1), d(3, 0)]);
        assert_eq!(s.num_entries(), 3);
        assert!(!s.is_current_level_empty());
        assert!(s.pop_frame().is_some());
        assert_eq!(s.num_entries(), 0);
        assert!(s.is_current_level_empty());
        assert!(s.is_empty());
    }

    #[test]
    fn test_current_frame_entries_and_all_entries_slices() {
        let mut s = SearchStack::new();

        // Empty slices when no frame
        assert_eq!(s.current_frame_entries(), &[]);
        assert_eq!(s.all_entries(), &[]);

        s.push_frame();
        s.extend([d(1, 1), d(2, 2)]);
        assert_eq!(s.current_frame_entries(), &[d(1, 1), d(2, 2)]);
        assert_eq!(s.all_entries(), &[d(1, 1), d(2, 2)]);

        s.push_frame(); // new frame
        s.extend([d(3, 0)]);
        assert_eq!(s.current_frame_entries(), &[d(3, 0)]);
        assert_eq!(s.all_entries(), &[d(1, 1), d(2, 2), d(3, 0)]);
    }

    #[test]
    fn test_allocated_memory_bytes_matches_formula_and_is_monotonic() {
        let s = SearchStack::preallocated(2, 3);
        let expected = s.entries.capacity() * core::mem::size_of::<Decision>()
            + s.frames.capacity() * core::mem::size_of::<usize>();
        assert_eq!(s.allocated_memory_bytes(), expected);

        let mut s2 = SearchStack::new();
        let initial = s2.allocated_memory_bytes();

        s2.ensure_capacity(4, 5);
        let after = s2.allocated_memory_bytes();
        assert!(after >= initial);

        // Push decisions should not reduce allocated bytes
        s2.push_frame();
        s2.extend([d(0, 0), d(1, 1), d(2, 2)]);
        let after_push = s2.allocated_memory_bytes();
        assert!(after_push >= after);

        // Reset keeps capacity
        s2.reset();
        assert_eq!(s2.allocated_memory_bytes(), after_push);
    }

    #[test]
    fn test_display_includes_counts() {
        let mut s = SearchStack::new();
        let s0 = format!("{}", s);
        assert!(s0.contains("entries: 0"));
        assert!(s0.contains("frames: 0"));

        s.push_frame();
        s.push(d(0, 0));
        let s1 = format!("{}", s);
        assert!(s1.contains("entries: 1"));
        assert!(s1.contains("frames: 1"));
    }

    #[test]
    fn test_pop_on_empty_stack_returns_none_and_is_safe() {
        let mut s = SearchStack::new();
        assert!(s.pop().is_none());

        s.push_frame();
        assert!(s.pop().is_none()); // empty frame
        s.extend([d(0, 0)]);
        assert!(s.pop().is_some()); // now has one
        assert!(s.pop().is_none());
    }

    #[test]
    fn test_frame_truncation_discards_stray_entries() {
        let mut s = SearchStack::new();

        s.push_frame();
        s.extend([d(0, 0), d(1, 1)]);
        s.push_frame();
        s.extend([d(2, 0), d(3, 1)]);
        // Manually pop frame and ensure entries truncate to start index
        let start_depth2 = s.current_level_start().unwrap();
        assert_eq!(start_depth2, 2);
        assert!(s.pop_frame().is_some());
        // After pop, entries should have truncated to 2
        assert_eq!(s.num_entries(), 2);
        assert_eq!(s.current_frame_entries(), &[d(0, 0), d(1, 1)]);
    }
}
