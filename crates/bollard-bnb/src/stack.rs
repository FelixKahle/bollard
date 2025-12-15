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
pub struct SearchStack<T> {
    /// The linear stack of pending decisions.
    entries: Vec<Decision<T>>,
    /// A stack of indices pointing to `entries`.
    /// `frames[i]` stores the index in `entries` where depth `i` began.
    frames: Vec<usize>,
}

impl<T> Default for SearchStack<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SearchStack<T> {
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
    pub fn push(&mut self, decision: Decision<T>) {
        self.entries.push(decision);
    }

    /// Extends the stack with multiple decision entries.
    #[inline]
    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Decision<T>>,
    {
        self.entries.extend(iter);
    }

    /// Pops the next decision (LIFO) from the stack.
    #[inline]
    pub fn pop(&mut self) -> Option<Decision<T>> {
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
    pub fn current_frame_entries(&self) -> &[Decision<T>] {
        match self.frames.last() {
            Some(&start) => &self.entries[start..],
            None => &[],
        }
    }

    /// Returns a slice of all decisions in the stack.
    #[inline]
    pub fn all_entries(&self) -> &[Decision<T>] {
        &self.entries
    }

    /// Returns the total allocated memory in bytes.
    #[inline]
    pub fn allocated_memory_bytes(&self) -> usize {
        let entries_size = self.entries.capacity() * core::mem::size_of::<Decision<T>>();
        let frames_size = self.frames.capacity() * core::mem::size_of::<usize>();
        entries_size + frames_size
    }

    /// Returns an iterator over all decisions in the stack.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, Decision<T>> {
        self.entries.iter()
    }
}

impl<T> std::fmt::Display for SearchStack<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SearchStack(entries: {}, frames: {})",
            self.entries.len(),
            self.frames.len()
        )
    }
}
