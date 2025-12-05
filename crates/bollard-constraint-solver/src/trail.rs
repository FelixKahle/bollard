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

use crate::state::SearchState;
use bollard_model::index::{BerthIndex, VesselIndex};
use num_traits::Zero;

/// An entry in the search trail, recording the previous state
/// before an assignment was made.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TrailEntry<T> {
    old_berth_time: T,
    old_objective: T,
    berth_index: BerthIndex,
    vessel_index: VesselIndex,
}

impl<T> TrailEntry<T>
where
    T: Copy,
{
    #[inline]
    pub fn new(
        old_berth_time: T,
        old_objective: T,
        berth_index: BerthIndex,
        vessel_index: VesselIndex,
    ) -> Self {
        Self {
            old_berth_time,
            old_objective,
            berth_index,
            vessel_index,
        }
    }

    #[inline]
    pub fn old_berth_time(&self) -> T {
        self.old_berth_time
    }

    #[inline]
    pub fn old_objective(&self) -> T {
        self.old_objective
    }

    #[inline]
    pub fn berth_index(&self) -> BerthIndex {
        self.berth_index
    }

    #[inline]
    pub fn vessel_index(&self) -> VesselIndex {
        self.vessel_index
    }
}

impl<T> std::fmt::Display for TrailEntry<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TrailEntry(berth: {}, vessel: {}, old_time: {}, old_obj: {})",
            self.berth_index.get(),
            self.vessel_index.get(),
            self.old_berth_time,
            self.old_objective
        )
    }
}

#[derive(Debug, Clone)]
pub struct SearchTrail<T> {
    /// The linear history of all changes made to the state.
    entries: Vec<TrailEntry<T>>,
    /// A stack of indices pointing to `entries`.
    /// `frames[i]` stores the index in `entries` where depth `i` began.
    frames: Vec<usize>,
}

impl<T> Default for SearchTrail<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SearchTrail<T> {
    /// Creates a new, empty `SearchTrail`.
    #[inline]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            frames: Vec::new(),
        }
    }

    /// Creates a new `SearchTrail` with specified capacities.
    #[inline]
    pub fn with_capacity(entry_capacity: usize, frame_capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(entry_capacity),
            frames: Vec::with_capacity(frame_capacity),
        }
    }

    /// Creates a new `SearchTrail` preallocating space based on the number of vessels.
    #[inline]
    pub fn preallocated(num_vessels: usize) -> Self {
        let entry_capacity = num_vessels;
        let frame_capacity = num_vessels + 1;

        Self {
            entries: Vec::with_capacity(entry_capacity),
            frames: Vec::with_capacity(frame_capacity),
        }
    }

    /// Ensures the trail has capacity for the given problem size.
    pub fn ensure_capacity(&mut self, num_vessels: usize) {
        if self.entries.capacity() < num_vessels {
            self.entries.reserve(num_vessels - self.entries.capacity());
        }
        if self.frames.capacity() < num_vessels + 1 {
            self.frames
                .reserve((num_vessels + 1) - self.frames.capacity());
        }
    }

    /// Returns the number of entries in the trail.
    #[inline]
    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    /// Returns the number of frames (depth) in the trail.
    #[inline]
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Returns the current depth of the search trail (alias for num_frames).
    #[inline]
    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    /// Returns true if there are no frames tracked.
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

    /// Applies an assignment to the search state and pushes an entry to the trail.
    ///
    /// # Panics
    ///
    /// Panics if `berth_index` is out of `0..num_berths` or
    /// `vessel_index` is out of `0..num_vessels`, or in
    /// debug builds if the vessel is already assigned.
    pub fn apply_assignment(
        &mut self,
        state: &mut SearchState<T>,
        berth_index: BerthIndex,
        vessel_index: VesselIndex,
        new_berth_time: T,
        new_objective: T,
    ) where
        T: Copy + Zero,
    {
        debug_assert!(
            !state.is_vessel_assigned(vessel_index),
            "Attempted to assign vessel {} which is already assigned!",
            vessel_index.get()
        );

        let old_berth_time = state.berth_free_time(berth_index);
        let old_objective = state.current_objective();

        self.entries.push(TrailEntry::new(
            old_berth_time,
            old_objective,
            berth_index,
            vessel_index,
        ));

        state.set_berth_free_time(berth_index, new_berth_time);
        state.assign_vessel(vessel_index);
        state.set_current_objective(new_objective);
    }

    /// Applies an assignment without bounds checking in the state.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the vessel is already assigned.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `berth_index` is within `0..num_berths`
    /// and `vessel_index` is within `0..num_vessels`.
    pub unsafe fn apply_assignment_unchecked(
        &mut self,
        state: &mut SearchState<T>,
        berth_index: BerthIndex,
        vessel_index: VesselIndex,
        new_berth_time: T,
        new_objective: T,
    ) where
        T: Copy + Zero,
    {
        debug_assert!(
            !state.is_vessel_assigned(vessel_index),
            "Attempted to assign vessel {} which is already assigned!",
            vessel_index.get()
        );

        let old_berth_time = unsafe { state.berth_free_time_unchecked(berth_index) };
        let old_objective = state.current_objective();

        self.entries.push(TrailEntry::new(
            old_berth_time,
            old_objective,
            berth_index,
            vessel_index,
        ));

        unsafe {
            state.set_berth_free_time_unchecked(berth_index, new_berth_time);
            state.assign_vessel_unchecked(vessel_index);
            state.set_current_objective(new_objective);
        }
    }

    /// Backtracks to the previous frame, undoing all assignments made since then.
    ///
    /// Effectively pops the current frame.
    pub fn backtrack(&mut self, state: &mut SearchState<T>)
    where
        T: Copy + Zero,
    {
        let start_index = match self.frames.pop() {
            Some(index) => index,
            None => return,
        };

        while self.entries.len() > start_index {
            let entry = unsafe { self.entries.pop().unwrap_unchecked() };
            self.undo_entry(state, entry);
        }
    }

    /// Clears the entire trail, undoing all assignments made across all frames.
    pub fn clear(&mut self, state: &mut SearchState<T>)
    where
        T: Copy + Zero,
    {
        for entry in self.entries.iter().rev() {
            self.undo_entry(state, *entry);
        }
        self.entries.clear();
        self.frames.clear();
    }

    /// Resets the trail markers without undoing any state changes.
    #[inline]
    pub fn reset(&mut self) {
        self.entries.clear();
        self.frames.clear();
    }

    /// Returns the total allocated memory in bytes.
    #[inline]
    pub fn allocated_memory_bytes(&self) -> usize {
        let entries_size = self.entries.capacity() * std::mem::size_of::<TrailEntry<T>>();
        let frames_size = self.frames.capacity() * std::mem::size_of::<usize>();
        entries_size + frames_size
    }

    /// Helper to undo a single entry.
    fn undo_entry(&self, state: &mut SearchState<T>, entry: TrailEntry<T>)
    where
        T: Copy + Zero,
    {
        unsafe {
            state.set_berth_free_time_unchecked(entry.berth_index, entry.old_berth_time);
            state.unassign_vessel_unchecked(entry.vessel_index);
            state.set_current_objective(entry.old_objective);
        }
    }

    #[inline]
    pub fn iter_entries(&self) -> std::slice::Iter<'_, TrailEntry<T>> {
        self.entries.iter()
    }
}

impl<T> std::fmt::Display for SearchTrail<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SearchTrail(entries: {}, frames: {})",
            self.entries.len(),
            self.frames.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::SearchState;
    use bollard_model::index::{BerthIndex, VesselIndex};
    use num_traits::Zero;

    type I = i32;

    fn make_state(num_berths: usize, num_vessels: usize) -> SearchState<I> {
        SearchState::<I>::new(num_berths, num_vessels)
    }

    #[test]
    fn test_trail_entry_accessors_and_display() {
        let e = TrailEntry::new(7, 42, BerthIndex::new(3), VesselIndex::new(5));
        assert_eq!(e.old_berth_time(), 7);
        assert_eq!(e.old_objective(), 42);
        assert_eq!(e.berth_index().get(), 3);
        assert_eq!(e.vessel_index().get(), 5);

        let fmt = format!("{}", e);
        assert!(fmt.contains("TrailEntry("));
        assert!(fmt.contains("berth: 3"));
        assert!(fmt.contains("vessel: 5"));
        assert!(fmt.contains("old_time: 7"));
        assert!(fmt.contains("old_obj: 42"));
    }

    #[test]
    fn test_new_with_capacity_preallocated_and_basic_props() {
        let t = SearchTrail::<I>::new();
        assert_eq!(t.num_entries(), 0);
        assert_eq!(t.num_frames(), 0);
        assert_eq!(t.depth(), 0);
        assert!(t.is_empty());

        let t2 = SearchTrail::<I>::with_capacity(10, 4);
        assert_eq!(t2.num_entries(), 0);
        assert_eq!(t2.num_frames(), 0);
        assert!(t2.is_empty());
        assert!(t2.allocated_memory_bytes() > 0);

        let t3 = SearchTrail::<I>::preallocated(7);
        assert_eq!(t3.num_entries(), 0);
        assert_eq!(t3.num_frames(), 0);
        assert!(t3.is_empty());

        let disp = format!("{}", t3);
        assert!(disp.contains("SearchTrail(entries: 0, frames: 0)"));
    }

    #[test]
    fn test_ensure_capacity_grows_as_expected() {
        let mut t = SearchTrail::<I>::with_capacity(1, 1);
        let init_entries_cap = t.entries.capacity();
        let init_frames_cap = t.frames.capacity();
        let init_bytes = t.allocated_memory_bytes();

        // Request larger capacity
        t.ensure_capacity(10);

        // Capacities should not shrink, and typically grow
        assert!(t.entries.capacity() >= init_entries_cap);
        assert!(t.frames.capacity() >= init_frames_cap);

        // Allocated bytes should be monotonic (non-decreasing)
        let after_bytes = t.allocated_memory_bytes();
        assert!(after_bytes >= init_bytes);

        // Idempotence when requirements are below current capacities
        let cap_entries = t.entries.capacity();
        let cap_frames = t.frames.capacity();
        t.ensure_capacity(5);
        assert_eq!(t.entries.capacity(), cap_entries);
        assert_eq!(t.frames.capacity(), cap_frames);
    }

    #[test]
    fn test_push_frame_and_is_empty_depth_num_frames() {
        let mut t = SearchTrail::<I>::new();
        assert!(t.is_empty());
        assert_eq!(t.depth(), 0);

        t.push_frame();
        assert!(!t.is_empty());
        assert_eq!(t.depth(), 1);
        assert_eq!(t.num_frames(), 1);

        t.push_frame();
        assert_eq!(t.depth(), 2);
        assert_eq!(t.num_frames(), 2);
    }

    #[test]
    fn test_apply_assignment_and_backtrack_single_frame() {
        let mut state = make_state(2, 3);
        let mut t = SearchTrail::<I>::new();

        let v = VesselIndex::new(1);
        let b = BerthIndex::new(0);

        // Frame
        t.push_frame();

        // Apply assignment: change berth 0 free time to 10, objective to 5
        t.apply_assignment(&mut state, b, v, 10, 5);
        assert_eq!(t.num_entries(), 1);
        assert_eq!(t.num_frames(), 1);

        // State changed
        assert_eq!(state.berth_free_time(b), 10);
        assert!(state.is_vessel_assigned(v));
        assert_eq!(state.current_objective(), 5);

        // Backtrack: undo to frame start
        t.backtrack(&mut state);
        assert_eq!(t.num_entries(), 0);
        assert_eq!(t.num_frames(), 0);
        assert!(t.is_empty());

        // State restored
        assert_eq!(state.berth_free_time(b), I::zero());
        assert!(!state.is_vessel_assigned(v));
        assert_eq!(state.current_objective(), I::zero());
    }

    #[test]
    fn test_apply_assignment_unchecked_and_nested_frames_backtrack() {
        let mut state = make_state(3, 3);
        let mut t = SearchTrail::<I>::new();

        // Frame 0
        t.push_frame();
        unsafe {
            t.apply_assignment_unchecked(&mut state, BerthIndex::new(1), VesselIndex::new(0), 7, 3)
        };
        assert_eq!(t.num_entries(), 1);
        assert_eq!(t.depth(), 1);
        assert_eq!(state.berth_free_time(BerthIndex::new(1)), 7);
        assert!(state.is_vessel_assigned(VesselIndex::new(0)));
        assert_eq!(state.current_objective(), 3);

        // Frame 1
        t.push_frame();
        unsafe {
            t.apply_assignment_unchecked(&mut state, BerthIndex::new(2), VesselIndex::new(2), 12, 9)
        };
        assert_eq!(t.num_entries(), 2);
        assert_eq!(t.depth(), 2);
        assert_eq!(state.berth_free_time(BerthIndex::new(2)), 12);
        assert!(state.is_vessel_assigned(VesselIndex::new(2)));
        assert_eq!(state.current_objective(), 9);

        // Backtrack inner frame only
        t.backtrack(&mut state);
        assert_eq!(t.num_entries(), 1);
        assert_eq!(t.depth(), 1);
        // Inner assignment undone, outer remains
        assert_eq!(state.berth_free_time(BerthIndex::new(2)), I::zero());
        assert!(!state.is_vessel_assigned(VesselIndex::new(2)));
        assert_eq!(state.current_objective(), 3); // restored to cost after first assign

        assert_eq!(state.berth_free_time(BerthIndex::new(1)), 7);
        assert!(state.is_vessel_assigned(VesselIndex::new(0)));

        // Backtrack outer frame
        t.backtrack(&mut state);
        assert_eq!(t.num_entries(), 0);
        assert_eq!(t.depth(), 0);
        assert!(t.is_empty());

        assert_eq!(state.berth_free_time(BerthIndex::new(1)), I::zero());
        assert!(!state.is_vessel_assigned(VesselIndex::new(0)));
        assert_eq!(state.current_objective(), I::zero());
    }

    #[test]
    fn test_backtrack_no_frames_is_noop() {
        let mut state = make_state(1, 1);
        let mut t = SearchTrail::<I>::new();

        // No frames; backtrack must do nothing and not panic
        t.backtrack(&mut state);
        assert_eq!(t.num_entries(), 0);
        assert_eq!(t.num_frames(), 0);
        assert!(t.is_empty());
    }

    #[test]
    fn test_clear_undoes_all_entries_and_resets_frames() {
        let mut state = make_state(2, 2);
        let mut t = SearchTrail::<I>::new();

        // Frame 0
        t.push_frame();
        t.apply_assignment(&mut state, BerthIndex::new(0), VesselIndex::new(0), 5, 1);

        // Frame 1
        t.push_frame();
        t.apply_assignment(&mut state, BerthIndex::new(1), VesselIndex::new(1), 8, 3);

        assert_eq!(t.num_entries(), 2);
        assert_eq!(t.depth(), 2);
        assert_eq!(state.current_objective(), 3);
        assert!(state.is_vessel_assigned(VesselIndex::new(0)));
        assert!(state.is_vessel_assigned(VesselIndex::new(1)));
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), 5);
        assert_eq!(state.berth_free_time(BerthIndex::new(1)), 8);

        // Clear should undo both entries and empty frames
        t.clear(&mut state);
        assert_eq!(t.num_entries(), 0);
        assert_eq!(t.num_frames(), 0);
        assert_eq!(t.depth(), 0);
        assert!(t.is_empty());

        assert_eq!(state.current_objective(), I::zero());
        assert!(!state.is_vessel_assigned(VesselIndex::new(0)));
        assert!(!state.is_vessel_assigned(VesselIndex::new(1)));
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), I::zero());
        assert_eq!(state.berth_free_time(BerthIndex::new(1)), I::zero());
    }

    #[test]
    fn test_reset_clears_trail_but_does_not_touch_state() {
        let mut state = make_state(2, 2);
        let mut t = SearchTrail::<I>::new();

        // Modify state directly and via trail
        state.set_berth_free_time(BerthIndex::new(0), 11);
        state.set_current_objective(7);
        t.push_frame();
        t.apply_assignment(&mut state, BerthIndex::new(1), VesselIndex::new(0), 9, 13);

        assert_eq!(t.num_entries(), 1);
        assert_eq!(t.num_frames(), 1);
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), 11);
        assert_eq!(state.berth_free_time(BerthIndex::new(1)), 9);
        assert!(state.is_vessel_assigned(VesselIndex::new(0)));
        assert_eq!(state.current_objective(), 13);

        // Reset trail markers only
        t.reset();
        assert_eq!(t.num_entries(), 0);
        assert_eq!(t.num_frames(), 0);
        assert!(t.is_empty());

        // State should remain unchanged
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), 11);
        assert_eq!(state.berth_free_time(BerthIndex::new(1)), 9);
        assert!(state.is_vessel_assigned(VesselIndex::new(0)));
        assert_eq!(state.current_objective(), 13);
    }

    #[test]
    fn test_allocated_memory_bytes_matches_formula() {
        let t = SearchTrail::<I>::with_capacity(16, 8);
        let expected = t.entries.capacity() * std::mem::size_of::<TrailEntry<I>>()
            + t.frames.capacity() * std::mem::size_of::<usize>();
        assert_eq!(t.allocated_memory_bytes(), expected);

        // Ensure increasing capacity increases reported bytes
        let mut t2 = SearchTrail::<I>::with_capacity(1, 1);
        let initial = t2.allocated_memory_bytes();
        t2.ensure_capacity(32);
        let after = t2.allocated_memory_bytes();
        assert!(after >= initial);
    }

    #[test]
    fn test_current_objective_restoration_order_with_multiple_entries_in_one_frame() {
        let mut state = make_state(1, 3);
        let mut t = SearchTrail::<I>::new();

        // One frame, two consecutive assignments
        t.push_frame();

        // Assign vessel 0, objective -> 10, berth 0 -> 4
        t.apply_assignment(&mut state, BerthIndex::new(0), VesselIndex::new(0), 4, 10);
        // Assign vessel 1, objective -> 25, berth 0 -> 7
        t.apply_assignment(&mut state, BerthIndex::new(0), VesselIndex::new(1), 7, 25);

        assert_eq!(t.num_entries(), 2);
        assert_eq!(state.current_objective(), 25);
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), 7);
        assert!(state.is_vessel_assigned(VesselIndex::new(0)));
        assert!(state.is_vessel_assigned(VesselIndex::new(1)));

        // Backtrack should undo in reverse order restoring objective properly
        t.backtrack(&mut state);
        assert_eq!(t.num_entries(), 0);
        assert_eq!(state.current_objective(), I::zero());
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), I::zero());
        assert!(!state.is_vessel_assigned(VesselIndex::new(0)));
        assert!(!state.is_vessel_assigned(VesselIndex::new(1)));
    }

    #[test]
    fn test_backtrack_over_empty_frame_does_not_change_state() {
        let mut state = make_state(1, 1);
        let mut t = SearchTrail::<I>::new();

        // Push an empty frame
        t.push_frame();
        assert_eq!(t.num_frames(), 1);
        assert_eq!(t.num_entries(), 0);

        // Backtrack: should remove frame and leave state unchanged
        t.backtrack(&mut state);
        assert_eq!(t.num_frames(), 0);
        assert_eq!(t.num_entries(), 0);
        assert!(t.is_empty());
        assert_eq!(state.current_objective(), I::zero());
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), I::zero());
        assert!(!state.is_vessel_assigned(VesselIndex::new(0)));
    }

    #[test]
    fn test_multiple_frames_mixed_empty_and_nonempty_sequences() {
        let mut state = make_state(2, 3);
        let mut t = SearchTrail::<I>::new();

        // Frame A (non-empty)
        t.push_frame();
        t.apply_assignment(&mut state, BerthIndex::new(0), VesselIndex::new(1), 5, 7);

        // Frame B (empty)
        t.push_frame();

        // Frame C (non-empty)
        t.push_frame();
        t.apply_assignment(&mut state, BerthIndex::new(1), VesselIndex::new(2), 11, 15);

        assert_eq!(t.num_frames(), 3);
        assert_eq!(t.num_entries(), 2);
        assert_eq!(state.current_objective(), 15);
        assert!(state.is_vessel_assigned(VesselIndex::new(1)));
        assert!(state.is_vessel_assigned(VesselIndex::new(2)));

        // Backtrack C: undo vessel 2
        t.backtrack(&mut state);
        assert_eq!(t.num_frames(), 2);
        assert_eq!(t.num_entries(), 1);
        assert_eq!(state.current_objective(), 7);
        assert!(!state.is_vessel_assigned(VesselIndex::new(2)));
        assert!(state.is_vessel_assigned(VesselIndex::new(1)));
        assert_eq!(state.berth_free_time(BerthIndex::new(1)), I::zero());

        // Backtrack B: empty frame, no change in entries or state
        t.backtrack(&mut state);
        assert_eq!(t.num_frames(), 1);
        assert_eq!(t.num_entries(), 1);
        assert_eq!(state.current_objective(), 7);

        // Backtrack A: undo vessel 1
        t.backtrack(&mut state);
        assert_eq!(t.num_frames(), 0);
        assert_eq!(t.num_entries(), 0);
        assert_eq!(state.current_objective(), I::zero());
        assert!(!state.is_vessel_assigned(VesselIndex::new(1)));
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), I::zero());
    }

    #[test]
    fn test_display_includes_counts() {
        let mut t = SearchTrail::<I>::new();
        assert!(format!("{}", t).contains("entries: 0"));
        assert!(format!("{}", t).contains("frames: 0"));

        t.push_frame();
        let s = format!("{}", t);
        assert!(s.contains("entries: 0"));
        assert!(s.contains("frames: 1"));

        let mut state = make_state(1, 1);
        t.apply_assignment(&mut state, BerthIndex::new(0), VesselIndex::new(0), 3, 5);
        let s2 = format!("{}", t);
        assert!(s2.contains("entries: 1"));
        assert!(s2.contains("frames: 1"));
    }

    #[test]
    fn test_apply_assignment_unchecked_respects_invariants() {
        let mut state = make_state(1, 2);
        let mut t = SearchTrail::<I>::new();

        // Frame
        t.push_frame();

        // Unsafe apply for vessel 1 on berth 0
        unsafe {
            t.apply_assignment_unchecked(&mut state, BerthIndex::new(0), VesselIndex::new(1), 6, 9)
        }
        assert_eq!(t.num_entries(), 1);
        assert!(state.is_vessel_assigned(VesselIndex::new(1)));
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), 6);
        assert_eq!(state.current_objective(), 9);

        // Backtrack: should restore to initial
        t.backtrack(&mut state);
        assert_eq!(t.num_entries(), 0);
        assert!(!state.is_vessel_assigned(VesselIndex::new(1)));
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), I::zero());
        assert_eq!(state.current_objective(), I::zero());
    }
}
