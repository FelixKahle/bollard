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
use num_traits::{PrimInt, Zero};

/// An entry in the search trail, recording the previous state
/// before an assignment was made.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TrailEntry<T> {
    old_berth_time: T,
    old_objective: T,
    berth_index: BerthIndex,
    vessel_index: VesselIndex,
    prev_last_decision_time: T,
    prev_last_decision_vessel: VesselIndex,
}

impl<T> TrailEntry<T>
where
    T: Copy,
{
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
        actual_start_time: T,
    ) where
        T: PrimInt,
    {
        debug_assert!(
            !state.is_vessel_assigned(vessel_index),
            "Attempted to assign vessel {} which is already assigned!",
            vessel_index.get()
        );

        let old_berth_time = state.berth_free_time(berth_index);
        let old_objective = state.current_objective();
        let prev_time = state.last_decision_time();
        let prev_vessel = state.last_decision_vessel();

        self.entries.push(TrailEntry {
            old_berth_time,
            old_objective,
            berth_index,
            vessel_index,
            prev_last_decision_time: prev_time,
            prev_last_decision_vessel: prev_vessel,
        });

        state.set_berth_free_time(berth_index, new_berth_time);
        state.assign_vessel(vessel_index);
        state.set_current_objective(new_objective);
        state.set_last_decision(actual_start_time, vessel_index);
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
        actual_start_time: T,
    ) where
        T: PrimInt,
    {
        debug_assert!(
            !state.is_vessel_assigned(vessel_index),
            "Attempted to assign vessel {} which is already assigned!",
            vessel_index.get()
        );

        let old_berth_time = unsafe { state.berth_free_time_unchecked(berth_index) };
        let old_objective = state.current_objective();
        let prev_time = state.last_decision_time();
        let prev_vessel = state.last_decision_vessel();

        self.entries.push(TrailEntry {
            old_berth_time,
            old_objective,
            berth_index,
            vessel_index,
            prev_last_decision_time: prev_time,
            prev_last_decision_vessel: prev_vessel,
        });

        unsafe {
            state.set_berth_free_time_unchecked(berth_index, new_berth_time);
            state.assign_vessel_unchecked(vessel_index);
            state.set_current_objective(new_objective);
            state.set_last_decision(actual_start_time, vessel_index);
        }
    }

    /// Backtracks to the previous frame, undoing all assignments made since then.
    ///
    /// Effectively pops the current frame.
    pub fn backtrack(&mut self, state: &mut SearchState<T>)
    where
        T: PrimInt,
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
        T: PrimInt,
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
        T: PrimInt + Zero,
    {
        unsafe {
            state.set_berth_free_time_unchecked(entry.berth_index, entry.old_berth_time);
            state.unassign_vessel_unchecked(entry.vessel_index);
            state.set_current_objective(entry.old_objective);

            state.set_last_decision(
                entry.prev_last_decision_time,
                entry.prev_last_decision_vessel,
            );
        }
    }

    /// Returns an iterator over all trail entries.
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

    type IntegerType = i64;

    fn make_state(num_berths: usize, num_vessels: usize) -> SearchState<IntegerType> {
        SearchState::<IntegerType>::new(num_berths, num_vessels)
    }

    #[test]
    fn test_new_and_basic_properties() {
        let trail = SearchTrail::<IntegerType>::new();
        assert_eq!(trail.num_entries(), 0);
        assert_eq!(trail.num_frames(), 0);
        assert!(trail.is_empty());
        assert_eq!(trail.depth(), 0);

        let trail2 = SearchTrail::<IntegerType>::with_capacity(10, 5);
        assert_eq!(trail2.num_entries(), 0);
        assert_eq!(trail2.num_frames(), 0);
        assert!(trail2.is_empty());
    }

    #[test]
    fn test_preallocated_and_ensure_capacity() {
        let mut trail = SearchTrail::<IntegerType>::preallocated(7);
        let before_alloc = trail.allocated_memory_bytes();
        trail.ensure_capacity(12);
        let after_alloc = trail.allocated_memory_bytes();
        assert!(after_alloc >= before_alloc);
        assert_eq!(trail.num_entries(), 0);
        assert_eq!(trail.num_frames(), 0);
    }

    #[test]
    fn test_push_frame_and_backtrack_empty() {
        let mut trail = SearchTrail::<IntegerType>::new();
        let mut state = make_state(2, 3);
        trail.backtrack(&mut state);
        assert_eq!(trail.num_frames(), 0);
        assert_eq!(trail.num_entries(), 0);

        trail.push_frame();
        assert_eq!(trail.num_frames(), 1);
        assert_eq!(trail.num_entries(), 0);

        trail.backtrack(&mut state);
        assert_eq!(trail.num_frames(), 0);
        assert_eq!(trail.num_entries(), 0);
    }

    #[test]
    fn test_apply_assignment_creates_entry_and_updates_state() {
        let mut trail = SearchTrail::<IntegerType>::new();
        let mut state = make_state(2, 3);
        trail.push_frame();

        assert_eq!(state.berth_free_time(BerthIndex::new(1)), 0i64);
        assert_eq!(state.current_objective(), 0i64);
        assert_eq!(state.last_decision_time(), i64::MIN);

        let berth = BerthIndex::new(1);
        let vessel = VesselIndex::new(2);
        let new_berth_time = 7 as IntegerType;
        let new_objective = 13 as IntegerType;
        let actual_start_time = 5 as IntegerType;

        trail.apply_assignment(
            &mut state,
            berth,
            vessel,
            new_berth_time,
            new_objective,
            actual_start_time,
        );

        assert_eq!(trail.num_entries(), 1);
        assert_eq!(trail.num_frames(), 1);

        assert!(state.is_vessel_assigned(vessel));
        assert_eq!(state.berth_free_time(berth), new_berth_time);
        assert_eq!(state.current_objective(), new_objective);
        assert_eq!(state.last_decision_time(), actual_start_time);
        assert_eq!(state.last_decision_vessel(), vessel);

        let entry = trail.iter_entries().next().unwrap();
        assert_eq!(entry.berth_index(), berth);
        assert_eq!(entry.vessel_index(), vessel);
        assert_eq!(entry.old_berth_time(), 0i64);
        assert_eq!(entry.old_objective(), 0i64);

        let s = format!("{}", entry);
        assert!(s.contains("TrailEntry("));
        assert!(s.contains("berth: 1"));
        assert!(s.contains("vessel: 2"));
        assert!(s.contains("old_time: 0"));
        assert!(s.contains("old_obj: 0"));
    }

    #[test]
    fn test_backtrack_undo_single_assignment() {
        let mut trail = SearchTrail::<IntegerType>::new();
        let mut state = make_state(3, 4);

        trail.push_frame();

        let berth = BerthIndex::new(0);
        let vessel = VesselIndex::new(1);
        let new_berth_time = 10 as IntegerType;
        let new_objective = 20 as IntegerType;
        let actual_start_time = 10 as IntegerType;

        trail.apply_assignment(
            &mut state,
            berth,
            vessel,
            new_berth_time,
            new_objective,
            actual_start_time,
        );

        assert!(state.is_vessel_assigned(vessel));
        assert_eq!(state.berth_free_time(berth), 10i64);
        assert_eq!(trail.num_entries(), 1);

        trail.backtrack(&mut state);

        assert!(!state.is_vessel_assigned(vessel));
        assert_eq!(state.berth_free_time(berth), 0i64);
        assert_eq!(state.current_objective(), 0i64);
        assert_eq!(state.last_decision_time(), i64::MIN);
        assert_eq!(state.last_decision_vessel().get(), 0);

        assert_eq!(trail.num_entries(), 0);
        assert_eq!(trail.num_frames(), 0);
    }

    #[test]
    fn test_backtrack_multiple_entries_in_single_frame() {
        let mut trail = SearchTrail::<IntegerType>::new();
        let mut state = make_state(2, 3);
        trail.push_frame();

        trail.apply_assignment(
            &mut state,
            BerthIndex::new(0),
            VesselIndex::new(0),
            5i64,
            5i64,
            5i64,
        );
        trail.apply_assignment(
            &mut state,
            BerthIndex::new(1),
            VesselIndex::new(2),
            9i64,
            14i64,
            9i64,
        );

        assert_eq!(trail.num_entries(), 2);
        assert!(state.is_vessel_assigned(VesselIndex::new(0)));
        assert!(state.is_vessel_assigned(VesselIndex::new(2)));
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), 5i64);
        assert_eq!(state.berth_free_time(BerthIndex::new(1)), 9i64);
        assert_eq!(state.current_objective(), 14i64);

        trail.backtrack(&mut state);

        assert_eq!(trail.num_entries(), 0);
        assert_eq!(trail.num_frames(), 0);
        assert!(!state.is_vessel_assigned(VesselIndex::new(0)));
        assert!(!state.is_vessel_assigned(VesselIndex::new(2)));
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), 0i64);
        assert_eq!(state.berth_free_time(BerthIndex::new(1)), 0i64);
        assert_eq!(state.current_objective(), 0i64);
    }

    #[test]
    fn test_nested_frames_and_partial_backtrack() {
        let mut trail = SearchTrail::<IntegerType>::new();
        let mut state = make_state(1, 3);

        trail.push_frame();
        trail.apply_assignment(
            &mut state,
            BerthIndex::new(0),
            VesselIndex::new(0),
            3i64,
            3i64,
            3i64,
        );

        trail.push_frame();
        trail.apply_assignment(
            &mut state,
            BerthIndex::new(0),
            VesselIndex::new(1),
            6i64,
            9i64,
            6i64,
        );

        assert_eq!(trail.num_frames(), 2);
        assert_eq!(trail.num_entries(), 2);

        trail.backtrack(&mut state);
        assert_eq!(trail.num_frames(), 1);
        assert_eq!(trail.num_entries(), 1);

        assert!(state.is_vessel_assigned(VesselIndex::new(0)));
        assert!(!state.is_vessel_assigned(VesselIndex::new(1)));
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), 3i64);
        assert_eq!(state.current_objective(), 3i64);

        trail.backtrack(&mut state);
        assert_eq!(trail.num_frames(), 0);
        assert_eq!(trail.num_entries(), 0);

        assert!(!state.is_vessel_assigned(VesselIndex::new(0)));
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), 0i64);
        assert_eq!(state.current_objective(), 0i64);
    }

    #[test]
    fn test_clear_undoes_all_without_needing_frames() {
        let mut trail = SearchTrail::<IntegerType>::new();
        let mut state = make_state(2, 3);

        trail.apply_assignment(
            &mut state,
            BerthIndex::new(0),
            VesselIndex::new(1),
            4i64,
            4i64,
            4i64,
        );
        trail.apply_assignment(
            &mut state,
            BerthIndex::new(1),
            VesselIndex::new(2),
            7i64,
            11i64,
            7i64,
        );

        assert_eq!(trail.num_entries(), 2);
        assert_eq!(trail.num_frames(), 0);

        trail.clear(&mut state);

        assert_eq!(trail.num_entries(), 0);
        assert_eq!(trail.num_frames(), 0);
        assert!(!state.is_vessel_assigned(VesselIndex::new(1)));
        assert!(!state.is_vessel_assigned(VesselIndex::new(2)));
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), 0i64);
        assert_eq!(state.berth_free_time(BerthIndex::new(1)), 0i64);
        assert_eq!(state.current_objective(), 0i64);
    }

    #[test]
    fn test_reset_clears_markers_but_not_state_changes() {
        let mut trail = SearchTrail::<IntegerType>::new();
        let mut state = make_state(1, 2);

        trail.push_frame();
        trail.apply_assignment(
            &mut state,
            BerthIndex::new(0),
            VesselIndex::new(0),
            5i64,
            5i64,
            5i64,
        );

        assert_eq!(trail.num_entries(), 1);
        assert_eq!(trail.num_frames(), 1);
        assert!(state.is_vessel_assigned(VesselIndex::new(0)));
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), 5i64);

        trail.reset();

        assert_eq!(trail.num_entries(), 0);
        assert_eq!(trail.num_frames(), 0);

        assert!(state.is_vessel_assigned(VesselIndex::new(0)));
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), 5i64);
        assert_eq!(state.current_objective(), 5i64);
    }

    #[test]
    fn test_apply_assignment_unchecked_matches_checked_behavior() {
        let mut trail_checked = SearchTrail::<IntegerType>::new();
        let mut state_checked = make_state(2, 2);

        let mut trail_unchecked = SearchTrail::<IntegerType>::new();
        let mut state_unchecked = make_state(2, 2);

        trail_checked.apply_assignment(
            &mut state_checked,
            BerthIndex::new(1),
            VesselIndex::new(0),
            8i64,
            8i64,
            8i64,
        );

        unsafe {
            trail_unchecked.apply_assignment_unchecked(
                &mut state_unchecked,
                BerthIndex::new(1),
                VesselIndex::new(0),
                8i64,
                8i64,
                8i64,
            );
        }

        assert_eq!(
            state_checked.berth_free_time(BerthIndex::new(1)),
            state_unchecked.berth_free_time(BerthIndex::new(1))
        );
        assert_eq!(
            state_checked.is_vessel_assigned(VesselIndex::new(0)),
            state_unchecked.is_vessel_assigned(VesselIndex::new(0))
        );
        assert_eq!(trail_checked.num_entries(), trail_unchecked.num_entries());
        assert_eq!(
            trail_checked.iter_entries().count(),
            trail_unchecked.iter_entries().count()
        );

        trail_checked.push_frame();
        trail_checked.backtrack(&mut state_checked);
        trail_checked.clear(&mut state_checked);

        trail_unchecked.clear(&mut state_unchecked);

        assert_eq!(trail_checked.num_entries(), 0);
        assert_eq!(trail_unchecked.num_entries(), 0);
        assert_eq!(state_checked.berth_free_time(BerthIndex::new(1)), 0i64);
        assert_eq!(state_unchecked.berth_free_time(BerthIndex::new(1)), 0i64);
        assert!(!state_checked.is_vessel_assigned(VesselIndex::new(0)));
        assert!(!state_unchecked.is_vessel_assigned(VesselIndex::new(0)));
    }

    #[test]
    fn test_display_for_search_trail() {
        let mut trail = SearchTrail::<IntegerType>::new();
        assert_eq!(format!("{}", trail), "SearchTrail(entries: 0, frames: 0)");

        let mut state = make_state(1, 1);
        trail.push_frame();
        trail.apply_assignment(
            &mut state,
            BerthIndex::new(0),
            VesselIndex::new(0),
            1i64,
            1i64,
            1i64,
        );

        let s = format!("{}", trail);
        assert!(s.contains("SearchTrail(entries: 1, frames: 1)"));
    }

    #[test]
    fn test_iter_entries_order_lifo_backtrack() {
        let mut trail = SearchTrail::<IntegerType>::new();
        let mut state = make_state(2, 3);
        trail.push_frame();

        trail.apply_assignment(
            &mut state,
            BerthIndex::new(0),
            VesselIndex::new(0),
            1i64,
            1i64,
            1i64,
        );
        trail.apply_assignment(
            &mut state,
            BerthIndex::new(1),
            VesselIndex::new(1),
            2i64,
            3i64,
            2i64,
        );
        trail.apply_assignment(
            &mut state,
            BerthIndex::new(1),
            VesselIndex::new(2),
            5i64,
            8i64,
            5i64,
        );

        let entries: Vec<_> = trail.iter_entries().copied().collect();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].vessel_index().get(), 0);
        assert_eq!(entries[1].vessel_index().get(), 1);
        assert_eq!(entries[2].vessel_index().get(), 2);

        trail.backtrack(&mut state);
        assert_eq!(trail.num_entries(), 0);
        assert!(!state.is_vessel_assigned(VesselIndex::new(0)));
        assert!(!state.is_vessel_assigned(VesselIndex::new(1)));
        assert!(!state.is_vessel_assigned(VesselIndex::new(2)));
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), 0i64);
        assert_eq!(state.berth_free_time(BerthIndex::new(1)), 0i64);
        assert_eq!(state.current_objective(), 0i64);
    }

    #[test]
    fn test_backtrack_with_no_frames_is_noop() {
        let mut trail = SearchTrail::<IntegerType>::new();
        let mut state = make_state(1, 1);

        trail.apply_assignment(
            &mut state,
            BerthIndex::new(0),
            VesselIndex::new(0),
            3i64,
            3i64,
            3i64,
        );

        let prev_entries = trail.num_entries();
        let prev_frames = trail.num_frames();

        trail.backtrack(&mut state);

        assert_eq!(trail.num_entries(), prev_entries, "entries unchanged");
        assert_eq!(trail.num_frames(), prev_frames, "frames unchanged");
        assert!(state.is_vessel_assigned(VesselIndex::new(0)));
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), 3i64);
    }

    #[test]
    fn test_clear_on_empty_is_safe() {
        let mut trail = SearchTrail::<IntegerType>::new();
        let mut state = make_state(1, 1);
        trail.clear(&mut state);
        assert_eq!(trail.num_entries(), 0);
        assert_eq!(trail.num_frames(), 0);
    }

    #[test]
    fn test_allocated_memory_bytes_is_nonzero_after_capacity() {
        let trail = SearchTrail::<IntegerType>::with_capacity(16, 8);
        let bytes = trail.allocated_memory_bytes();
        assert!(bytes > 0);
    }
}
