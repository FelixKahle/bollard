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
    // Note: We don't store `old_vessel_assignment` (bool) because
    // we strictly toggle False -> True during search, so undo is always True -> False.
    berth_index: BerthIndex,
    vessel_index: VesselIndex,
}

impl<T> TrailEntry<T>
where
    T: Copy,
{
    /// Creates a new `TrailEntry`.
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

    /// Returns the old berth free time before the assignment.
    #[inline]
    pub fn old_berth_time(&self) -> T {
        self.old_berth_time
    }

    /// Returns the old objective value before the assignment.
    #[inline]
    pub fn old_objective(&self) -> T {
        self.old_objective
    }

    /// Returns the berth index involved in the assignment.
    #[inline]
    pub fn berth_index(&self) -> BerthIndex {
        self.berth_index
    }

    /// Returns the vessel index involved in the assignment.
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
            "TrailEntry(berth_index: {}, vessel_index: {}, old_berth_time: {}, old_objective: {})",
            self.berth_index.get(),
            self.vessel_index.get(),
            self.old_berth_time,
            self.old_objective
        )
    }
}

/// Represents an assignment of a vessel to a berth,
/// along with the new berth free time and objective value
/// after the assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Assignment<T> {
    /// The new berth free time after the assignment.
    pub new_berth_time: T,

    /// The new objective value after the assignment.
    pub new_objective: T,

    /// The index of the vessel being assigned.
    pub vessel_index: VesselIndex,

    /// The index of the berth to which the vessel is assigned.
    pub berth_index: BerthIndex,
}

impl<T> Assignment<T> {
    /// Creates a new `Assignment`.
    #[inline]
    pub const fn new(
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        new_berth_time: T,
        new_objective: T,
    ) -> Self {
        Self {
            vessel_index,
            berth_index,
            new_berth_time,
            new_objective,
        }
    }
}

impl<T> std::fmt::Display for Assignment<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Assignment(vessel_index: {}, berth_index: {}, new_berth_time: {}, new_objective: {})",
            self.vessel_index.get(),
            self.berth_index.get(),
            self.new_berth_time,
            self.new_objective
        )
    }
}

#[derive(Debug, Clone)]
pub struct SearchTrail<T> {
    /// The linear history of all changes made to the state.
    entries: Vec<TrailEntry<T>>,
    /// A stack of indices pointing to `entries`.
    /// `frames[i]` stores the index in `entries` where depth `i` began.
    frame: Vec<usize>,
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
            frame: Vec::new(),
        }
    }

    /// Creates a new `SearchTrail` with specified capacities.
    /// `entry_capacity` is the initial capacity for the entries,
    /// and `frame_capacity` is the initial capacity for the frame stack.
    ///
    /// # Note
    ///
    /// It's recommended to use `preallocated` when the number of vessels is known,
    /// as it sets the allocated to the upper bound for the search trail.
    #[inline]
    pub fn with_capacity(entry_capacity: usize, frame_capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(entry_capacity),
            frame: Vec::with_capacity(frame_capacity),
        }
    }

    /// Creates a new `SearchTrail` preallocating space based on the number of vessels.
    #[inline]
    pub fn preallocated(num_vessels: usize) -> Self {
        let entry_capacity = num_vessels;
        let frame_capacity = num_vessels + 1;

        Self {
            entries: Vec::with_capacity(entry_capacity),
            frame: Vec::with_capacity(frame_capacity),
        }
    }

    /// Returns the number of entries in the trail.
    #[inline]
    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    /// Returns the number of frames in the trail.
    #[inline]
    pub fn num_frames(&self) -> usize {
        self.frame.len()
    }

    /// Pushes a new checkpoint onto the frame stack.
    #[inline]
    pub fn push_checkpoint(&mut self) {
        self.frame.push(self.entries.len());
    }

    /// Applies an assignment to the search state, recording the previous state in the trail.
    ///
    /// # Panics
    ///
    /// If `assignment.vessel_index` is not in 0..num_vessels() or `assignment.berth_index` is not in 0..num_berths().
    /// In debug mode, also panics if the vessel is already assigned.
    pub fn apply_assignment(&mut self, state: &mut SearchState<T>, assignment: Assignment<T>)
    where
        T: Copy + Zero,
    {
        debug_assert!(
            !state.is_vessel_assigned(assignment.vessel_index),
            "Attempted to assign vessel {} which is already assigned!",
            assignment.vessel_index.get()
        );

        let old_berth_time = state.berth_free_time(assignment.berth_index);
        let old_objective = state.current_objective();

        self.entries.push(TrailEntry::new(
            old_berth_time,
            old_objective,
            assignment.berth_index,
            assignment.vessel_index,
        ));

        state.set_berth_free_time(assignment.berth_index, assignment.new_berth_time);
        state.assign_vessel(assignment.vessel_index);
        state.set_current_objective(assignment.new_objective);
    }

    /// # Safety
    ///
    /// # Panics
    ///
    /// In debug mode, panics if the vessel is already assigned.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `assignment.vessel_index` is in 0..num_vessels()
    /// and `assignment.berth_index` is in 0..num_berths().
    pub unsafe fn apply_assignment_unchecked(
        &mut self,
        state: &mut SearchState<T>,
        assignment: Assignment<T>,
    ) where
        T: Copy + Zero,
    {
        debug_assert!(
            !state.is_vessel_assigned(assignment.vessel_index),
            "Attempted to assign vessel {} which is already assigned!",
            assignment.vessel_index.get()
        );

        let old_berth_time = unsafe { state.berth_free_time_unchecked(assignment.berth_index) };
        let old_objective = state.current_objective();

        self.entries.push(TrailEntry::new(
            old_berth_time,
            old_objective,
            assignment.berth_index,
            assignment.vessel_index,
        ));

        unsafe {
            state.set_berth_free_time_unchecked(assignment.berth_index, assignment.new_berth_time);
            state.assign_vessel_unchecked(assignment.vessel_index);
            state.set_current_objective(assignment.new_objective);
        }
    }

    /// Backtracks to the previous checkpoint, undoing all assignments made since then.
    /// If there are no checkpoints, this is a no-op.
    pub fn backtrack(&mut self, state: &mut SearchState<T>)
    where
        T: Copy + Zero,
    {
        let start_index = match self.frame.pop() {
            Some(index) => index,
            None => return,
        };

        while self.entries.len() > start_index {
            let entry = unsafe { self.entries.pop().unwrap_unchecked() };
            self.undo_entry(state, entry);
        }
    }

    /// Clears the entire trail, undoing all assignments made.
    pub fn clear(&mut self, state: &mut SearchState<T>)
    where
        T: Copy + Zero,
    {
        for entry in self.entries.iter().rev() {
            self.undo_entry(state, *entry);
        }

        self.entries.clear();
        self.frame.clear();
    }

    /// Undoes a single trail entry, restoring the previous state.
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

    /// Returns the current depth of the search trail.
    #[inline]
    pub fn depth(&self) -> usize {
        self.frame.len()
    }

    /// Returns the total allocated memory in bytes used by the search trail.
    #[inline]
    pub fn allocated_memory_bytes(&self) -> usize {
        let entries_size = self.entries.capacity() * std::mem::size_of::<TrailEntry<T>>();
        let frame_size = self.frame.capacity() * std::mem::size_of::<usize>();
        entries_size + frame_size
    }
}

impl<T> std::fmt::Display for SearchTrail<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SearchTrail(num_entries: {}, num_frames: {})",
            self.entries.len(),
            self.frame.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Zero;

    type IntegerType = i32;

    fn make_state(num_berths: usize, num_vessels: usize) -> SearchState<IntegerType> {
        SearchState::<IntegerType>::new(num_berths, num_vessels)
    }

    #[test]
    fn test_new_and_capacity() {
        let trail = SearchTrail::<IntegerType>::new();
        assert_eq!(trail.num_entries(), 0);
        assert_eq!(trail.num_frames(), 0);
        assert_eq!(trail.depth(), 0);

        let trail2 = SearchTrail::<IntegerType>::with_capacity(10, 5);
        assert_eq!(trail2.num_entries(), 0);
        assert_eq!(trail2.num_frames(), 0);

        let trail3 = SearchTrail::<IntegerType>::preallocated(7);
        assert_eq!(trail3.num_entries(), 0);
        assert_eq!(trail3.num_frames(), 0);

        // Display sanity
        let formatted = format!("{}", trail);
        assert!(formatted.contains("SearchTrail(num_entries: 0, num_frames: 0)"));
    }

    #[test]
    fn test_assignment_display_and_entry_accessors() {
        let assign = Assignment::new(
            VesselIndex::new(2),
            BerthIndex::new(1),
            10, // new berth time
            5,  // new objective
        );
        let formatted = format!("{}", assign);
        assert!(formatted.contains("Assignment(vessel_index: 2"));
        assert!(formatted.contains("berth_index: 1"));
        assert!(formatted.contains("new_berth_time: 10"));
        assert!(formatted.contains("new_objective: 5"));

        let entry = TrailEntry::new(3, 7, BerthIndex::new(4), VesselIndex::new(9));
        assert_eq!(entry.old_berth_time(), 3);
        assert_eq!(entry.old_objective(), 7);
        assert_eq!(entry.berth_index().get(), 4);
        assert_eq!(entry.vessel_index().get(), 9);

        let entry_fmt = format!("{}", entry);
        assert!(entry_fmt.contains(
            "TrailEntry(berth_index: 4, vessel_index: 9, old_berth_time: 3, old_objective: 7)"
        ));
    }

    #[test]
    fn test_apply_assignment_and_backtrack_single() {
        // Setup: 2 berths, 3 vessels. All zeros initially.
        let mut state = make_state(2, 3);
        let mut trail = SearchTrail::<IntegerType>::new();

        let v = VesselIndex::new(1);
        let b = BerthIndex::new(0);

        // Checkpoint
        trail.push_checkpoint();
        assert_eq!(trail.depth(), 1);

        // Apply assignment: set berth 0 free time to 10, objective to 5, assign vessel 1
        let assignment = Assignment::new(v, b, 10, 5);
        trail.apply_assignment(&mut state, assignment);

        // Assert state changed
        assert_eq!(state.berth_free_time(b), 10);
        assert!(state.is_vessel_assigned(v));
        assert_eq!(state.current_objective(), 5);

        // Backtrack: should undo the single entry
        trail.backtrack(&mut state);

        // After backtrack, depth back to 0, state restored
        assert_eq!(trail.depth(), 0);
        assert_eq!(state.berth_free_time(b), IntegerType::zero());
        assert!(!state.is_vessel_assigned(v));
        assert_eq!(state.current_objective(), IntegerType::zero());

        // Trail entries should be empty
        assert_eq!(trail.num_entries(), 0);
    }

    #[test]
    fn test_apply_assignment_unchecked_and_backtrack_multiple_frames() {
        let mut state = make_state(3, 4);
        let mut trail = SearchTrail::<IntegerType>::new();

        // Frame 0
        trail.push_checkpoint();
        let a0 = Assignment::new(VesselIndex::new(0), BerthIndex::new(2), 7, 3);
        unsafe {
            trail.apply_assignment_unchecked(&mut state, a0);
        }
        assert_eq!(state.berth_free_time(BerthIndex::new(2)), 7);
        assert!(state.is_vessel_assigned(VesselIndex::new(0)));
        assert_eq!(state.current_objective(), 3);

        // Frame 1 (nested)
        trail.push_checkpoint();
        let a1 = Assignment::new(VesselIndex::new(2), BerthIndex::new(1), 12, 9);
        unsafe {
            trail.apply_assignment_unchecked(&mut state, a1);
        }
        assert_eq!(state.berth_free_time(BerthIndex::new(1)), 12);
        assert!(state.is_vessel_assigned(VesselIndex::new(2)));
        assert_eq!(state.current_objective(), 9);

        // Verify trail depth and counts
        assert_eq!(trail.depth(), 2);
        assert_eq!(trail.num_frames(), 2);
        assert_eq!(trail.num_entries(), 2);

        // Backtrack only the inner frame: should undo a1
        trail.backtrack(&mut state);
        assert_eq!(trail.depth(), 1);
        assert_eq!(trail.num_frames(), 1);
        assert_eq!(trail.num_entries(), 1);

        // State after undoing a1: berth 1 back to 0, vessel 2 unassigned, objective restored to 3
        assert_eq!(
            state.berth_free_time(BerthIndex::new(1)),
            IntegerType::zero()
        );
        assert!(!state.is_vessel_assigned(VesselIndex::new(2)));
        assert_eq!(state.current_objective(), 3);

        // Outer frame still applied (a0 still in effect)
        assert_eq!(state.berth_free_time(BerthIndex::new(2)), 7);
        assert!(state.is_vessel_assigned(VesselIndex::new(0)));

        // Backtrack outer frame: undo a0
        trail.backtrack(&mut state);
        assert_eq!(trail.depth(), 0);
        assert_eq!(trail.num_frames(), 0);
        assert_eq!(trail.num_entries(), 0);

        // State fully restored
        assert_eq!(
            state.berth_free_time(BerthIndex::new(2)),
            IntegerType::zero()
        );
        assert!(!state.is_vessel_assigned(VesselIndex::new(0)));
        assert_eq!(state.current_objective(), IntegerType::zero());
    }

    #[test]
    fn test_clear_resets_frames_and_entries_and_undoes_state() {
        let mut state = make_state(2, 2);
        let mut trail = SearchTrail::<IntegerType>::new();

        // Frame 0
        trail.push_checkpoint();
        trail.apply_assignment(
            &mut state,
            Assignment::new(VesselIndex::new(0), BerthIndex::new(0), 5, 1),
        );

        // Frame 1
        trail.push_checkpoint();
        trail.apply_assignment(
            &mut state,
            Assignment::new(VesselIndex::new(1), BerthIndex::new(1), 8, 3),
        );

        assert_eq!(trail.depth(), 2);
        assert_eq!(trail.num_entries(), 2);
        assert_eq!(state.current_objective(), 3);
        assert!(state.is_vessel_assigned(VesselIndex::new(0)));
        assert!(state.is_vessel_assigned(VesselIndex::new(1)));
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), 5);
        assert_eq!(state.berth_free_time(BerthIndex::new(1)), 8);

        // Clear should unwind all frames and restore state, plus empty entries
        trail.clear(&mut state);

        assert_eq!(trail.depth(), 0);
        assert_eq!(trail.num_frames(), 0);
        assert_eq!(trail.num_entries(), 0);

        assert_eq!(state.current_objective(), IntegerType::zero());
        assert!(!state.is_vessel_assigned(VesselIndex::new(0)));
        assert!(!state.is_vessel_assigned(VesselIndex::new(1)));
        assert_eq!(
            state.berth_free_time(BerthIndex::new(0)),
            IntegerType::zero()
        );
        assert_eq!(
            state.berth_free_time(BerthIndex::new(1)),
            IntegerType::zero()
        );
    }

    #[test]
    fn test_depth_and_frame_tracking_with_empty_backtrack() {
        let mut state = make_state(1, 1);
        let mut trail = SearchTrail::<IntegerType>::new();

        // Backtrack on empty frame stack should be a no-op
        trail.backtrack(&mut state);
        assert_eq!(trail.depth(), 0);
        assert_eq!(trail.num_frames(), 0);
        assert_eq!(trail.num_entries(), 0);

        // Push 3 frames, but assign only in first and third
        trail.push_checkpoint(); // depth 1
        trail.apply_assignment(
            &mut state,
            Assignment::new(VesselIndex::new(0), BerthIndex::new(0), 4, 2),
        );
        assert_eq!(trail.depth(), 1);
        assert_eq!(trail.num_entries(), 1);

        trail.push_checkpoint(); // depth 2 (no assignment)
        assert_eq!(trail.num_entries(), 1);

        trail.push_checkpoint(); // depth 3
        // No assignment here either
        assert_eq!(trail.num_entries(), 1);

        // Backtrack third frame (empty)
        trail.backtrack(&mut state);
        assert_eq!(trail.depth(), 2);
        assert_eq!(trail.num_entries(), 1);
        // State unchanged because third frame was empty
        assert!(state.is_vessel_assigned(VesselIndex::new(0)));
        assert_eq!(state.berth_free_time(BerthIndex::new(0)), 4);
        assert_eq!(state.current_objective(), 2);

        // Backtrack second frame (empty)
        trail.backtrack(&mut state);
        assert_eq!(trail.depth(), 1);
        assert_eq!(trail.num_entries(), 1);

        // Backtrack first frame (contains one entry)
        trail.backtrack(&mut state);
        assert_eq!(trail.depth(), 0);
        assert_eq!(trail.num_entries(), 0);

        // State restored
        assert!(!state.is_vessel_assigned(VesselIndex::new(0)));
        assert_eq!(
            state.berth_free_time(BerthIndex::new(0)),
            IntegerType::zero()
        );
        assert_eq!(state.current_objective(), IntegerType::zero());
    }

    #[test]
    fn test_allocated_memory_bytes_with_capacity() {
        // Known capacities
        let entry_capacity = 10;
        let frame_capacity = 5;
        let trail = SearchTrail::<IntegerType>::with_capacity(entry_capacity, frame_capacity);

        // Compute expected size using the same formula
        let expected = trail.entries.capacity() * std::mem::size_of::<TrailEntry<IntegerType>>()
            + trail.frame.capacity() * std::mem::size_of::<usize>();

        // Sanity: capacities should match the requested ones initially
        assert_eq!(trail.entries.capacity(), entry_capacity);
        assert_eq!(trail.frame.capacity(), frame_capacity);

        assert_eq!(trail.allocated_memory_bytes(), expected);
    }

    #[test]
    fn test_allocated_memory_bytes_preallocated() {
        // Preallocated uses num_vessels for entries, num_vessels+1 for frame
        let num_vessels = 7;
        let trail = SearchTrail::<IntegerType>::preallocated(num_vessels);

        let expected = trail.entries.capacity() * std::mem::size_of::<TrailEntry<IntegerType>>()
            + trail.frame.capacity() * std::mem::size_of::<usize>();

        assert_eq!(trail.entries.capacity(), num_vessels);
        assert_eq!(trail.frame.capacity(), num_vessels + 1);

        assert_eq!(trail.allocated_memory_bytes(), expected);
    }
}
