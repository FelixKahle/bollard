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

use crate::state::SearchState;
use bollard_model::index::{BerthIndex, VesselIndex};
use num_traits::{PrimInt, Zero};

/// A compact record of a single assignment mutation applied to the search state.
///
/// `TrailEntry` captures sufficient information to undo one assignment during backtracking:
/// - the berth that was affected,
/// - the vessel that was assigned,
/// - the previous berth free time (`old_berth_time`),
/// - and the previous objective value (`old_objective`).
///
/// This type is intended to be stored in a linear log (`SearchTrail::entries`) and
/// consumed in reverse when undoing decisions. It is cheap to copy and designed for
/// high-throughput search loops.
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
            "TrailEntry(berth: {}, vessel: {}, old_time: {}, old_obj: {})",
            self.berth_index.get(),
            self.vessel_index.get(),
            self.old_berth_time,
            self.old_objective
        )
    }
}

/// A frame marker describing the boundaries of a decision level on the trail.
///
/// `FrameEntry` stores:
/// - the previous `last_decision_time` and `last_decision_vessel` (to restore metadata on backtrack),
/// - the `entry_start_index` in the trail where this frame began.
///
/// Frames are pushed before exploring a new decision level and popped on backtrack.
/// All `TrailEntry`s added after `entry_start_index` belong to this frame and will be undone
/// during backtracking.
#[derive(Copy, Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FrameEntry<T> {
    previous_last_decision_time: T,
    previous_last_decision_vessel: VesselIndex,
    entry_start_index: usize,
}

impl<T> FrameEntry<T>
where
    T: Copy,
{
    /// Creates a new `FrameEntry`.
    #[inline(always)]
    pub fn new(
        previous_last_decision_time: T,
        previous_last_decision_vessel: VesselIndex,
        entry_start_index: usize,
    ) -> Self {
        Self {
            previous_last_decision_time,
            previous_last_decision_vessel,
            entry_start_index,
        }
    }

    /// Returns the previous last decision time before this frame.
    #[inline]
    pub fn previous_last_decision_time(&self) -> T {
        self.previous_last_decision_time
    }

    /// Returns the previous last decision vessel before this frame.
    #[inline]
    pub fn previous_last_decision_vessel(&self) -> VesselIndex {
        self.previous_last_decision_vessel
    }

    /// Returns the entry start index for this frame.
    #[inline]
    pub fn entry_start_index(&self) -> usize {
        self.entry_start_index
    }
}

impl<T> std::fmt::Display for FrameEntry<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FrameEntry(prev_last_decision_time: {}, prev_last_decision_vessel: {}, entry_start_index: {})",
            self.previous_last_decision_time,
            self.previous_last_decision_vessel.get(),
            self.entry_start_index
        )
    }
}

/// A linear undo log with frame markers for efficient backtracking.
///
/// `SearchTrail` records all assignments applied to `SearchState` along with frame boundaries,
/// enabling O(k) rollback of k mutations when backtracking a frame. Typical usage:
/// 1. Call `push_frame(state)` before expanding a node/decision level,
/// 2. For each applied decision, call `apply_assignment(...)`,
/// 3. On prune or completion, call `backtrack(state)` to restore the state to the start of the frame.
///
/// Performance notes:
/// - `entries` grows linearly with the number of applied decisions; backtracking iterates entries in reverse.
/// - `frames` stores start indices and previous decision metadata to restore state efficiently.
/// - Prefer `preallocated` to reduce reallocations in large searches.
#[derive(Debug, Clone)]
pub struct SearchTrail<T> {
    /// The linear history of all changes made to the state.
    entries: Vec<TrailEntry<T>>,
    /// A stack of indices pointing to `entries`.
    /// `frames[i]` stores the index in `entries` where depth `i` began.
    frames: Vec<FrameEntry<T>>,
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

    /// Creates a new `SearchTrail` preallocating space based on the number of vessels.
    ///
    /// This helps avoid frequent reallocations during search by reserving:
    /// - `num_vessels` capacity for entries,
    /// - `num_vessels + 1` capacity for frames.
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
    ///
    /// If the internal `entries`/`frames` capacities are smaller than the recommended
    /// `num_vessels`/`num_vessels + 1`, they will be increased to reduce allocations
    /// during deep search and backtracking.
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
    pub fn push_frame(&mut self, state: &SearchState<T>)
    where
        T: PrimInt,
    {
        self.frames.push(FrameEntry::new(
            state.last_decision_time(),
            state.last_decision_vessel(),
            self.entries.len(),
        ));
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
            vessel_index.get() < state.num_vessels(),
            "called `SearchTrail::apply_assignment` with vessel index out of bounds: the len is {} but the index is {}",
            vessel_index.get(),
            state.num_vessels()
        );
        debug_assert!(
            berth_index.get() < state.num_berths(),
            "called `SearchTrail::apply_assignment` with berth index out of bounds: the len is {} but the index is {}",
            berth_index.get(),
            state.num_berths()
        );
        debug_assert!(
            !state.is_vessel_assigned(vessel_index),
            "called SearchTrail::apply_assignment with vessel {} which is already assigned!",
            vessel_index.get()
        );

        let old_berth_time = state.berth_free_time(berth_index);
        let old_objective = state.current_objective();

        self.entries.push(TrailEntry {
            old_berth_time,
            old_objective,
            berth_index,
            vessel_index,
        });

        state.set_berth_free_time(berth_index, new_berth_time);
        state.assign_vessel(vessel_index, berth_index, actual_start_time);
        state.set_current_objective(new_objective);
        state.set_last_decision(actual_start_time, vessel_index);
    }

    /// Applies an assignment without bounds checking in the state.
    ///
    /// # Panics
    ///
    /// Panics if `berth_index` is out of `0..num_berths` or
    /// `vessel_index` is out of `0..num_vessels`,
    /// or the vessel is already assigned.
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
            vessel_index.get() < state.num_vessels(),
            "called `SearchTrail::apply_assignment_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            vessel_index.get(),
            state.num_vessels()
        );
        debug_assert!(
            berth_index.get() < state.num_berths(),
            "called `SearchTrail::apply_assignment_unchecked` with berth index out of bounds: the len is {} but the index is {}",
            berth_index.get(),
            state.num_berths()
        );
        debug_assert!(
            !state.is_vessel_assigned(vessel_index),
            "called SearchTrail::apply_assignment_unchecked with vessel {} which is already assigned!",
            vessel_index.get()
        );

        let old_berth_time = unsafe { state.berth_free_time_unchecked(berth_index) };
        let old_objective = state.current_objective();

        self.entries.push(TrailEntry {
            old_berth_time,
            old_objective,
            berth_index,
            vessel_index,
        });

        unsafe {
            state.set_berth_free_time_unchecked(berth_index, new_berth_time);
            state.assign_vessel_unchecked(vessel_index, berth_index, actual_start_time);
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
        let frame = match self.frames.pop() {
            Some(f) => f,
            None => return,
        };

        state.set_last_decision(
            frame.previous_last_decision_time,
            frame.previous_last_decision_vessel,
        );

        while self.entries.len() > frame.entry_start_index {
            debug_assert!(
                !self.entries.is_empty(),
                "called `SearchTrail::backtrack` on an empty trail"
            );

            let entry = unsafe { self.entries.pop().unwrap_unchecked() };

            unsafe {
                state.set_berth_free_time_unchecked(entry.berth_index, entry.old_berth_time);
                state.unassign_vessel_unchecked(entry.vessel_index);
                state.set_current_objective(entry.old_objective);
            }
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
        let frames_size = self.frames.capacity() * std::mem::size_of::<FrameEntry<T>>();
        entries_size + frames_size
    }

    /// Helper to undo a single entry.
    #[inline]
    fn undo_entry(&self, state: &mut SearchState<T>, entry: TrailEntry<T>)
    where
        T: PrimInt + Zero,
    {
        debug_assert!(
            entry.vessel_index.get() < state.num_vessels(),
            "called `SearchTrail::undo_entry` with vessel index out of bounds: the len is {} but the index is {}",
            entry.vessel_index.get(),
            state.num_vessels()
        );
        debug_assert!(
            entry.berth_index.get() < state.num_berths(),
            "called `SearchTrail::undo_entry` with berth index out of bounds: the len is {} but the index is {}",
            entry.berth_index.get(),
            state.num_berths()
        );

        unsafe {
            state.set_berth_free_time_unchecked(entry.berth_index, entry.old_berth_time);
            state.unassign_vessel_unchecked(entry.vessel_index);
            state.set_current_objective(entry.old_objective);
        }
    }

    /// Returns an iterator over all trail entries.
    #[inline]
    pub fn iter_entries(&self) -> std::slice::Iter<'_, TrailEntry<T>> {
        self.entries.iter()
    }

    /// Returns an iterator over all frame entries.
    #[inline]
    pub fn iter_frames(&self) -> std::slice::Iter<'_, FrameEntry<T>> {
        self.frames.iter()
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
