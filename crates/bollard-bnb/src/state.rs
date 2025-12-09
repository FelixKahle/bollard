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

//! Search state management for the constraint solver.
//!
//! This module provides `SearchState`, a compact, mutable container for tracking the
//! incremental assignment of vessels to berths over time during search and optimization.
//!
//! Key responsibilities:
//! - Maintain assignment status for each vessel, including berth and start time.
//! - Track berth availability (`berth_free_times`) at any point in the search.
//! - Maintain counters and invariants like `num_assigned_vessels <= num_vessels`.
//! - Track the current objective value and the last decision (time and vessel).
//!
//! Performance considerations:
//! - Provides both checked and unchecked (unsafe) accessors and mutators. Unchecked variants
//!   avoid bounds checks for hot loops under the assumption that the caller ensures validity.
//! - Uses `FixedBitSet` to track assignments efficiently.
//!
//! Safety and invariants:
//! - All methods with `unsafe` in their name require the caller to ensure the provided indices
//!   are within bounds and the logical preconditions (e.g., assignment status) are satisfied.
//! - Debug assertions are used extensively to catch invariant violations in debug builds.
//!   In release builds, callers must uphold invariants to avoid UB when using unchecked methods.

use bollard_model::{
    index::{BerthIndex, VesselIndex},
    solution::Solution,
};
use fixedbitset::FixedBitSet;
use num_traits::{Bounded, PrimInt, Signed, Zero};

/// A compact, mutable container holding the incremental search state for the
/// berth-vessel constraint solver.
///
/// The state tracks:
/// - `current_objective`: the current objective value of the partial solution.
/// - `last_decision_time` and `last_decision_vessel`: metadata on the most recent search decision.
/// - `num_assigned_vessels` and `num_vessels`: assignment progress tracking.
/// - `berth_free_times`: when each berth becomes available next.
/// - `vessel_assignments`: bitset indicating whether a vessel is currently assigned.
/// - `vessel_start_times`: the scheduled start time per vessel.
/// - `vessel_berths`: the assigned berth per vessel.
///
/// Invariants (debug-checked):
/// - `num_assigned_vessels <= num_vessels`
/// - For any vessel `i`: if `vessel_assignments[i]` then `vessel_start_times[i]` and
///   `vessel_berths[i]` contain valid data.
///
/// Safety contract for `*_unchecked` methods:
/// - Caller must ensure indices are within bounds and logical preconditions are met.
/// - Violations may cause undefined behavior in release builds.
#[derive(Debug, Clone)]
pub struct SearchState<T> {
    // We optimize layout for T=i64 here,
    // grouping fields by alignment to minimize padding.

    // Heap-managed, pointer-sized fields (8-aligned on 64-bit)
    berth_free_times: Vec<T>,
    vessel_start_times: Vec<T>,
    vessel_berths: Vec<BerthIndex>,
    vessel_assignments: FixedBitSet,

    // 8-aligned scalars grouped together
    current_objective: T,
    last_decision_time: T,

    // Counters and indices (usize) grouped at the end
    last_decision_vessel: VesselIndex,
    num_vessels: usize,
    num_assigned_vessels: usize,
}

impl<T> SearchState<T> {
    /// Creates a new `SearchState` with the specified number of berths and vessels.
    /// The initial state has all berths free at time zero, no vessels assigned,
    /// and the objective value set to zero.
    #[inline]
    pub fn new(num_berths: usize, num_vessels: usize) -> Self
    where
        T: Copy + Zero + Bounded,
    {
        Self {
            berth_free_times: vec![T::zero(); num_berths],
            vessel_assignments: FixedBitSet::with_capacity(num_vessels),
            vessel_start_times: vec![T::zero(); num_vessels],
            vessel_berths: vec![BerthIndex::new(0); num_vessels],
            num_assigned_vessels: 0,
            num_vessels,
            current_objective: T::zero(),
            last_decision_time: T::min_value(),
            last_decision_vessel: VesselIndex::new(0),
        }
    }

    /// Returns the number of berths in this state.
    #[inline]
    pub fn num_berths(&self) -> usize {
        self.berth_free_times.len()
    }

    /// Returns the number of vessels in this state.
    #[inline]
    pub fn num_vessels(&self) -> usize {
        self.num_vessels
    }

    /// Returns the number of assigned vessels in this state.
    #[inline]
    pub fn num_assigned_vessels(&self) -> usize {
        self.num_assigned_vessels
    }

    /// Returns the current objective value of this state.
    #[inline]
    pub fn current_objective(&self) -> T
    where
        T: Copy,
    {
        self.current_objective
    }

    /// Sets the current objective value of this state.
    #[inline]
    pub fn set_current_objective(&mut self, objective: T) {
        self.current_objective = objective;
    }

    /// Returns the time of the last decision made in this state.
    #[inline]
    pub fn last_decision_time(&self) -> T
    where
        T: Copy,
    {
        self.last_decision_time
    }

    /// Returns the vessel index of the last decision made in this state.
    #[inline]
    pub fn last_decision_vessel(&self) -> VesselIndex {
        self.last_decision_vessel
    }

    /// Sets the last decision made in this state.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is out of bounds `0..num_vessels`.
    #[inline]
    pub fn set_last_decision(&mut self, time: T, vessel_index: VesselIndex) {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels,
            "called `SearchState::set_last_decision` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels
        );

        self.last_decision_time = time;
        self.last_decision_vessel = vessel_index;
    }

    /// Sets the last decision made in this state without bounds checking.
    ///
    /// # Panics
    ///
    /// In debug mode, panics if `vessel_index` is out of bounds `0..num_vessels`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is within bounds `0..num_vessels`.
    #[inline]
    pub unsafe fn set_last_decision_unchecked(&mut self, time: T, vessel_index: VesselIndex) {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels,
            "called `SearchState::set_last_decision_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels
        );

        self.last_decision_time = time;
        self.last_decision_vessel = vessel_index;
    }

    /// Checks if the specified vessel is assigned in this state.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is out of bounds `0..num_vessels`.
    #[inline]
    pub fn is_vessel_assigned(&self, vessel_index: VesselIndex) -> bool {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels,
            "called `SearchState::is_vessel_assigned` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels
        );

        self.vessel_assignments.contains(index)
    }

    /// Checks if the specified vessel is assigned in this state without bounds checking.
    ///
    /// # Panics
    ///
    /// In debug mode, panics if `vessel_index` is out of bounds `0..num_vessels`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is within bounds `0..num_vessels`.
    #[inline]
    pub unsafe fn is_vessel_assigned_unchecked(&self, vessel_index: VesselIndex) -> bool {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels,
            "called `SearchState::is_vessel_assigned_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels
        );

        unsafe { self.vessel_assignments.contains_unchecked(index) }
    }

    /// Assigns a vessel to a berth starting at a specific time.
    ///
    /// # Panics
    ///
    /// In debug mode, panics if `vessel_index` is out of bounds `0..num_vessels`,
    /// if the vessel is already assigned, or if the number of assigned vessels
    /// has reached the total number of vessels.
    #[inline]
    pub fn assign_vessel(
        &mut self,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
    ) {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels,
            "called `SearchState::assign_vessel` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels
        );
        debug_assert!(
            !self.vessel_assignments.contains(index),
            "called `SearchState::assign_vessel` with vessel {} already assigned",
            index
        );
        debug_assert!(
            self.num_assigned_vessels < self.num_vessels,
            "called `SearchState::assign_vessel` but the internal assignment count is already at the limit {}",
            self.num_vessels
        );

        self.vessel_assignments.insert(index);
        self.vessel_start_times[index] = start_time;
        self.vessel_berths[index] = berth_index;

        self.num_assigned_vessels += 1;

        debug_assert!(self.num_assigned_vessels <= self.num_vessels);
    }

    /// Assigns a vessel to a berth starting at a specific time without bounds checking.
    ///
    /// # Panics
    ///
    /// In debug mode, panics if `vessel_index` is out of bounds `0..num_vessels`,
    /// if the vessel is already assigned, or if the number of assigned vessels
    /// has reached the total number of vessels.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is within bounds `0..num_vessels`.
    #[inline]
    pub unsafe fn assign_vessel_unchecked(
        &mut self,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
    ) {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels,
            "called `SearchState::assign_vessel_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels
        );
        debug_assert!(
            !self.vessel_assignments.contains(index),
            "called `SearchState::assign_vessel_unchecked` with vessel {} already assigned",
            index
        );
        debug_assert!(
            self.num_assigned_vessels < self.num_vessels,
            "called `SearchState::assign_vessel_unchecked` but the internal assignment count is already at the limit {}",
            self.num_vessels
        );

        unsafe {
            self.vessel_assignments.insert_unchecked(index);
            *self.vessel_start_times.get_unchecked_mut(index) = start_time;
            *self.vessel_berths.get_unchecked_mut(index) = berth_index;
        }

        self.num_assigned_vessels += 1;

        debug_assert!(self.num_assigned_vessels <= self.num_vessels);
    }

    /// Unassigns a vessel from its berth.
    ///
    /// # Panics
    ///
    /// In debug mode, panics if `vessel_index` is out of bounds `0..num_vessels`,
    /// if the vessel is already unassigned, or if the number of assigned vessels
    /// is already zero.
    #[inline]
    pub fn unassign_vessel(&mut self, vessel_index: VesselIndex) {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels,
            "called `SearchState::unassign_vessel` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels
        );
        debug_assert!(
            self.vessel_assignments.contains(index),
            "called `SearchState::unassign_vessel` with vessel {} already unassigned",
            index
        );
        debug_assert!(
            self.num_assigned_vessels > 0,
            "called `SearchState::unassign_vessel` but the internal assignment count is already at 0",
        );

        self.vessel_assignments.set(index, false);
        self.num_assigned_vessels -= 1;

        debug_assert!(self.num_assigned_vessels <= self.num_vessels);
    }

    /// Unassigns a vessel from its berth without bounds checking.
    ///
    /// # Panics
    ///
    /// In debug mode, panics if `vessel_index` is out of bounds `0..num_vessels`,
    /// if the vessel is already unassigned, or if the number of assigned vessels
    /// is already zero.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is within bounds `0..num_vessels`.
    #[inline]
    pub unsafe fn unassign_vessel_unchecked(&mut self, vessel_index: VesselIndex) {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels,
            "called `SearchState::unassign_vessel_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels
        );
        debug_assert!(
            self.vessel_assignments.contains(index),
            "called `SearchState::unassign_vessel_unchecked` with vessel {} already unassigned",
            index
        );
        debug_assert!(
            self.num_assigned_vessels > 0,
            "called `SearchState::unassign_vessel_unchecked` but the internal assignment count is already at 0",
        );

        unsafe {
            self.vessel_assignments.set_unchecked(index, false);
        }

        self.num_assigned_vessels -= 1;

        debug_assert!(self.num_assigned_vessels <= self.num_vessels);
    }

    /// Returns the start time of the specified vessel.
    ///
    /// # Panics
    ///
    /// In debug mode, panics if `vessel_index` is out of bounds `0..num_vessels`,
    /// or if the vessel is unassigned.
    #[inline]
    pub fn vessel_start_time(&self, vessel_index: VesselIndex) -> T
    where
        T: Copy,
    {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels,
            "called `SearchState::vessel_start_time` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels
        );
        debug_assert!(
            self.vessel_assignments.contains(index),
            "called `SearchState::vessel_start_time` with vessel {} unassigned",
            index
        );

        self.vessel_start_times[index]
    }

    /// Returns the start time of the specified vessel without bounds checking.
    ///
    /// # Panics
    ///
    /// In debug mode, panics if `vessel_index` is out of bounds `0..num_vessels`,
    /// or if the vessel is unassigned.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is within bounds `0..num_vessels`.
    #[inline]
    pub unsafe fn vessel_start_time_unchecked(&self, vessel_index: VesselIndex) -> T
    where
        T: Copy,
    {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels,
            "called `SearchState::vessel_start_time_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels
        );
        debug_assert!(
            self.vessel_assignments.contains(index),
            "called `SearchState::vessel_start_time_unchecked` with vessel {} unassigned",
            index
        );

        unsafe { *self.vessel_start_times.get_unchecked(index) }
    }

    /// Returns the berth index of the specified vessel.
    ///
    /// # Panics
    ///
    /// In debug mode, panics if `vessel_index` is out of bounds `0..num_vessels`,
    /// or if the vessel is unassigned.
    #[inline]
    pub fn vessel_berth(&self, vessel_index: VesselIndex) -> BerthIndex {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels,
            "called `SearchState::set_last_decision` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels
        );
        debug_assert!(
            self.vessel_assignments.contains(index),
            "called `SearchState::vessel_berth` with vessel {} unassigned",
            index
        );

        self.vessel_berths[index]
    }

    /// Returns the berth index of the specified vessel without bounds checking.
    ///
    /// # Panics
    ///
    /// In debug mode, panics if `vessel_index` is out of bounds `0..num_vessels`,
    /// or if the vessel is unassigned.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is within bounds `0..num_vessels`.
    #[inline]
    pub unsafe fn vessel_berth_unchecked(&self, vessel_index: VesselIndex) -> BerthIndex {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels,
            "called `SearchState::set_last_decision` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels
        );
        debug_assert!(
            self.vessel_assignments.contains(index),
            "called `SearchState::vessel_berth_unchecked` with vessel {} unassigned",
            index
        );

        unsafe { *self.vessel_berths.get_unchecked(index) }
    }

    /// Returns a slice of the berth free times.
    #[inline]
    pub fn berth_free_times(&self) -> &[T] {
        &self.berth_free_times
    }

    /// Returns the free time of the specified berth.
    ///
    /// # Panics
    ///
    /// Panics if `berth_index` is out of bounds `0..num_berths`.
    #[inline]
    pub fn berth_free_time(&self, berth_index: BerthIndex) -> T
    where
        T: Copy,
    {
        let index = berth_index.get();
        debug_assert!(index < self.num_berths());

        self.berth_free_times[index]
    }

    /// Returns the free time of the specified berth without bounds checking.
    ///
    /// # Panics
    ///
    /// In debug mode, panics if `berth_index` is out of bounds `0..num_berths`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `berth_index` is within bounds `0..num_berths`.
    #[inline]
    pub unsafe fn berth_free_time_unchecked(&self, berth_index: BerthIndex) -> T
    where
        T: Copy,
    {
        let index = berth_index.get();
        debug_assert!(index < self.num_berths());
        unsafe { *self.berth_free_times.get_unchecked(index) }
    }

    /// Sets the free time of the specified berth.
    ///
    /// # Panics
    ///
    /// Panics if `berth_index` is out of bounds `0..num_berths`.
    #[inline]
    pub fn set_berth_free_time(&mut self, berth_index: BerthIndex, time: T) {
        let index = berth_index.get();
        debug_assert!(index < self.num_berths());

        self.berth_free_times[index] = time;
    }

    /// Sets the free time of the specified berth without bounds checking.
    ///
    /// # Panics
    ///
    /// In debug mode, panics if `berth_index` is out of bounds `0..num_berths`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `berth_index` is within bounds `0..num_berths`.
    #[inline]
    pub unsafe fn set_berth_free_time_unchecked(&mut self, berth_index: BerthIndex, time: T) {
        let index = berth_index.get();
        debug_assert!(index < self.num_berths());

        unsafe {
            *self.berth_free_times.get_unchecked_mut(index) = time;
        }
    }

    /// Resets the search state to its initial configuration.
    /// This method clears all vessel assignments, sets all berth free times to zero,
    /// and resets the objective value and decision tracking variables.
    #[inline]
    pub fn reset(&mut self)
    where
        T: Copy + Zero + Bounded,
    {
        self.berth_free_times.fill(T::zero());
        self.vessel_assignments.clear();
        self.num_assigned_vessels = 0;
        self.current_objective = T::zero();
        self.vessel_start_times.fill(T::zero());
        self.vessel_berths.fill(BerthIndex::new(0));
        self.last_decision_time = T::min_value();
        self.last_decision_vessel = VesselIndex::new(0);
    }
}

impl<T> std::fmt::Display for SearchState<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "State(objective: {}, assigned_vessels: {}/{})",
            self.current_objective, self.num_assigned_vessels, self.num_vessels
        )
    }
}

/// Error indicating that a solution is incomplete.
/// This error is returned when attempting to convert a `SearchState`
/// into a `Solution`, but not all vessels have been assigned.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IncompleteSolutionError {
    assigned_vessels: usize,
    total_vessels: usize,
}

impl std::fmt::Display for IncompleteSolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Incomplete solution: assigned {}/{} vessels",
            self.assigned_vessels, self.total_vessels
        )
    }
}

impl std::error::Error for IncompleteSolutionError {}

impl<T> TryInto<Solution<T>> for SearchState<T>
where
    T: PrimInt + Signed,
{
    type Error = IncompleteSolutionError;

    fn try_into(self) -> Result<Solution<T>, Self::Error> {
        if self.num_assigned_vessels() != self.num_vessels {
            return Err(IncompleteSolutionError {
                assigned_vessels: self.num_assigned_vessels,
                total_vessels: self.num_vessels,
            });
        }

        Ok(Solution::new(
            self.current_objective,
            self.vessel_berths.to_vec(),
            self.vessel_start_times.to_vec(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::state::IncompleteSolutionError;

    use super::SearchState;
    use bollard_model::index::{BerthIndex, VesselIndex};
    use num_traits::Zero;
    use std::convert::TryInto;

    // Helper constructors for indices
    fn v(i: usize) -> VesselIndex {
        VesselIndex::new(i)
    }
    fn b(i: usize) -> BerthIndex {
        BerthIndex::new(i)
    }

    #[test]
    fn test_new_initial_state_and_sizes() {
        let s: SearchState<i64> = SearchState::new(3, 5);

        // Sizes
        assert_eq!(s.num_berths(), 3);
        assert_eq!(s.num_vessels(), 5);
        assert_eq!(s.num_assigned_vessels(), 0);

        // Defaults for objective and decision tracking
        assert_eq!(s.current_objective(), i64::zero());
        assert_eq!(s.last_decision_time(), i64::min_value());
        assert_eq!(s.last_decision_vessel().get(), 0);

        // Collections initialized
        assert_eq!(s.berth_free_times().len(), 3);
        assert!(s.berth_free_times().iter().all(|&t| t == 0));

        assert_eq!(s.vessel_start_times.len(), 5);
        assert!(s.vessel_start_times.iter().all(|&t| t == 0));

        assert_eq!(s.vessel_berths.len(), 5);
        assert!(s.vessel_berths.iter().all(|&bi| bi.get() == 0));

        // No assignments
        for i in 0..5 {
            assert!(!s.is_vessel_assigned(v(i)));
            unsafe { assert!(!s.is_vessel_assigned_unchecked(v(i))) }
        }
    }

    #[test]
    fn test_new_with_zero_berths_and_zero_vessels() {
        let s: SearchState<i64> = SearchState::new(0, 0);

        assert_eq!(s.num_berths(), 0);
        assert_eq!(s.num_vessels(), 0);
        assert_eq!(s.num_assigned_vessels(), 0);

        assert!(s.berth_free_times().is_empty());
        assert!(s.vessel_start_times.is_empty());
        assert!(s.vessel_berths.is_empty());

        assert_eq!(s.current_objective(), i64::zero());
        assert_eq!(s.last_decision_time(), i64::min_value());
        assert_eq!(s.last_decision_vessel().get(), 0);

        let display = format!("{}", s);
        assert!(display.contains("assigned_vessels: 0/0"));
    }

    #[test]
    fn test_objective_set_and_get() {
        let mut s: SearchState<i64> = SearchState::new(1, 1);
        assert_eq!(s.current_objective(), i64::zero());
        s.set_current_objective(42);
        assert_eq!(s.current_objective(), 42);
        s.set_current_objective(i64::zero());
        assert_eq!(s.current_objective(), i64::zero());
    }

    #[test]
    fn test_last_decision_checked_and_unchecked() {
        let mut s: SearchState<i64> = SearchState::new(2, 3);

        // Checked setter/getters
        s.set_last_decision(100, v(2));
        assert_eq!(s.last_decision_time(), 100);
        assert_eq!(s.last_decision_vessel().get(), 2);

        // Unsafe unchecked (if present per outline)
        unsafe {
            // If there is a set_last_decision_unchecked, call it (outline lists it)
            // Note: signature is not shown in detail, but we assume same as checked version
            s.set_last_decision_unchecked(200, v(1));
        }
        assert_eq!(s.last_decision_time(), 200);
        assert_eq!(s.last_decision_vessel().get(), 1);

        // Boundary times
        s.set_last_decision(i64::min_value(), v(0));
        assert_eq!(s.last_decision_time(), i64::min_value());
        s.set_last_decision(i64::max_value(), v(0));
        assert_eq!(s.last_decision_time(), i64::max_value());
    }

    #[test]
    fn test_berth_free_time_checked_and_unchecked() {
        let mut s: SearchState<i64> = SearchState::new(3, 2);

        // Checked set/get
        s.set_berth_free_time(b(0), 5);
        s.set_berth_free_time(b(1), 10);
        s.set_berth_free_time(b(2), 15);
        assert_eq!(s.berth_free_time(b(0)), 5);
        assert_eq!(s.berth_free_time(b(1)), 10);
        assert_eq!(s.berth_free_time(b(2)), 15);

        // Unchecked get/set
        unsafe {
            assert_eq!(s.berth_free_time_unchecked(b(0)), 5);
            assert_eq!(s.berth_free_time_unchecked(b(1)), 10);
            assert_eq!(s.berth_free_time_unchecked(b(2)), 15);

            s.set_berth_free_time_unchecked(b(0), 7);
            s.set_berth_free_time_unchecked(b(1), 12);
            s.set_berth_free_time_unchecked(b(2), 17);
        }
        assert_eq!(s.berth_free_time(b(0)), 7);
        assert_eq!(s.berth_free_time(b(1)), 12);
        assert_eq!(s.berth_free_time(b(2)), 17);

        // Bulk accessor matches individual
        let bulk = s.berth_free_times().to_vec();
        for i in 0..s.num_berths() {
            assert_eq!(bulk[i], s.berth_free_time(b(i)));
        }
    }

    #[test]
    fn test_vessel_assignment_lifecycle_checked() {
        let mut s: SearchState<i64> = SearchState::new(2, 3);

        // Initial
        assert_eq!(s.num_assigned_vessels(), 0);
        for i in 0..3 {
            assert!(!s.is_vessel_assigned(v(i)));
        }

        // Assign vessel 0
        s.assign_vessel(v(0), b(1), 100);
        assert!(s.is_vessel_assigned(v(0)));
        assert_eq!(s.vessel_start_time(v(0)), 100);
        assert_eq!(s.vessel_berth(v(0)).get(), 1);
        assert_eq!(s.num_assigned_vessels(), 1);

        // Assign vessel 2
        s.assign_vessel(v(2), b(0), 200);
        assert!(s.is_vessel_assigned(v(2)));
        assert_eq!(s.vessel_start_time(v(2)), 200);
        assert_eq!(s.vessel_berth(v(2)).get(), 0);
        assert_eq!(s.num_assigned_vessels(), 2);

        // Unassign vessel 0
        s.unassign_vessel(v(0));
        assert!(!s.is_vessel_assigned(v(0)));
        assert_eq!(s.num_assigned_vessels(), 1);

        // Reassign with new data
        s.assign_vessel(v(0), b(0), 300);
        assert!(s.is_vessel_assigned(v(0)));
        assert_eq!(s.vessel_start_time(v(0)), 300);
        assert_eq!(s.vessel_berth(v(0)).get(), 0);
        assert_eq!(s.num_assigned_vessels(), 2);
    }

    #[test]
    fn test_vessel_assignment_lifecycle_unchecked() {
        let mut s: SearchState<i64> = SearchState::new(3, 4);

        unsafe {
            s.assign_vessel_unchecked(v(1), b(2), 50);
            s.assign_vessel_unchecked(v(3), b(1), 75);
        }
        assert_eq!(s.num_assigned_vessels(), 2);

        unsafe {
            assert!(s.is_vessel_assigned_unchecked(v(1)));
            assert!(s.is_vessel_assigned_unchecked(v(3)));
        }

        assert_eq!(s.vessel_start_time(v(1)), 50);
        assert_eq!(s.vessel_berth(v(1)).get(), 2);
        assert_eq!(s.vessel_start_time(v(3)), 75);
        assert_eq!(s.vessel_berth(v(3)).get(), 1);

        unsafe {
            s.unassign_vessel_unchecked(v(1));
        }
        assert_eq!(s.num_assigned_vessels(), 1);
        unsafe {
            assert!(!s.is_vessel_assigned_unchecked(v(1)));
            assert!(s.is_vessel_assigned_unchecked(v(3)));
        }

        s.assign_vessel(v(0), b(0), 10);
        assert_eq!(s.num_assigned_vessels(), 2);
    }

    #[test]
    fn test_vessel_accessors_checked_and_unchecked() {
        let mut s: SearchState<i64> = SearchState::new(1, 2);

        s.assign_vessel(v(0), b(0), 123);
        s.assign_vessel(v(1), b(0), 456);

        assert_eq!(s.vessel_start_time(v(0)), 123);
        assert_eq!(s.vessel_start_time(v(1)), 456);
        assert_eq!(s.vessel_berth(v(0)).get(), 0);
        assert_eq!(s.vessel_berth(v(1)).get(), 0);

        unsafe {
            assert_eq!(s.vessel_start_time_unchecked(v(0)), 123);
            assert_eq!(s.vessel_start_time_unchecked(v(1)), 456);
            assert_eq!(s.vessel_berth_unchecked(v(0)).get(), 0);
            assert_eq!(s.vessel_berth_unchecked(v(1)).get(), 0);
        }
    }

    #[test]
    fn test_bulk_vessel_accessors_match_individual() {
        let mut s: SearchState<i64> = SearchState::new(2, 3);

        // Assign all vessels to avoid unassigned-access panics.
        s.assign_vessel(v(0), b(1), 100);
        s.assign_vessel(v(1), b(0), 150);
        s.assign_vessel(v(2), b(0), 200);

        let starts = s.vessel_start_times.to_vec();
        let berths = s.vessel_berths.to_vec();

        for i in 0..s.num_vessels() {
            assert!(
                s.is_vessel_assigned(v(i)),
                "vessel {i} should be assigned for this test"
            );
            assert_eq!(starts[i], s.vessel_start_time(v(i)));
            assert_eq!(berths[i].get(), s.vessel_berth(v(i)).get());
        }
    }

    #[test]
    fn test_reset_restores_defaults_and_clears_assignments() {
        let mut s: SearchState<i64> = SearchState::new(4, 3);

        s.set_current_objective(777);
        s.set_last_decision(999, v(2));
        s.set_berth_free_time(b(0), 5);
        s.set_berth_free_time(b(1), 6);
        s.set_berth_free_time(b(2), 7);
        s.set_berth_free_time(b(3), 8);

        s.assign_vessel(v(0), b(1), 10);
        s.assign_vessel(v(1), b(2), 20);
        s.assign_vessel(v(2), b(3), 30);

        assert_eq!(s.num_assigned_vessels(), 3);
        assert_eq!(s.current_objective(), 777);
        assert_eq!(s.last_decision_time(), 999);
        assert_eq!(s.last_decision_vessel().get(), 2);

        s.reset();

        assert_eq!(s.num_assigned_vessels(), 0);
        assert_eq!(s.current_objective(), i64::zero());
        assert_eq!(s.last_decision_time(), i64::min_value());
        assert_eq!(s.last_decision_vessel().get(), 0);
        assert!(s.berth_free_times().iter().all(|&t| t == 0));
        assert!(s.vessel_start_times.iter().all(|&t| t == 0));
        assert!(s.vessel_berths.iter().all(|&bi| bi.get() == 0));
        for i in 0..s.num_vessels() {
            assert!(!s.is_vessel_assigned(v(i)));
        }
    }

    #[test]
    fn test_display_formats_summary() {
        let mut s: SearchState<i64> = SearchState::new(2, 3);
        s.set_current_objective(15);
        s.assign_vessel(v(0), b(1), 10);
        s.assign_vessel(v(2), b(0), 20);

        let formatted = format!("{}", s);
        assert!(formatted.contains("State(objective: 15"));
        assert!(formatted.contains("assigned_vessels: 2/3"));
    }

    #[test]
    fn test_assignment_count_invariants() {
        let mut s: SearchState<i64> = SearchState::new(2, 2);

        s.assign_vessel(v(0), b(0), 1);
        assert_eq!(s.num_assigned_vessels(), 1);
        s.assign_vessel(v(1), b(1), 2);
        assert_eq!(s.num_assigned_vessels(), 2);

        // Capacity reached
        assert_eq!(s.num_assigned_vessels(), s.num_vessels());

        // Unassign and reassign keep within bounds
        s.unassign_vessel(v(0));
        assert_eq!(s.num_assigned_vessels(), 1);
        s.assign_vessel(v(0), b(1), 3);
        assert_eq!(s.num_assigned_vessels(), 2);
        assert!(s.num_assigned_vessels() <= s.num_vessels());
    }

    #[test]
    fn test_is_vessel_assigned_checked_vs_unchecked_consistency() {
        let mut s: SearchState<i64> = SearchState::new(1, 3);

        for i in 0..3 {
            assert!(!s.is_vessel_assigned(v(i)));
            unsafe {
                assert!(!s.is_vessel_assigned_unchecked(v(i)));
            }
        }

        s.assign_vessel(v(1), b(0), 10);
        assert!(s.is_vessel_assigned(v(1)));
        unsafe {
            assert!(s.is_vessel_assigned_unchecked(v(1)));
        }
    }

    #[test]
    fn test_index_bounds_near_edges() {
        let mut s: SearchState<i64> = SearchState::new(2, 2);

        // Valid edge indices
        s.set_berth_free_time(b(0), 1);
        s.set_berth_free_time(b(1), 2);
        assert_eq!(s.berth_free_time(b(0)), 1);
        assert_eq!(s.berth_free_time(b(1)), 2);

        s.assign_vessel(v(0), b(0), 111);
        s.assign_vessel(v(1), b(1), 222);
        assert!(s.is_vessel_assigned(v(0)));
        assert!(s.is_vessel_assigned(v(1)));
    }

    #[test]
    fn test_try_into_solution_success_and_failure() {
        let mut full: SearchState<i64> = SearchState::new(2, 2);
        full.assign_vessel(v(0), b(0), 10);
        full.assign_vessel(v(1), b(1), 20);

        // Try converting a fully-assigned state
        let full_result: Result<super::Solution<i64>, IncompleteSolutionError> = full.try_into();
        assert!(
            full_result.is_ok(),
            "expected TryInto<Solution> to succeed for fully-assigned state"
        );

        let mut partial: SearchState<i64> = SearchState::new(2, 2);
        partial.assign_vessel(v(0), b(0), 10);

        let partial_result: Result<super::Solution<i64>, IncompleteSolutionError> =
            partial.try_into();
        assert!(
            partial_result.is_err(),
            "expected TryInto<Solution> to fail for incomplete state"
        );

        // Validate error contents if fields are exposed via Display
        let err = partial_result.err().unwrap();
        let msg = format!("{}", err);
        assert!(
            msg.contains("assigned"),
            "error message should mention assigned vessels"
        );
    }

    #[test]
    fn test_incomplete_solution_error_display_includes_counts() {
        use super::IncompleteSolutionError;

        // Build an incomplete state: 2 assigned out of 3
        let mut s: SearchState<i64> = SearchState::new(3, 3);
        s.assign_vessel(VesselIndex::new(0), BerthIndex::new(0), 1);
        s.assign_vessel(VesselIndex::new(1), BerthIndex::new(1), 2);
        // One vessel remains unassigned -> conversion should fail
        let res: Result<super::Solution<i64>, IncompleteSolutionError> = s.try_into();
        let err = res.expect_err("conversion should fail for incomplete assignment");
        let text = format!("{}", err);

        assert!(!text.is_empty(), "error display should not be empty");
        assert!(
            text.contains("2"),
            "error display should include assigned count (2)"
        );
        assert!(
            text.contains("3"),
            "error display should include total count (3)"
        );
    }
}
