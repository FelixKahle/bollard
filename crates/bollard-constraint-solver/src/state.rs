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

use bollard_model::index::{BerthIndex, VesselIndex};
use num_traits::Zero;

#[derive(Debug, Clone)]
pub struct SearchState<T> {
    berth_free_times: Vec<T>,      // len = num_berths
    vessel_assignments: Vec<bool>, // len = num_vessels
    num_assigned_vessels: usize,
    current_objective: T,
}

impl<T> SearchState<T>
where
    T: Copy + Zero,
{
    /// Creates a new initial search state with all berths free at time zero
    /// and no vessels assigned.
    #[inline]
    pub fn new(num_berths: usize, num_vessels: usize) -> Self {
        Self {
            berth_free_times: vec![T::zero(); num_berths],
            vessel_assignments: vec![false; num_vessels],
            num_assigned_vessels: 0,
            current_objective: T::zero(),
        }
    }

    /// Returns the number of berths in the state.
    #[inline]
    pub fn num_berths(&self) -> usize {
        self.berth_free_times.len()
    }

    /// Returns the number of vessels in the state.
    #[inline]
    pub fn num_vessels(&self) -> usize {
        self.vessel_assignments.len()
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
    /// Panics if `berth_index` is not within 0..num_berths().
    #[inline]
    pub fn berth_free_time(&self, berth_index: BerthIndex) -> T {
        debug_assert!(berth_index.get() < self.num_berths());

        self.berth_free_times[berth_index.get()]
    }

    /// Returns the free time of the specified berth without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `berth_index` is within 0..num_berths().
    /// Otherwise this function may cause undefined behavior.
    #[inline]
    pub unsafe fn berth_free_time_unchecked(&self, berth_index: BerthIndex) -> T {
        debug_assert!(berth_index.get() < self.num_berths());

        unsafe { *self.berth_free_times.get_unchecked(berth_index.get()) }
    }

    /// Sets the free time of the specified berth.
    ///
    /// # Panics
    ///
    /// Panics if `berth_index` is not within 0..num_berths().
    #[inline]
    pub fn set_berth_free_time(&mut self, berth_index: BerthIndex, time: T) {
        debug_assert!(berth_index.get() < self.num_berths());

        self.berth_free_times[berth_index.get()] = time;
    }

    /// Sets the free time of the specified berth without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `berth_index` is within 0..num_berths().
    /// Otherwise this function may cause undefined behavior.
    #[inline]
    pub unsafe fn set_berth_free_time_unchecked(&mut self, berth_index: BerthIndex, time: T) {
        debug_assert!(berth_index.get() < self.num_berths());

        unsafe {
            *self.berth_free_times.get_unchecked_mut(berth_index.get()) = time;
        }
    }

    /// Returns a slice of the vessel assignments.
    #[inline]
    pub fn vessel_assignments(&self) -> &[bool] {
        &self.vessel_assignments
    }

    /// Returns whether the specified vessel is assigned.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is out not within 0..num_vessels().
    #[inline]
    pub fn is_vessel_assigned(&self, vessel_index: VesselIndex) -> bool {
        debug_assert!(vessel_index.get() < self.num_vessels());

        self.vessel_assignments[vessel_index.get()]
    }

    /// Returns whether the specified vessel is assigned without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is within 0..num_vessels().
    /// Otherwise this function may cause undefined behavior.
    #[inline]
    pub unsafe fn is_vessel_assigned_unchecked(&self, vessel_index: VesselIndex) -> bool {
        debug_assert!(vessel_index.get() < self.num_vessels());

        unsafe { *self.vessel_assignments.get_unchecked(vessel_index.get()) }
    }

    /// Marks the specified vessel as assigned.
    ///
    /// # Panics
    ///
    /// Panics if the vessel is already assigned or if `vessel_index`
    /// is not within 0..num_vessels().
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is out not within 0..num_vessels().
    #[inline]
    pub fn assign_vessel(&mut self, vessel_index: VesselIndex) {
        debug_assert!(vessel_index.get() < self.num_vessels());
        debug_assert!(
            !self.vessel_assignments[vessel_index.get()],
            "assign_vessel: vessel {} already assigned",
            vessel_index.get()
        );

        self.vessel_assignments[vessel_index.get()] = true;
        self.num_assigned_vessels += 1;

        debug_assert!(self.num_assigned_vessels <= self.num_vessels());
    }

    /// Marks the specified vessel as assigned without bounds checking.
    ///
    /// # Panics
    ///
    /// Panics if the vessel is already assigned.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is within 0..num_vessels().
    /// Otherwise this function may cause undefined behavior.
    #[inline]
    pub unsafe fn assign_vessel_unchecked(&mut self, vessel_index: VesselIndex) {
        debug_assert!(vessel_index.get() < self.num_vessels());
        debug_assert!(
            !self.vessel_assignments[vessel_index.get()],
            "assign_vessel: vessel {} already assigned",
            vessel_index.get()
        );

        unsafe {
            *self
                .vessel_assignments
                .get_unchecked_mut(vessel_index.get()) = true;
        }
        self.num_assigned_vessels += 1;

        debug_assert!(self.num_assigned_vessels <= self.num_vessels());
    }

    /// Marks the specified vessel as assigned.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is out not within 0..num_vessels() or if the vessel
    /// is already unassigned.
    #[inline]
    pub fn unassign_vessel(&mut self, vessel_index: VesselIndex) {
        debug_assert!(vessel_index.get() < self.num_vessels());
        debug_assert!(
            self.vessel_assignments[vessel_index.get()],
            "unassign_vessel: vessel {} already unassigned",
            vessel_index.get()
        );

        self.vessel_assignments[vessel_index.get()] = false;
        self.num_assigned_vessels -= 1;

        debug_assert!(self.num_assigned_vessels <= self.num_vessels());
    }

    /// Marks the specified vessel as unassigned without bounds checking.
    ///
    /// # Panics
    ///
    /// Panics if the vessel is already unassigned.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is within 0..num_vessels().
    /// Otherwise this function may cause undefined behavior.
    #[inline]
    pub unsafe fn unassign_vessel_unchecked(&mut self, vessel_index: VesselIndex) {
        debug_assert!(vessel_index.get() < self.num_vessels());
        debug_assert!(
            self.vessel_assignments[vessel_index.get()],
            "unassign_vessel: vessel {} already unassigned",
            vessel_index.get()
        );

        unsafe {
            *self
                .vessel_assignments
                .get_unchecked_mut(vessel_index.get()) = false;
        }
        self.num_assigned_vessels -= 1;

        debug_assert!(self.num_assigned_vessels <= self.num_vessels());
    }

    /// Returns the number of assigned vessels.
    #[inline]
    pub fn num_assigned_vessels(&self) -> usize {
        self.num_assigned_vessels
    }

    /// Returns the current objective value.
    #[inline]
    pub fn current_objective(&self) -> T {
        self.current_objective
    }

    /// Sets the current objective value.
    #[inline]
    pub fn set_current_objective(&mut self, objective: T) {
        self.current_objective = objective;
    }

    /// Resets the state to the initial configuration.
    #[inline]
    pub fn reset(&mut self) {
        self.berth_free_times.fill(T::zero());
        self.vessel_assignments.fill(false);
        self.num_assigned_vessels = 0;
        self.current_objective = T::zero();
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
            self.current_objective,
            self.num_assigned_vessels,
            self.vessel_assignments.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Zero;

    type IntegerType = i32;

    #[test]
    fn test_new_initial_state() {
        let num_berths = 3;
        let num_vessels = 5;
        let state = SearchState::<IntegerType>::new(num_berths, num_vessels);

        // Sizes
        assert_eq!(state.num_berths(), num_berths);
        assert_eq!(state.num_vessels(), num_vessels);

        // Initial berth free times are zero
        assert_eq!(state.berth_free_times().len(), num_berths);
        assert!(
            state
                .berth_free_times()
                .iter()
                .all(|&t| t == IntegerType::zero())
        );

        // Initial vessel assignments are false
        assert_eq!(state.vessel_assignments().len(), num_vessels);
        assert!(state.vessel_assignments().iter().all(|&a| a == false));

        // No vessels assigned, objective zero
        assert_eq!(state.num_assigned_vessels(), 0);
        assert_eq!(state.current_objective(), IntegerType::zero());
    }

    #[test]
    fn test_assign_unassign_vessel_and_counts() {
        let num_berths = 2;
        let num_vessels = 4;
        let mut state = SearchState::<IntegerType>::new(num_berths, num_vessels);

        // Assign two vessels
        state.assign_vessel(VesselIndex::new(1));
        state.assign_vessel(VesselIndex::new(3));
        assert_eq!(state.num_assigned_vessels(), 2);
        assert!(state.is_vessel_assigned(VesselIndex::new(1)));
        assert!(state.is_vessel_assigned(VesselIndex::new(3)));
        assert!(!state.is_vessel_assigned(VesselIndex::new(0)));
        assert!(!state.is_vessel_assigned(VesselIndex::new(2)));

        // Unassign one vessel
        state.unassign_vessel(VesselIndex::new(1));
        assert_eq!(state.num_assigned_vessels(), 1);
        assert!(!state.is_vessel_assigned(VesselIndex::new(1)));
        assert!(state.is_vessel_assigned(VesselIndex::new(3)));
    }

    #[test]
    fn test_display_format() {
        let mut state = SearchState::<IntegerType>::new(1, 3);

        // Assign 2 vessels to influence the display counts
        state.assign_vessel(VesselIndex::new(0));
        state.assign_vessel(VesselIndex::new(2));

        let formatted = format!("{}", state);
        // Example: "State(objective: 0, assigned_vessels: 2/3)"
        assert!(formatted.contains("State(objective: "));
        assert!(formatted.contains("assigned_vessels: 2/3"));
    }
}
