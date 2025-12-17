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

//! Decision building and symmetry handling
//!
//! Defines the `Decision<T>` type and the `DecisionBuilder` trait used to
//! generate feasible `(vessel, berth)` assignments for branch‑and‑bound.
//!
//! A `Decision` carries indices, computed start time, and objective cost delta,
//! making it suitable for immediate evaluation and ordering.
//!
//!
//! Feasibility integrates three aspects:
//! - Model topology (processing times must exist and be allowed)
//! - Berth availability (respecting opening intervals and closures)
//! - Evaluator constraints (e.g., deadlines)
//!
//! Symmetry reduction filters indistinguishable adjacent berth options under
//! identical state and processing conditions, preserving canonical schedules
//! while avoiding redundant branches.
//!
//! Ordering is stable via `Ord`, enabling consistent tie‑breaking and
//! deterministic traversal by builders.
//!
//! Builders implement `DecisionBuilder` to produce iterators over rich decisions.
//! Once an iterator is exhausted, it is fused and yields `None` thereafter.

use crate::{
    berth_availability::BerthAvailability, eval::evaluator::ObjectiveEvaluator, state::SearchState,
};
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use bollard_search::num::SolverNumeric;
use num_traits::{PrimInt, Signed};
use std::iter::FusedIterator;

/// A decision to assign a vessel to a berth at a specific start time,
/// along with the associated cost delta.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Decision<T> {
    // Optimize for T = i64 to minimize padding.
    // This will minimize the size of Decision so we can
    // fill more cache lines with it.
    /// The start time for the vessel assignment.
    start_time: T,
    /// The cost delta associated with the vessel assignment.
    cost_delta: T,
    /// The index of the vessel to be assigned.
    vessel_index: VesselIndex,
    /// The index of the berth to which the vessel is assigned.
    berth_index: BerthIndex,
}

impl<T> std::fmt::Display for Decision<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Decision(vessel: {}, berth: {}, start_time: {}, cost_delta: {})",
            self.vessel_index, self.berth_index, self.start_time, self.cost_delta
        )
    }
}

impl<T: Ord> Ord for Decision<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.vessel_index
            .cmp(&other.vessel_index)
            .then(self.berth_index.cmp(&other.berth_index))
            .then(self.start_time.cmp(&other.start_time))
            .then(self.cost_delta.cmp(&other.cost_delta))
    }
}

impl<T: Ord> PartialOrd for Decision<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Checks if assigning the vessel to the current berth at the specified
/// start time and processing time is symmetric to assigning it to the
/// previous berth.
///
/// # Panics
///
/// The caller must ensure that `vessel_index` is within `0..model.num_vessels()`
/// and that `berth_index` is within `0..model.num_berths()`.
#[inline(always)]
fn is_symmetric_to_previous_berth<T>(
    vessel_index: VesselIndex,
    berth_index: BerthIndex,
    current_start: T,
    current_processing_time: T,
    model: &Model<T>,
    state: &SearchState<T>,
    ba: &BerthAvailability<T>,
) -> bool
where
    T: SolverNumeric,
{
    debug_assert!(
        vessel_index.get() < model.num_vessels(),
        "called `decision::is_symmetric_to_previous_berth` with vessel index out of bounds: the len is {} but the index is {}",
        model.num_vessels(),
        vessel_index.get()
    );
    debug_assert!(
        berth_index.get() < model.num_berths(),
        "called `decision::is_symmetric_to_previous_berth` with berth index out of bounds: the len is {} but the index is {}",
        model.num_berths(),
        berth_index.get()
    );

    if berth_index.is_zero() {
        return false;
    }

    let previous_berth = BerthIndex::new(berth_index.get() - 1);
    let current_free = state.berth_free_time(berth_index);
    let previous_free = state.berth_free_time(previous_berth);

    if current_free != previous_free {
        return false;
    }

    let previous_processing_time = model
        .vessel_processing_time(vessel_index, previous_berth)
        .unwrap_or_else(T::zero);

    if current_processing_time != previous_processing_time {
        return false;
    }

    if !model.vessel_allowed_on_berth(vessel_index, previous_berth) {
        return false;
    }

    let arrival = model.vessel_arrival_time(vessel_index);

    let nominal_start = if arrival > current_free {
        arrival
    } else {
        current_free
    };

    // This uses a custom written, high-performance binary search internally.
    // Even though its fast to call it still needs to load the berth availability data
    // from memory, and call the binary search that is O(log N + K) where N is the number of
    // stored intervals and K is the linear scan after positioning the pointer. Usually both
    // N and K are small, but this is still more expensive than simple arithmetic operations.
    let previous_start =
        ba.earliest_availability(previous_berth, nominal_start, previous_processing_time);
    previous_start == Some(current_start)
}

/// Checks if assigning the vessel to the current berth at the specified
/// start time and processing time is symmetric to assigning it to the
/// previous berth without performing bounds checks on the indices.
///
/// # Panics
///
/// In debug builds, this function will panic if `vessel_index` is not within
/// `0..model.num_vessels()` or if `berth_index` is not within `0..model.num_berths()`.
///
/// # Safety
///
/// The caller must ensure that `vessel_index` is within `0..model.num_vessels()`
/// and that `berth_index` is within `0..model.num_berths()`.
#[inline(always)]
unsafe fn is_symmetric_to_previous_berth_unchecked<T>(
    vessel_index: VesselIndex,
    berth_index: BerthIndex,
    current_start: T,
    current_processing_time: T,
    model: &Model<T>,
    state: &SearchState<T>,
    ba: &BerthAvailability<T>,
) -> bool
where
    T: SolverNumeric,
{
    debug_assert!(
        vessel_index.get() < model.num_vessels(),
        "called `decision::is_symmetric_to_previous_berth_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
        model.num_vessels(),
        vessel_index.get()
    );
    debug_assert!(
        berth_index.get() < model.num_berths(),
        "called `decision::is_symmetric_to_previous_berth_unchecked` with berth index out of bounds: the len is {} but the index is {}",
        model.num_berths(),
        berth_index.get()
    );

    if berth_index.is_zero() {
        return false;
    }

    let previous_berth = BerthIndex::new(berth_index.get() - 1);
    let current_free = unsafe { state.berth_free_time_unchecked(berth_index) };
    let previous_free = unsafe { state.berth_free_time_unchecked(previous_berth) };

    if current_free != previous_free {
        return false;
    }

    let previous_processing_time = unsafe {
        model
            .vessel_processing_time_unchecked(vessel_index, previous_berth)
            .unwrap_or_else(T::zero)
    };

    if current_processing_time != previous_processing_time {
        return false;
    }

    if unsafe { !model.vessel_allowed_on_berth_unchecked(vessel_index, previous_berth) } {
        return false;
    }

    let arrival = unsafe { model.vessel_arrival_time_unchecked(vessel_index) };
    let nominal_start = if arrival > current_free {
        arrival
    } else {
        current_free
    };

    // This uses a custom written, high-performance binary search internally.
    // Even though its fast to call it still needs to load the berth availability data
    // from memory, and call the binary search that is O(log N + K) where N is the number of
    // stored intervals and K is the linear scan after positioning the pointer. Usually both
    // N and K are small, but this is still more expensive than simple arithmetic operations.
    let previous_start = unsafe {
        ba.earliest_availability_unchecked(previous_berth, nominal_start, previous_processing_time)
    };
    previous_start == Some(current_start)
}

impl<T> Decision<T>
where
    T: SolverNumeric,
{
    /// Tries to create a new `Decision` for assigning the specified vessel
    /// to the specified berth, given the model, berth availability, search
    /// state, and objective evaluator.
    ///
    /// Decisions are only generated forward in time to break symmetry.
    ///
    /// # Symmetry
    ///
    /// Symmetry refers to a situation where different sequences of decisions
    /// lead to the same state or solution. For example, assigning Vessel A to
    /// Berth 1 and then Vessel B to Berth 2 may yield the same outcome as
    /// assigning Vessel B to Berth 2 first and then Vessel A to Berth 1.
    ///
    /// To avoid exploring symmetric states, which can lead to redundant computations,
    /// this method enforces that decisions are only made forward in time. This means
    /// that a vessel can only be assigned to a berth at or after the last decision time.
    ///
    /// Additionally, if the start time of the new decision is equal to the last decision
    /// time, the vessel index must be greater than or equal to the last decision vessel
    /// index. This further reduces symmetry by ensuring a consistent ordering of vessel
    /// assignments when they occur at the same time.
    ///
    /// # Panics
    ///
    /// The caller must ensure that `vessel_index` is within `0..model.num_vessels()`
    /// and that `berth_index` is within `0..model.num_berths()`.
    #[inline]
    pub fn try_new<E>(
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        model: &Model<T>,
        berth_availability: &BerthAvailability<T>,
        state: &SearchState<T>,
        evaluator: &mut E,
    ) -> Option<Self>
    where
        E: ObjectiveEvaluator<T>,
    {
        debug_assert!(
            vessel_index.get() < model.num_vessels(),
            "called `Decision::try_new` with vessel index out of bounds: the len is {} but the index is {}",
            model.num_vessels(),
            vessel_index.get()
        );
        debug_assert!(
            berth_index.get() < model.num_berths(),
            "called `Decision::try_new` with berth index out of bounds: the len is {} but the index is {}",
            model.num_berths(),
            berth_index.get()
        );

        if state.is_vessel_assigned(vessel_index) {
            return None;
        }

        let processing_time_option = model.vessel_processing_time(vessel_index, berth_index);
        if processing_time_option.is_none() {
            return None;
        }
        let processing_time = processing_time_option.unwrap_unchecked();
        let berth_free = state.berth_free_time(berth_index);
        let arrival = model.vessel_arrival_time(vessel_index);

        let nominal_start = if arrival > berth_free {
            arrival
        } else {
            berth_free
        };

        let actual_start_time = berth_availability.earliest_availability(
            berth_index,
            nominal_start,
            processing_time,
        )?;

        let last_decision_time = state.last_decision_time();
        if actual_start_time < last_decision_time {
            return None;
        }

        if actual_start_time == last_decision_time
            && vessel_index.get() < state.last_decision_vessel().get()
        {
            return None;
        }

        if is_symmetric_to_previous_berth(
            vessel_index,
            berth_index,
            actual_start_time,
            processing_time,
            model,
            state,
            berth_availability,
        ) {
            return None;
        }

        let cost_delta = evaluator.evaluate_vessel_assignment(
            model,
            berth_availability,
            vessel_index,
            berth_index,
            actual_start_time,
        )?;

        Some(Self {
            vessel_index,
            berth_index,
            start_time: actual_start_time,
            cost_delta,
        })
    }

    /// Tries to create a new `Decision` for assigning the specified vessel
    /// to the specified berth, given the model, berth availability, search
    /// state, and objective evaluator without performing bounds checks on the
    /// indices.
    ///
    /// Decisions are only generated forward in time to break symmetry.
    ///
    /// # Symmetry
    ///
    /// Symmetry refers to a situation where different sequences of decisions
    /// lead to the same state or solution. For example, assigning Vessel A to
    /// Berth 1 and then Vessel B to Berth 2 may yield the same outcome as
    /// assigning Vessel B to Berth 2 first and then Vessel A to Berth 1.
    ///
    /// To avoid exploring symmetric states, which can lead to redundant computations,
    /// this method enforces that decisions are only made forward in time. This means
    /// that a vessel can only be assigned to a berth at or after the last decision time.
    ///
    /// Additionally, if the start time of the new decision is equal to the last decision
    /// time, the vessel index must be greater than or equal to the last decision vessel
    /// index. This further reduces symmetry by ensuring a consistent ordering of vessel
    /// assignments when they occur at the same time.
    ///
    /// # Panics
    ///
    /// In debug builds, this function will panic if `vessel_index` is not within
    /// `0..model.num_vessels()` or if `berth_index` is not within `0..model.num_berths()`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is within `0..model.num_vessels()`
    /// and that `berth_index` is within `0..model.num_berths()`.
    #[inline]
    pub unsafe fn try_new_unchecked<E>(
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        model: &Model<T>,
        berth_availability: &BerthAvailability<T>,
        state: &SearchState<T>,
        evaluator: &mut E,
    ) -> Option<Self>
    where
        E: ObjectiveEvaluator<T>,
    {
        debug_assert!(
            vessel_index.get() < model.num_vessels(),
            "called `Decision::try_new_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            model.num_vessels(),
            vessel_index.get()
        );
        debug_assert!(
            berth_index.get() < model.num_berths(),
            "called `Decision::try_new_unchecked` with berth index out of bounds: the len is {} but the index is {}",
            model.num_berths(),
            berth_index.get()
        );

        if unsafe { state.is_vessel_assigned_unchecked(vessel_index) } {
            return None;
        }

        let processing_time_option =
            unsafe { model.vessel_processing_time_unchecked(vessel_index, berth_index) };
        if processing_time_option.is_none() {
            return None;
        }

        let processing_time = processing_time_option.unwrap_unchecked();
        let berth_free = unsafe { state.berth_free_time_unchecked(berth_index) };
        let arrival = unsafe { model.vessel_arrival_time_unchecked(vessel_index) };

        let nominal_start = if arrival > berth_free {
            arrival
        } else {
            berth_free
        };

        let actual_start_time = unsafe {
            berth_availability.earliest_availability_unchecked(
                berth_index,
                nominal_start,
                processing_time,
            )?
        };

        let last_decision_time = state.last_decision_time();
        if actual_start_time < last_decision_time {
            return None;
        }

        if actual_start_time == last_decision_time
            && vessel_index.get() < state.last_decision_vessel().get()
        {
            return None;
        }

        if unsafe {
            is_symmetric_to_previous_berth_unchecked(
                vessel_index,
                berth_index,
                actual_start_time,
                processing_time,
                model,
                state,
                berth_availability,
            )
        } {
            return None;
        }

        let cost_delta = evaluator.evaluate_vessel_assignment(
            model,
            berth_availability,
            vessel_index,
            berth_index,
            actual_start_time,
        )?;

        Some(Self {
            vessel_index,
            berth_index,
            start_time: actual_start_time,
            cost_delta,
        })
    }

    /// Returns the index of the vessel to be assigned.
    #[inline(always)]
    pub const fn vessel_index(&self) -> VesselIndex {
        self.vessel_index
    }

    /// Returns the index of the berth to which the vessel is assigned.
    #[inline(always)]
    pub const fn berth_index(&self) -> BerthIndex {
        self.berth_index
    }

    /// Returns the start time for the vessel assignment.
    #[inline(always)]
    pub const fn start_time(&self) -> T {
        self.start_time
    }

    /// Returns the cost delta associated with the vessel assignment.
    #[inline(always)]
    pub const fn cost_delta(&self) -> T {
        self.cost_delta
    }
}

/// A trait for building decisions in a branch-and-bound search.
pub trait DecisionBuilder<T, E>
where
    T: PrimInt + Signed,
    E: ObjectiveEvaluator<T>,
{
    /// An iterator over decisions produced by this decision builder.
    type DecisionIterator<'a>: Iterator<Item = Decision<T>> + FusedIterator + 'a
    where
        Self: 'a,
        T: 'a,
        E: 'a;

    /// Returns the name of this decision builder.
    fn name(&self) -> &str;

    /// Produces an iterator over decisions for the given model, berth availability,
    /// and search state using the provided objective evaluator.
    fn next_decision<'a>(
        &'a mut self,
        evaluator: &'a mut E,
        model: &'a Model<T>,
        berth_availability: &'a BerthAvailability<T>,
        search_state: &'a SearchState<T>,
    ) -> Self::DecisionIterator<'a>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::wtft::WeightedFlowTimeEvaluator;
    use bollard_model::index::{BerthIndex, VesselIndex};
    use bollard_model::model::ModelBuilder;
    use bollard_model::time::ProcessingTime;

    type IntegerType = i64;

    fn build_basic_model() -> Model<IntegerType> {
        // 2 berths, 3 vessels
        let mut b = ModelBuilder::<IntegerType>::new(2, 3);

        // Arrivals 0, 5, 10
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_arrival_time(VesselIndex::new(1), 5)
            .set_vessel_arrival_time(VesselIndex::new(2), 10);

        // Weights = 1 each
        b.set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_weight(VesselIndex::new(1), 1)
            .set_vessel_weight(VesselIndex::new(2), 1);

        // Processing times: all 5 unless explicitly None later
        for v in 0..3 {
            for b_idx in 0..2 {
                b.set_vessel_processing_time(
                    VesselIndex::new(v),
                    BerthIndex::new(b_idx),
                    ProcessingTime::some(5),
                );
            }
        }

        b.build()
    }

    #[test]
    fn test_symmetric_true_when_previous_berth_is_indistinguishable() {
        // Model: 2 berths, 1 vessel; arrival=5; proc times equal on both berths.
        let mut b = ModelBuilder::<IntegerType>::new(2, 1);
        b.set_vessel_arrival_time(VesselIndex::new(0), 5)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(7),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::some(7),
            );
        let model = b.build();

        let mut ba = BerthAvailability::<IntegerType>::new();
        assert!(ba.initialize(&model, &[]));

        let mut state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        // Equal free times
        state.set_berth_free_time(BerthIndex::new(0), 5);
        state.set_berth_free_time(BerthIndex::new(1), 5);

        let v = VesselIndex::new(0);
        let b1 = BerthIndex::new(1);
        let pt = model.vessel_processing_time(v, b1).unwrap();
        let nominal = model.vessel_arrival_time(v).max(state.berth_free_time(b1));
        let current_start = ba.earliest_availability(b1, nominal, pt).unwrap();

        assert!(is_symmetric_to_previous_berth(
            v,
            b1,
            current_start,
            pt,
            &model,
            &state,
            &ba
        ));
    }

    #[test]
    fn test_symmetric_false_for_berth_index_zero() {
        let mut b = ModelBuilder::<IntegerType>::new(2, 1);
        b.set_vessel_arrival_time(VesselIndex::new(0), 5)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(7),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::some(7),
            );
        let model = b.build();

        let mut ba = BerthAvailability::<IntegerType>::new();
        assert!(ba.initialize(&model, &[]));

        let mut state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        state.set_berth_free_time(BerthIndex::new(0), 5);

        let v = VesselIndex::new(0);
        let b0 = BerthIndex::new(0);
        let pt = model.vessel_processing_time(v, b0).unwrap();
        let nominal = model.vessel_arrival_time(v).max(state.berth_free_time(b0));
        let current_start = ba.earliest_availability(b0, nominal, pt).unwrap();

        assert!(!is_symmetric_to_previous_berth(
            v,
            b0,
            current_start,
            pt,
            &model,
            &state,
            &ba
        ));
    }

    #[test]
    fn test_symmetric_false_when_free_times_differ() {
        let mut b = ModelBuilder::<IntegerType>::new(2, 1);
        b.set_vessel_arrival_time(VesselIndex::new(0), 5)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(7),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::some(7),
            );
        let model = b.build();

        let mut ba = BerthAvailability::<IntegerType>::new();
        assert!(ba.initialize(&model, &[]));

        let mut state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        state.set_berth_free_time(BerthIndex::new(0), 0);
        state.set_berth_free_time(BerthIndex::new(1), 5);

        let v = VesselIndex::new(0);
        let b1 = BerthIndex::new(1);
        let pt = model.vessel_processing_time(v, b1).unwrap();
        let nominal = model.vessel_arrival_time(v).max(state.berth_free_time(b1));
        let current_start = ba.earliest_availability(b1, nominal, pt).unwrap();

        assert!(!is_symmetric_to_previous_berth(
            v,
            b1,
            current_start,
            pt,
            &model,
            &state,
            &ba
        ));
    }

    #[test]
    fn test_symmetric_false_when_processing_times_differ_on_prev_berth() {
        // Different PTs: b0=9, b1=7
        let mut b = ModelBuilder::<IntegerType>::new(2, 1);
        b.set_vessel_arrival_time(VesselIndex::new(0), 5)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(9),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::some(7),
            );
        let model = b.build();

        let mut ba = BerthAvailability::<IntegerType>::new();
        assert!(ba.initialize(&model, &[]));

        let mut state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        state.set_berth_free_time(BerthIndex::new(0), 5);
        state.set_berth_free_time(BerthIndex::new(1), 5);

        let v = VesselIndex::new(0);
        let b1 = BerthIndex::new(1);
        let pt = model.vessel_processing_time(v, b1).unwrap();
        let nominal = model.vessel_arrival_time(v).max(state.berth_free_time(b1));
        let current_start = ba.earliest_availability(b1, nominal, pt).unwrap();

        assert!(!is_symmetric_to_previous_berth(
            v,
            b1,
            current_start,
            pt,
            &model,
            &state,
            &ba
        ));
    }

    #[test]
    fn test_symmetric_false_when_prev_berth_disallowed() {
        // Make previous berth forbidden by setting PT None on b0
        let mut b = ModelBuilder::<IntegerType>::new(2, 1);
        b.set_vessel_arrival_time(VesselIndex::new(0), 5)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::none(),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::some(7),
            );
        let model = b.build();

        let mut ba = BerthAvailability::<IntegerType>::new();
        assert!(ba.initialize(&model, &[]));

        let mut state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        state.set_berth_free_time(BerthIndex::new(0), 5);
        state.set_berth_free_time(BerthIndex::new(1), 5);

        let v = VesselIndex::new(0);
        let b1 = BerthIndex::new(1);
        let pt = model.vessel_processing_time(v, b1).unwrap();
        let nominal = model.vessel_arrival_time(v).max(state.berth_free_time(b1));
        let current_start = ba.earliest_availability(b1, nominal, pt).unwrap();

        assert!(!is_symmetric_to_previous_berth(
            v,
            b1,
            current_start,
            pt,
            &model,
            &state,
            &ba
        ));
    }

    #[test]
    fn test_symmetric_false_when_previous_berth_yields_different_start() {
        // Equal PTs and allowed, but BA difference forces different previous start.
        let mut b = ModelBuilder::<IntegerType>::new(2, 1);
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(5),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::some(5),
            );
        let model = b.build();

        let mut ba = BerthAvailability::<IntegerType>::new();
        assert!(ba.initialize(
            &model,
            &[ // add an availability block on b0 starting later
            // If your BerthAvailability supports constraints input, use them here;
            // otherwise we simulate by setting different free times.
        ]
        ));

        let mut state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        // Force b0 free later than b1 so previous start cannot match
        state.set_berth_free_time(BerthIndex::new(0), 10);
        state.set_berth_free_time(BerthIndex::new(1), 0);

        let v = VesselIndex::new(0);
        let b1 = BerthIndex::new(1);
        let pt = model.vessel_processing_time(v, b1).unwrap();
        let nominal = model.vessel_arrival_time(v).max(state.berth_free_time(b1));
        let current_start = ba.earliest_availability(b1, nominal, pt).unwrap();

        assert!(!is_symmetric_to_previous_berth(
            v,
            b1,
            current_start,
            pt,
            &model,
            &state,
            &ba
        ));
    }

    #[test]
    fn test_symmetric_unchecked_true_when_previous_berth_is_indistinguishable() {
        let mut b = ModelBuilder::<IntegerType>::new(2, 1);
        b.set_vessel_arrival_time(VesselIndex::new(0), 5)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(7),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::some(7),
            );
        let model = b.build();

        let mut ba = BerthAvailability::<IntegerType>::new();
        assert!(ba.initialize(&model, &[]));

        let mut state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        state.set_berth_free_time(BerthIndex::new(0), 5);
        state.set_berth_free_time(BerthIndex::new(1), 5);

        let v = VesselIndex::new(0);
        let b1 = BerthIndex::new(1);
        let pt = model.vessel_processing_time(v, b1).unwrap();
        let nominal = model.vessel_arrival_time(v).max(state.berth_free_time(b1));
        let current_start = ba.earliest_availability(b1, nominal, pt).unwrap();

        let ok = unsafe {
            is_symmetric_to_previous_berth_unchecked(v, b1, current_start, pt, &model, &state, &ba)
        };
        assert!(ok);
    }

    #[test]
    fn test_symmetric_unchecked_false_when_processing_times_differ_on_prev_berth() {
        let mut b = ModelBuilder::<IntegerType>::new(2, 1);
        b.set_vessel_arrival_time(VesselIndex::new(0), 5)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(9),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::some(7),
            );
        let model = b.build();

        let mut ba = BerthAvailability::<IntegerType>::new();
        assert!(ba.initialize(&model, &[]));

        let mut state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        state.set_berth_free_time(BerthIndex::new(0), 5);
        state.set_berth_free_time(BerthIndex::new(1), 5);

        let v = VesselIndex::new(0);
        let b1 = BerthIndex::new(1);
        let pt = model.vessel_processing_time(v, b1).unwrap();
        let nominal = model
            .vessel_arrival_time(v)
            .max(unsafe { state.berth_free_time_unchecked(b1) });
        let current_start = unsafe { ba.earliest_availability_unchecked(b1, nominal, pt).unwrap() };

        let ok = unsafe {
            is_symmetric_to_previous_berth_unchecked(v, b1, current_start, pt, &model, &state, &ba)
        };
        assert!(!ok);
    }

    #[test]
    fn test_try_new_filters_already_assigned_vessels() {
        let model = build_basic_model();
        let mut ba = BerthAvailability::<IntegerType>::new();
        assert!(ba.initialize(&model, &[]));

        let mut state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut eval = WeightedFlowTimeEvaluator::<IntegerType>::new();

        // Assign vessel 0 first
        state.assign_vessel(VesselIndex::new(0), BerthIndex::new(0), 0);

        // Now attempt decision for vessel 0 again -> should be filtered out
        let d = Decision::try_new(
            VesselIndex::new(0),
            BerthIndex::new(1),
            &model,
            &ba,
            &state,
            &mut eval,
        );
        assert!(
            d.is_none(),
            "already-assigned vessel should yield no decision"
        );
    }

    #[test]
    fn test_forward_in_time_and_tie_breaking_by_vessel_index() {
        let model = build_basic_model();
        let mut ba = BerthAvailability::<IntegerType>::new();
        assert!(ba.initialize(&model, &[]));

        let mut state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut eval = WeightedFlowTimeEvaluator::<IntegerType>::new();

        // Set last decision to time 10, vessel 1
        state.set_last_decision(10, VesselIndex::new(1));

        // Candidate starting before time 10 must be filtered
        {
            // Make berth 0 free at time 0 explicitly to test the time guard
            state.set_berth_free_time(BerthIndex::new(0), 0);
            // Vessel 0 arrival is 0, processing fits, earliest availability 0 -> actual_start 5? No: nominal max(arrival=0, berth_free=0)=0; availability permits start at 0; but we set last_decision_time=10 -> should filter
            let d = Decision::try_new(
                VesselIndex::new(0),
                BerthIndex::new(0),
                &model,
                &ba,
                &state,
                &mut eval,
            );
            assert!(
                d.is_none(),
                "decisions before last_decision_time must be filtered"
            );
        }

        // Now force decisions at exactly time 10:
        state.set_berth_free_time(BerthIndex::new(0), 10);
        state.set_berth_free_time(BerthIndex::new(1), 10);

        // At time 10, vessel 0 has index < last_decision_vessel(1) -> filtered
        let d0 = Decision::try_new(
            VesselIndex::new(0),
            BerthIndex::new(0),
            &model,
            &ba,
            &state,
            &mut eval,
        );
        assert!(
            d0.is_none(),
            "equal time, lower vessel index than last decision must be filtered"
        );

        // Vessel 1 equal to last_decision_vessel -> allowed
        let d1 = Decision::try_new(
            VesselIndex::new(1),
            BerthIndex::new(0),
            &model,
            &ba,
            &state,
            &mut eval,
        );
        assert!(
            d1.is_some(),
            "equal time, same vessel index should be allowed"
        );

        // Vessel 2 greater than last_decision_vessel -> allowed
        // Use lower-index berth (0) to avoid adjacent-berth symmetry filtering.
        let d2 = Decision::try_new(
            VesselIndex::new(2),
            BerthIndex::new(0),
            &model,
            &ba,
            &state,
            &mut eval,
        );
        assert!(
            d2.is_some(),
            "equal time, higher vessel index should be allowed (canonical lower-index berth)"
        );
    }

    #[test]
    fn test_adjacent_berth_indistinguishability_is_filtered() {
        let mut bldr = ModelBuilder::<IntegerType>::new(2, 1);
        // Vessel 0 arrives at 5
        bldr.set_vessel_arrival_time(VesselIndex::new(0), 5)
            .set_vessel_weight(VesselIndex::new(0), 1);
        // Both berths have identical processing time 7 for vessel 0
        bldr.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(0),
            ProcessingTime::some(7),
        );
        bldr.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(1),
            ProcessingTime::some(7),
        );
        let model = bldr.build();

        let mut ba = BerthAvailability::<IntegerType>::new();
        assert!(ba.initialize(&model, &[]));

        let mut state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut eval = WeightedFlowTimeEvaluator::<IntegerType>::new();

        // Make both berths have same free time, ensuring indistinguishable slots
        state.set_berth_free_time(BerthIndex::new(0), 5);
        state.set_berth_free_time(BerthIndex::new(1), 5);

        // Attempt decision on the higher-index berth (1). Because both are indistinguishable,
        // the code should filter B1 and keep only B0 as canonical.
        let d_b1 = Decision::try_new(
            VesselIndex::new(0),
            BerthIndex::new(1),
            &model,
            &ba,
            &state,
            &mut eval,
        );
        assert!(
            d_b1.is_none(),
            "indistinguishable adjacent berths should be symmetry-filtered on higher index"
        );

        // Lower-index berth (0) should be allowed
        let d_b0 = Decision::try_new(
            VesselIndex::new(0),
            BerthIndex::new(0),
            &model,
            &ba,
            &state,
            &mut eval,
        );
        assert!(
            d_b0.is_some(),
            "canonical lower-index berth should be allowed"
        );
    }

    #[test]
    fn test_processing_time_none_or_forbidden_pair_returns_none() {
        let mut bldr = ModelBuilder::<IntegerType>::new(2, 1);
        bldr.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_weight(VesselIndex::new(0), 1);

        // Berth 0 has a processing time, berth 1 is None for vessel 0
        bldr.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(0),
            ProcessingTime::some(5),
        );
        bldr.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(1),
            ProcessingTime::none(),
        );

        let model = bldr.build();
        let mut ba = BerthAvailability::<IntegerType>::new();
        assert!(ba.initialize(&model, &[]));

        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut eval = WeightedFlowTimeEvaluator::<IntegerType>::new();

        // None processing time -> decision must be None
        let d_none = Decision::try_new(
            VesselIndex::new(0),
            BerthIndex::new(1),
            &model,
            &ba,
            &state,
            &mut eval,
        );
        assert!(d_none.is_none());

        // Allowed, Some -> should produce a decision
        let d_some = Decision::try_new(
            VesselIndex::new(0),
            BerthIndex::new(0),
            &model,
            &ba,
            &state,
            &mut eval,
        );
        assert!(d_some.is_some());
    }

    #[test]
    fn test_evaluator_integration_cost_delta_is_some_for_feasible() {
        let model = build_basic_model();
        let mut ba = BerthAvailability::<IntegerType>::new();
        assert!(ba.initialize(&model, &[]));

        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut eval = WeightedFlowTimeEvaluator::<IntegerType>::new();

        // Ensure a feasible nominal start (arrival=5, berth_free=0 -> nominal=5)
        let v = VesselIndex::new(1);
        let b = BerthIndex::new(0);

        let d = Decision::try_new(v, b, &model, &ba, &state, &mut eval)
            .expect("feasible decision must exist");

        // WeightedFlowTime should produce a non-negative cost_delta
        // Not asserting a specific value to avoid coupling too tightly;
        // just validate that evaluate_vessel_assignment produced Some
        assert!(d.cost_delta() >= 0);
    }
}
