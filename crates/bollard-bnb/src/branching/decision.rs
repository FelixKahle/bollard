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

use crate::{eval::evaluator::ObjectiveEvaluator, state::SearchState};
use bollard_core::num::constants::MinusOne;
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use num_traits::{PrimInt, Signed};
use std::iter::FusedIterator;

/// A distinct decision in the decision tree.
/// Represents assigning a specific vessel to a specific berth.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Decision {
    vessel_index: VesselIndex,
    berth_index: BerthIndex,
}

impl Decision {
    /// Creates a new decision.
    #[inline(always)]
    pub const fn new(vessel_index: VesselIndex, berth_index: BerthIndex) -> Self {
        Self {
            vessel_index,
            berth_index,
        }
    }

    /// Creates a new decision if it is valid in the given model and state.
    /// Returns `None` if the decision is not valid.
    ///
    /// # Panics
    ///
    /// In debug mode, this function will panic if `vessel_index` is not in bounds
    /// `0..model.num_vessels()` or if `berth_index` is
    /// not in bounds `0..model.num_berths()`.
    #[inline(always)]
    pub fn try_new<T>(
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        model: &Model<T>,
        state: &SearchState<T>,
    ) -> Option<Self>
    where
        T: PrimInt + Signed + MinusOne,
    {
        debug_assert!(
            vessel_index.get() < model.num_vessels(),
            "called `shared::is_valid_decision` with vessel index out of bounds: the len is {} but the index is {}",
            model.num_vessels(),
            vessel_index.get()
        );

        debug_assert!(
            berth_index.get() < model.num_berths(),
            "called `shared::is_valid_decision` with berth index out of bounds: the len is {} but the index is {}",
            model.num_berths(),
            berth_index.get()
        );

        if is_valid_decision(model, state, vessel_index, berth_index) {
            Some(Self::new(vessel_index, berth_index))
        } else {
            None
        }
    }

    /// Creates a new decision if it is valid in the given model and state.
    /// Returns `None` if the decision is not valid.
    ///
    /// # Panics
    ///
    /// In debug mode, this function will panic if `vessel_index` is not in bounds
    /// `0..model.num_vessels()` or if `berth_index` is
    /// not in bounds `0..model.num_berths()`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is in bounds
    /// `0..model.num_vessels()` and that `berth_index` is in bounds
    /// `0..model.num_berths()`.
    #[inline(always)]
    pub unsafe fn try_new_unchecked<T>(
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        model: &Model<T>,
        state: &SearchState<T>,
    ) -> Option<Self>
    where
        T: PrimInt + Signed + MinusOne,
    {
        debug_assert!(
            vessel_index.get() < model.num_vessels(),
            "called `shared::is_valid_decision_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            model.num_vessels(),
            vessel_index.get()
        );

        debug_assert!(
            berth_index.get() < model.num_berths(),
            "called `shared::is_valid_decision_unchecked` with berth index out of bounds: the len is {} but the index is {}",
            model.num_berths(),
            berth_index.get()
        );

        if unsafe { is_valid_decision_unchecked(model, state, vessel_index, berth_index) } {
            Some(Self::new(vessel_index, berth_index))
        } else {
            None
        }
    }

    #[inline(always)]
    pub const fn vessel_index(&self) -> VesselIndex {
        self.vessel_index
    }

    #[inline(always)]
    pub const fn berth_index(&self) -> BerthIndex {
        self.berth_index
    }

    #[inline(always)]
    pub const fn into_inner(self) -> (VesselIndex, BerthIndex) {
        (self.vessel_index, self.berth_index)
    }
}

impl std::fmt::Display for Decision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Decision(v: {}, b: {})",
            self.vessel_index, self.berth_index
        )
    }
}

/// A pluggable decision builder for search strategies.
pub trait DecisionBuilder<T, E>
where
    T: PrimInt + Signed,
    E: ObjectiveEvaluator<T>,
{
    /// The iterator type returned by `next_decision`.
    type DecisionIterator<'a>: Iterator<Item = Decision> + FusedIterator + 'a
    where
        Self: 'a,
        T: 'a,
        E: 'a;

    /// Returns the name of the decision builder.
    fn name(&self) -> &str;

    /// Returns an iterator over the next decisions to try.
    fn next_decision<'a>(
        &'a mut self,
        evaluator: &'a mut E,
        model: &'a Model<T>,
        search_state: &'a SearchState<T>,
    ) -> Self::DecisionIterator<'a>;
}

/// Checks whether assigning the given vessel to the given berth is a valid decision,
/// based on the model and the current search state.
///
/// # Panics
///
/// In debug mode, this function will panic if `vessel_index` is not in bounds
/// `0..model.num_vessels()` or if `berth_index` is not in bounds `0..model.num_berths()`.
#[inline(always)]
fn is_valid_decision<T>(
    model: &Model<T>,
    state: &SearchState<T>,
    vessel_index: VesselIndex,
    berth_index: BerthIndex,
) -> bool
where
    T: PrimInt + Signed + MinusOne,
{
    debug_assert!(
        vessel_index.get() < model.num_vessels(),
        "called `shared::is_valid_decision_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
        model.num_vessels(),
        vessel_index.get()
    );

    debug_assert!(
        berth_index.get() < model.num_berths(),
        "called `shared::is_valid_decision_unchecked` with berth index out of bounds: the len is {} but the index is {}",
        model.num_berths(),
        berth_index.get()
    );

    let berth_free = state.berth_free_time(berth_index);
    let arrival = model.vessel_arrival_time(vessel_index);
    let last_decision_time = state.last_decision_time();
    let last_decision_vessel = state.last_decision_vessel();

    if !model.vessel_allowed_on_berth(vessel_index, berth_index) {
        return false;
    }

    let start_time = if arrival > berth_free {
        arrival
    } else {
        berth_free
    };

    if start_time < last_decision_time {
        return false;
    }
    if start_time == last_decision_time && vessel_index.get() < last_decision_vessel.get() {
        return false;
    }

    if berth_index.get() > 0 {
        let previous_berth_index = BerthIndex::new(berth_index.get() - 1);
        let previous_free = state.berth_free_time(previous_berth_index);

        if previous_free == berth_free {
            let current_processing_time = model.vessel_processing_time(vessel_index, berth_index);
            let prev_proc = model.vessel_processing_time(vessel_index, previous_berth_index);

            if current_processing_time == prev_proc
                && model.vessel_allowed_on_berth(vessel_index, previous_berth_index)
            {
                return false;
            }
        }
    }

    true
}

/// Checks whether assigning the given vessel to the given berth is a valid decision,
/// based on the model and the current search state.
///
/// # Panics
///
/// In debug mode, this function will panic if `vessel_index` is not in bounds
/// `0..model.num_vessels()` or if `berth_index` is not in bounds `0..model.num_berths()`.
///
/// # Safety
///
/// The caller must ensure that `vessel_index` is in bounds
/// `0..model.num_vessels()` and that `berth_index` is in bounds `0..model.num_berths()`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn is_valid_decision_unchecked<T>(
    model: &Model<T>,
    state: &SearchState<T>,
    vessel_index: VesselIndex,
    berth_index: BerthIndex,
) -> bool
where
    T: PrimInt + Signed + MinusOne,
{
    debug_assert!(
        vessel_index.get() < model.num_vessels(),
        "called `shared::is_valid_decision_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
        model.num_vessels(),
        vessel_index.get()
    );

    debug_assert!(
        berth_index.get() < model.num_berths(),
        "called `shared::is_valid_decision_unchecked` with berth index out of bounds: the len is {} but the index is {}",
        model.num_berths(),
        berth_index.get()
    );

    let berth_free = unsafe { state.berth_free_time_unchecked(berth_index) };
    let arrival = unsafe { model.vessel_arrival_time_unchecked(vessel_index) };
    let last_decision_time = state.last_decision_time();
    let last_decision_vessel = state.last_decision_vessel();

    if unsafe { !model.vessel_allowed_on_berth_unchecked(vessel_index, berth_index) } {
        return false;
    }

    let start_time = if arrival > berth_free {
        arrival
    } else {
        berth_free
    };

    if start_time < last_decision_time {
        return false;
    }
    if start_time == last_decision_time && vessel_index.get() < last_decision_vessel.get() {
        return false;
    }

    if berth_index.get() > 0 {
        let previous_berth_index = BerthIndex::new(berth_index.get() - 1);
        let previous_free = unsafe { state.berth_free_time_unchecked(previous_berth_index) };

        if previous_free == berth_free {
            let current_processing_time =
                unsafe { model.vessel_processing_time_unchecked(vessel_index, berth_index) };
            let prev_proc = unsafe {
                model.vessel_processing_time_unchecked(vessel_index, previous_berth_index)
            };

            if current_processing_time == prev_proc
                && unsafe {
                    model.vessel_allowed_on_berth_unchecked(vessel_index, previous_berth_index)
                }
            {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod decision_tests {
    use super::Decision;
    use crate::state::SearchState;
    use bollard_model::{
        index::{BerthIndex, VesselIndex},
        model::ModelBuilder,
        time::ProcessingTime,
    };

    type T = i64;

    fn v(i: usize) -> VesselIndex {
        VesselIndex::new(i)
    }
    fn b(i: usize) -> BerthIndex {
        BerthIndex::new(i)
    }

    fn build_model(
        num_berths: usize,
        num_vessels: usize,
        allowed: impl Fn(VesselIndex, BerthIndex) -> bool,
        proc: impl Fn(VesselIndex, BerthIndex) -> T,
    ) -> bollard_model::model::Model<T> {
        let mut mb = ModelBuilder::<T>::new(num_berths, num_vessels);
        for i in 0..num_vessels {
            let vi = v(i);
            mb.set_vessel_arrival_time(vi, i as T);
            mb.set_vessel_weight(vi, 1);
            for j in 0..num_berths {
                let bj = b(j);
                if allowed(vi, bj) {
                    mb.set_vessel_processing_time(vi, bj, ProcessingTime::some(proc(vi, bj)));
                } else {
                    mb.set_vessel_processing_time(vi, bj, ProcessingTime::none());
                }
            }
        }
        mb.build()
    }

    #[test]
    fn test_try_new_rejects_disallowed_pair_and_accepts_allowed() {
        // Only (v0,b0) allowed; others disallowed
        let model = build_model(2, 2, |v, b| v.get() == 0 && b.get() == 0, |_v, _b| 5);
        let state: SearchState<T> = SearchState::new(model.num_berths(), model.num_vessels());

        // Disallowed => None
        assert!(
            Decision::try_new(v(1), b(1), &model, &state).is_none(),
            "disallowed vessel-berth should be rejected"
        );

        // Allowed => Some
        let d = Decision::try_new(v(0), b(0), &model, &state);
        assert!(d.is_some(), "allowed vessel-berth should be accepted");
        let (vi, bi) = d.unwrap().into_inner();
        assert_eq!(vi.get(), 0);
        assert_eq!(bi.get(), 0);
    }

    #[test]
    fn test_try_new_enforces_chronological_order_by_start_time() {
        // All allowed, identical processing times
        let mut mb = ModelBuilder::<T>::new(1, 2);
        // Explicit arrivals so we can test exact times
        mb.set_vessel_arrival_time(v(0), 0);
        mb.set_vessel_arrival_time(v(1), 5);
        // weights irrelevant here
        mb.set_vessel_weight(v(0), 1);
        mb.set_vessel_weight(v(1), 1);
        // processing times (any positive)
        mb.set_vessel_processing_time(v(0), b(0), ProcessingTime::some(5));
        mb.set_vessel_processing_time(v(1), b(0), ProcessingTime::some(5));
        let model = mb.build();

        let mut state: SearchState<T> = SearchState::new(model.num_berths(), model.num_vessels());

        // last_decision = (time=5, vessel=0)
        state.set_last_decision(5, v(0));
        // berth free at 0
        state.set_berth_free_time(b(0), 0);

        // arrival=1, berth_free=0 => start=1 < last_time => None
        // For vessel 1 we set arrival to 5; to test earlier, check vessel 0 on b0 with arrival 0 -> start 0 < 5
        assert!(
            Decision::try_new(v(0), b(0), &model, &state).is_none(),
            "start time earlier than last decision should be rejected"
        );

        // arrival=5 (v1), berth_free=0 => start=5 == last_time => Some
        assert!(
            Decision::try_new(v(1), b(0), &model, &state).is_some(),
            "equal start time should be allowed"
        );

        // arrival=10 (set explicitly), berth_free=0 => start=10 > last_time => Some
        let mut mb2 = ModelBuilder::<T>::new(1, 2);
        mb2.set_vessel_arrival_time(v(0), 0);
        mb2.set_vessel_arrival_time(v(1), 10);
        mb2.set_vessel_weight(v(0), 1);
        mb2.set_vessel_weight(v(1), 1);
        mb2.set_vessel_processing_time(v(0), b(0), ProcessingTime::some(5));
        mb2.set_vessel_processing_time(v(1), b(0), ProcessingTime::some(5));
        let model2 = mb2.build();

        state.set_last_decision(5, v(0)); // keep last time and vessel
        state.set_berth_free_time(b(0), 0);
        assert!(
            Decision::try_new(v(1), b(0), &model2, &state).is_some(),
            "greater start time should be allowed"
        );
    }

    #[test]
    fn test_try_new_tie_breaks_on_equal_times() {
        // Build model where arrivals produce equal start times at last_decision_time=10
        let mut mb = ModelBuilder::<T>::new(1, 4);
        // All arrivals set to 10 so start=max(arrival, berth_free)=10
        mb.set_vessel_arrival_time(v(0), 10);
        mb.set_vessel_arrival_time(v(1), 10);
        mb.set_vessel_arrival_time(v(2), 10);
        mb.set_vessel_arrival_time(v(3), 10);
        // weights irrelevant
        mb.set_vessel_weight(v(0), 1);
        mb.set_vessel_weight(v(1), 1);
        mb.set_vessel_weight(v(2), 1);
        mb.set_vessel_weight(v(3), 1);
        // processing times (any positive)
        mb.set_vessel_processing_time(v(0), b(0), ProcessingTime::some(3));
        mb.set_vessel_processing_time(v(1), b(0), ProcessingTime::some(3));
        mb.set_vessel_processing_time(v(2), b(0), ProcessingTime::some(3));
        mb.set_vessel_processing_time(v(3), b(0), ProcessingTime::some(3));
        let model = mb.build();

        let mut state: SearchState<T> = SearchState::new(model.num_berths(), model.num_vessels());

        // last_decision = (time=10, vessel=2), berth free at 0
        state.set_last_decision(10, v(2));
        state.set_berth_free_time(b(0), 0);

        // Equal time for all: arrival=10, berth_free=0 -> start=10
        // v(1) < v(2) => reject
        assert!(
            Decision::try_new(v(1), b(0), &model, &state).is_none(),
            "lower vessel index on equal time should be rejected"
        );

        // v(2) == v(2) => allowed
        assert!(
            Decision::try_new(v(2), b(0), &model, &state).is_some(),
            "same vessel index on equal time should be allowed"
        );

        // v(3) > v(2) => allowed
        assert!(
            Decision::try_new(v(3), b(0), &model, &state).is_some(),
            "higher vessel index on equal time should be allowed"
        );
    }

    #[test]
    fn test_try_new_applies_symmetry_breaking_on_equal_berth_free_and_processing() {
        // One vessel, two berths; allowed on both; identical processing time
        let model = build_model(2, 1, |_v, _b| true, |_v, _b| 7);
        let mut state: SearchState<T> = SearchState::new(model.num_berths(), model.num_vessels());

        // Both berths free at the same time
        state.set_berth_free_time(b(0), 0);
        state.set_berth_free_time(b(1), 0);
        state.set_last_decision(0, v(0));

        // First berth should be valid
        assert!(
            Decision::try_new(v(0), b(0), &model, &state).is_some(),
            "first symmetric berth should be allowed"
        );

        // Second berth is symmetric (same free time and processing) and previous berth allowed,
        // so it should be rejected
        assert!(
            Decision::try_new(v(0), b(1), &model, &state).is_none(),
            "second symmetric berth should be rejected"
        );
    }

    #[test]
    fn test_try_new_unchecked_matches_checked() {
        // Mixed allowed/disallowed and varied processing
        let model = build_model(
            3,
            3,
            |v, b| !(v.get() == 1 && b.get() == 0),
            |_v, b| {
                if b.get() == 0 {
                    5
                } else if b.get() == 1 {
                    5
                } else {
                    9
                }
            },
        );
        let mut state: SearchState<T> = SearchState::new(model.num_berths(), model.num_vessels());
        state.set_last_decision(0, v(0));

        // Checked and unchecked should agree on validity
        let cases = [
            (v(0), b(0)),
            (v(0), b(1)),
            (v(0), b(2)),
            (v(1), b(0)), // disallowed
            (v(1), b(1)),
            (v(2), b(2)),
        ];

        for (vi, bi) in cases {
            let checked = Decision::try_new(vi, bi, &model, &state);
            let unchecked = unsafe { Decision::try_new_unchecked(vi, bi, &model, &state) };
            assert_eq!(
                checked.is_some(),
                unchecked.is_some(),
                "checked vs unchecked validity should match for ({},{})",
                vi.get(),
                bi.get()
            );
            if checked.is_some() {
                assert_eq!(
                    checked.unwrap().into_inner(),
                    unchecked.unwrap().into_inner()
                );
            }
        }
    }
}
