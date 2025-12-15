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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Decision<T> {
    start_time: T,
    cost_delta: T,
    vessel_index: VesselIndex,
    berth_index: BerthIndex,
}

impl<T: Ord> Ord for Decision<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.vessel_index.cmp(&other.vessel_index) {
            std::cmp::Ordering::Equal => self.berth_index.cmp(&other.berth_index),
            ord => ord,
        }
    }
}

impl<T: Ord> PartialOrd for Decision<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Decision<T>
where
    T: SolverNumeric,
{
    #[inline(always)]
    pub const fn new(
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
        cost_delta: T,
    ) -> Self {
        Self {
            vessel_index,
            berth_index,
            start_time,
            cost_delta,
        }
    }

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

        if !model.vessel_allowed_on_berth(vessel_index, berth_index) {
            return None;
        }

        let processing_time_option = model.vessel_processing_time(vessel_index, berth_index);
        if processing_time_option.is_none() {
            return None;
        }
        let pt = processing_time_option.unwrap_unchecked();
        let berth_free = state.berth_free_time(berth_index);
        let arrival = model.vessel_arrival_time(vessel_index);

        let nominal_start = if arrival > berth_free {
            arrival
        } else {
            berth_free
        };

        let actual_start_time =
            berth_availability.earliest_availability(berth_index, nominal_start, pt)?;

        let last_decision_time = state.last_decision_time();
        if actual_start_time < last_decision_time {
            return None;
        }

        if actual_start_time == last_decision_time
            && vessel_index.get() < state.last_decision_vessel().get()
        {
            return None;
        }

        if berth_index.get() > 0 {
            let prev_berth_idx = BerthIndex::new(berth_index.get() - 1);
            let prev_free = state.berth_free_time(prev_berth_idx);

            if prev_free == berth_free {
                let prev_actual_start =
                    berth_availability.earliest_availability(prev_berth_idx, nominal_start, pt);

                if prev_actual_start == Some(actual_start_time) {
                    let prev_proc = model
                        .vessel_processing_time(vessel_index, prev_berth_idx)
                        .unwrap_or_else(T::zero);

                    if pt == prev_proc
                        && model.vessel_allowed_on_berth(vessel_index, prev_berth_idx)
                    {
                        return None;
                    }
                }
            }
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

        let processing_time_option =
            unsafe { model.vessel_processing_time_unchecked(vessel_index, berth_index) };

        if processing_time_option.is_none() {
            return None;
        }

        let pt = processing_time_option.unwrap_unchecked();
        let berth_free = unsafe { state.berth_free_time_unchecked(berth_index) };
        let arrival = unsafe { model.vessel_arrival_time_unchecked(vessel_index) };

        let nominal_start = if arrival > berth_free {
            arrival
        } else {
            berth_free
        };

        let actual_start_time = unsafe {
            berth_availability.earliest_availability_unchecked(berth_index, nominal_start, pt)?
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

        if berth_index.get() > 0 {
            let prev_berth_idx = BerthIndex::new(berth_index.get() - 1);
            let prev_free = unsafe { state.berth_free_time_unchecked(prev_berth_idx) };

            if prev_free == berth_free {
                let prev_actual_start = unsafe {
                    berth_availability.earliest_availability_unchecked(
                        prev_berth_idx,
                        nominal_start,
                        pt,
                    )
                };

                if prev_actual_start == Some(actual_start_time) {
                    let prev_proc = unsafe {
                        model
                            .vessel_processing_time_unchecked(vessel_index, prev_berth_idx)
                            .unwrap_or_else(T::zero)
                    };

                    if pt == prev_proc
                        && unsafe {
                            model.vessel_allowed_on_berth_unchecked(vessel_index, prev_berth_idx)
                        }
                    {
                        return None;
                    }
                }
            }
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

    #[inline(always)]
    pub const fn vessel_index(&self) -> VesselIndex {
        self.vessel_index
    }

    #[inline(always)]
    pub const fn berth_index(&self) -> BerthIndex {
        self.berth_index
    }

    #[inline(always)]
    pub const fn start_time(&self) -> T {
        self.start_time
    }

    #[inline(always)]
    pub const fn cost_delta(&self) -> T {
        self.cost_delta
    }

    #[inline(always)]
    pub const fn into_inner(self) -> (VesselIndex, BerthIndex) {
        (self.vessel_index, self.berth_index)
    }
}

impl<T: std::fmt::Display> std::fmt::Display for Decision<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Decision(v: {}, b: {}, start: {}, cost: {})",
            self.vessel_index, self.berth_index, self.start_time, self.cost_delta
        )
    }
}

pub trait DecisionBuilder<T, E>
where
    T: PrimInt + Signed,
    E: ObjectiveEvaluator<T>,
{
    type DecisionIterator<'a>: Iterator<Item = Decision<T>> + FusedIterator + 'a
    where
        Self: 'a,
        T: 'a,
        E: 'a;

    fn name(&self) -> &str;

    fn next_decision<'a>(
        &'a mut self,
        evaluator: &'a mut E,
        model: &'a Model<T>,
        berth_availability: &'a BerthAvailability<T>,
        search_state: &'a SearchState<T>,
    ) -> Self::DecisionIterator<'a>;
}
