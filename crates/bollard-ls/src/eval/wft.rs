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

use crate::eval::evaluator::{AssignmentEvaluator, Evaluation};
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use bollard_search::num::SolverNumeric;

#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WeightedFlowTimeEvaluator<T>
where
    T: SolverNumeric,
{
    _phantom: std::marker::PhantomData<T>,
}

impl<T> WeightedFlowTimeEvaluator<T>
where
    T: SolverNumeric,
{
    /// Creates a new WeightedFlowTimeEvaluator.
    #[inline]
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> std::fmt::Display for WeightedFlowTimeEvaluator<T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WeightedFlowTimeEvaluator")
    }
}

impl<T> Default for WeightedFlowTimeEvaluator<T>
where
    T: SolverNumeric,
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> WeightedFlowTimeEvaluator<T>
where
    T: SolverNumeric,
{
    /// Calculates the weighted flow time cost for a vessel assigned to a berth at a given start time.
    ///
    /// # Panics
    ///
    /// In debug builds, this function will panic if `vessel_index` is not within `0..model.num_vessels()`
    /// or if `berth_index` is not within `0..model.num_berths()`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is within `0..model.num_vessels()` and
    /// `berth_index` is within `0..model.num_berths()`.
    #[inline(always)]
    unsafe fn calculate_flow_time_cost(
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
    ) -> Option<T> {
        debug_assert!(
            vessel_index.get() < model.num_vessels(),
            "called `WeightedFlowTimeEvaluator::calculate_flow_time_cost` with vessel index out of bounds: the len is {} but the index is {}",
            model.num_vessels(),
            vessel_index.get()
        );

        debug_assert!(
            berth_index.get() < model.num_berths(),
            "called `WeightedFlowTimeEvaluator::calculate_flow_time_cost` with berth index out of bounds: the len is {} but the index is {}",
            model.num_berths(),
            berth_index.get()
        );

        let deadline = unsafe { model.vessel_latest_departure_time_unchecked(vessel_index) };
        let pt_opt = unsafe { model.vessel_processing_time_unchecked(vessel_index, berth_index) };

        if pt_opt.is_none() {
            return None;
        }

        let pt = pt_opt.unwrap_unchecked();
        let completion_time = start_time.saturating_add_val(pt);

        if completion_time > deadline {
            return None;
        }

        let weight = unsafe { model.vessel_weight_unchecked(vessel_index) };
        Some(completion_time.saturating_mul_val(weight))
    }
}

impl<T> AssignmentEvaluator<T> for WeightedFlowTimeEvaluator<T>
where
    T: SolverNumeric,
{
    fn name(&self) -> &str {
        "WeightedFlowTimeEvaluator"
    }

    fn evaluate(
        &self,
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
    ) -> Option<Evaluation<T>> {
        assert!(
            vessel_index.get() < model.num_vessels(),
            "called `WeightedFlowTimeEvaluator::evaluate` with vessel index out of bounds: the len is {} but the index is {}",
            model.num_vessels(),
            vessel_index.get()
        );

        assert!(
            berth_index.get() < model.num_berths(),
            "called `WeightedFlowTimeEvaluator::evaluate` with berth index out of bounds: the len is {} but the index is {}",
            model.num_berths(),
            berth_index.get()
        );

        unsafe { self.evaluate_unchecked(model, vessel_index, berth_index, start_time) }
    }

    unsafe fn evaluate_unchecked(
        &self,
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
    ) -> Option<Evaluation<T>> {
        let cost = unsafe {
            Self::calculate_flow_time_cost(model, vessel_index, berth_index, start_time)?
        };

        Some(Evaluation {
            score: cost,
            objective_delta: cost,
        })
    }
}
