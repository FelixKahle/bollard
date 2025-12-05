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
use bollard_core::num::{constants::MinusOne, ops::saturating_arithmetic};
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use num_traits::{PrimInt, Signed};

#[allow(dead_code)]
pub trait ObjectiveEvaluator<T>
where
    T: PrimInt + Signed,
{
    /// Returns the name of the objective evaluator.
    fn name(&self) -> &str;

    /// Evaluates the cost of assigning a vessel to a berth starting at a given time.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is not within `0..model.num_vessels()` or if
    /// `berth_index` is not within `0..model.num_berths()`.
    fn evaluate_vessel_assignment(
        &self,
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        berth_ready: T,
    ) -> T
    where
        T: MinusOne
            + saturating_arithmetic::SaturatingAddVal
            + saturating_arithmetic::SaturatingMulVal;

    /// Evaluates the cost of assigning a vessel to a berth starting at a given time
    /// without performing bounds checking on the indices.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is within `0..model.num_vessels()`
    /// and that `berth_index` is within `0..model.num_berths()`.
    unsafe fn evaluate_vessel_assignment_unchecked(
        &self,
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        berth_ready: T,
    ) -> T
    where
        T: MinusOne
            + saturating_arithmetic::SaturatingAddVal
            + saturating_arithmetic::SaturatingMulVal;

    fn lower_bound(&self, model: &Model<T>, state: &SearchState<T>) -> T
    where
        T: MinusOne
            + saturating_arithmetic::SaturatingAddVal
            + saturating_arithmetic::SaturatingMulVal;
}

impl<T> std::fmt::Debug for dyn ObjectiveEvaluator<T>
where
    T: PrimInt + Signed,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ObjectiveEvaluator({})", self.name())
    }
}

impl<T> std::fmt::Display for dyn ObjectiveEvaluator<T>
where
    T: PrimInt + Signed,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ObjectiveEvaluator({})", self.name())
    }
}

pub struct WeightedFlowTimeEvaluator<T>
where
    T: PrimInt + Signed,
{
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for WeightedFlowTimeEvaluator<T>
where
    T: PrimInt + Signed,
 {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> WeightedFlowTimeEvaluator<T>
where
    T: PrimInt + Signed,
{
    #[inline]
    pub const fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> ObjectiveEvaluator<T> for WeightedFlowTimeEvaluator<T>
where
    T: PrimInt + Signed,
{
    #[inline]
    fn name(&self) -> &str {
        "WeightedFlowTimeEvaluator"
    }

    #[inline]
    fn evaluate_vessel_assignment(
        &self,
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        berth_ready: T,
    ) -> T
    where
        T: MinusOne
            + saturating_arithmetic::SaturatingAddVal
            + saturating_arithmetic::SaturatingMulVal,
    {
        let arrival_time = model.arrival_time(vessel_index);
        let weight = model.vessel_weight(vessel_index);
        let opt_pt = model.processing_time(vessel_index, berth_index);
        if opt_pt.is_none() {
            return T::max_value();
        }
        let processing_time = opt_pt.unwrap();
        let effective_start = if arrival_time > berth_ready {
            arrival_time
        } else {
            berth_ready
        };
        let finish_time = effective_start.saturating_add_val(processing_time);
        finish_time.saturating_mul_val(weight)
    }

    unsafe fn evaluate_vessel_assignment_unchecked(
        &self,
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        berth_ready: T,
    ) -> T
    where
        T: MinusOne
            + saturating_arithmetic::SaturatingAddVal
            + saturating_arithmetic::SaturatingMulVal,
    {
        let arrival_time = unsafe { model.arrival_time_unchecked(vessel_index) };
        let weight = unsafe { model.vessel_weight_unchecked(vessel_index) };
        let opt_pt = unsafe { model.processing_time_unchecked(vessel_index, berth_index) };
        if opt_pt.is_none() {
            return T::max_value();
        }
        let processing_time = opt_pt.unwrap();

        let effective_start = if arrival_time > berth_ready {
            arrival_time
        } else {
            berth_ready
        };
        let finish_time = effective_start.saturating_add_val(processing_time);
        finish_time.saturating_mul_val(weight)
    }

    #[inline]
    fn lower_bound(&self, model: &Model<T>, state: &SearchState<T>) -> T
    where
        T: MinusOne
            + saturating_arithmetic::SaturatingAddVal
            + saturating_arithmetic::SaturatingMulVal,
    {
        let mut estimated_cost = state.current_objective();

        for i in 0..model.num_vessels() {
            let vessel_index = VesselIndex::new(i);

            if unsafe { state.is_vessel_assigned_unchecked(vessel_index) } {
                continue;
            }

            let arrival_time = unsafe { model.arrival_time_unchecked(vessel_index) };
            let weight = unsafe { model.vessel_weight_unchecked(vessel_index) };
            let mut min_vessel_cost = T::max_value();
            let mut feasible_berth_found = false;

            for b in 0..model.num_berths() {
                let berth_index = BerthIndex::new(b);

                let processing_time =
                    unsafe { model.processing_time_unchecked(vessel_index, berth_index) };

                if processing_time.is_some() {
                    let duration = processing_time.unwrap();
                    feasible_berth_found = true;

                    let berth_ready = unsafe { state.berth_free_time_unchecked(berth_index) };
                    let start_time = if arrival_time > berth_ready {
                        arrival_time
                    } else {
                        berth_ready
                    };

                    let finish_time = start_time.saturating_add_val(duration);
                    let cost = finish_time.saturating_mul_val(weight);

                    if cost < min_vessel_cost {
                        min_vessel_cost = cost;
                    }
                }
            }

            if !feasible_berth_found {
                return T::max_value();
            }
            estimated_cost = estimated_cost.saturating_add_val(min_vessel_cost);
        }

        estimated_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bollard_model::{model::ModelBuilder, time::ProcessingTime};

    type IntegerType = i64;

    fn build_model_waiting() -> Model<IntegerType> {
        // 2 berths, 2 vessels
        let mut builder = ModelBuilder::<IntegerType>::new(2, 2);
        // arrivals
        builder.set_arrival_time(VesselIndex::new(0), 5); // vessel 0 arrives at t=5
        builder.set_arrival_time(VesselIndex::new(1), 2); // vessel 1 arrives at t=2
        // weights
        builder.set_vessel_weight(VesselIndex::new(0), 3);
        builder.set_vessel_weight(VesselIndex::new(1), 4);
        // processing times
        builder.set_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(0),
            ProcessingTime::some(7),
        );
        builder.set_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(1),
            ProcessingTime::some(3),
        );
        builder.set_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(0),
            ProcessingTime::none(),
        ); // infeasible on berth 0
        builder.set_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(1),
            ProcessingTime::some(6),
        );
        builder.build()
    }

    #[test]
    fn test_name_and_display_debug() {
        let eval = WeightedFlowTimeEvaluator::<IntegerType>::new();
        assert_eq!(eval.name(), "WeightedFlowTimeEvaluator");

        let dbg = format!("{:?}", &eval as &dyn ObjectiveEvaluator<IntegerType>);
        let disp = format!("{}", &eval as &dyn ObjectiveEvaluator<IntegerType>);
        assert!(dbg.contains("ObjectiveEvaluator(WeightedFlowTimeEvaluator)"));
        assert!(disp.contains("ObjectiveEvaluator(WeightedFlowTimeEvaluator)"));
    }

    #[test]
    fn test_evaluate_respects_waiting_time() {
        let model = build_model_waiting();
        let eval = WeightedFlowTimeEvaluator::<IntegerType>::new();

        // Vessel 0, Berth 0:
        // arrival = 5, proc = 7, weight = 3
        // Case A: berth_ready = 4 => effective_start = max(5,4) = 5
        // finish = 5 + 7 = 12 => cost = 12 * 3 = 36
        let cost_a =
            eval.evaluate_vessel_assignment(&model, VesselIndex::new(0), BerthIndex::new(0), 4);
        assert_eq!(cost_a, 36);

        // Case B: berth_ready = 10 => effective_start = max(5,10) = 10
        // finish = 10 + 7 = 17 => cost = 17 * 3 = 51
        let cost_b =
            eval.evaluate_vessel_assignment(&model, VesselIndex::new(0), BerthIndex::new(0), 10);
        assert_eq!(cost_b, 51);

        // Vessel 0, Berth 1:
        // arrival = 5, proc = 3, weight = 3
        // berth_ready = 0 => start=max(5,0)=5, finish=8, cost=8*3=24
        let cost_c =
            eval.evaluate_vessel_assignment(&model, VesselIndex::new(0), BerthIndex::new(1), 0);
        assert_eq!(cost_c, 24);
    }

    #[test]
    fn test_evaluate_infeasible_returns_max_value() {
        let model = build_model_waiting();
        let eval = WeightedFlowTimeEvaluator::<IntegerType>::new();

        // Vessel 1 on berth 0 is infeasible (None processing time) => max_value
        let cost =
            eval.evaluate_vessel_assignment(&model, VesselIndex::new(1), BerthIndex::new(0), 3);
        assert_eq!(cost, IntegerType::max_value());
    }

    #[test]
    fn test_evaluate_unchecked_matches_checked_and_waiting() {
        let model = build_model_waiting();
        let eval = WeightedFlowTimeEvaluator::<IntegerType>::new();

        // Feasible case parity
        let checked = eval.evaluate_vessel_assignment(
            &model,
            VesselIndex::new(0),
            BerthIndex::new(1),
            9, // max(arrival=5, berth_ready=9)=9, finish=9+3=12, cost=12*3=36
        );
        let unchecked = unsafe {
            eval.evaluate_vessel_assignment_unchecked(
                &model,
                VesselIndex::new(0),
                BerthIndex::new(1),
                9,
            )
        };
        assert_eq!(checked, unchecked);
        assert_eq!(unchecked, 36);

        // Infeasible parity
        let checked_inf =
            eval.evaluate_vessel_assignment(&model, VesselIndex::new(1), BerthIndex::new(0), 0);
        let unchecked_inf = unsafe {
            eval.evaluate_vessel_assignment_unchecked(
                &model,
                VesselIndex::new(1),
                BerthIndex::new(0),
                0,
            )
        };
        assert_eq!(checked_inf, IntegerType::max_value());
        assert_eq!(unchecked_inf, IntegerType::max_value());
    }

    #[test]
    fn test_lower_bound_respects_waiting_and_feasibility() {
        let model = build_model_waiting();
        let eval = WeightedFlowTimeEvaluator::<IntegerType>::new();

        // State: 2 berths, both free at t=0, objective=0, all vessels unassigned
        let mut state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());

        // Vessel 0:
        //  - berth 0: start=max(arrival=5, ready=0)=5, finish=5+7=12, cost=12*3=36
        //  - berth 1: start=max(5,0)=5, finish=5+3=8, cost=8*3=24 -> min=24
        //
        // Vessel 1:
        //  - berth 0 infeasible
        //  - berth 1: start=max(arrival=2, ready=0)=2, finish=2+6=8, cost=8*4=32
        //
        // LB = 0 + 24 + 32 = 56
        let lb = eval.lower_bound(&model, &state);
        assert_eq!(lb, 56);

        // Assign vessel 0; LB should only consider vessel 1 now => 32
        state.assign_vessel(VesselIndex::new(0));

        let lb_after_assign = eval.lower_bound(&model, &state);
        assert_eq!(lb_after_assign, 32);

        // Make vessel 1 infeasible on its only berth => LB becomes max_value
        let mut builder = ModelBuilder::<IntegerType>::new(2, 2);
        builder.set_arrival_time(VesselIndex::new(0), 5);
        builder.set_arrival_time(VesselIndex::new(1), 2);
        builder.set_vessel_weight(VesselIndex::new(0), 3);
        builder.set_vessel_weight(VesselIndex::new(1), 4);
        builder.set_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(0),
            ProcessingTime::some(7),
        );
        builder.set_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(1),
            ProcessingTime::some(3),
        );
        builder.set_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(0),
            ProcessingTime::none(),
        );
        builder.set_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(1),
            ProcessingTime::none(),
        );
        let model_inf = builder.build();

        state.reset();
        let lb_inf = eval.lower_bound(&model_inf, &state);
        assert_eq!(lb_inf, IntegerType::max_value());
    }
}
