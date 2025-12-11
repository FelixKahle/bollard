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
    branching::decision::{Decision, DecisionBuilder},
    eval::evaluator::ObjectiveEvaluator,
    state::SearchState,
};
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use bollard_search::num::SolverNumeric;
use std::iter::FusedIterator;

/// A decision builder that implements a **Dynamic Weighted Shortest Processing Time (WSPT)** heuristic.
///
/// Instead of exploring branches in arbitrary order, this builder:
/// 1. Generates all feasible assignments for the current state.
/// 2. Calculates the immediate objective increase (weighted completion time) for each.
/// 3. Sorts branches so that the "cheapest" moves are explored first.
///
/// This "Best-First" strategy helps the solver find high-quality incumbents early,
/// maximizing the effectiveness of bound-based pruning.
#[derive(Debug, Clone, Default)]
pub struct WsptHeuristicBuilder<T> {
    candidates: Vec<(T, Decision)>,
}

impl<T> WsptHeuristicBuilder<T> {
    /// Creates a new `WsptHeuristicBuilder` with an empty candidate buffer.
    #[inline]
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
        }
    }

    /// Creates a new `WsptHeuristicBuilder` with a preallocated candidate buffer.
    #[inline]
    pub fn preallocated(num_berths: usize, num_vessels: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(num_berths * num_vessels),
        }
    }

    /// Creates a new `WsptHeuristicBuilder` with a candidate buffer of the specified capacity.
    /// You may use `WsptHeuristicBuilder::preallocated` for a more specific preallocation based
    /// on the number of berths and vessels.
    #[inline]
    pub fn with_capacity(size: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(size),
        }
    }
}

impl<T, E> DecisionBuilder<T, E> for WsptHeuristicBuilder<T>
where
    T: SolverNumeric,
    E: ObjectiveEvaluator<T>,
{
    type DecisionIterator<'a>
        = CostGuidedIter<'a, T>
    where
        T: 'a,
        E: 'a,
        Self: 'a;

    fn name(&self) -> &str {
        "CostGuidedBuilder"
    }

    fn next_decision<'a>(
        &'a mut self,
        evaluator: &'a mut E,
        model: &'a Model<T>,
        state: &'a SearchState<T>,
    ) -> Self::DecisionIterator<'a> {
        self.candidates.clear();

        let num_vessels = model.num_vessels();
        let num_berths = model.num_berths();

        for v in 0..num_vessels {
            let vessel_index = VesselIndex::new(v);

            if unsafe { state.is_vessel_assigned_unchecked(vessel_index) } {
                continue;
            }

            for b in 0..num_berths {
                let berth_index = BerthIndex::new(b);

                if let Some(decision) =
                    unsafe { Decision::try_new_unchecked(vessel_index, berth_index, model, state) }
                {
                    let berth_free_time = unsafe { state.berth_free_time_unchecked(berth_index) };

                    if let Some(cost) = evaluator.evaluate_vessel_assignment(
                        model,
                        vessel_index,
                        berth_index,
                        berth_free_time,
                    ) {
                        // Push to our scratch buffer
                        self.candidates.push((cost, decision));
                    }
                }
            }
        }

        self.candidates
            .sort_unstable_by(|(cost_a, _), (cost_b, _)| cost_a.cmp(cost_b));
        CostGuidedIter {
            iter: self.candidates.iter(),
        }
    }
}

/// A lightweight iterator wrapper that yields decisions from the slice.
/// It holds a reference to the builder's scratch buffer.
pub struct CostGuidedIter<'a, T> {
    iter: std::slice::Iter<'a, (T, Decision)>,
}

impl<'a, T> Iterator for CostGuidedIter<'a, T>
where
    T: Copy,
{
    type Item = Decision;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        // Map the reference (T, Decision) to just the Decision copy.
        // This compiles to a simple pointer increment.
        self.iter.next().map(|(_, decision)| *decision)
    }
}

impl<'a, T> FusedIterator for CostGuidedIter<'a, T> where T: Copy {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::evaluator::ObjectiveEvaluator;
    use crate::eval::wtft::WeightedFlowTimeEvaluator;
    use bollard_model::{
        index::{BerthIndex, VesselIndex},
        model::ModelBuilder,
        time::ProcessingTime,
    };

    type IntegerType = i64;

    // Build a small model:
    // - 3 vessels, 2 berths
    // - All arrivals at 0, deadlines very large
    // - Weights vary to make costs distinct
    // - Processing times vary across berths to produce distinct immediate costs
    fn build_small_model() -> bollard_model::model::Model<IntegerType> {
        let mut b = ModelBuilder::<IntegerType>::new(2, 3);
        // Vessel 0: weight 3
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(0), IntegerType::MAX)
            .set_vessel_weight(VesselIndex::new(0), 3)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(10), // cost ~ (start+10)*3
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::some(2), // cheaper
            );

        // Vessel 1: weight 2
        b.set_vessel_arrival_time(VesselIndex::new(1), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(1), IntegerType::MAX)
            .set_vessel_weight(VesselIndex::new(1), 2)
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(0),
                ProcessingTime::some(5),
            )
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(1),
                ProcessingTime::some(7),
            );

        // Vessel 2: make it infeasible on b0 (None), feasible on b1
        b.set_vessel_arrival_time(VesselIndex::new(2), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(2), IntegerType::MAX)
            .set_vessel_weight(VesselIndex::new(2), 1)
            .set_vessel_processing_time(
                VesselIndex::new(2),
                BerthIndex::new(0),
                ProcessingTime::none(), // infeasible on b0
            )
            .set_vessel_processing_time(
                VesselIndex::new(2),
                BerthIndex::new(1),
                ProcessingTime::some(4),
            );

        b.build()
    }

    #[test]
    fn test_wspt_orders_by_increasing_cost() {
        let model = build_small_model();
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = WsptHeuristicBuilder::<IntegerType>::new();

        // Generate decisions
        let mut iter = builder.next_decision(&mut evaluator, &model, &state);
        let decisions: Vec<Decision> = iter.by_ref().collect();
        assert!(
            !decisions.is_empty(),
            "Expected at least one feasible decision"
        );

        // Compute costs for the yielded decisions using the evaluator and the current berth free time
        let observed_costs: Vec<IntegerType> = decisions
            .iter()
            .map(|d| {
                let v = d.vessel_index();
                let b = d.berth_index();
                let ready = unsafe { state.berth_free_time_unchecked(b) };
                evaluator
                    .evaluate_vessel_assignment(&model, v, b, ready)
                    .expect("decision should be feasible and evaluable")
            })
            .collect();

        // Ensure non-decreasing costs (sorted ascending)
        let mut sorted = observed_costs.clone();
        sorted.sort_unstable();
        assert_eq!(
            observed_costs, sorted,
            "WSPT builder must yield decisions ordered by increasing immediate cost"
        );

        // Sanity-check a plausible prefix given the model setup:
        // - v0,b1 has short proc 2 with weight 3 => cheap
        // - v1,b0 has proc 5 with weight 2 => moderate
        // - v2,b1 has proc 4 with weight 1 => also moderate; exact order depends on weight*finish
        // We ensure infeasible (v2,b0) is not present.
        assert!(
            !decisions
                .iter()
                .any(|d| d.vessel_index().get() == 2 && d.berth_index().get() == 0),
            "Infeasible pair (v2,b0) must not be yielded"
        );
    }

    #[test]
    fn test_wspt_empty_when_no_feasible_candidates() {
        // One vessel, one berth, infeasible processing time everywhere
        let mut b = ModelBuilder::<IntegerType>::new(1, 1);
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(0), IntegerType::MAX)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::none(), // no feasible processing time
            );
        let model = b.build();

        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = WsptHeuristicBuilder::<IntegerType>::new();

        let mut iter = builder.next_decision(&mut evaluator, &model, &state);
        assert_eq!(
            iter.next(),
            None,
            "Iterator should be empty when no feasible assignments exist"
        );
    }
}
