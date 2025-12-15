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
    berth_availability::BerthAvailability,
    branching::decision::{Decision, DecisionBuilder},
    eval::evaluator::ObjectiveEvaluator,
    state::SearchState,
};
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use bollard_search::num::SolverNumeric;
use num_traits::{PrimInt, Signed};
use std::iter::FusedIterator;

/// Internal candidate structure for FCFS (First-Come-First-Serve) sorting.
///
/// Orders candidates primarily by **Arrival Time**, then by **Cost** (using the
/// pre-calculated delta in `Decision`), then by **Decision** (indices) for
/// deterministic tie-breaking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FcfsCandidate<T> {
    arrival_time: T,
    // The decision carries the cost_delta, so we don't need a separate field.
    decision: Decision<T>,
}

impl<T> FcfsCandidate<T> {
    /// Creates a new `FcfsCandidate`.
    #[inline(always)]
    fn new(arrival_time: T, decision: Decision<T>) -> Self {
        Self {
            arrival_time,
            decision,
        }
    }
}

impl<T> PartialOrd for FcfsCandidate<T>
where
    T: SolverNumeric,
{
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for FcfsCandidate<T>
where
    T: SolverNumeric,
{
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.arrival_time
            .cmp(&other.arrival_time)
            // Secondary sort: Cost (Greedy preference for cheaper moves among equal arrivals)
            .then_with(|| self.decision.cost_delta().cmp(&other.decision.cost_delta()))
            // Tertiary sort: Deterministic indices
            .then_with(|| self.decision.cmp(&other.decision))
    }
}

impl<T> std::fmt::Display for FcfsCandidate<T>
where
    T: std::fmt::Display + PrimInt + Signed,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Arrival: {}, Decision: {}",
            self.arrival_time, self.decision
        )
    }
}

/// A decision builder that implements a **First-Come-First-Serve (FCFS)** branching heuristic.
///
/// This builder prioritizes vessels by their arrival time. For each unassigned vessel:
/// 1. It generates fully computed `Decision<T>` objects (validating availability and calculating cost).
/// 2. It collects all feasible decisions into a buffer.
/// 3. It sorts them by:
///    - **Arrival Time** (ascending)
///    - **Cost Delta** (ascending)
///    - **Indices** (deterministic tie-break)
///
/// This strategy enforces a time-first exploration order, which can be useful for schedules
/// where respecting arrival chronology improves feasibility.
#[derive(Debug, Clone, Default)]
pub struct FcfsHeuristicBuilder<T> {
    candidates: Vec<FcfsCandidate<T>>,
}

impl<T> FcfsHeuristicBuilder<T> {
    /// Creates a new `FcfsHeuristicBuilder` with an empty candidate buffer.
    #[inline]
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
        }
    }

    /// Creates a new `FcfsHeuristicBuilder` with preallocated capacity.
    #[inline]
    pub fn preallocated(num_berths: usize, num_vessels: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(num_berths * num_vessels),
        }
    }

    /// Creates a new `FcfsHeuristicBuilder` with specific capacity.
    #[inline]
    pub fn with_capacity(size: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(size),
        }
    }
}

impl<T, E> DecisionBuilder<T, E> for FcfsHeuristicBuilder<T>
where
    T: SolverNumeric,
    E: ObjectiveEvaluator<T>,
{
    type DecisionIterator<'a>
        = FcfsHeuristicIter<'a, T>
    where
        T: 'a,
        E: 'a,
        Self: 'a;

    fn name(&self) -> &str {
        "FirstComeFirstServeBuilder"
    }

    fn next_decision<'a>(
        &'a mut self,
        evaluator: &'a mut E,
        model: &'a Model<T>,
        berth_availability: &'a BerthAvailability<T>,
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

            let arrival_time = unsafe { model.vessel_arrival_time_unchecked(vessel_index) };

            for b in 0..num_berths {
                let berth_index = BerthIndex::new(b);

                // Use the Rich Decision pipeline.
                // This validates feasibility, calculates the actual start time (with availability),
                // and computes the cost delta.
                if let Some(decision) = unsafe {
                    Decision::try_new_unchecked(
                        vessel_index,
                        berth_index,
                        model,
                        berth_availability,
                        state,
                        evaluator,
                    )
                } {
                    self.candidates
                        .push(FcfsCandidate::new(arrival_time, decision));
                }
            }
        }

        // Sort to enforce FCFS order
        self.candidates.sort_unstable();

        FcfsHeuristicIter {
            iter: self.candidates.iter(),
        }
    }
}

/// A lightweight iterator that yields `Decision`s from the FCFS-sorted candidate slice.
pub struct FcfsHeuristicIter<'a, T> {
    iter: std::slice::Iter<'a, FcfsCandidate<T>>,
}

impl<'a, T> Iterator for FcfsHeuristicIter<'a, T>
where
    T: Copy,
{
    type Item = Decision<T>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|c| c.decision)
    }
}

impl<'a, T> FusedIterator for FcfsHeuristicIter<'a, T> where T: Copy {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::wtft::WeightedFlowTimeEvaluator;
    use bollard_model::{
        index::{BerthIndex, VesselIndex},
        model::ModelBuilder,
        time::ProcessingTime,
    };

    type IntegerType = i64;

    // Helper to build a model where FCFS and WSPT would disagree.
    //
    // Vessel 0: Arrives at 0 (Early). Processing: 100 (Expensive).
    // Vessel 1: Arrives at 50 (Late). Processing: 1 (Cheap).
    //
    // WSPT (Cost-based) would prefer V1 because it's cheaper/shorter.
    // FCFS (Time-based) must prefer V0 because it arrived earlier.
    fn build_fcfs_test_model() -> bollard_model::model::Model<IntegerType> {
        let mut b = ModelBuilder::<IntegerType>::new(1, 2);

        // Vessel 0: Early Arrival, Heavy Weight/Processing
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(100),
            );

        // Vessel 1: Late Arrival, Light Weight/Processing
        b.set_vessel_arrival_time(VesselIndex::new(1), 50)
            .set_vessel_weight(VesselIndex::new(1), 1)
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(0),
                ProcessingTime::some(1),
            );

        b.build()
    }

    #[test]
    fn test_fcfs_orders_by_arrival_time() {
        let model = build_fcfs_test_model();
        let mut berth_availability = BerthAvailability::new();
        berth_availability.initialize(&model, &[]);
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = FcfsHeuristicBuilder::<IntegerType>::new();

        let mut iter = builder.next_decision(&mut evaluator, &model, &berth_availability, &state);

        // Expect Vessel 0 first (Arrival 0), even though it has high cost
        let first = iter.next().expect("Should have a decision");
        assert_eq!(
            first.vessel_index().get(),
            0,
            "FCFS should pick Vessel 0 first (Arrival 0)"
        );

        // Expect Vessel 1 second (Arrival 50)
        let second = iter.next().expect("Should have a second decision");
        assert_eq!(
            second.vessel_index().get(),
            1,
            "FCFS should pick Vessel 1 second (Arrival 50)"
        );
    }

    #[test]
    fn test_fcfs_ties_broken_by_cost() {
        // Two vessels arrive at same time.
        // V0: Costly. V1: Cheap.
        // FCFS should defer to cost (greedy) when arrivals match.
        let mut b = ModelBuilder::<IntegerType>::new(1, 2);

        b.set_vessel_arrival_time(VesselIndex::new(0), 10)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(100),
            ); // Cost ~ 110

        b.set_vessel_arrival_time(VesselIndex::new(1), 10)
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(0),
                ProcessingTime::some(10),
            ); // Cost ~ 20

        let model = b.build();
        let mut berth_availability = BerthAvailability::new();
        berth_availability.initialize(&model, &[]);
        let state = SearchState::<IntegerType>::new(1, 2);
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = FcfsHeuristicBuilder::<IntegerType>::new();

        let mut iter = builder.next_decision(&mut evaluator, &model, &berth_availability, &state);

        let first = iter.next().expect("decision");
        assert_eq!(
            first.vessel_index().get(),
            1,
            "Should pick V1 first because arrivals are equal but V1 is cheaper"
        );
    }

    #[test]
    fn test_fcfs_excludes_infeasible_assignments_and_empty_when_none() {
        // One vessel, two berths; all infeasible processing times
        let mut b = ModelBuilder::<IntegerType>::new(2, 1);
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::none(),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::none(),
            );
        let model = b.build();
        let mut berth_availability = BerthAvailability::new();
        berth_availability.initialize(&model, &[]);
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = FcfsHeuristicBuilder::<IntegerType>::new();

        let mut iter = builder.next_decision(&mut evaluator, &model, &berth_availability, &state);
        assert_eq!(
            iter.next(),
            None,
            "Iterator should be empty when all assignments are infeasible"
        );
    }

    #[test]
    fn test_fcfs_tie_breaks_by_decision_when_arrival_and_cost_equal() {
        // Two vessels arriving at the same time and with identical costs.
        // With equal arrival_time and cost, FCFS falls back to Decision ordering.
        // Decision tie-break order is deterministic (vessel index first, then berth index).
        let mut b = ModelBuilder::<IntegerType>::new(1, 2);

        // Same arrival time and identical processing times -> identical costs
        b.set_vessel_arrival_time(VesselIndex::new(0), 5)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(10),
            );

        b.set_vessel_arrival_time(VesselIndex::new(1), 5)
            .set_vessel_weight(VesselIndex::new(1), 1)
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(0),
                ProcessingTime::some(10),
            );

        let model = b.build();
        let mut berth_availability = BerthAvailability::new();
        berth_availability.initialize(&model, &[]);
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = FcfsHeuristicBuilder::<IntegerType>::new();

        let decisions: Vec<Decision<IntegerType>> = builder
            .next_decision(&mut evaluator, &model, &berth_availability, &state)
            .collect();

        assert_eq!(decisions.len(), 2, "Expected two feasible decisions");

        // Both have same arrival and cost; ordering should be by Decision's natural order.
        // VesselIndex(0) should come before VesselIndex(1).
        assert_eq!(
            decisions[0].vessel_index().get(),
            0,
            "With equal arrival and cost, decision ordering should favor lower vessel index first"
        );
        assert_eq!(
            decisions[1].vessel_index().get(),
            1,
            "Second should be the higher vessel index"
        );
    }

    #[test]
    fn test_fcfs_multiple_berths_respects_arrival_primary_then_cost_secondary() {
        // Two vessels, two berths. Same arrival for both vessels.
        // V0: cheap on b0, expensive on b1.
        // V1: cheap on b1, expensive on b0.
        // FCFS should first group by arrival (same here), then sort by cost ascending per candidate,
        // and tie-break further by decision.
        let mut b = ModelBuilder::<IntegerType>::new(2, 2);

        // Arrival equal for both
        b.set_vessel_arrival_time(VesselIndex::new(0), 10)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(2), // cheap
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::some(50), // expensive
            );

        b.set_vessel_arrival_time(VesselIndex::new(1), 10)
            .set_vessel_weight(VesselIndex::new(1), 1)
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(0),
                ProcessingTime::some(50), // expensive
            )
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(1),
                ProcessingTime::some(2), // cheap
            );

        let model = b.build();
        let mut berth_availability = BerthAvailability::new();
        berth_availability.initialize(&model, &[]);
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = FcfsHeuristicBuilder::<IntegerType>::new();

        let decisions: Vec<Decision<IntegerType>> = builder
            .next_decision(&mut evaluator, &model, &berth_availability, &state)
            .collect();

        assert_eq!(decisions.len(), 4, "Expected four feasible decisions");

        // Verify sorted order using the costs embedded in the decisions
        // Expected order:
        // 1. V0 on B0 (Cost ~ 2)
        // 2. V1 on B1 (Cost ~ 2) -> Tie break on Vessel Index puts V0 first? No, V0 vs V1.
        //    Wait: V0 on B0 (v=0, b=0, cost=2). V1 on B1 (v=1, b=1, cost=2).
        //    Arrivals same. Costs same.
        //    Tie breaker: Decision order. Decision compares VesselIndex first.
        //    So V0 (0) < V1 (1). V0 on B0 should be first.
        // 3. V0 on B1 (Cost ~ 50)
        // 4. V1 on B0 (Cost ~ 50) -> Tie break V0 < V1.

        // Actually, let's verify what the builder produced vs manual expectation
        let actual_costs: Vec<(usize, usize, IntegerType)> = decisions
            .iter()
            .map(|d| {
                (
                    d.vessel_index().get(),
                    d.berth_index().get(),
                    d.cost_delta(),
                )
            })
            .collect();

        // Sort manually to verify expectations
        let mut expected = actual_costs.clone();
        expected.sort_by(|a, b| {
            // Primary: Cost (since arrivals are all 10)
            a.2.cmp(&b.2)
                // Secondary: Vessel Index
                .then_with(|| a.0.cmp(&b.0))
                // Tertiary: Berth Index
                .then_with(|| a.1.cmp(&b.1))
        });

        assert_eq!(
            actual_costs, expected,
            "With equal arrivals, FCFS must sort by cost ascending then decision"
        );
    }
}
