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

/// Internal candidate structure for FCFS (First-Come-First-Serve) sorting.
///
/// Orders candidates primarily by **Arrival Time**, then by **Cost**,
/// then by **Decision** (deterministic tie-breaking).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FcfsCandidate<T> {
    // We optimize padding for T = i64
    // as `Decision` contains two usize fields
    // this order minimizes struct size.
    arrival_time: T,
    cost: T,
    decision: Decision,
}

impl<T> FcfsCandidate<T> {
    /// Creates a new `FcfsCandidate`.
    #[inline(always)]
    fn new(arrival_time: T, cost: T, decision: Decision) -> Self {
        Self {
            arrival_time,
            cost,
            decision,
        }
    }
}

impl<T> PartialOrd for FcfsCandidate<T>
where
    T: PartialOrd + Ord,
{
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for FcfsCandidate<T>
where
    T: Ord,
{
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.arrival_time
            .cmp(&other.arrival_time)
            .then_with(|| self.cost.cmp(&other.cost))
            .then_with(|| self.decision.cmp(&other.decision))
    }
}

impl<T> std::fmt::Display for FcfsCandidate<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Arrival: {}, Cost: {}, Decision: {}",
            self.arrival_time, self.cost, self.decision
        )
    }
}

/// A decision builder that implements a **First-Come-First-Serve (FCFS)** branching heuristic.
///
/// This builder prioritizes vessels by their arrival time. For each unassigned vessel:
/// 1. It enumerates admissible berths and forms feasible `(vessel, berth)` decisions.
/// 2. It computes the immediate objective increase (`cost`) via the evaluator for each feasible decision.
/// 3. It inserts all feasible candidates into a global list annotated with:
///    - the vessel’s `arrival_time`,
///    - the decision’s `cost`,
///    - the `Decision` itself.
/// 4. It globally sorts candidates by:
///    - primary: ascending `arrival_time` (earlier vessels first),
///    - secondary: ascending `cost` (prefer cheaper among equally early arrivals),
///    - tertiary: deterministic `Decision` order (tie-break).
///
/// This strategy enforces a time-first exploration order, which can be useful for schedules
/// where respecting arrival chronology improves feasibility and reduces backtracking. Among
/// vessels with equal arrival times, the builder prefers lower-cost assignments to quickly
/// identify strong incumbents.
///
/// Notes:
/// - Feasibility is determined via `Decision::try_new_unchecked` and the evaluator returning `Some(cost)`.
/// - Vessels already assigned in `state` are skipped.
/// - Only feasible pairs contribute candidates.
/// - Sorting uses `sort_unstable` on a total order, yielding deterministic iteration.
#[derive(Debug, Clone, Default)]
pub struct FcfsHeuristicBuilder<T> {
    candidates: Vec<FcfsCandidate<T>>,
}

impl<T> FcfsHeuristicBuilder<T> {
    /// Creates a new `FcfsHeuristicBuilder` with an empty candidate buffer.
    ///
    /// Use this when instance sizes are unknown. For larger instances, prefer `preallocated`
    /// to reduce reallocations while building candidate sets.
    #[inline]
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
        }
    }

    /// Creates a new `FcfsHeuristicBuilder` with a preallocated candidate buffer sized for
    /// the given number of berths and vessels.
    ///
    /// Capacity:
    /// - `candidates`: up to `num_berths * num_vessels` global candidates.
    #[inline]
    pub fn preallocated(num_berths: usize, num_vessels: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(num_berths * num_vessels),
        }
    }

    /// Creates a new `FcfsHeuristicBuilder` with a specific capacity for the internal buffer.
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

                if let Some(decision) =
                    unsafe { Decision::try_new_unchecked(vessel_index, berth_index, model, state) }
                {
                    let berth_free_time = unsafe { state.berth_free_time_unchecked(berth_index) };
                    if let Some(cost) = unsafe {
                        evaluator.evaluate_vessel_assignment_unchecked(
                            model,
                            vessel_index,
                            berth_index,
                            berth_free_time,
                        )
                    } {
                        self.candidates
                            .push(FcfsCandidate::new(arrival_time, cost, decision));
                    }
                }
            }
        }

        self.candidates.sort_unstable();

        FcfsHeuristicIter {
            iter: self.candidates.iter(),
        }
    }
}

/// A lightweight iterator that yields `Decision`s from the FCFS-sorted candidate slice.
///
/// Properties:
/// - Borrowing: holds a reference into the builder’s internal buffer; no allocations.
/// - Order: yields decisions sorted by arrival time, then cost, then decision.
/// - Fused: once exhausted, subsequent calls return `None`.
pub struct FcfsHeuristicIter<'a, T> {
    iter: std::slice::Iter<'a, FcfsCandidate<T>>,
}

impl<'a, T> Iterator for FcfsHeuristicIter<'a, T>
where
    T: Copy,
{
    type Item = Decision;

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
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = FcfsHeuristicBuilder::<IntegerType>::new();

        let mut iter = builder.next_decision(&mut evaluator, &model, &state);

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
        let state = SearchState::<IntegerType>::new(1, 2);
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = FcfsHeuristicBuilder::<IntegerType>::new();

        let mut iter = builder.next_decision(&mut evaluator, &model, &state);

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

        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = FcfsHeuristicBuilder::<IntegerType>::new();

        let mut iter = builder.next_decision(&mut evaluator, &model, &state);
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
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = FcfsHeuristicBuilder::<IntegerType>::new();

        let decisions: Vec<Decision> = builder
            .next_decision(&mut evaluator, &model, &state)
            .by_ref()
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
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = FcfsHeuristicBuilder::<IntegerType>::new();

        let decisions: Vec<Decision> = builder
            .next_decision(&mut evaluator, &model, &state)
            .by_ref()
            .collect();

        assert_eq!(decisions.len(), 4, "Expected four feasible decisions");

        // Compute costs for asserted ordering checks
        let costs: Vec<(usize, usize, IntegerType)> = decisions
            .iter()
            .map(|d| {
                let v = d.vessel_index();
                let b = d.berth_index();
                let ready = unsafe { state.berth_free_time_unchecked(b) };
                let c = evaluator
                    .evaluate_vessel_assignment(&model, v, b, ready)
                    .unwrap();
                (v.get(), b.get(), c)
            })
            .collect();

        // Arrival times are equal for all decisions, so the ordering should be by cost ascending,
        // then by decision as a tie-breaker.
        let mut sorted = costs.clone();
        sorted.sort_by(|a, b| {
            a.2.cmp(&b.2)
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.1.cmp(&b.1))
        });
        assert_eq!(
            costs, sorted,
            "With equal arrivals, FCFS must sort by cost ascending then decision"
        );
    }
}
