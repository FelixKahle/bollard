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

//! Most Constrained Variable (MRV) branching
//!
//! Prioritizes vessels that have the fewest feasible assignment options remaining.
//! This is the premier heuristic for "Feasibility First" problems (sparse constraints).
//!
//! Strategy:
//! 1. Generate all valid decisions for all unassigned vessels.
//! 2. Group decisions by vessel.
//! 3. Sort vessels by the count of their valid decisions (ascending).
//! 4. Tie-break by Arrival Time (earlier first).
//!
//! This forces the solver to assign the "pickiest" ships first. If a ship has 0 options,
//! it floats to the top and causes an immediate backtrack (pruning the tree instantly).

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

#[derive(Debug, Clone, PartialEq, Eq)]
struct MostConstrainedCandidate<T> {
    decisions: Vec<Decision<T>>,
    arrival_time: T,
}

impl<T> Ord for MostConstrainedCandidate<T>
where
    T: SolverNumeric,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.decisions
            .len()
            .cmp(&other.decisions.len())
            .then_with(|| self.arrival_time.cmp(&other.arrival_time))
    }
}

impl<T> PartialOrd for MostConstrainedCandidate<T>
where
    T: SolverNumeric,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> MostConstrainedCandidate<T>
where
    T: SolverNumeric,
{
    fn new() -> Self {
        Self {
            decisions: Vec::with_capacity(16), // Pre-allocate for typical berth count
            arrival_time: T::zero(),
        }
    }
}

/// A decision builder focused on finding ANY feasible solution as fast as possible.
#[derive(Debug, Clone, Default)]
pub struct MostConstrainedBuilder<T> {
    candidates: Vec<MostConstrainedCandidate<T>>,
    active_indices: Vec<usize>,
    decision_queue: Vec<Decision<T>>,
}

impl<T> MostConstrainedBuilder<T>
where
    T: SolverNumeric,
{
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
            active_indices: Vec::new(),
            decision_queue: Vec::new(),
        }
    }

    /// Pre-allocates memory for the specific problem size.
    /// Call this once before starting the search.
    pub fn preallocated(num_berths: usize, num_vessels: usize) -> Self {
        let mut candidates = Vec::with_capacity(num_vessels);
        for _ in 0..num_vessels {
            candidates.push(MostConstrainedCandidate {
                decisions: Vec::with_capacity(num_berths),
                arrival_time: T::zero(),
            });
        }

        Self {
            candidates,
            active_indices: Vec::with_capacity(num_vessels),
            decision_queue: Vec::with_capacity(num_berths),
        }
    }
}

impl<T, E> DecisionBuilder<T, E> for MostConstrainedBuilder<T>
where
    T: SolverNumeric,
    E: ObjectiveEvaluator<T>,
{
    type DecisionIterator<'a>
        = std::vec::IntoIter<Decision<T>>
    where
        T: 'a,
        E: 'a,
        Self: 'a;

    fn name(&self) -> &str {
        "MostConstrainedBuilder"
    }

    fn next_decision<'a>(
        &'a mut self,
        evaluator: &'a mut E,
        model: &'a Model<T>,
        berth_availability: &'a BerthAvailability<T>,
        state: &'a SearchState<T>,
    ) -> Self::DecisionIterator<'a> {
        self.decision_queue.clear();
        self.active_indices.clear();

        let num_vessels = model.num_vessels();
        let num_berths = model.num_berths();

        if self.candidates.len() < num_vessels {
            self.candidates
                .resize(num_vessels, MostConstrainedCandidate::new());
        }

        for vessel_index in 0..num_vessels {
            let vessel = VesselIndex::new(vessel_index);

            if unsafe { state.is_vessel_assigned_unchecked(vessel) } {
                continue;
            }

            let candidate = unsafe { self.candidates.get_unchecked_mut(vessel_index) };

            candidate.decisions.clear();
            candidate.arrival_time = unsafe { model.vessel_arrival_time_unchecked(vessel) };

            for berth_index in 0..num_berths {
                let berth = BerthIndex::new(berth_index);

                if let Some(decision) = unsafe {
                    Decision::try_new_unchecked(
                        vessel,
                        berth,
                        model,
                        berth_availability,
                        state,
                        evaluator,
                    )
                } {
                    candidate.decisions.push(decision);
                }
            }

            if candidate.decisions.is_empty() {
                return Vec::new().into_iter();
            }

            self.active_indices.push(vessel_index);
        }

        let candidates_ref = &self.candidates;
        self.active_indices.sort_unstable_by(|&a_idx, &b_idx| {
            let a = unsafe { candidates_ref.get_unchecked(a_idx) };
            let b = unsafe { candidates_ref.get_unchecked(b_idx) };
            a.decisions
                .len()
                .cmp(&b.decisions.len())
                .then_with(|| a.arrival_time.cmp(&b.arrival_time))
        });

        if let Some(&best_vessel_idx) = self.active_indices.first() {
            let best_candidate = unsafe { self.candidates.get_unchecked(best_vessel_idx) };
            self.decision_queue
                .extend_from_slice(&best_candidate.decisions);
        }

        self.decision_queue.clone().into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::wtft::WeightedFlowTimeEvaluator;
    use bollard_core::math::interval::ClosedOpenInterval;
    use bollard_model::{
        index::{BerthIndex, VesselIndex},
        model::ModelBuilder,
        time::ProcessingTime,
    };

    type IntegerType = i64;

    #[test]
    fn test_mrv_picks_most_constrained_then_tiebreaks_by_arrival() {
        // Setup: 2 Berths, 3 Vessels
        let mut b = ModelBuilder::<IntegerType>::new(2, 3);

        // Open berths and ensure they are usable
        b.add_berth_opening_time(BerthIndex::new(0), ClosedOpenInterval::new(0, 1000))
            .add_berth_opening_time(BerthIndex::new(1), ClosedOpenInterval::new(0, 1000));

        // Generous deadlines to avoid time-based infeasibility
        b.set_vessel_latest_departure_time(VesselIndex::new(0), 1000)
            .set_vessel_latest_departure_time(VesselIndex::new(1), 1000)
            .set_vessel_latest_departure_time(VesselIndex::new(2), 1000);

        // Vessel 0: Arrival 5. Valid on b0 ONLY. (Count = 1)
        b.set_vessel_arrival_time(VesselIndex::new(0), 5)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(10),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::none(),
            );

        // Vessel 1: Arrival 0. Valid on b0 AND b1. (Count = 2)
        // Break symmetry by making processing times differ across berths.
        b.set_vessel_arrival_time(VesselIndex::new(1), 0)
            .set_vessel_weight(VesselIndex::new(1), 1)
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(0),
                ProcessingTime::some(10),
            )
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(1),
                ProcessingTime::some(8), // differs to avoid symmetry pruning
            );

        // Vessel 2: Arrival 3. Valid on b0 ONLY. (Count = 1)
        b.set_vessel_arrival_time(VesselIndex::new(2), 3)
            .set_vessel_weight(VesselIndex::new(2), 1)
            .set_vessel_processing_time(
                VesselIndex::new(2),
                BerthIndex::new(0),
                ProcessingTime::some(10),
            )
            .set_vessel_processing_time(
                VesselIndex::new(2),
                BerthIndex::new(1),
                ProcessingTime::none(),
            );

        let model = b.build();
        let mut berth_availability = BerthAvailability::new();
        berth_availability.initialize(&model, &[]);
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = MostConstrainedBuilder::<IntegerType>::new();

        let decisions: Vec<Decision<IntegerType>> = builder
            .next_decision(&mut evaluator, &model, &berth_availability, &state)
            .collect();

        assert!(!decisions.is_empty(), "Decisions should not be empty");

        // Expectation:
        // Counts: V0=1, V1=2, V2=1.
        // Candidates (Min Count): V0, V2.
        // Tie-break (Arrival): V2(3) < V0(5).
        // Winner: V2 (all decisions should be for vessel 2 only).
        assert!(
            decisions.iter().all(|d| d.vessel_index().get() == 2),
            "Expected Vessel 2 (Most Constrained + Earliest). Got: {:?}",
            decisions
                .iter()
                .map(|d| d.vessel_index().get())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            decisions.len(),
            1,
            "MRV should return only the selected vessel's decisions; v2 has exactly one"
        );
    }

    #[test]
    fn test_mrv_returns_empty_when_any_vessel_has_zero_options() {
        // 2 berths, 2 vessels
        // v0: 2 options
        // v1: 0 options -> MRV should return empty to trigger immediate backtrack
        let mut b = ModelBuilder::<IntegerType>::new(2, 2);

        b.add_berth_opening_time(BerthIndex::new(0), ClosedOpenInterval::new(0, 1000))
            .add_berth_opening_time(BerthIndex::new(1), ClosedOpenInterval::new(0, 1000));

        b.set_vessel_latest_departure_time(VesselIndex::new(0), 1000)
            .set_vessel_latest_departure_time(VesselIndex::new(1), 1000);

        // Vessel 0
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(1),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::some(1),
            );

        // Vessel 1: no feasible berths
        b.set_vessel_arrival_time(VesselIndex::new(1), 0)
            .set_vessel_weight(VesselIndex::new(1), 1)
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(0),
                ProcessingTime::none(),
            )
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(1),
                ProcessingTime::none(),
            );

        let model = b.build();
        let mut berth_availability = BerthAvailability::new();
        berth_availability.initialize(&model, &[]);
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = MostConstrainedBuilder::<IntegerType>::new();

        let mut iter = builder.next_decision(&mut evaluator, &model, &berth_availability, &state);
        assert_eq!(
            iter.next(),
            None,
            "Iterator should be empty when any unassigned vessel has zero feasible options"
        );
    }

    #[test]
    fn test_mrv_tie_by_arrival_when_option_counts_equal() {
        // 2 berths, 2 vessels, both with exactly 1 feasible option.
        // v0: arrival 10, option at b1
        // v1: arrival 5,  option at b0  => selected due to earlier arrival
        let mut b = ModelBuilder::<IntegerType>::new(2, 2);

        b.add_berth_opening_time(BerthIndex::new(0), ClosedOpenInterval::new(0, 1000))
            .add_berth_opening_time(BerthIndex::new(1), ClosedOpenInterval::new(0, 1000));

        b.set_vessel_latest_departure_time(VesselIndex::new(0), 1000)
            .set_vessel_latest_departure_time(VesselIndex::new(1), 1000);

        // Vessel 0: only b1 feasible
        b.set_vessel_arrival_time(VesselIndex::new(0), 10)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::none(),
            )
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::some(2),
            );

        // Vessel 1: only b0 feasible; earlier arrival
        b.set_vessel_arrival_time(VesselIndex::new(1), 5)
            .set_vessel_weight(VesselIndex::new(1), 1)
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(0),
                ProcessingTime::some(2),
            )
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(1),
                ProcessingTime::none(),
            );

        let model = b.build();
        let mut berth_availability = BerthAvailability::new();
        berth_availability.initialize(&model, &[]);
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::new();
        let mut builder = MostConstrainedBuilder::<IntegerType>::new();

        let decisions: Vec<Decision<IntegerType>> = builder
            .next_decision(&mut evaluator, &model, &berth_availability, &state)
            .collect();

        assert!(
            !decisions.is_empty(),
            "Expected one decision for the earlier-arriving vessel"
        );
        assert!(
            decisions.iter().all(|d| d.vessel_index().get() == 1),
            "Tie on option counts should be broken by earlier arrival (v1)"
        );
        assert_eq!(
            decisions.len(),
            1,
            "Each vessel has 1 option; MRV should return only the selected vessel's options"
        );
    }
}
