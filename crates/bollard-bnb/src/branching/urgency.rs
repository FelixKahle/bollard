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

//! Urgency‑Guided Regret Branching
//!
//! Implements a hybrid decision builder that fuses **Earliest Deadline First (EDF)**
//! with **Regret-Based Optimization**.
//!
//! ### Strategy
//! 1.  **Feasibility (Primary):** Sort by **Decision Slack** (`Deadline - Finish`).
//!     This aligns perfectly with EDF, ensuring the tightest time windows are filled first.
//! 2.  **Optimization (Secondary):** Sort by **Vessel Regret**.
//!     If multiple options have the same slack (e.g., parallel berths), prioritize the
//!     vessel where the cost gap between best and 2nd-best is largest.
//! 3.  **Cost (Tertiary):** Cheapest move first.

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
use std::cmp::Ordering;
use std::iter::FusedIterator;

/// Internal candidate structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UrgencyCandidate<T> {
    decision_slack: T, // Primary: The slack of THIS specific choice
    vessel_regret: T,  // Secondary: The regret context of the VESSEL
    decision: Decision<T>,
}

impl<T> UrgencyCandidate<T> {
    #[inline(always)]
    fn new(decision_slack: T, vessel_regret: T, decision: Decision<T>) -> Self {
        Self {
            decision_slack,
            vessel_regret,
            decision,
        }
    }
}

impl<T> Ord for UrgencyCandidate<T>
where
    T: SolverNumeric,
{
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        // 1. Feasibility: Smallest Slack First (Ascending)
        self.decision_slack
            .cmp(&other.decision_slack)
            // 2. Optimization: Highest Regret First (Descending)
            .then_with(|| other.vessel_regret.cmp(&self.vessel_regret))
            // 3. Greedy: Cheapest Cost First (Ascending)
            .then_with(|| self.decision.cost_delta().cmp(&other.decision.cost_delta()))
            // 4. Determinism
            .then_with(|| self.decision.cmp(&other.decision))
    }
}

impl<T> PartialOrd for UrgencyCandidate<T>
where
    T: SolverNumeric,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Default)]
pub struct UrgencyRegretBuilder<T> {
    candidates: Vec<UrgencyCandidate<T>>,
    scratch_options: Vec<(Decision<T>, T)>, // (Decision, DecisionSlack)
}

impl<T> UrgencyRegretBuilder<T> {
    #[inline]
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
            scratch_options: Vec::new(),
        }
    }

    #[inline]
    pub fn preallocated(num_berths: usize, num_vessels: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(num_berths * num_vessels),
            scratch_options: Vec::with_capacity(num_berths),
        }
    }
}

impl<T, E> DecisionBuilder<T, E> for UrgencyRegretBuilder<T>
where
    T: SolverNumeric,
    E: ObjectiveEvaluator<T>,
{
    type DecisionIterator<'a>
        = UrgencyRegretIter<'a, T>
    where
        T: 'a,
        E: 'a,
        Self: 'a;

    fn name(&self) -> &str {
        "UrgencyRegretBuilder"
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

            self.scratch_options.clear();

            // 1. Collect all feasible options for this vessel
            for b in 0..num_berths {
                let berth_index = BerthIndex::new(b);

                if let Some(decision) = unsafe {
                    Decision::try_new_unchecked(
                        vessel_index,
                        berth_index,
                        model,
                        berth_availability,
                        state,
                        evaluator,
                    )
                } && let Some(duration) = unsafe {
                    model
                        .vessel_processing_time_unchecked(vessel_index, berth_index)
                        .into()
                } {
                    let finish = decision.start_time() + duration;
                    let deadline =
                        unsafe { model.vessel_latest_departure_time_unchecked(vessel_index) };
                    // Calculate Decision Slack
                    let slack = deadline.saturating_sub(finish);
                    self.scratch_options.push((decision, slack));
                }
            }

            if self.scratch_options.is_empty() {
                continue;
            }

            // 2. Compute Regret for the *Vessel*
            // Regret = Cost(2nd Best) - Cost(Best)
            self.scratch_options
                .sort_unstable_by_key(|(d, _)| d.cost_delta());

            let best_cost = self.scratch_options[0].0.cost_delta();
            let regret = if self.scratch_options.len() > 1 {
                self.scratch_options[1].0.cost_delta() - best_cost
            } else {
                T::max_value() // Must assign!
            };

            // 3. Promote options to global list
            // Using Decision Slack + Vessel Regret Context
            for (decision, slack) in self.scratch_options.iter() {
                self.candidates
                    .push(UrgencyCandidate::new(*slack, regret, *decision));
            }
        }

        // 4. Sort globally: [Best ... Worst]
        self.candidates.sort_unstable();

        // 5. Return Forward Iterator (Same as EDF)
        UrgencyRegretIter {
            iter: self.candidates.iter(),
        }
    }
}

pub struct UrgencyRegretIter<'a, T> {
    iter: std::slice::Iter<'a, UrgencyCandidate<T>>,
}

impl<'a, T: Copy> Iterator for UrgencyRegretIter<'a, T> {
    type Item = Decision<T>;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|c| c.decision)
    }
}

impl<'a, T: Copy> FusedIterator for UrgencyRegretIter<'a, T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::wct::WeightedCompletionTimeEvaluator;
    use bollard_model::{
        index::{BerthIndex, VesselIndex},
        model::ModelBuilder,
        time::ProcessingTime,
    };

    type IntegerType = i64;

    #[test]
    fn test_urgency_regret_prioritizes_decision_slack_not_vessel_min_slack() {
        let mut b = ModelBuilder::<IntegerType>::new(2, 2);

        // V0
        b.set_vessel_arrival_time(VesselIndex::new(0), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(0), 100)
            .set_vessel_weight(VesselIndex::new(0), 1)
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(0),
                ProcessingTime::some(100),
            ) // Slack 0
            .set_vessel_processing_time(
                VesselIndex::new(0),
                BerthIndex::new(1),
                ProcessingTime::some(0),
            ); // Slack 100

        // V1
        b.set_vessel_arrival_time(VesselIndex::new(1), 0)
            .set_vessel_latest_departure_time(VesselIndex::new(1), 100)
            .set_vessel_weight(VesselIndex::new(1), 1)
            .set_vessel_processing_time(
                VesselIndex::new(1),
                BerthIndex::new(0),
                ProcessingTime::some(50),
            ); // Slack 50

        let model = b.build();
        let mut berth_availability = BerthAvailability::new();
        berth_availability.initialize(&model, &[]);
        let state = SearchState::<IntegerType>::new(model.num_berths(), model.num_vessels());
        let mut evaluator = WeightedCompletionTimeEvaluator::<IntegerType>::new();
        let mut builder = UrgencyRegretBuilder::<IntegerType>::new();

        let decisions: Vec<Decision<IntegerType>> = builder
            .next_decision(&mut evaluator, &model, &berth_availability, &state)
            .collect();

        // Check order (Forward Iterator)
        // [Best (0), Medium (50), Worst (100)]

        assert_eq!(decisions.len(), 3);

        // 1. Best: V0 on B0 (Slack 0)
        assert_eq!(decisions[0].vessel_index().get(), 0);
        assert_eq!(decisions[0].berth_index().get(), 0);

        // 2. Medium: V1 on B0 (Slack 50)
        assert_eq!(decisions[1].vessel_index().get(), 1);
        assert_eq!(decisions[1].berth_index().get(), 0);

        // 3. Worst: V0 on B1 (Slack 100)
        assert_eq!(decisions[2].vessel_index().get(), 0);
        assert_eq!(decisions[2].berth_index().get(), 1);
    }
}
