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

//! Objective evaluation for local search.
//!
//! This module defines the scoring logic used by the local search decoder to
//! compare feasible assignments and accumulate the objective value. The design
//! separates the metric used to choose among candidates from the cost recorded
//! on the schedule, so heuristics can steer decisions without contaminating the
//! reported objective.
//!
//! Evaluators read from an immutable `Model` and accept typed vessel and berth
//! indices with a proposed start time. The checked path validates bounds and is
//! suitable for general use, while the unchecked path is optimized for tight
//! inner loops once invariants are established elsewhere. The default
//! `WeightedFlowTimeEvaluator` computes completion time multiplied by vessel
//! weight and treats assignments that exceed a deadline or lack a processing
//! time as infeasible, making it both fast and consistent with common flow‑time
//! objectives in local search.

use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use bollard_search::num::SolverNumeric;

/// The result of an assignment evaluation.
///
/// This separates the "Decision Metric" (Score) from the "Business Metric" (Objective Delta).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Evaluation<T> {
    /// The heuristic value used by the Decoder to pick the "best" berth.
    /// This may include penalties, noise, or other guidance terms.
    /// Lower is better.
    pub score: T,

    /// The actual cost to be added to the Schedule's objective value.
    /// This represents the physical/business cost (e.g., Weighted Flow Time).
    pub objective_delta: T,
}

impl<T> std::fmt::Display for Evaluation<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Evaluation(score: {}, objective_delta: {})",
            self.score, self.objective_delta
        )
    }
}

impl<T> Evaluation<T> {
    /// Creates a new Evaluation with the given score and objective delta.
    #[inline(always)]
    pub fn new(score: T, objective_delta: T) -> Self {
        Self {
            score,
            objective_delta,
        }
    }
}

/// A trait for evaluating the cost and score of assigning a vessel to a berth.
pub trait AssignmentEvaluator<T>
where
    T: SolverNumeric,
{
    /// Returns the name of the evaluator.
    fn name(&self) -> &str;

    /// Evaluates the assignment of a vessel to a berth at a given start time.
    fn evaluate(
        &self,
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
    ) -> Option<Evaluation<T>>;

    /// Evaluates the assignment without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is within `0..model.num_vessels()` and
    /// `berth_index` is within `0..model.num_berths()`.
    unsafe fn evaluate_unchecked(
        &self,
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
    ) -> Option<Evaluation<T>>;
}
