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
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use bollard_search::num::SolverNumeric;
use num_traits::{PrimInt, Signed};

/// A strategy for scoring decisions and computing a global lower bound.
///
/// `ObjectiveEvaluator` decouples the solver from a particular objective function.
/// The solver calls:
/// - `evaluate_vessel_assignment` to compute the incremental cost of assigning a vessel to a berth
///   given the berth's ready time,
/// - `lower_bound` to estimate a tight bound on the total objective from the current state.
///
/// `None` represents an infeasible tree branch or assignment.
pub trait ObjectiveEvaluator<T>
where
    T: PrimInt + Signed,
{
    /// Returns the name of the objective evaluator.
    fn name(&self) -> &str;

    /// Evaluates the cost of assigning a vessel to a berth starting at a given time.
    ///
    /// The cost function depends on the concrete evaluator. For weighted flow time, it is:
    /// `cost = weight(vessel) * finish_time`, where
    /// `finish_time = max(arrival_time(vessel), berth_ready) + processing_time(vessel, berth)`.
    ///
    /// Returns `Some(cost)` if the assignment is feasible, otherwise `None`.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is not within `0..model.num_vessels()` or if
    /// `berth_index` is not within `0..model.num_berths()`.
    fn evaluate_vessel_assignment(
        &mut self,
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        berth_ready: T,
    ) -> Option<T>
    where
        T: SolverNumeric;

    /// Evaluates the cost of assigning a vessel to a berth starting at a given time
    /// without performing bounds checking on the indices.
    ///
    /// Returns `Some(cost)` if the assignment is feasible, otherwise `None`.
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
    ) -> Option<T>
    where
        T: SolverNumeric;

    /// Computes a global lower bound on the objective from the current state.
    ///
    /// The bound should be fast and optimistic (never exceed the optimal value for completion).
    /// The solver prunes branches when `lower_bound >= best_objective`. Also it does not
    /// take into account the current cost of the partial state.
    ///
    /// Implementations typically:
    /// - accumulate the current objective,
    /// - add the minimum incremental cost for each remaining vessel across feasible berths
    ///   using current berth free times,
    /// - return `None` if any vessel has no feasible berth.
    fn estimate_remaining_cost(&mut self, model: &Model<T>, state: &SearchState<T>) -> Option<T>
    where
        T: SolverNumeric;

    /// Computes the total Lower Bound ($f(n)$) for the current branch.
    ///
    /// The default implementation calculates `f(n) = g(n) + h(n)`:
    /// - `g(n)`: The cost already incurred (`state.current_objective()`).
    /// - `h(n)`: The estimated remaining cost (`estimate_remaining_cost()`).
    ///
    /// Returns `None` if the heuristic determines the branch is infeasible.
    fn lower_bound(&mut self, model: &Model<T>, state: &SearchState<T>) -> Option<T>
    where
        T: SolverNumeric,
    {
        let h_n = self.estimate_remaining_cost(model, state)?;
        let g_n = state.current_objective();

        // Use saturating_add to avoid panic on overflow near T::MAX
        Some(g_n.saturating_add(h_n))
    }
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
