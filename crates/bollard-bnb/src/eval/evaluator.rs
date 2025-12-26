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

use crate::{berth_availability::BerthAvailability, state::SearchState};
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
/// - `evaluate_vessel_assignment` to compute the incremental cost of assigning a vessel.
/// - `estimate_remaining_cost` to predict the minimum cost required to complete the schedule.
///
/// Both methods now have access to `BerthAvailability`, allowing cost calculations and
/// bounds to account for static constraints like maintenance windows.
///
/// # Requirements: Regular Objective Function
///
/// Implementations of this trait **must** represent a **regular objective function**.
///
/// A regular objective function is non-decreasing with respect to the completion times of
/// the vessels. In practical terms, this means that completing a vessel earlier (or at the
/// same time) should never result in a higher cost than completing it later.
///
/// **The solver relies on this property for correctness.**
///
/// If a non-regular objective is used (for example, one that includes earliness penalties
/// where finishing *too* early increases the cost), the solver's dominance rules and
/// bounding logic may incorrectly prune the optimal solution, leading to valid
/// schedules being discarded.
pub trait ObjectiveEvaluator<T>
where
    T: PrimInt + Signed,
{
    /// Returns the name of the objective evaluator.
    fn name(&self) -> &str;

    /// Evaluates the cost of assigning a vessel to a berth starting at a given time.
    ///
    /// The `start_time` passed here is the **actual start time**, already adjusted for
    /// availability (maintenance windows) by the `DecisionBuilder`.
    ///
    /// While many evaluators (like Weighted Flow Time) only need the `start_time`,
    /// `berth_availability` is provided for complex objectives that might depend on
    /// the specific interval characteristics or future lookahead.
    ///
    /// Returns `Some(cost)` if the assignment is feasible, otherwise `None`.
    fn evaluate_vessel_assignment(
        &mut self,
        model: &Model<T>,
        berth_availability: &BerthAvailability<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
    ) -> Option<T>
    where
        T: SolverNumeric;

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
        berth_availability: &BerthAvailability<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
    ) -> Option<T>
    where
        T: SolverNumeric;

    /// Computes a global lower bound on the objective from the current state.
    ///
    /// The bound must be optimistic (never exceed the true optimal remaining cost).
    ///
    /// **Crucial:** Implementations should use `berth_availability` to determine
    /// when berths *actually* become free for the remaining vessels. Ignoring
    /// maintenance windows here will result in a loose bound and poor pruning performance.
    fn estimate_remaining_cost(
        &mut self,
        model: &Model<T>,
        berth_availability: &BerthAvailability<T>,
        state: &SearchState<T>,
    ) -> Option<T>
    where
        T: SolverNumeric;

    /// Computes the total Lower Bound ($f(n)$) for the current branch.
    ///
    /// The default implementation calculates `f(n) = g(n) + h(n)`:
    /// - `g(n)`: The cost already incurred (`state.current_objective()`).
    /// - `h(n)`: The estimated remaining cost (`estimate_remaining_cost()`).
    fn lower_bound(
        &mut self,
        model: &Model<T>,
        berth_availability: &BerthAvailability<T>,
        state: &SearchState<T>,
    ) -> Option<T>
    where
        T: SolverNumeric,
    {
        let h_n = self.estimate_remaining_cost(model, berth_availability, state)?;
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
