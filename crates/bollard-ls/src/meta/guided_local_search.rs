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

//! Guided Local Search (GLS) metaheuristic.
//!
//! GLS augments the base objective with adaptive penalties to escape local minima. It
//! increases the cost of revisiting certain solution features so the search explores
//! alternative neighborhoods rather than repeatedly settling on the same structures.
//!
//! The type `GuidedLocalSearch` implements the `Metaheuristic` trait and orchestrates
//! the search. The scoring is provided by `GuidedEvaluator<T>`, which wraps a
//! `WeightedFlowTimeEvaluator<T>` and adds a penalty term scaled by `lambda`. Penalties
//! are stored in a compact, row‑major `PenaltyMatrix<T>` indexed by `(vessel, berth)`.
//!
//! The acceptance decision is based on an augmented objective:
//! `augmented = base_objective + lambda * sum_penalty(schedule)`,
//! where `sum_penalty(schedule)` accumulates the per‑assignment counts from the penalty matrix.
//!
//! During stagnation, GLS identifies high‑utility assignments and raises their penalties
//! before continuing the search. Utility is computed as
//! `feature_cost(v) / (1 + penalty(v, b))`, where the feature cost derives from the vessel
//! weight and its scheduled start time. Increasing the penalty lowers the utility of that
//! assignment in future iterations, nudging the search away from it.
//!
//! Conversions to and from `f64` are handled defensively. Non‑convertible or non‑finite values
//! result in conservative behavior such as skipping a feature, treating an objective as
//! undesirable, or rejecting a candidate. Unchecked paths are reserved for hot loops where
//! invariants are upheld by the caller, with debug assertions acting as guardrails in debug
//! builds.
//!
//! The penalty matrix is contiguous in memory for cache efficiency and uses saturating
//! increments. Hot‑path methods are inlined where appropriate, and bounds checks can be
//! elided through the unchecked variants when indices are known to be valid.
//!
//! Type and trait bounds follow the rest of the solver: `T: SolverNumeric` is required
//! throughout. The guided evaluator uses `T: FromPrimitive` to convert penalty costs back
//! into `T`, and the metaheuristic layer also uses `ToPrimitive` for inspecting objectives.

use crate::eval::{AssignmentEvaluator, Evaluation, WeightedFlowTimeEvaluator};
use crate::memory::Schedule;
use crate::meta::metaheuristic::Metaheuristic;
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use bollard_search::{monitor::search_monitor::SearchCommand, num::SolverNumeric};
use num_traits::{PrimInt, ToPrimitive, Unsigned};

#[inline(always)]
fn flatten_index(vessel: usize, berth: usize, num_berths: usize) -> usize {
    vessel * num_berths + berth
}

/// Dense penalty store for vessel–berth assignments.
///
/// Penalties are kept in a contiguous row‑major array indexed by `(vessel, berth)`.
/// Lookups and increments are constant time and cache‑friendly. The matrix is sized
/// for the current problem dimensions and is reset to zero on resize. The unchecked
/// accessors trade bounds checking for speed in hot paths where the caller guarantees
/// valid indices. Increments use saturating semantics to avoid numeric overflow.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct PenaltyMatrix<T>
where
    T: PrimInt,
{
    data: Vec<T>,       // Row-major storage: vessel-major, then berth
    num_berths: usize,  // Number of berths (columns)
    num_vessels: usize, // Number of vessels (rows)
}

impl<T> PenaltyMatrix<T>
where
    T: PrimInt,
{
    /// Creates a new penalty matrix with the specified dimensions, initialized to zero.
    #[inline]
    fn new(num_vessels: usize, num_berths: usize) -> Self {
        let size = num_vessels * num_berths;
        Self {
            data: vec![T::zero(); size],
            num_berths,
            num_vessels,
        }
    }

    /// Resizes the penalty matrix to the specified dimensions, resetting all penalties to zero.
    #[inline]
    fn resize(&mut self, num_vessels: usize, num_berths: usize) {
        let new_size = num_vessels * num_berths;

        if self.data.len() != new_size {
            self.data = vec![T::zero(); new_size];
        } else {
            self.data.fill(T::zero());
        }

        self.num_berths = num_berths;
        self.num_vessels = num_vessels;
    }

    /// Retrieves the penalty for the specified `(vessel, berth)` assignment.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` or `berth` indices are out of bounds.
    #[inline(always)]
    fn get(&self, vessel: VesselIndex, berth: BerthIndex) -> T {
        let vessel_index = vessel.get();
        let berth_index = berth.get();

        debug_assert!(
            vessel_index < self.num_vessels,
            "called `PenaltyMatrix::get` with vessel index out of bounds: the len is {} but the index is {}",
            self.num_vessels,
            vessel_index
        );

        debug_assert!(
            berth_index < self.num_berths,
            "called `PenaltyMatrix::get` with berth index out of bounds: the len is {} but the index is {}",
            self.num_berths,
            berth_index
        );

        let index = flatten_index(vessel_index, berth_index, self.num_berths);
        debug_assert!(
            index < self.data.len(),
            "called `PenaltyMatrix::get` with computed index out of bounds: the len is {} but the index is {}",
            self.data.len(),
            index
        );

        self.data.get(index).copied().unwrap_or(T::zero())
    }

    /// Retrieves the penalty for the specified `(vessel, berth)` assignment without bounds checking.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` or `berth` indices are out of bounds.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel` and `berth` indices are within bounds.
    #[inline(always)]
    unsafe fn get_unchecked(&self, vessel: VesselIndex, berth: BerthIndex) -> T {
        let vessel_index = vessel.get();
        let berth_index = berth.get();

        debug_assert!(
            vessel_index < self.num_vessels,
            "called `PenaltyMatrix::get_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            self.num_vessels,
            vessel_index
        );

        debug_assert!(
            berth_index < self.num_berths,
            "called `PenaltyMatrix::get_unchecked` with berth index out of bounds: the len is {} but the index is {}",
            self.num_berths,
            berth_index
        );

        let index = flatten_index(vessel_index, berth_index, self.num_berths);

        debug_assert!(
            index < self.data.len(),
            "called `PenaltyMatrix::get_unchecked` with computed index out of bounds: the len is {} but the index is {}",
            self.data.len(),
            index
        );

        *unsafe { self.data.get_unchecked(index) }
    }

    /// Increments the penalty for the specified `(vessel, berth)` assignment by one, using saturating arithmetic.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` or `berth` indices are out of bounds.
    #[allow(dead_code)]
    #[inline(always)]
    pub fn increment(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        let vessel_index = vessel.get();
        let berth_index = berth.get();

        debug_assert!(
            vessel_index < self.num_vessels,
            "called `PenaltyMatrix::increment` with vessel index out of bounds: the len is {} but the index is {}",
            self.num_vessels,
            vessel_index
        );
        debug_assert!(
            berth_index < self.num_berths,
            "called `PenaltyMatrix::increment` with berth index out of bounds: the len is {} but the index is {}",
            self.num_berths,
            berth_index
        );

        if self.num_berths == 0 {
            return;
        }

        let index = flatten_index(vessel_index, berth_index, self.num_berths);

        debug_assert!(
            index < self.data.len(),
            "called `PenaltyMatrix::increment` with computed index out of bounds: the len is {} but the index is {}",
            self.data.len(),
            index
        );

        if let Some(val) = self.data.get_mut(index) {
            *val = val.saturating_add(T::one());
        }
    }

    /// Increments the penalty for the specified `(vessel, berth)` assignment by one without bounds checking, using saturating arithmetic.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` or `berth` indices are out of bounds.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel` and `berth` indices are within bounds.
    #[inline(always)]
    unsafe fn increment_unchecked(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        let vessel_index = vessel.get();
        let berth_index = berth.get();

        debug_assert!(
            vessel_index < self.num_vessels,
            "called `PenaltyMatrix::increment_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            self.num_vessels,
            vessel_index
        );
        debug_assert!(
            berth_index < self.num_berths,
            "called `PenaltyMatrix::increment_unchecked` with berth index out of bounds: the len is {} but the index is {}",
            self.num_berths,
            berth_index
        );

        if self.num_berths == 0 {
            return;
        }

        let index = flatten_index(vessel_index, berth_index, self.num_berths);

        debug_assert!(
            index < self.data.len(),
            "called `PenaltyMatrix::increment_unchecked` with computed index out of bounds: the len is {} but the index is {}",
            self.data.len(),
            index
        );

        let val = unsafe { self.data.get_unchecked_mut(index) };
        *val = val.saturating_add(T::one());
    }

    /// Resets all penalties in the matrix to zero.
    #[inline(always)]
    fn reset(&mut self) {
        self.data.fill(T::zero());
    }
}

/// Augmented evaluator that adds GLS penalties to the base score.
///
/// This wrapper delegates base scoring to `WeightedFlowTimeEvaluator` and adds a
/// penalty term scaled by `lambda` for the specific `(vessel, berth)` assignment
/// being evaluated. When `lambda` is zero or the penalty is zero, the inner score
/// is returned unchanged. Conversions to and from `f64` are handled defensively;
/// if a penalty cannot be represented, the evaluation returns `None` to signal that
/// the candidate should be rejected by the caller. The `resize` method synchronizes
/// the internal penalty matrix with the model dimensions at runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct GuidedEvaluator<T, B = u32>
where
    T: SolverNumeric,
    B: Unsigned + PrimInt,
{
    inner: WeightedFlowTimeEvaluator<T>, // Base evaluator
    penalties: PenaltyMatrix<B>,         // Penalty counts
    lambda: f64,                         // Penalty scaling factor
}

impl<T> GuidedEvaluator<T>
where
    T: SolverNumeric,
{
    /// Creates a new `GuidedEvaluator` with the specified dimensions and penalty scaling factor.
    #[inline]
    fn new(num_vessels: usize, num_berths: usize, lambda: f64) -> Self {
        Self {
            inner: WeightedFlowTimeEvaluator::default(),
            penalties: PenaltyMatrix::new(num_vessels, num_berths),
            lambda,
        }
    }

    /// Resizes the internal penalty matrix to match the specified dimensions.
    #[inline]
    fn resize(&mut self, num_vessels: usize, num_berths: usize) {
        self.penalties.resize(num_vessels, num_berths);
    }
}

impl<T, B> AssignmentEvaluator<T> for GuidedEvaluator<T, B>
where
    T: SolverNumeric + num_traits::FromPrimitive,
    B: Unsigned + PrimInt,
{
    fn name(&self) -> &str {
        "GuidedEvaluator"
    }

    fn evaluate(
        &self,
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
    ) -> Option<Evaluation<T>> {
        let inner_eval = self
            .inner
            .evaluate(model, vessel_index, berth_index, start_time)?;

        // Fast path: no penalty or degenerate lambda
        if self.lambda == 0.0 || !self.lambda.is_finite() {
            return Some(inner_eval);
        }
        let penalty_count = self.penalties.get(vessel_index, berth_index);
        if penalty_count == B::zero() {
            return Some(inner_eval);
        }

        // Convert penalty_count -> f64 robustly
        let count_f64 = match penalty_count.to_f64() {
            Some(v) if v.is_finite() => v,
            // If we can't represent the penalty, treat this move as invalid
            _ => return None,
        };

        // Compute penalty cost; ensure it's finite and non-negative
        let penalty_cost = self.lambda * count_f64;
        if !penalty_cost.is_finite() || penalty_cost < 0.0 {
            return None;
        }

        // Convert back to T; if it can't be represented, skip this move
        let penalty = T::from_f64(penalty_cost)?;

        Some(Evaluation {
            score: inner_eval.score + penalty,
            objective_delta: inner_eval.objective_delta,
        })
    }

    unsafe fn evaluate_unchecked(
        &self,
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
    ) -> Option<Evaluation<T>> {
        let inner_eval = unsafe {
            self.inner
                .evaluate_unchecked(model, vessel_index, berth_index, start_time)
        }?;

        // Fast path: no penalty or degenerate lambda
        if self.lambda == 0.0 || !self.lambda.is_finite() {
            return Some(inner_eval);
        }
        let penalty_count = self.penalties.get(vessel_index, berth_index);
        if penalty_count == B::zero() {
            return Some(inner_eval);
        }

        // Convert penalty_count -> f64 robustly
        let count_f64 = match penalty_count.to_f64() {
            Some(v) if v.is_finite() => v,
            _ => return None,
        };

        // Compute penalty cost; ensure it's finite and non-negative
        let penalty_cost = self.lambda * count_f64;
        if !penalty_cost.is_finite() || penalty_cost < 0.0 {
            return None;
        }

        // Convert back to T; if it can't be represented, skip this move
        let penalty = T::from_f64(penalty_cost)?;

        Some(Evaluation {
            score: inner_eval.score + penalty,
            objective_delta: inner_eval.objective_delta,
        })
    }
}

/// Guided Local Search controller.
///
/// Implements the GLS metaheuristic over the solver’s neighborhood moves. The
/// acceptance decision uses an augmented objective equal to the base objective plus
/// `lambda` times the sum of per‑assignment penalties. A stagnation counter tracks
/// consecutive non‑improving iterations; once it reaches `stagnation_limit`, the
/// current schedule is analyzed to identify high‑utility assignments and their
/// penalties are increased to encourage diversification. Numeric conversions used
/// to compute utilities and augmented objectives are validated for finiteness to
/// prevent undefined behavior from propagating through the search.
#[derive(Debug, Clone, PartialEq)]
pub struct GuidedLocalSearch<T>
where
    T: SolverNumeric,
{
    lambda: f64,                   // Penalty scaling factor
    stagnation_limit: u64,         // Limit before penalization
    stagnation_counter: u64,       // Current stagnation counter
    evaluator: GuidedEvaluator<T>, // Guided evaluator with penalties
    current_augmented_score: f64,  // Cached augmented score of current solution
}

impl<T> GuidedLocalSearch<T>
where
    T: SolverNumeric + num_traits::ToPrimitive,
{
    /// Creates a new `GuidedLocalSearch` with the specified penalty scaling factor and stagnation limit.
    #[inline]
    pub fn new(lambda: f64, stagnation_limit: u64) -> Self {
        let evaluator = GuidedEvaluator::new(0, 0, lambda);
        Self {
            lambda,
            stagnation_limit,
            evaluator,
            stagnation_counter: 0,
            current_augmented_score: f64::INFINITY,
        }
    }

    /// Creates a new `GuidedLocalSearch` with preallocated penalty matrix dimensions.
    #[inline]
    pub fn preallocated(
        lambda: f64,
        stagnation_limit: u64,
        num_vessels: usize,
        num_berths: usize,
    ) -> Self {
        let evaluator = GuidedEvaluator::new(num_vessels, num_berths, lambda);
        Self {
            lambda,
            stagnation_limit,
            evaluator,
            stagnation_counter: 0,
            current_augmented_score: f64::INFINITY,
        }
    }

    /// Calculates the full augmented objective from scratch.
    /// O(N) complexity. Use sparingly (initialization, updates).
    fn calculate_augmented_score(&self, schedule: &Schedule<T>) -> f64 {
        let base_objective = match schedule.objective_value().to_f64() {
            Some(v) if v.is_finite() => v,
            _ => return f64::INFINITY,
        };

        // Optimization for simple case
        if self.lambda == 0.0 {
            return base_objective;
        }

        let mut total_penalty = 0.0;

        for (vessel_index_raw, &assigned_berth) in schedule.berths().iter().enumerate() {
            let vessel_index = VesselIndex::new(vessel_index_raw);
            let penalty_value = self.evaluator.penalties.get(vessel_index, assigned_berth);

            let penalty = match penalty_value.to_f64() {
                Some(p) if p.is_finite() => p,
                _ => return f64::INFINITY,
            };

            total_penalty += penalty;
        }

        base_objective + (self.lambda * total_penalty)
    }

    /// Identifies high-utility assignments in the current schedule and increments their penalties.
    /// Utility is defined as (feature_cost) / (1 + penalty), where feature_cost is derived from vessel
    /// weight and start time O(N) complexity.
    fn penalize(&mut self, model: &Model<T>, current: &Schedule<T>) {
        let mut max_utility = f64::NEG_INFINITY;
        let mut candidates: Vec<(VesselIndex, BerthIndex)> = Vec::new();

        for (vessel_index, &berth) in current.berths().iter().enumerate() {
            let vessel = VesselIndex::new(vessel_index);

            let weight = match unsafe { model.vessel_weight_unchecked(vessel).to_f64() } {
                Some(w) if w.is_finite() => w,
                _ => continue,
            };

            let start_time =
                match unsafe { current.start_time_for_vessel_unchecked(vessel).to_f64() } {
                    Some(s) if s.is_finite() => s,
                    _ => continue,
                };

            let feature_cost = weight * start_time;

            let penalty_count = unsafe { self.evaluator.penalties.get_unchecked(vessel, berth) };
            let penalty = match penalty_count.to_f64() {
                Some(p) if p.is_finite() => p,
                _ => continue,
            };

            let denom = 1.0 + penalty;
            if denom <= 0.0 || !denom.is_finite() {
                continue;
            }

            let utility = feature_cost / denom;

            // Using epsilon for float comparison stability
            const EPSILON: f64 = 1e-9;
            if utility > max_utility + EPSILON {
                max_utility = utility;
                candidates.clear();
                candidates.push((vessel, berth));
            } else if (utility - max_utility).abs() < EPSILON {
                candidates.push((vessel, berth));
            }
        }

        for (vessel, berth) in candidates {
            unsafe { self.evaluator.penalties.increment_unchecked(vessel, berth) };
        }
    }
}

impl<T> Metaheuristic<T> for GuidedLocalSearch<T>
where
    T: SolverNumeric + num_traits::FromPrimitive + num_traits::ToPrimitive,
{
    type Evaluator = GuidedEvaluator<T>;

    fn name(&self) -> &str {
        "GuidedLocalSearch"
    }

    fn evaluator(&self) -> &GuidedEvaluator<T> {
        &self.evaluator
    }

    fn on_start(&mut self, model: &Model<T>, initial_solution: &Schedule<T>) {
        self.evaluator
            .resize(model.num_vessels(), model.num_berths());

        self.evaluator.penalties.reset();
        self.stagnation_counter = 0;

        // Initialize cache
        self.current_augmented_score = self.calculate_augmented_score(initial_solution);
    }

    fn search_command(
        &mut self,
        _iter: u64,
        _model: &Model<T>,
        _best: &Schedule<T>,
    ) -> SearchCommand {
        SearchCommand::Continue
    }

    fn should_accept(
        &mut self,
        _model: &Model<T>,
        _current: &Schedule<T>,
        candidate: &Schedule<T>,
        _best: &Schedule<T>,
    ) -> bool {
        let aug_candidate = self.calculate_augmented_score(candidate);
        aug_candidate < self.current_augmented_score - 1e-9
    }

    fn on_accept(&mut self, _model: &Model<T>, new_current: &Schedule<T>) {
        self.stagnation_counter = 0;
        self.current_augmented_score = self.calculate_augmented_score(new_current);
    }

    fn on_reject(&mut self, model: &Model<T>, current_schedule: &Schedule<T>) {
        self.stagnation_counter += 1;
        if self.stagnation_counter >= self.stagnation_limit {
            self.penalize(model, current_schedule);
            self.stagnation_counter = 0;

            // The penalties changed, so the augmented score of the
            // *current* solution (which hasn't moved) has increased.
            // We must update the cache so future candidates are compared correctly.
            self.current_augmented_score = self.calculate_augmented_score(current_schedule);
        }
    }

    fn on_new_best(&mut self, _model: &Model<T>, _new_best: &Schedule<T>) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use bollard_model::index::BerthIndex;
    use bollard_model::model::ModelBuilder;
    use bollard_model::solution::Solution;
    use bollard_search::monitor::search_monitor::SearchCommand;

    use crate::memory::Schedule;
    use bollard_search::num::SolverNumeric;

    fn sched<T: SolverNumeric>(obj: T, berths: Vec<BerthIndex>, starts: Vec<T>) -> Schedule<T> {
        Schedule::from(Solution::new(obj, berths, starts))
    }

    #[test]
    fn test_name_and_evaluator_access() {
        let gls: GuidedLocalSearch<i64> = GuidedLocalSearch::new(1.0, 100);
        assert_eq!(gls.name(), "GuidedLocalSearch");
        let _eval = gls.evaluator();
    }

    #[test]
    fn test_constructors_new_and_preallocated() {
        let gls_a: GuidedLocalSearch<i64> = GuidedLocalSearch::new(0.5, 50);
        let gls_b: GuidedLocalSearch<i64> = GuidedLocalSearch::preallocated(0.5, 50, 0, 0);
        assert_eq!(gls_a.name(), "GuidedLocalSearch");
        assert_eq!(gls_b.name(), "GuidedLocalSearch");
    }

    #[test]
    fn test_on_start_and_hooks_noops_on_empty_model() {
        // Empty model as requested
        let model = ModelBuilder::<i64>::new(0, 0).build();

        // Empty schedules
        let s0 = sched(0_i64, vec![], vec![]);
        let s1 = sched(0_i64, vec![], vec![]);

        let mut gls: GuidedLocalSearch<i64> = GuidedLocalSearch::new(1.0, 10);

        // These should not panic and maintain internal consistency
        gls.on_start(&model, &s0);
        gls.on_accept(&model, &s1);
        gls.on_reject(&model, &s0);
        gls.on_new_best(&model, &s1);

        // Evaluator remains accessible
        let _eval = gls.evaluator();
    }

    #[test]
    fn test_search_command_continue_under_neutral_conditions() {
        let mut gls: GuidedLocalSearch<i64> = GuidedLocalSearch::new(0.1, 5);

        // Empty model
        let model = ModelBuilder::<i64>::new(0, 0).build();
        let best = sched(0_i64, vec![], vec![]);

        // Under neutral conditions, search_command should continue
        let cmd = gls.search_command(0, &model, &best);
        assert_eq!(cmd, SearchCommand::Continue);
    }

    #[test]
    fn test_should_accept_considers_candidate_vs_current_objective_basics() {
        // Guided Local Search augments objective with penalties. We don't assert
        // penalty-specific behavior here; just exercise the acceptance path with simple inputs.
        let mut gls: GuidedLocalSearch<i64> = GuidedLocalSearch::new(0.5, 10);

        // Model must match the schedule size; use 1 vessel, 1 berth
        let model = ModelBuilder::<i64>::new(1, 1).build();

        // Simple schedules; same single berth/start layout, different objective values.
        let current = sched(100_i64, vec![BerthIndex::new(0)], vec![10_i64]);
        let better = sched(90_i64, vec![BerthIndex::new(0)], vec![10_i64]);
        let equal = sched(100_i64, vec![BerthIndex::new(0)], vec![10_i64]);
        let worse = sched(110_i64, vec![BerthIndex::new(0)], vec![10_i64]);

        // Initialize GLS so its evaluator and penalty matrix are sized correctly
        gls.on_start(&model, &current);

        // Best can be the current for this basic exercise
        let best = current.clone();

        // Exercise the method; we do not guarantee penalty behavior, but typical GLS should favor improvements.
        let accept_better = gls.should_accept(&model, &current, &better, &best);
        let accept_equal = gls.should_accept(&model, &current, &equal, &best);
        let accept_worse = gls.should_accept(&model, &current, &worse, &best);

        // Minimal expectations: prefer strictly better, avoid worse; equal is typically rejected.
        assert!(
            accept_better,
            "GLS should accept a strictly better candidate in basic scenarios"
        );
        assert!(
            !accept_worse,
            "GLS should reject a strictly worse candidate in basic scenarios"
        );
        assert!(
            !accept_equal,
            "GLS should reject equal-cost moves in basic scenarios (to avoid plateaus)"
        );
    }
}
