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

//! Metaheuristic interface for local search control.
//!
//! This module defines the trait used to steer a local search run, separating
//! move generation and decoding from acceptance policy and termination logic.
//! Implementations can express strategies such as hill climbing, simulated
//! annealing, or tabu-like acceptance while interacting with the engine through
//! clear lifecycle hooks: initialization on start, a lightweight per-iteration
//! command to continue or terminate, and callbacks on accept, reject, and new
//! best. The design aims to keep the hot path cheap and predictable so that
//! metaheuristics can inject guidance without disrupting tight inner loops.

use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
    solution::Solution,
};
use bollard_search::{monitor::search_monitor::SearchCommand, num::SolverNumeric};

use crate::meta::shared;

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
    /// This represents the physical/business cost.
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

/// A trait governing the acceptance logic and termination of the local search.
///
/// This decouples the search mechanism (moving through neighborhoods) from the
/// strategy (Hill Climbing, Simulated Annealing, Tabu Search, etc.).
pub trait Metaheuristic<T>: Send + Sync
where
    T: SolverNumeric,
{
    /// Returns the name of the metaheuristic.
    fn name(&self) -> &str;

    /// Evaluates the assignment of a vessel to a berth at a given start time.
    fn evaluate_assignment(
        &self,
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
    ) -> Option<Evaluation<T>> {
        let weighted_flow_time = shared::calculate_weighted_completion_time(
            model,
            vessel_index,
            berth_index,
            start_time,
        )?;
        Some(Evaluation::new(weighted_flow_time, weighted_flow_time))
    }

    /// Evaluates the assignment without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is within `0..model.num_vessels()` and
    /// `berth_index` is within `0..model.num_berths()`.
    unsafe fn evaluate_assignment_unchecked(
        &self,
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
    ) -> Option<Evaluation<T>> {
        let weighted_flow_time = unsafe {
            shared::calculate_weighted_completion_time_unchecked(
                model,
                vessel_index,
                berth_index,
                start_time,
            )
        }?;
        Some(Evaluation::new(weighted_flow_time, weighted_flow_time))
    }

    /// Called at the start of the search.
    fn on_start(&mut self, model: &Model<T>, initial_solution: &Solution<T>);

    /// Called at the end of the search.
    fn on_end(&mut self, model: &Model<T>, final_solution: &Solution<T>);

    /// Called when the neighbourhood of a solution has been exhausted.
    /// This will return `true` if the search should continue, or `false` if it should terminate.
    fn on_neighbourhood_exhausted(
        &mut self,
        _model: &Model<T>,
        _current: &Solution<T>,
        _best: &Solution<T>,
    ) -> bool {
        false
    }

    /// Determines if the search should proceed to the next iteration.
    fn search_command(
        &mut self,
        iteration: u64,
        model: &Model<T>,
        current_solution: &Solution<T>,
        best_solution: &Solution<T>,
    ) -> SearchCommand;

    /// Decides whether to accept the `candidate` solution over the `current` one.
    ///
    /// The `current` solution is the baseline for the move.
    /// The `best` solution found so far is also provided for context (e.g., aspiration criteria).
    fn should_accept(
        &mut self,
        model: &Model<T>,
        current: &Solution<T>,
        candidate: &Solution<T>,
        best: &Solution<T>,
    ) -> bool;

    /// Called when a move is accepted.
    fn on_accept(&mut self, model: &Model<T>, new_current: &Solution<T>);

    /// Called when a move is rejected.
    fn on_reject(&mut self, model: &Model<T>, rejected_candidate: &Solution<T>);

    /// Called when a new global best solution is found.
    fn on_new_best(&mut self, model: &Model<T>, new_best: &Solution<T>);
}

impl<T> std::fmt::Debug for dyn Metaheuristic<T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Metaheuristic {{ name: {} }}", self.name())
    }
}

impl<T> std::fmt::Display for dyn Metaheuristic<T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Metaheuristic: {}", self.name())
    }
}
