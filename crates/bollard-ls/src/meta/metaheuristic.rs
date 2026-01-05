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

use crate::{eval::AssignmentEvaluator, memory::Schedule};
use bollard_model::model::Model;
use bollard_search::{monitor::search_monitor::SearchCommand, num::SolverNumeric};

/// A trait governing the acceptance logic and termination of the local search.
///
/// This decouples the search mechanism (moving through neighborhoods) from the
/// strategy (Hill Climbing, Simulated Annealing, Tabu Search, etc.).
pub trait Metaheuristic<T>: Send + Sync
where
    T: SolverNumeric,
{
    /// The type of assignment evaluator used by the metaheuristic.
    type Evaluator: AssignmentEvaluator<T>;

    /// Returns the name of the metaheuristic.
    fn name(&self) -> &str;

    /// Provides mutable access to the assignment evaluator used by the search.
    fn evaluator(&self) -> &Self::Evaluator;

    /// Called at the start of the search.
    fn on_start(&mut self, model: &Model<T>, initial_solution: &Schedule<T>);

    /// Determines if the search should proceed to the next iteration.
    fn search_command(
        &mut self,
        iteration: u64,
        model: &Model<T>,
        best_solution: &Schedule<T>,
    ) -> SearchCommand;

    /// Decides whether to accept the `candidate` solution over the `current` one.
    ///
    /// The `current` solution is the baseline for the move.
    /// The `best` solution found so far is also provided for context (e.g., aspiration criteria).
    fn should_accept(
        &mut self,
        model: &Model<T>,
        current: &Schedule<T>,
        candidate: &Schedule<T>,
        best: &Schedule<T>,
    ) -> bool;

    /// Called when a move is accepted.
    fn on_accept(&mut self, model: &Model<T>, new_current: &Schedule<T>);

    /// Called when a move is rejected.
    fn on_reject(&mut self, model: &Model<T>, rejected_candidate: &Schedule<T>);

    /// Called when a new global best solution is found.
    fn on_new_best(&mut self, model: &Model<T>, new_best: &Schedule<T>);
}

impl<T, E> std::fmt::Debug for dyn Metaheuristic<T, Evaluator = E>
where
    T: SolverNumeric,
    E: AssignmentEvaluator<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Metaheuristic {{ name: {} }}", self.name())
    }
}

impl<T, E> std::fmt::Display for dyn Metaheuristic<T, Evaluator = E>
where
    T: SolverNumeric,
    E: AssignmentEvaluator<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Metaheuristic: {}", self.name())
    }
}
