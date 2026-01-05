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

//! Greedy Descent (first-improvement) metaheuristic.
//!
//! This module provides a simple, fast local search strategy that moves to the
//! first neighbor encountered that strictly reduces the objective. In the context
//! of the engine, this behaves as a first‑improvement hill climber: once a better
//! candidate is found, the move is committed immediately and the search continues
//! from the new point. The approach is deterministic for a fixed neighborhood
//! generator and evaluation order, and it tends to reach a local optimum quickly.
//!
//! The acceptance rule compares the candidate and current objective values and
//! proceeds only when the candidate is strictly better. Equal‑cost moves are not
//! accepted in order to avoid random walks on plateaus; this keeps the method
//! simple and eliminates the need for cycle detection. Termination is emergent:
//! when no improving neighbor can be found by the surrounding search engine, the
//! procedure has reached a local optimum under the given move set.
//!
//! # Integration details
//!
//! The type `GreedyDescent` implements the `Metaheuristic` trait and uses
//! `WeightedFlowTimeEvaluator` as its evaluator. It maintains no internal memory
//! or parameters beyond the evaluator, which makes it a useful baseline strategy
//! and a lightweight component to combine with more advanced methods. Because it
//! performs only strict comparisons on the objective, numerical behavior follows
//! the semantics of the solver’s numeric type; ties are treated as non‑improvements.
//!
//! This module is a good fit when you want a fast baseline, to polish solutions
//! before applying more exploratory metaheuristics, or to serve as a deterministic
//! fallback when diversification is not required.

use crate::meta::metaheuristic::Metaheuristic;
use crate::{eval::WeightedFlowTimeEvaluator, memory::Schedule};
use bollard_model::model::Model;
use bollard_search::{monitor::search_monitor::SearchCommand, num::SolverNumeric};

/// A Greedy Descent metaheuristic (First Improvement).
///
/// This strategy navigates the search landscape by accepting any move that reduces
/// the objective cost. Due to the architecture of the `LocalSearchEngine` (which
/// commits to a move immediately upon acceptance), this behaves as a **First Improvement**
/// descent rather than a Best Improvement (Steepest) descent.
///
/// # Attributes
/// * **Acceptance:** Strict improvement (`candidate < current`).
/// * **Termination:** Natural exhaustion (Local Optimum).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GreedyDescent<T>
where
    T: SolverNumeric,
{
    evaluator: WeightedFlowTimeEvaluator<T>,
}

impl<T> Default for GreedyDescent<T>
where
    T: SolverNumeric,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> GreedyDescent<T>
where
    T: SolverNumeric,
{
    /// Creates a new `GreedyDescent` that runs until a local optimum is reached.
    pub fn new() -> Self {
        Self {
            evaluator: WeightedFlowTimeEvaluator::new(),
        }
    }
}

impl<T> Metaheuristic<T> for GreedyDescent<T>
where
    T: SolverNumeric,
{
    type Evaluator = WeightedFlowTimeEvaluator<T>;

    fn name(&self) -> &str {
        "GreedyDescent"
    }

    fn on_start(&mut self, _model: &Model<T>, _initial_solution: &Schedule<T>) {
        // Stateless strategy; no setup required.
    }

    fn search_command(
        &mut self,
        _iteration: u64,
        _model: &Model<T>,
        _best_solution: &Schedule<T>,
    ) -> SearchCommand {
        SearchCommand::Continue
    }

    fn should_accept(
        &mut self,
        _model: &Model<T>,
        current: &Schedule<T>,
        candidate: &Schedule<T>,
        _best: &Schedule<T>,
    ) -> bool {
        // Strict inequality is crucial here.
        // Accepting equal moves (<=) turns this into a random walk on plateaus,
        // which requires cycle detection to be safe. We stick to strict descent.
        candidate.objective_value() < current.objective_value()
    }

    fn on_accept(&mut self, _model: &Model<T>, _new_current: &Schedule<T>) {
        // No internal state (like Tabu lists) to update.
    }

    fn on_reject(&mut self, _model: &Model<T>, _rejected_candidate: &Schedule<T>) {
        // No internal state to update.
    }

    fn on_new_best(&mut self, _model: &Model<T>, _new_best: &Schedule<T>) {
        // No internal state to update.
    }

    fn evaluator(&self) -> &WeightedFlowTimeEvaluator<T> {
        &self.evaluator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bollard_model::index::BerthIndex;
    use bollard_model::model::ModelBuilder;
    use bollard_model::solution::Solution;

    fn sched<T: SolverNumeric>(obj: T, berths: Vec<BerthIndex>, starts: Vec<T>) -> Schedule<T> {
        Schedule::from(Solution::new(obj, berths, starts))
    }

    #[test]
    fn test_name_and_default_new_equivalence() {
        let a: GreedyDescent<i64> = GreedyDescent::default();
        let b: GreedyDescent<i64> = GreedyDescent::new();
        assert_eq!(a.name(), "GreedyDescent");
        assert_eq!(b.name(), "GreedyDescent");
        // Evaluator presence check
        let _ea = a.evaluator();
        let _eb = b.evaluator();
    }

    #[test]
    fn test_search_command_is_continue() {
        let mut mh: GreedyDescent<i64> = GreedyDescent::new();

        // Build a minimal model via ModelBuilder; content is irrelevant to greedy logic.
        let model = ModelBuilder::<i64>::new(0, 0).build();

        // Using empty schedules for best/current context.
        let best = sched(0_i64, vec![], vec![]);

        let cmd = mh.search_command(0, &model, &best);
        assert_eq!(cmd, SearchCommand::Continue);
    }

    #[test]
    fn test_should_accept_strict_improvement() {
        let mut mh: GreedyDescent<i64> = GreedyDescent::new();

        // Build a minimal model via ModelBuilder; acceptance does not consult the model.
        let model = ModelBuilder::<i64>::new(0, 0).build();

        // Create simple schedules with objective values only; berth/start times are arbitrary but valid.
        let current = sched(100_i64, vec![BerthIndex::new(0)], vec![10_i64]);
        let better = sched(99_i64, vec![BerthIndex::new(0)], vec![10_i64]);
        let equal = sched(100_i64, vec![BerthIndex::new(0)], vec![10_i64]);
        let worse = sched(101_i64, vec![BerthIndex::new(0)], vec![10_i64]);

        // Best is irrelevant for strict greedy acceptance; provide a dummy equal to current
        let best = sched(100_i64, vec![BerthIndex::new(0)], vec![10_i64]);

        assert!(
            mh.should_accept(&model, &current, &better, &best),
            "must accept strictly better objective"
        );
        assert!(
            !mh.should_accept(&model, &current, &equal, &best),
            "must reject equal objective to avoid plateau random walk"
        );
        assert!(
            !mh.should_accept(&model, &current, &worse, &best),
            "must reject worse objective"
        );
    }

    #[test]
    fn test_on_start_and_hooks_are_noops() {
        let mut mh: GreedyDescent<i64> = GreedyDescent::new();
        let model = ModelBuilder::<i64>::new(0, 0).build();

        // Use small valid schedules.
        let s0 = sched(5_i64, vec![BerthIndex::new(0)], vec![1_i64]);
        let s1 = sched(4_i64, vec![BerthIndex::new(0)], vec![1_i64]);

        // These should be no-ops and not change internal state nor panic.
        mh.on_start(&model, &s0);
        mh.on_accept(&model, &s1);
        mh.on_reject(&model, &s0);
        mh.on_new_best(&model, &s1);

        // evaluator remains accessible
        let _e = mh.evaluator();
    }
}
