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

//! Tabu Search metaheuristic.
//!
//! This module implements a memory-based search strategy that guides the local search
//! out of local optima by forbidding recently visited solutions. It maintains a
//! "Tabu List" of solution signatures (hashes) with a fixed tenure. If a candidate
//! solution matches an entry in the Tabu List, it is rejected unless it satisfies
//! the **Aspiration Criterion** (i.e., it is better than the global best solution found so far).
//!
//! # Architecture
//!
//! The `TabuSearch<T>` struct implements the `Metaheuristic` trait. It uses a
//! `VecDeque` as a ring buffer to efficiently manage the Tabu List (FIFO). To
//! avoid tight coupling with specific neighborhood operators (e.g., Swap, Relocate),
//! this implementation defines "Tabu" based on the **Solution State** (via hashing)
//! rather than the move attributes. This makes the metaheuristic operator-agnostic
//! and robust across different problem formulations.
//!
//! # Mechanics
//!
//! 1.  **Short-Term Memory:** Recently visited solutions are stored in the Tabu List.
//! 2.  **Tabu Check:** Before accepting a move, the search checks if the candidate's
//!     hash exists in the list.
//! 3.  **Aspiration:** If a candidate is Tabu but its objective value is strictly
//!     better than the global best, the Tabu status is overridden, and the move is accepted.
//! 4.  **Tenure:** The list has a fixed capacity (`tenure`). When full, the oldest
//!     entry is removed to make room for the new one.
//!
//! This strategy is deterministic and pairs well with deterministic operators. It effectively
//! prevents short-term cycling (returning to the same local optimum immediately) and forces
//! the search to explore new regions of the solution space.

use crate::eval::WeightedFlowTimeEvaluator;
use crate::memory::Schedule;
use crate::meta::metaheuristic::Metaheuristic;
use bollard_model::model::Model;
use bollard_search::{monitor::search_monitor::SearchCommand, num::SolverNumeric};
use std::collections::{HashSet, VecDeque};
use std::hash::{Hash, Hasher};

/// A Tabu Search metaheuristic with fixed tenure and solution-hash memory.
///
/// Use `TabuSearch::new(tenure)` to create an instance. A tenure of `7-20` is often
/// a good starting point for scheduling problems.
#[derive(Debug, Clone)]
pub struct TabuSearch<T>
where
    T: SolverNumeric,
{
    evaluator: WeightedFlowTimeEvaluator<T>, // Standard evaluator
    tenure: usize,                           // Maximum size of the tabu list
    tabu_queue: VecDeque<u64>,               // FIFO queue for tenure management
    tabu_set: HashSet<u64>,                  // Hash set for O(1) lookups
}

impl<T> TabuSearch<T>
where
    T: SolverNumeric,
{
    /// Creates a new `TabuSearch` instance with the specified tabu tenure.
    ///
    /// # Panics
    ///
    /// Panics if `tenure` is 0.
    #[inline]
    pub fn new(tenure: usize) -> Self {
        assert!(
            tenure > 0,
            "called `TabuSearch::new()` with tenure {}, but tenure must be greater than 0",
            tenure
        );

        Self {
            evaluator: WeightedFlowTimeEvaluator::new(),
            tenure,
            tabu_queue: VecDeque::with_capacity(tenure),
            tabu_set: HashSet::with_capacity(tenure),
        }
    }

    /// Computes a lightweight hash of the schedule to serve as its signature.
    ///
    /// We hash the objective value and the start times sequence. This is generally
    /// sufficient to distinguish solutions in the local search landscape without
    /// the cost of deep structural hashing.
    #[inline]
    fn hash_schedule(&self, schedule: &Schedule<T>) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        // Hash the objective value (primary discriminator)
        schedule.objective_value().hash(&mut hasher);
        // Hash the berth assignments (structural discriminator)
        schedule.berths().hash(&mut hasher);
        // Hash the start times (temporal discriminator)
        schedule.start_times().hash(&mut hasher);
        hasher.finish()
    }

    /// Adds a solution signature to the tabu list, managing tenure.
    #[inline]
    fn make_tabu(&mut self, hash: u64) {
        // If the list is full, expire the oldest entry
        if self.tabu_queue.len() >= self.tenure
            && let Some(oldest) = self.tabu_queue.pop_front() {
                self.tabu_set.remove(&oldest);
            }

        // Add the new entry
        if self.tabu_set.insert(hash) {
            self.tabu_queue.push_back(hash);
        }
    }
}

impl<T> Metaheuristic<T> for TabuSearch<T>
where
    T: SolverNumeric + Hash,
{
    type Evaluator = WeightedFlowTimeEvaluator<T>;

    fn name(&self) -> &str {
        "TabuSearch"
    }

    fn evaluator(&self) -> &WeightedFlowTimeEvaluator<T> {
        &self.evaluator
    }

    fn on_start(&mut self, _model: &Model<T>, initial_solution: &Schedule<T>) {
        // Clear history on restart
        self.tabu_queue.clear();
        self.tabu_set.clear();

        // Mark initial solution as visited
        let hash = self.hash_schedule(initial_solution);
        self.make_tabu(hash);
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
        best: &Schedule<T>,
    ) -> bool {
        // Basic Improvement Check: Is it better than current?
        // Tabu Search is generally a Best-Improvement method, but in a First-Improvement
        // engine, we accept any improving move that isn't Tabu.
        let is_improvement = candidate.objective_value() < current.objective_value();

        // Tabu Check
        let candidate_hash = self.hash_schedule(candidate);
        let is_tabu = self.tabu_set.contains(&candidate_hash);

        // Aspiration Criterion:
        // Ideally, we accept if it beats the GLOBAL best.
        let is_global_best = candidate.objective_value() < best.objective_value();

        if is_tabu {
            // Only accept a Tabu move if it's a new global best (Aspiration)
            is_global_best
        } else {
            // Accept non-Tabu moves if they improve the current solution
            // (Standard Descent behavior)
            is_improvement
        }
    }

    fn on_accept(&mut self, _model: &Model<T>, new_current: &Schedule<T>) {
        // Record the new current solution in the Tabu list to prevent returning to it
        let hash = self.hash_schedule(new_current);
        self.make_tabu(hash);
    }

    fn on_reject(&mut self, _model: &Model<T>, _rejected_candidate: &Schedule<T>) {
        // No state update on rejection.
        // Some variations add visited candidates to Tabu, but standard TS
        // only tabus the trajectory taken.
    }

    fn on_new_best(&mut self, _model: &Model<T>, _new_best: &Schedule<T>) {
        // No specific action needed; Aspiration is handled in `should_accept`.
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
    fn test_initialization() {
        let ts: TabuSearch<i64> = TabuSearch::new(5);
        assert_eq!(ts.name(), "TabuSearch");
        assert_eq!(ts.tenure, 5);
        assert!(ts.tabu_queue.is_empty());
    }

    #[test]
    #[should_panic(
        expected = "called `TabuSearch::new()` with tenure 0, but tenure must be greater than 0"
    )]
    fn test_zero_tenure_panics() {
        let _ts: TabuSearch<i64> = TabuSearch::new(0);
    }

    #[test]
    fn test_hashing_and_tabu_logic() {
        let mut ts: TabuSearch<i64> = TabuSearch::new(2);
        let model = ModelBuilder::<i64>::new(0, 0).build();

        // S1: Obj 100
        let s1 = sched(100_i64, vec![BerthIndex::new(0)], vec![10]);
        // S2: Obj 90 (Better)
        let s2 = sched(90_i64, vec![BerthIndex::new(0)], vec![20]);
        // S3: Obj 100 (Same as S1 but different time -> different hash)
        let s3 = sched(100_i64, vec![BerthIndex::new(0)], vec![30]);

        // Start with S1
        ts.on_start(&model, &s1);

        // S1 should be in tabu list now
        let h1 = ts.hash_schedule(&s1);
        assert!(ts.tabu_set.contains(&h1));

        // Accept S2 (Normal move)
        assert!(ts.should_accept(&model, &s1, &s2, &s1));
        ts.on_accept(&model, &s2);

        // S2 should be in tabu list
        let h2 = ts.hash_schedule(&s2);
        assert!(ts.tabu_set.contains(&h2));
        assert!(ts.tabu_set.contains(&h1)); // Tenure is 2, so S1 still there

        // Accept S3
        ts.on_accept(&model, &s3);

        // Now tenure (2) is full: [S2, S3]. S1 should be evicted.
        assert!(!ts.tabu_set.contains(&h1));
        assert!(ts.tabu_set.contains(&h2));
        let h3 = ts.hash_schedule(&s3);
        assert!(ts.tabu_set.contains(&h3));
    }

    #[test]
    fn test_aspiration_criterion() {
        let mut ts: TabuSearch<i64> = TabuSearch::new(10);
        // Model dimensions match the schedules (1 vessel, 1 berth)
        let model = ModelBuilder::<i64>::new(1, 1).build();

        // Consistent 1-vessel schedules
        let current = sched(100_i64, vec![BerthIndex::new(0)], vec![0]);
        let best = sched(50_i64, vec![BerthIndex::new(0)], vec![0]); // Global best is 50
        // Candidate strictly better than global best (40)
        let candidate_awesome = sched(40_i64, vec![BerthIndex::new(0)], vec![1]);

        // Initialize lifecycle
        ts.on_start(&model, &current);

        // Force candidate to be Tabu
        let h_awesome = ts.hash_schedule(&candidate_awesome);
        ts.make_tabu(h_awesome);
        assert!(ts.tabu_set.contains(&h_awesome));

        // Candidate is Tabu, BUT it beats global best (40 < 50) -> Aspiration accept
        let accept = ts.should_accept(&model, &current, &candidate_awesome, &best);
        assert!(accept, "Aspiration criterion should override Tabu status");
    }

    #[test]
    fn test_tabu_blocks_non_aspiration_move() {
        let mut ts: TabuSearch<i64> = TabuSearch::new(10);
        // Model dimensions match the schedules (1 vessel, 1 berth)
        let model = ModelBuilder::<i64>::new(1, 1).build();

        let current = sched(100_i64, vec![BerthIndex::new(0)], vec![0]);
        let best = sched(50_i64, vec![BerthIndex::new(0)], vec![0]);
        // Candidate is better than current (90 < 100), but worse than best (90 > 50).
        let candidate_ok = sched(90_i64, vec![BerthIndex::new(0)], vec![1]);

        // Initialize lifecycle
        ts.on_start(&model, &current);

        // Force candidate to be Tabu
        let h_ok = ts.hash_schedule(&candidate_ok);
        ts.make_tabu(h_ok);

        // Candidate is Tabu and does NOT beat the global best -> Reject
        let accept = ts.should_accept(&model, &current, &candidate_ok, &best);
        assert!(
            !accept,
            "Tabu should block moves that do not meet aspiration"
        );
    }
}
