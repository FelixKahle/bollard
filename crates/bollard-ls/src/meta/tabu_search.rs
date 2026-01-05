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
//! A clean, literature-standard implementation of Tabu Search designed for
//! "First-Improvement" engines. It uses short-term memory to prevent the
//! search from cycling back to recently visited solutions.
//!
//! # Mechanisms
//!
//! 1.  **Tabu List:** A FIFO queue of solution signatures (hashes) with a fixed
//!     `tenure`. Prevents returning to specific states for $N$ iterations.
//! 2.  **Aspiration Criterion:** A "Golden Rule" that overrides the Tabu status
//!     if a candidate solution is better than the global best found so far.
//!
//! # Configuration
//!
//! * **`tenure`:** The length of the memory. A value of $\sqrt{N}$ (where $N$
//!   is the number of vessels) is a common rule of thumb.

use crate::eval::WeightedFlowTimeEvaluator;
use crate::memory::Schedule;
use crate::meta::metaheuristic::Metaheuristic;
use bollard_model::model::Model;
use bollard_search::{monitor::search_monitor::SearchCommand, num::SolverNumeric};
use rustc_hash::FxHasher;
use std::collections::{HashSet, VecDeque};
use std::hash::{Hash, Hasher};

/// A Tabu Search metaheuristic with fixed tenure and lightweight solution signatures.
///
/// This variant stores hashes of visited schedules in a short‑term memory to
/// discourage immediate cycling. A candidate that matches a stored signature is
/// rejected unless it satisfies the aspiration criterion (strictly better than
/// the global best). The memory is managed as a FIFO queue with a bounded
/// `tenure`, expiring the oldest entries as new ones arrive. The evaluator
/// remains standard (`WeightedFlowTimeEvaluator`), keeping scoring consistent
/// across metaheuristics.
#[derive(Debug, Clone)]
pub struct TabuSearch<T>
where
    T: SolverNumeric,
{
    evaluator: WeightedFlowTimeEvaluator<T>, // Evaluator for the objective
    tenure: usize,                           // Size of the Tabu memory
    tabu_queue: VecDeque<u64>,               // FIFO for expiring old entries
    tabu_set: HashSet<u64>,                  // HashSet for O(1) lookups
}

impl<T> TabuSearch<T>
where
    T: SolverNumeric,
{
    /// Creates a new Tabu Search with the specified memory size (tenure).
    ///
    /// # Panics
    ///
    /// Panics if `tenure` is 0.
    pub fn new(tenure: usize) -> Self {
        assert!(tenure > 0, "called `TabuSearch::new()` with zero tenure");
        Self {
            evaluator: WeightedFlowTimeEvaluator::new(),
            tenure,
            tabu_queue: VecDeque::with_capacity(tenure),
            tabu_set: HashSet::with_capacity(tenure),
        }
    }

    /// Generates a lightweight signature for the schedule.
    /// We hash the Objective and the Structure (Berths + Start Times).
    #[inline]
    fn hash_schedule(&self, schedule: &Schedule<T>) -> u64 {
        let mut hasher = FxHasher::default(); // Fast non-cryptographic hash
        schedule.objective_value().hash(&mut hasher);
        schedule.berths().hash(&mut hasher);
        schedule.start_times().hash(&mut hasher);
        hasher.finish()
    }

    /// Records a solution in the Tabu list, handling tenure expiration.
    #[inline]
    fn make_tabu(&mut self, hash: u64) {
        if self.tabu_queue.len() >= self.tenure
            && let Some(oldest) = self.tabu_queue.pop_front()
        {
            self.tabu_set.remove(&oldest);
        }

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
        self.tabu_queue.clear();
        self.tabu_set.clear();

        let hash = self.hash_schedule(initial_solution);
        self.make_tabu(hash);
    }

    fn search_command(&mut self, _: u64, _: &Model<T>, _: &Schedule<T>) -> SearchCommand {
        SearchCommand::Continue
    }

    fn should_accept(
        &mut self,
        _model: &Model<T>,
        current: &Schedule<T>,
        candidate: &Schedule<T>,
        best: &Schedule<T>,
    ) -> bool {
        let is_improvement = candidate.objective_value() < current.objective_value();
        let is_global_best = candidate.objective_value() < best.objective_value();

        let hash = self.hash_schedule(candidate);
        let is_tabu = self.tabu_set.contains(&hash);

        if is_tabu {
            is_global_best
        } else {
            is_improvement
        }
    }

    fn on_accept(&mut self, _model: &Model<T>, new_current: &Schedule<T>) {
        // We moved to a new solution. Mark it as Tabu so we don't
        // cycle back to it immediately.
        let hash = self.hash_schedule(new_current);
        self.make_tabu(hash);
    }

    // No-ops for these hooks in standard Tabu Search
    fn on_reject(&mut self, _model: &Model<T>, _rejected: &Schedule<T>) {}
    fn on_new_best(&mut self, _model: &Model<T>, _new_best: &Schedule<T>) {}
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
        assert!(ts.tabu_set.is_empty());
    }

    #[test]
    #[should_panic(expected = "called `TabuSearch::new()` with zero tenure")]
    fn test_zero_tenure_panics() {
        let _ts: TabuSearch<i64> = TabuSearch::new(0);
    }

    #[test]
    fn test_search_command_continue() {
        let mut ts: TabuSearch<i64> = TabuSearch::new(3);
        let model = ModelBuilder::<i64>::new(1, 1).build();
        let best = sched(0_i64, vec![BerthIndex::new(0)], vec![0_i64]);
        assert_eq!(ts.search_command(0, &model, &best), SearchCommand::Continue);
    }

    #[test]
    fn test_on_start_clears_and_marks_initial() {
        let mut ts: TabuSearch<i64> = TabuSearch::new(3);
        let model = ModelBuilder::<i64>::new(1, 1).build();

        // Pre-populate with some junk
        ts.tabu_queue.push_back(111);
        ts.tabu_set.insert(111);
        ts.tabu_queue.push_back(222);
        ts.tabu_set.insert(222);

        let init = sched(100_i64, vec![BerthIndex::new(0)], vec![0_i64]);
        let h_init = ts.hash_schedule(&init);

        ts.on_start(&model, &init);

        // Cleared previous and added initial
        assert_eq!(ts.tabu_queue.len(), 1);
        assert_eq!(ts.tabu_set.len(), 1);
        assert_eq!(ts.tabu_queue.back().copied(), Some(h_init));
        assert!(ts.tabu_set.contains(&h_init));
        assert!(!ts.tabu_set.contains(&111));
        assert!(!ts.tabu_set.contains(&222));
    }

    #[test]
    fn test_should_accept_rules_non_tabu() {
        let mut ts: TabuSearch<i64> = TabuSearch::new(5);
        let model = ModelBuilder::<i64>::new(1, 1).build();

        let current = sched(100_i64, vec![BerthIndex::new(0)], vec![0_i64]);
        let better = sched(90_i64, vec![BerthIndex::new(0)], vec![1_i64]);
        let equal = sched(100_i64, vec![BerthIndex::new(0)], vec![2_i64]);
        let worse = sched(110_i64, vec![BerthIndex::new(0)], vec![3_i64]);
        let best = current.clone();

        ts.on_start(&model, &current);

        assert!(
            ts.should_accept(&model, &current, &better, &best),
            "Non‑tabu strict improvement should be accepted"
        );
        assert!(
            !ts.should_accept(&model, &current, &equal, &best),
            "Equal objective is not a strict improvement"
        );
        assert!(
            !ts.should_accept(&model, &current, &worse, &best),
            "Worse objective should be rejected when not tabu and not an improvement"
        );
    }

    #[test]
    fn test_make_tabu_eviction_fifo() {
        // Tenure 2 forces FIFO behavior
        let mut ts: TabuSearch<i64> = TabuSearch::new(2);

        let s1 = sched(100_i64, vec![BerthIndex::new(0)], vec![10_i64]);
        let s2 = sched(90_i64, vec![BerthIndex::new(0)], vec![20_i64]);
        let s3 = sched(80_i64, vec![BerthIndex::new(0)], vec![30_i64]);

        let h1 = ts.hash_schedule(&s1);
        let h2 = ts.hash_schedule(&s2);
        let h3 = ts.hash_schedule(&s3);

        ts.make_tabu(h1);
        ts.make_tabu(h2);
        assert!(ts.tabu_set.contains(&h1));
        assert!(ts.tabu_set.contains(&h2));
        assert_eq!(ts.tabu_queue.len(), 2);

        // Adding a third should evict h1
        ts.make_tabu(h3);
        assert!(!ts.tabu_set.contains(&h1));
        assert!(ts.tabu_set.contains(&h2));
        assert!(ts.tabu_set.contains(&h3));
        assert_eq!(ts.tabu_queue.len(), 2);
        assert_eq!(ts.tabu_queue.front().copied(), Some(h2));
        assert_eq!(ts.tabu_queue.back().copied(), Some(h3));
    }

    #[test]
    fn test_readding_same_hash_does_not_duplicate() {
        let mut ts: TabuSearch<i64> = TabuSearch::new(3);

        let s = sched(100_i64, vec![BerthIndex::new(0)], vec![10_i64]);
        let h = ts.hash_schedule(&s);

        ts.make_tabu(h);
        ts.make_tabu(h); // re-add same
        ts.make_tabu(h); // again

        // Only inserted once
        assert!(ts.tabu_set.contains(&h));
        assert_eq!(ts.tabu_set.len(), 1);
        assert_eq!(ts.tabu_queue.len(), 1);
        assert_eq!(ts.tabu_queue.front().copied(), Some(h));
    }

    #[test]
    fn test_on_accept_marks_new_current_as_tabu() {
        let mut ts: TabuSearch<i64> = TabuSearch::new(3);
        let model = ModelBuilder::<i64>::new(1, 1).build();

        let current = sched(100_i64, vec![BerthIndex::new(0)], vec![0_i64]);
        let next = sched(90_i64, vec![BerthIndex::new(0)], vec![1_i64]);

        ts.on_start(&model, &current);
        let h_current = ts.hash_schedule(&current);
        assert!(ts.tabu_set.contains(&h_current));

        ts.on_accept(&model, &next);
        let h_next = ts.hash_schedule(&next);
        assert!(ts.tabu_set.contains(&h_next));
        assert_eq!(ts.tabu_queue.back().copied(), Some(h_next));
    }

    #[test]
    fn test_hash_schedule_sensitivity() {
        let ts: TabuSearch<i64> = TabuSearch::new(3);

        // Base
        let s_base = sched(100_i64, vec![BerthIndex::new(0)], vec![10_i64]);
        let h_base = ts.hash_schedule(&s_base);

        // Same contents -> same hash
        let s_same = sched(100_i64, vec![BerthIndex::new(0)], vec![10_i64]);
        let h_same = ts.hash_schedule(&s_same);
        assert_eq!(h_base, h_same, "identical schedules must hash the same");

        // Different objective -> different hash
        let s_obj = sched(101_i64, vec![BerthIndex::new(0)], vec![10_i64]);
        let h_obj = ts.hash_schedule(&s_obj);
        assert_ne!(h_base, h_obj, "objective value should affect the hash");

        // Different berth assignment -> different hash
        // Note: With a single vessel, varying berth index changes structure.
        let s_berth = sched(100_i64, vec![BerthIndex::new(1)], vec![10_i64]);
        let h_berth = ts.hash_schedule(&s_berth);
        assert_ne!(h_base, h_berth, "berths should affect the hash");

        // Different start time -> different hash
        let s_start = sched(100_i64, vec![BerthIndex::new(0)], vec![11_i64]);
        let h_start = ts.hash_schedule(&s_start);
        assert_ne!(h_base, h_start, "start times should affect the hash");
    }

    // Existing tests from earlier conversations (kept and validated)

    #[test]
    fn test_tabu_prevents_cycling() {
        // Tenure = 10
        let mut ts: TabuSearch<i64> = TabuSearch::new(10);
        let model = ModelBuilder::<i64>::new(0, 0).build();

        // Solution A (Cost 100)
        let s_a = sched(100, vec![BerthIndex::new(0)], vec![10]);
        // Solution B (Cost 90)
        let s_b = sched(90, vec![BerthIndex::new(0)], vec![20]);

        let best = s_a.clone();

        // Start at A
        ts.on_start(&model, &s_a);

        // A is now Tabu. Move to B.
        assert!(ts.should_accept(&model, &s_a, &s_b, &best));
        ts.on_accept(&model, &s_b);

        // B is now Tabu.
        // Try to move BACK to A (Cycling).
        // A is in the Tabu List (from start). A is NOT better than Global Best (100 !< 100).
        // Result: REJECT.
        let accept_back = ts.should_accept(&model, &s_b, &s_a, &best);
        assert!(
            !accept_back,
            "Tabu Search should prevent returning to Solution A"
        );
    }

    #[test]
    fn test_aspiration_overrides_tabu() {
        let mut ts: TabuSearch<i64> = TabuSearch::new(10);
        let model = ModelBuilder::<i64>::new(0, 0).build();

        let s_current = sched(100, vec![], vec![]);
        let s_best = sched(50, vec![], vec![]); // Global best is 50

        // Candidate is Tabu (we force it)
        let s_candidate = sched(40, vec![BerthIndex::new(0)], vec![1]);
        let h = ts.hash_schedule(&s_candidate);
        ts.make_tabu(h);

        // But Candidate (40) is better than Best (50).
        // Result: ACCEPT (Aspiration).
        let accept = ts.should_accept(&model, &s_current, &s_candidate, &s_best);
        assert!(accept, "Aspiration should override Tabu status");
    }
}
