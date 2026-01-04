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

//! Local search operators and a compound selector based on a Multi-Armed Bandit strategy.
//!
//! This module defines the `LocalSearchOperator` trait for neighborhood exploration in
//! local search, along with an adapter to use operators as iterators and a compound
//! operator that prioritizes sub-operators using UCB1-based scoring.
//!
//! An operator maintains state while exploring a neighborhood. It performs a preparation
//! step when a new incumbent solution is accepted, applies successive mutations via
//! the `next_neighbor` method, and can reset its internal cursor without repeating
//! the analysis. The external search engine handles decoding, evaluation, and rollback
//! of changes applied to the genotype.
//!
//! The `LocalSearchOperatorIterator` allows integrating an operator into iterator-style
//! workflows by repeatedly invoking `next_neighbor` until the neighborhood is exhausted.
//!
//! The `MultiArmedBanditCompoundOperator` coordinates multiple sub-operators. It tracks
//! sampling statistics and average improvements, normalizes the exploitation term by the
//! best improvement observed so far, and uses an exploration bonus that decreases with
//! the number of samples for each sub-operator and increases with the total number of
//! samples. The ranked order of sub-operators is updated using the current scores,
//! and ties are resolved deterministically by index.

use crate::{
    memory::Schedule, mutator::Mutator, neighborhood::neighborhoods::Neighborhoods,
    queue::VesselPriorityQueue,
};
use bollard_search::num::SolverNumeric;
use std::cmp::Ordering;

/// A stateful operator that explores a specific neighborhood in a local search.
///
/// This trait behaves similarly to an [Iterator], but is designed for the high-performance
/// requirements of local search where mutations are applied to an external state
/// and must be reversible.
///
/// ## Lifecycle
///
/// 1. **`prepare`**: Called once when the search reaches a new "best" solution. The operator
///    analyzes the solution and prepares its internal list of potential moves.
/// 2. **`next_neighbor`**: Called repeatedly. Each call applies exactly one mutation to the
///    genotype (queue). If it returns `true`, the search engine decodes and evaluates the candidate.
///    If it returns `false`, the neighborhood is exhausted.
/// 3. **`reset`**: Reverts the operator's internal cursor or counters to the start of the
///    current neighborhood without re-analyzing the schedule.
pub trait LocalSearchOperator<T, N>
where
    T: SolverNumeric,
    N: Neighborhoods,
{
    /// Returns the name of the operator for logging and identification purposes.
    fn name(&self) -> &str;

    /// Prepares the operator to explore the neighborhood of a new solution.
    ///
    /// This is the "heavy lifting" phase where the operator might:
    /// - Identify "bottleneck" vessels in the `schedule` to target for mutation.
    /// - Pre-calculate a list of vessel pairs for swapping.
    /// - Shuffle or re-order its internal move list if the operator is stochastic.
    fn prepare(&mut self, schedule: &Schedule<T>, queue: &VesselPriorityQueue, neighborhoods: &N);

    /// Applies the next mutation in the neighborhood sequence.
    ///
    /// The operator uses the provided `mutator` to modify the `VesselPriorityQueue`.
    /// The search engine will handle the evaluation and potential rollback of these changes.
    ///
    /// # Returns
    /// - `true`: A mutation was successfully applied. The search engine should now evaluate
    ///   the new candidate.
    /// - `false`: No more neighbors exist in this specific neighborhood.
    ///
    /// # Note
    /// It is expected that the operator manages an internal "cursor" to ensure that
    /// subsequent calls result in different neighbors.
    fn next_neighbor(&mut self, schedule: &Schedule<T>, mutator: &mut Mutator<T>, n: &N) -> bool;

    /// Resets the operator's internal state to the beginning of the neighborhood.
    ///
    /// Unlike `prepare`, `reset` should not perform expensive re-analysis of the schedule.
    /// It simply moves the internal iteration cursor back to the first potential neighbor.
    /// This is useful for multi-restart strategies or meta-heuristics that need to
    /// re-examine the same neighborhood multiple times.
    fn reset(&mut self);
}

impl<T, N> std::fmt::Debug for dyn LocalSearchOperator<T, N>
where
    T: SolverNumeric,
    N: Neighborhoods,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LocalSearchOperator {{ name: {} }}", self.name())
    }
}

impl<T, N> std::fmt::Display for dyn LocalSearchOperator<T, N>
where
    T: SolverNumeric,
    N: Neighborhoods,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// An iterator adapter that allows using a `LocalSearchOperator` in iterator contexts.
#[derive(Debug)]
pub struct LocalSearchOperatorIterator<'a, T, N, O>
where
    T: SolverNumeric,
    N: Neighborhoods,
    O: LocalSearchOperator<T, N>,
{
    operator: &'a mut O,
    neighborhoods: &'a N,
    schedule: &'a Schedule<T>,
    mutator: &'a mut Mutator<'a, T>,
}

impl<'a, T, N, O> LocalSearchOperatorIterator<'a, T, N, O>
where
    T: SolverNumeric,
    N: Neighborhoods,
    O: LocalSearchOperator<T, N>,
{
    pub fn new(
        operator: &'a mut O,
        neighborhoods: &'a N,
        schedule: &'a Schedule<T>,
        mutator: &'a mut Mutator<'a, T>,
    ) -> Self {
        Self {
            operator,
            neighborhoods,
            schedule,
            mutator,
        }
    }
}

impl<'a, T, N, O> Iterator for LocalSearchOperatorIterator<'a, T, N, O>
where
    T: SolverNumeric,
    N: Neighborhoods,
    O: LocalSearchOperator<T, N>,
{
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        if self
            .operator
            .next_neighbor(self.schedule, self.mutator, self.neighborhoods)
        {
            Some(())
        } else {
            None
        }
    }
}

/// A compound operator that selects one random sub-operator for each local search iteration.
///
/// Unlike a sequential compound operator, this operator introduces stochasticity by
/// picking a single random neighborhood strategy when `prepare` is called and sticking
/// with it until the next solution is found.
///
/// This is effective for:
/// - **Diversification**: Preventing the search from following deterministic paths.
/// - **Escape**: breaking out of local optima that a specific deterministic operator might get stuck in.
#[derive(Debug)]
pub struct RandomOperatorCompoundOperator<T, N, R>
where
    T: SolverNumeric,
    N: Neighborhoods,
    R: rand::Rng,
{
    operators: Vec<Box<dyn LocalSearchOperator<T, N>>>,
    rng: R,
    /// The index of the operator currently selected for the active neighborhood.
    current_index: usize,
}

impl<T, N, R> RandomOperatorCompoundOperator<T, N, R>
where
    T: SolverNumeric,
    N: Neighborhoods,
    R: rand::Rng,
{
    /// Creates a new `RandomOperatorCompoundOperator`.
    ///
    /// # Panics
    /// Panics if `operators` is empty.
    pub fn new(operators: Vec<Box<dyn LocalSearchOperator<T, N>>>, rng: R) -> Self {
        assert!(!operators.is_empty(), "Operators list cannot be empty");
        Self {
            operators,
            rng,
            current_index: 0,
        }
    }
}

impl<T, N, R> LocalSearchOperator<T, N> for RandomOperatorCompoundOperator<T, N, R>
where
    T: SolverNumeric,
    N: Neighborhoods,
    R: rand::Rng,
{
    fn name(&self) -> &str {
        "RandomOperatorCompoundOperator"
    }

    fn prepare(&mut self, schedule: &Schedule<T>, queue: &VesselPriorityQueue, neighborhoods: &N) {
        if self.operators.is_empty() {
            return;
        }

        self.current_index = self.rng.random_range(0..self.operators.len());
        if let Some(op) = self.operators.get_mut(self.current_index) {
            op.prepare(schedule, queue, neighborhoods);
        }
    }

    fn next_neighbor(
        &mut self,
        schedule: &Schedule<T>,
        mutator: &mut Mutator<T>,
        neighborhoods: &N,
    ) -> bool {
        if let Some(op) = self.operators.get_mut(self.current_index) {
            op.next_neighbor(schedule, mutator, neighborhoods)
        } else {
            false
        }
    }

    fn reset(&mut self) {
        if let Some(op) = self.operators.get_mut(self.current_index) {
            op.reset();
        }
    }
}

/// A compound operator that iterates through sub-operators in a fixed sequential order.
///
/// This operator provides a deterministic "Round Robin" strategy. It starts with the first
/// sub-operator and lets it run until it is exhausted (returns `false`). Then, it immediately
/// switches to the next operator in the list.
///
/// It returns `false` (exhausted) only when **all** sub-operators have been exhausted for
/// the current solution.
#[derive(Debug)]
pub struct RoundRobinCompoundOperator<T, N>
where
    T: SolverNumeric,
    N: Neighborhoods,
{
    operators: Vec<Box<dyn LocalSearchOperator<T, N>>>,
    current_index: usize,
    op_started: Vec<bool>,
}

impl<T, N> RoundRobinCompoundOperator<T, N>
where
    T: SolverNumeric,
    N: Neighborhoods,
{
    /// Creates a new `RoundRobinCompoundOperator`.
    pub fn new(operators: Vec<Box<dyn LocalSearchOperator<T, N>>>) -> Self {
        let len = operators.len();
        Self {
            operators,
            current_index: 0,
            op_started: vec![false; len],
        }
    }
}

impl<T, N> LocalSearchOperator<T, N> for RoundRobinCompoundOperator<T, N>
where
    T: SolverNumeric,
    N: Neighborhoods,
{
    fn name(&self) -> &str {
        "RoundRobinCompoundOperator"
    }

    fn prepare(
        &mut self,
        _schedule: &Schedule<T>,
        _queue: &VesselPriorityQueue,
        _neighborhoods: &N,
    ) {
        self.current_index = 0;
        self.op_started.fill(false);
    }

    fn next_neighbor(
        &mut self,
        schedule: &Schedule<T>,
        mutator: &mut Mutator<T>,
        neighborhoods: &N,
    ) -> bool {
        loop {
            if self.current_index >= self.operators.len() {
                return false;
            }

            let op = &mut self.operators[self.current_index];
            if !self.op_started[self.current_index] {
                let queue = mutator.queue();
                op.prepare(schedule, queue, neighborhoods);
                self.op_started[self.current_index] = true;
            }

            if op.next_neighbor(schedule, mutator, neighborhoods) {
                return true;
            }

            self.current_index += 1;
        }
    }

    fn reset(&mut self) {
        for op in self.operators.iter_mut() {
            op.reset();
        }
        self.current_index = 0;
        self.op_started.fill(false);
    }
}

/// Internal statistics for the Multi-Armed Bandit operator.
///
/// Tracks the number of samples and average improvements for each sub-operator.
/// Used to compute UCB1 scores for operator selection.
#[derive(Debug, Default)]
struct BanditStats {
    total_samples: usize,       // Total number of samples across all operators
    samples_per_op: Vec<usize>, // len = number of operators
    avg_improvements: Vec<f64>, // len = number of operators

    /// The maximum improvement seen so far across all operators.
    /// Used to normalize the exploitation term to the [0, 1] range.
    global_max_improvement: f64,
}

impl BanditStats {
    /// Creates a new `BanditStats` instance for the specified number of operators.
    #[inline]
    fn new(size: usize) -> Self {
        Self {
            total_samples: 0,
            samples_per_op: vec![0; size],
            avg_improvements: vec![0.0; size],
            // Initialize to 1.0 to prevent division by zero on the first iteration.
            // As soon as a real improvement > 1.0 is found, this will scale up.
            global_max_improvement: 1.0,
        }
    }

    /// Computes the UCB1 score for the specified operator index.
    ///
    /// # Panics
    ///
    /// In debug builds, this method will panic if `index` is not within `0..self.samples_per_op.len()`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` is within `0..self.samples_per_op.len()`.
    unsafe fn get_score_unchecked(&self, index: usize, exploration_coef: f64) -> f64 {
        debug_assert!(
            index < self.samples_per_op.len(),
            "called `BanditStats::get_score_unchecked` with index out of bounds: the len is {} but the index is {}",
            self.samples_per_op.len(),
            index,
        );

        let n_i = unsafe { *self.samples_per_op.get_unchecked(index) as f64 };
        if n_i == 0.0 {
            return f64::INFINITY;
        }

        let total_n = self.total_samples as f64;

        let raw_exploitation = unsafe { *self.avg_improvements.get_unchecked(index) };
        let normalized_exploitation = raw_exploitation / self.global_max_improvement;
        let bonus = exploration_coef * ((2.0 * (1.0 + total_n).ln()) / n_i).sqrt();

        normalized_exploitation + bonus
    }

    /// Updates the average improvement for the specified operator index using
    /// the provided delta and learning rate (alpha).
    ///
    /// # Panics
    ///
    /// In debug builds, this method will panic if `index` is not within `0..self.avg_improvements.len()`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` is within `0..self.avg_improvements.len()`.
    unsafe fn update_improvement_unchecked(&mut self, index: usize, delta: f64, alpha: f64) {
        debug_assert!(
            index < self.avg_improvements.len(),
            "called `BanditStats::update_improvement_unchecked` with index out of bounds: the len is {} but the index is {}",
            self.avg_improvements.len(),
            index,
        );

        // Track the global maximum improvement to maintain correct normalization scaling
        if delta > self.global_max_improvement {
            self.global_max_improvement = delta;
        }

        let current_avg = unsafe { *self.avg_improvements.get_unchecked(index) };
        let new_avg = current_avg + alpha * (delta - current_avg);
        unsafe {
            *self.avg_improvements.get_unchecked_mut(index) = new_avg;
        }
    }
}

/// A compound operator that selects sub-operators using a Multi-Armed Bandit (UCB1) strategy.
pub struct MultiArmedBanditCompoundOperator<T, N>
where
    T: SolverNumeric,
    N: Neighborhoods,
{
    operators: Vec<Box<dyn LocalSearchOperator<T, N>>>,
    memory_coeff: f64,
    exploration_coeff: f64,

    // State
    stats: BanditStats,
    ranked_indices: Vec<usize>,
    active_rank_idx: usize,
    op_started: Vec<bool>,
    last_obj: Option<T>,
}

impl<T, N> MultiArmedBanditCompoundOperator<T, N>
where
    T: SolverNumeric,
    N: Neighborhoods,
{
    /// Creates a new `MultiArmedBanditCompoundOperator` with the specified sub-operators
    /// and configuration parameters.
    #[inline]
    pub fn new(
        operators: Vec<Box<dyn LocalSearchOperator<T, N>>>,
        memory_coeff: f64,
        exploration_coeff: f64,
    ) -> Self {
        let n = operators.len();
        Self {
            operators,
            memory_coeff,
            exploration_coeff,
            stats: BanditStats::new(n),
            ranked_indices: (0..n).collect(),
            active_rank_idx: 0,
            op_started: vec![false; n],
            last_obj: None,
        }
    }

    #[inline]
    pub fn with_defaults(operators: Vec<Box<dyn LocalSearchOperator<T, N>>>) -> Self {
        // memory_coeff: 0.2 (React faster to changes in operator effectiveness)
        // exploration_coeff: 1.414 (Standard UCB1 constant)
        Self::new(operators, 0.2, std::f64::consts::SQRT_2)
    }

    fn sort_by_score(&mut self) {
        let c = self.exploration_coeff;
        let stats = &self.stats;
        self.ranked_indices.sort_by(|&a, &b| unsafe {
            stats
                .get_score_unchecked(b, c)
                .partial_cmp(&stats.get_score_unchecked(a, c))
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
    }
}

impl<T, N> LocalSearchOperator<T, N> for MultiArmedBanditCompoundOperator<T, N>
where
    T: SolverNumeric,
    N: Neighborhoods,
{
    fn name(&self) -> &str {
        "MultiArmedBanditCompoundOperator"
    }

    fn prepare(
        &mut self,
        schedule: &Schedule<T>,
        _queue: &VesselPriorityQueue,
        _neighborhoods: &N,
    ) {
        let current_obj = schedule.objective_value();

        // Learning Phase: Update stats based on the last accepted move
        if let Some(prev_obj) = self.last_obj {
            let improvement =
                (prev_obj.to_f64().unwrap_or(0.0) - current_obj.to_f64().unwrap_or(0.0)).max(0.0);

            // Note: We update even if improvement is 0.0 to decay the average
            if improvement >= 0.0 {
                let last_op_idx = self.ranked_indices[self.active_rank_idx];
                unsafe {
                    self.stats.update_improvement_unchecked(
                        last_op_idx,
                        improvement,
                        self.memory_coeff,
                    )
                };
            }
        }
        self.last_obj = Some(current_obj);

        // Strategy Phase: Re-rank operators and reset cursors
        self.sort_by_score();
        self.active_rank_idx = 0;
        self.op_started.fill(false);
    }

    fn next_neighbor(
        &mut self,
        schedule: &Schedule<T>,
        mutator: &mut Mutator<T>,
        neighborhoods: &N,
    ) -> bool {
        let num_ops = self.operators.len();
        if num_ops == 0 {
            return false;
        }

        let start_rank = self.active_rank_idx;
        loop {
            let op_idx = self.ranked_indices[self.active_rank_idx];
            let op = &mut self.operators[op_idx];

            // Lazy preparation
            if !self.op_started[op_idx] {
                op.prepare(schedule, mutator.queue(), neighborhoods);
                self.op_started[op_idx] = true;
            }

            if op.next_neighbor(schedule, mutator, neighborhoods) {
                self.stats.total_samples += 1;
                self.stats.samples_per_op[op_idx] += 1;
                return true;
            }

            // Move to next operator in ranked order
            self.active_rank_idx = (self.active_rank_idx + 1) % num_ops;
            if self.active_rank_idx == start_rank {
                return false;
            }
        }
    }

    fn reset(&mut self) {
        self.operators.iter_mut().for_each(|op| op.reset());
        self.active_rank_idx = 0;
        self.op_started.fill(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neighborhood::topology::StaticTopology;
    use bollard_search::num::SolverNumeric;
    use rand::{SeedableRng, rngs::StdRng};
    use std::cell::RefCell;
    use std::rc::Rc;

    /// A mock neighborhood topology that always returns false (not relevant for flow control).
    #[derive(Debug)]
    struct MockTopology;
    impl Neighborhoods for MockTopology {
        fn num_vessels(&self) -> usize {
            10
        }
        unsafe fn are_neighbors_unchecked(
            &self,
            _a: bollard_model::index::VesselIndex,
            _b: bollard_model::index::VesselIndex,
        ) -> bool {
            true
        }
        unsafe fn neighbors_of_unchecked(
            &self,
            _v: bollard_model::index::VesselIndex,
        ) -> &[bollard_model::index::VesselIndex] {
            &[]
        }
    }

    /// A smart operator that:
    /// 1. Has an ID.
    /// 2. Returns `true` (success) for a fixed number of iterations (`limit`), then `false`.
    /// 3. Logs its actions to a shared vector so we can verify execution order.
    struct TrackingOperator {
        id: usize,
        limit: usize,
        calls: usize,
        log: Rc<RefCell<Vec<usize>>>, // Logs the ID every time next_neighbor is called successfully
        prepare_log: Rc<RefCell<Vec<usize>>>, // Logs the ID when prepare is called
    }

    impl TrackingOperator {
        fn new(
            id: usize,
            limit: usize,
            log: Rc<RefCell<Vec<usize>>>,
            prepare_log: Rc<RefCell<Vec<usize>>>,
        ) -> Self {
            Self {
                id,
                limit,
                calls: 0,
                log,
                prepare_log,
            }
        }
    }

    impl<T: SolverNumeric, N: Neighborhoods> LocalSearchOperator<T, N> for TrackingOperator {
        fn name(&self) -> &str {
            "TrackingOperator"
        }

        fn prepare(&mut self, _schedule: &Schedule<T>, _queue: &VesselPriorityQueue, _n: &N) {
            self.calls = 0; // Reset internal counter on prepare
            self.prepare_log.borrow_mut().push(self.id);
        }

        fn next_neighbor(
            &mut self,
            _schedule: &Schedule<T>,
            _mutator: &mut Mutator<T>,
            _n: &N,
        ) -> bool {
            if self.calls < self.limit {
                self.calls += 1;
                self.log.borrow_mut().push(self.id);
                true
            } else {
                false
            }
        }

        fn reset(&mut self) {
            self.calls = 0;
        }
    }

    // Helper to generate minimal dummy structs needed for method signatures
    fn get_mocks() -> (
        Schedule<i64>,
        VesselPriorityQueue,
        crate::undo::UndoLog,
        MockTopology,
    ) {
        // Minimal empty schedule (operators in these tests don't use its contents)
        let sched = Schedule::new(0i64, Vec::new(), Vec::new());

        // Owned queue and undo log to build a Mutator in each test
        let q_owned = VesselPriorityQueue::new();
        let log_owned = crate::undo::UndoLog::new(16, 16);

        let topo = MockTopology;

        (sched, q_owned, log_owned, topo)
    }

    #[test]
    fn test_round_robin_sequence_and_exhaustion() {
        let execution_log = Rc::new(RefCell::new(Vec::new()));
        let prepare_log = Rc::new(RefCell::new(Vec::new()));

        // Op 0 runs 2 times, Op 1 runs 1 time, Op 2 runs 0 times.
        let op0 = Box::new(TrackingOperator::new(
            0,
            2,
            execution_log.clone(),
            prepare_log.clone(),
        ));
        let op1 = Box::new(TrackingOperator::new(
            1,
            1,
            execution_log.clone(),
            prepare_log.clone(),
        ));
        let op2 = Box::new(TrackingOperator::new(
            2,
            0,
            execution_log.clone(),
            prepare_log.clone(),
        ));

        let mut rr = RoundRobinCompoundOperator::new(vec![op0, op1, op2]);
        let (sched, mut q, mut log, topo) = get_mocks();

        // Prepare BEFORE creating the mutator to avoid overlapping borrows
        rr.prepare(&sched, &q, &topo);

        let mut m = Mutator::new(&mut q, &mut log);

        // Lazy preparation check
        assert!(
            prepare_log.borrow().is_empty(),
            "RoundRobin should lazy prepare"
        );

        // 2. Execution Loop
        assert!(rr.next_neighbor(&sched, &mut m, &topo)); // Op 0 (1/2)
        assert_eq!(
            *prepare_log.borrow(),
            vec![0],
            "Op 0 should be prepared on first access"
        );

        assert!(rr.next_neighbor(&sched, &mut m, &topo)); // Op 0 (2/2)

        assert!(rr.next_neighbor(&sched, &mut m, &topo)); // Switches to Op 1 -> Op 1 (1/1)
        assert_eq!(
            *prepare_log.borrow(),
            vec![0, 1],
            "Op 1 should be prepared when switched to"
        );

        // Op 1 exhausted -> Switches to Op 2 -> Op 2 exhausted -> Returns False
        assert!(!rr.next_neighbor(&sched, &mut m, &topo));
        assert_eq!(
            *prepare_log.borrow(),
            vec![0, 1, 2],
            "Op 2 should be prepared even if it yields no moves"
        );

        assert_eq!(*execution_log.borrow(), vec![0, 0, 1]);
    }

    #[test]
    fn test_round_robin_reset() {
        let execution_log = Rc::new(RefCell::new(Vec::new()));
        let prepare_log = Rc::new(RefCell::new(Vec::new()));

        let op0 = Box::new(TrackingOperator::new(
            0,
            5,
            execution_log.clone(),
            prepare_log.clone(),
        ));
        let op1 = Box::new(TrackingOperator::new(
            1,
            5,
            execution_log.clone(),
            prepare_log.clone(),
        ));

        let mut rr = RoundRobinCompoundOperator::new(vec![op0, op1]);
        let (sched, mut q, mut log, topo) = get_mocks();

        // Prepare BEFORE creating the mutator to avoid overlapping borrows
        rr.prepare(&sched, &q, &topo);

        let mut m = Mutator::new(&mut q, &mut log);

        rr.next_neighbor(&sched, &mut m, &topo);
        rr.next_neighbor(&sched, &mut m, &topo);

        rr.reset();

        // Should start at Op 0 again
        rr.next_neighbor(&sched, &mut m, &topo);

        assert_eq!(*execution_log.borrow(), vec![0, 0, 0]);
        // Reset clears op_started, so the next access triggers a second lazy prepare for Op 0
        assert_eq!(*prepare_log.borrow(), vec![0, 0]);
    }

    #[test]
    fn test_random_operator_stickiness() {
        // "Stickiness" means: Once an operator is picked in `prepare`,
        // subsequent `next_neighbor` calls must use that SAME operator until `prepare` is called again.

        let execution_log = Rc::new(RefCell::new(Vec::new()));
        let prepare_log = Rc::new(RefCell::new(Vec::new()));

        let op0 = Box::new(TrackingOperator::new(
            0,
            100,
            execution_log.clone(),
            prepare_log.clone(),
        ));
        let op1 = Box::new(TrackingOperator::new(
            1,
            100,
            execution_log.clone(),
            prepare_log.clone(),
        ));

        // Use a seeded RNG so the choice is deterministic for this test run.
        let mut random_op =
            RandomOperatorCompoundOperator::new(vec![op0, op1], StdRng::seed_from_u64(42));
        let (sched, mut q, mut log, topo) = get_mocks();

        // Prepare BEFORE creating the mutator to avoid overlapping borrows
        random_op.prepare(&sched, &q, &topo);

        let mut m = Mutator::new(&mut q, &mut log);

        // Determine which one was picked by checking the prepare log
        let picked_id = prepare_log.borrow()[0];

        // 2. Run multiple steps
        random_op.next_neighbor(&sched, &mut m, &topo);
        random_op.next_neighbor(&sched, &mut m, &topo);
        random_op.next_neighbor(&sched, &mut m, &topo);

        // 3. Verify that ALL executions came from the picked ID
        let log = execution_log.borrow();
        assert_eq!(log.len(), 3);
        for &executed_id in log.iter() {
            assert_eq!(
                executed_id, picked_id,
                "Random operator switched mid-stream! It should be sticky."
            );
        }
    }

    #[test]
    fn test_random_operator_distribution() {
        // Statistical test: ensure that over many `prepare` calls, both operators get picked.
        let execution_log = Rc::new(RefCell::new(Vec::new()));
        let prepare_log = Rc::new(RefCell::new(Vec::new()));

        let op0 = Box::new(TrackingOperator::new(
            0,
            1,
            execution_log.clone(),
            prepare_log.clone(),
        ));
        let op1 = Box::new(TrackingOperator::new(
            1,
            1,
            execution_log.clone(),
            prepare_log.clone(),
        ));

        // Different seed or loop with seed to ensure coverage
        let mut random_op =
            RandomOperatorCompoundOperator::new(vec![op0, op1], StdRng::seed_from_u64(12345));
        let (sched, q, _log, topo) = get_mocks();

        let iterations = 100;
        let mut picked_0 = 0;
        let mut picked_1 = 0;

        for _ in 0..iterations {
            random_op.prepare(&sched, &q, &topo);

            let last_picked = *prepare_log.borrow().last().unwrap();
            if last_picked == 0 {
                picked_0 += 1;
            } else {
                picked_1 += 1;
            }
        }

        assert!(picked_0 > 0, "Random operator never picked Op 0");
        assert!(picked_1 > 0, "Random operator never picked Op 1");
    }

    // A minimal dummy operator used only to construct the compound operator for stats testing
    struct DummyOperator;
    impl<T: SolverNumeric> LocalSearchOperator<T, StaticTopology> for DummyOperator {
        fn name(&self) -> &str {
            "DummyOperator"
        }
        fn prepare(&mut self, _s: &Schedule<T>, _q: &VesselPriorityQueue, _n: &StaticTopology) {}
        fn next_neighbor(
            &mut self,
            _s: &Schedule<T>,
            _m: &mut Mutator<T>,
            _n: &StaticTopology,
        ) -> bool {
            false
        }
        fn reset(&mut self) {}
    }

    fn build_mab(
        n: usize,
        memory: f64,
        exploration: f64,
    ) -> MultiArmedBanditCompoundOperator<i64, StaticTopology> {
        let mut ops: Vec<Box<dyn LocalSearchOperator<i64, StaticTopology>>> = Vec::with_capacity(n);
        for _ in 0..n {
            ops.push(Box::new(DummyOperator));
        }
        MultiArmedBanditCompoundOperator::new(ops, memory, exploration)
    }

    #[test]
    fn test_mab_ranking_logic() {
        let n = 3;
        let mut compound = build_mab(n, 0.2, std::f64::consts::SQRT_2);

        compound.stats.total_samples = 35;
        compound.stats.samples_per_op = vec![10, 5, 20];
        compound.stats.avg_improvements = vec![1.0, 2.0, 0.5];
        compound.stats.global_max_improvement = 2.0;

        // Recompute ranking
        compound.ranked_indices = (0..n).collect();
        compound.sort_by_score();

        assert_eq!(compound.ranked_indices, vec![1, 0, 2]);
    }
}
