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

use crate::{
    branching::decision::{Decision, DecisionBuilder},
    eval::evaluator::ObjectiveEvaluator,
    monitor::search_monitor::{SearchCommand, TreeSearchMonitor},
    result::{SearchOutcome, SolverResult, TerminationReason},
    stack::SearchStack,
    state::SearchState,
    stats::BnbSolverStatistics,
    trail::SearchTrail,
};
use bollard_core::num::{constants::MinusOne, ops::saturating_arithmetic};
use bollard_model::index::{BerthIndex, VesselIndex};
use bollard_model::{model::Model, solution::Solution};
use num_traits::{FromPrimitive, PrimInt, Signed};

/// A constraint branch and bound solver for the berth scheduling problem using
/// a backtracking search algorithm with constraint propagation and bounding.
/// Note that this is just the execution engine, the construction and navigation
/// of the search tree is done to a `DecisionBuilder` and the evaluation of
/// objectives and bounds is done by an `ObjectiveEvaluator`.
#[derive(Clone)]
pub struct BnbSolver<T>
where
    T: PrimInt + Signed,
{
    trail: SearchTrail<T>,
    stack: SearchStack,
}

impl<T> Default for BnbSolver<T>
where
    T: PrimInt + Signed,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> BnbSolver<T>
where
    T: PrimInt + Signed,
{
    /// Create a new constraint solver instance.
    #[inline]
    pub fn new() -> Self {
        Self {
            trail: SearchTrail::new(),
            stack: SearchStack::new(),
        }
    }

    /// Create a new constraint solver instance with preallocated
    /// storage for the given number of berths and vessels.
    ///
    /// # Note
    ///
    /// When you invoke the solver it will internally ensure that
    /// the trail and stack have sufficient capacity for the given model.
    /// Constructing this the solver with preallocated storage only moves
    /// the cost of the memory allocations to the construction time of the solver,
    /// but does not change the asymptotic memory usage of the solver.
    #[inline]
    pub fn preallocated(num_berths: usize, num_vessels: usize) -> Self {
        Self {
            trail: SearchTrail::preallocated(num_vessels),
            stack: SearchStack::preallocated(num_berths, num_vessels),
        }
    }

    /// Solve the given model using the provided decision builder,
    /// objective evaluator, and search monitor.
    #[inline]
    pub fn solve<B, E, S>(
        &mut self,
        model: &Model<T>,
        builder: &mut B,
        evaluator: &mut E,
        monitor: S,
    ) -> SolverResult<T>
    where
        B: DecisionBuilder<T, E>,
        E: ObjectiveEvaluator<T>,
        S: TreeSearchMonitor<T>,
        T: MinusOne
            + saturating_arithmetic::SaturatingAddVal
            + saturating_arithmetic::SaturatingMulVal
            + saturating_arithmetic::SaturatingSubVal
            + std::fmt::Display
            + FromPrimitive,
    {
        let mut monitor = monitor;
        let session = BnbSolverSearchSession::new(self, model, builder, evaluator, &mut monitor);
        let res = session.run();
        self.reset();
        res
    }

    /// Reset the internal state of the solver, clearing
    /// any stored trail and stack information.
    ///
    /// # Note
    ///
    /// This does not deallocate any memory used by the trail or stack,
    /// but only resets their logical state.
    #[inline]
    fn reset(&mut self) {
        self.trail.reset();
        self.stack.reset();
    }
}

/// A child node generated from a decision.
#[derive(Clone, Copy, Debug)]
struct ChildNode<T> {
    // The layout of this struct is optimized for cache efficiency
    // assuming T = i64 (8 bytes).
    // The layout ensures that padding is minimized.
    start_time: T,
    new_berth_time: T,
    new_objective: T,
    vessel_index: VesselIndex,
    berth_index: BerthIndex,
}

impl<T> std::fmt::Display for ChildNode<T>
where
    T: PrimInt + Signed + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ChildNode(vessel: {}, berth: {}, start: {}, new_berth_time: {}, new_objective: {})",
            self.vessel_index,
            self.berth_index,
            self.start_time,
            self.new_berth_time,
            self.new_objective
        )
    }
}

/// The result of a single search step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchStep {
    /// The search should continue.
    Continue,
    /// The search is finished.
    Finished,
}

impl std::fmt::Display for SearchStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchStep::Continue => write!(f, "Continue"),
            SearchStep::Finished => write!(f, "Finished"),
        }
    }
}

/// A search session for the constraint solver.
/// This struct encapsulates the state and logic
/// of a single search run.
struct BnbSolverSearchSession<'a, T, B, E, S>
where
    T: PrimInt + Signed,
{
    solver: &'a mut BnbSolver<T>,
    model: &'a Model<T>,
    builder: &'a mut B,
    evaluator: &'a mut E,
    monitor: &'a mut S,
    state: SearchState<T>,
    best_objective: T,
    best_solution: Option<Solution<T>>,
    stats: BnbSolverStatistics,
    start_time: std::time::Instant,
}

impl<'a, T, B, E, S> std::fmt::Debug for BnbSolverSearchSession<'a, T, B, E, S>
where
    T: PrimInt + Signed + std::fmt::Debug + MinusOne,
    B: DecisionBuilder<T, E>,
    E: ObjectiveEvaluator<T>,
    S: TreeSearchMonitor<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchSession")
            .field("model", &self.model)
            .field("state", &self.state)
            .field("best_objective", &self.best_objective)
            .field("best_solution", &self.best_solution)
            .field("stats", &self.stats)
            .finish()
    }
}

impl<'a, T, B, E, S> std::fmt::Display for BnbSolverSearchSession<'a, T, B, E, S>
where
    T: PrimInt + Signed + std::fmt::Display + MinusOne,
    B: DecisionBuilder<T, E>,
    E: ObjectiveEvaluator<T>,
    S: TreeSearchMonitor<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let solution_str = match &self.best_solution {
            Some(sol) => format!("Solution(objective: {})", sol.objective_value()),
            None => "No solution found".to_string(),
        };
        write!(
            f,
            "SearchSession(best_objective: {}, best_solution: {}, stats: {})",
            self.best_objective, solution_str, self.stats
        )
    }
}

impl<'a, T, B, E, S> BnbSolverSearchSession<'a, T, B, E, S>
where
    T: PrimInt
        + Signed
        + MinusOne
        + saturating_arithmetic::SaturatingAddVal
        + saturating_arithmetic::SaturatingMulVal
        + saturating_arithmetic::SaturatingSubVal
        + std::fmt::Display,
    B: DecisionBuilder<T, E>,
    E: ObjectiveEvaluator<T>,
    S: TreeSearchMonitor<T>,
{
    /// Create a new search session.
    #[inline]
    fn new(
        solver: &'a mut BnbSolver<T>,
        model: &'a Model<T>,
        builder: &'a mut B,
        evaluator: &'a mut E,
        monitor: &'a mut S,
    ) -> Self {
        let state = SearchState::new(model.num_berths(), model.num_vessels());
        Self {
            solver,
            model,
            builder,
            evaluator,
            state,
            monitor,
            best_objective: T::max_value(),
            best_solution: None,
            stats: BnbSolverStatistics::default(),
            start_time: std::time::Instant::now(),
        }
    }

    /// Run the search session.
    #[inline]
    fn run(mut self) -> SolverResult<T>
    where
        T: FromPrimitive,
    {
        self.monitor.on_enter_search(self.model);
        self.initialize();

        let termination_reason = loop {
            if let SearchCommand::Stop = self.monitor.check_termination(&self.state, &self.stats) {
                break TerminationReason::Aborted;
            }

            match self.step() {
                SearchStep::Continue => continue,
                SearchStep::Finished => {
                    if self.best_solution.is_some() {
                        break TerminationReason::OptimalityProven;
                    } else {
                        break TerminationReason::InfeasibilityProven;
                    }
                }
            }
        };

        self.stats.set_total_time(self.start_time.elapsed());
        self.monitor.on_exit_search(&self.stats);
        self.finalize_result(termination_reason)
    }

    /// Finalize the solver result based on the best solution found
    /// and the termination reason.
    #[inline]
    fn finalize_result(self, reason: TerminationReason) -> SolverResult<T> {
        let outcome = match (&self.best_solution, reason) {
            (Some(_), TerminationReason::OptimalityProven) => SearchOutcome::Optimal,
            (Some(_), TerminationReason::Aborted) => SearchOutcome::Feasible,
            (None, TerminationReason::InfeasibilityProven) => SearchOutcome::Infeasible,
            (None, TerminationReason::Aborted) => SearchOutcome::Unknown,
            _ => {
                // If everything is implemented correctly, this case should never occur!
                panic!("internal error: inconsistent solver result state");
            }
        };

        SolverResult {
            outcome,
            reason,
            solution: self.best_solution,
            statistics: self.stats,
            objective_evaluator: self.evaluator.name().to_string(),
            tree_builder: self.builder.name().to_string(),
        }
    }

    /// Perform a single search step.
    #[inline]
    fn step(&mut self) -> SearchStep
    where
        T: FromPrimitive,
    {
        if self.solver.stack.is_current_level_empty() {
            if self.solver.stack.depth() <= 1 {
                return SearchStep::Finished;
            }
            self.backtrack_step();
            return SearchStep::Continue;
        }

        unsafe {
            self.process_next_decision();
        }
        SearchStep::Continue
    }

    /// Initialize the search session.
    ///
    /// This sets up the initial trail and stack frames,
    /// makes sure we have enaugh memory allocated to *not*
    /// resize during the search, and pushes the first decisions
    /// onto the stack.
    #[inline]
    fn initialize(&mut self) {
        self.solver.trail.ensure_capacity(self.model.num_vessels());
        self.solver
            .stack
            .ensure_capacity(self.model.num_berths(), self.model.num_vessels());

        // Root frame. Crucial to have this before pushing decisions!
        self.solver.trail.push_frame(&self.state);
        self.solver.stack.push_frame();

        // Root node is effectively node #0
        self.stats.on_node_explored();

        self.solver.stack.extend(self.builder.next_decision(
            self.evaluator,
            self.model,
            &self.state,
        ));
    }

    #[inline]
    fn backtrack_step(&mut self) {
        self.stats.on_backtrack();
        self.monitor.on_backtrack(&self.state, &self.stats);

        self.solver.trail.backtrack(&mut self.state);
        self.solver.stack.pop_frame();
    }

    /// Process the next decision from the stack.
    ///
    /// # Panics
    ///
    /// In debug builds, this function will panic if called
    /// when the current decision stack level is empty.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the current decision stack level
    /// is not empty.
    #[inline(always)]
    unsafe fn process_next_decision(&mut self)
    where
        T: FromPrimitive,
    {
        debug_assert!(
            !self.solver.stack.is_current_level_empty(),
            "called `ConstraintSolverSearchSession::process_next_decision` with empty decision stack"
        );

        let decision = unsafe { self.solver.stack.pop().unwrap_unchecked() };

        self.stats.on_decision_generated();

        let child = match unsafe { self.build_child(&decision) } {
            Some(c) => c,
            // Pruned inside build_child (either infeasible or local bound)
            None => return,
        };

        self.descend(child);
    }

    /// Build a child node from the given decision.
    ///
    /// # Panics
    ///
    /// In debug builds, this function will panic if the vessel or berth index
    /// are out of bounds.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `decision.vessel_index()` and `decision.berth_index()`
    /// are valid indices within the model.
    #[inline(always)]
    unsafe fn build_child(&mut self, decision: &Decision) -> Option<ChildNode<T>> {
        let (vessel_index, berth_index) = (decision.vessel_index(), decision.berth_index());

        debug_assert!(
            vessel_index.get() < self.model.num_vessels(),
            "called `ConstraintSolverSearchSession::build_child` with vessel index out of bounds: the len is {} but the index is {}",
            self.model.num_vessels(),
            vessel_index.get()
        );
        debug_assert!(
            berth_index.get() < self.model.num_berths(),
            "called `ConstraintSolverSearchSession::build_child` with berth index out of bounds: the len is {} but the index is {}",
            self.model.num_berths(),
            berth_index.get()
        );

        if !self.is_structurally_feasible(decision) {
            self.stats.on_pruning_infeasible();
            return None;
        }

        let current_berth_time = unsafe { self.state.berth_free_time_unchecked(berth_index) };
        let move_cost = self.evaluator.evaluate_vessel_assignment(
            self.model,
            vessel_index,
            berth_index,
            current_berth_time,
        )?;

        let current_obj = self.state.current_objective();
        let new_objective = current_obj.saturating_add_val(move_cost);

        if new_objective >= self.best_objective {
            self.stats.on_pruning_local();
            return None;
        }

        let processing_time = unsafe {
            self.model
                .vessel_processing_time_unchecked(vessel_index, berth_index)
                .unwrap()
        };
        let arrival = unsafe { self.model.vessel_arrival_time_unchecked(vessel_index) };

        let start_time = if arrival > current_berth_time {
            arrival
        } else {
            current_berth_time
        };
        let new_berth_time = start_time.saturating_add_val(processing_time);

        Some(ChildNode {
            vessel_index,
            berth_index,
            start_time,
            new_berth_time,
            new_objective,
        })
    }

    /// Descend into the given child node, applying its assignment
    /// to the current state.
    #[inline(always)]
    fn descend(&mut self, child: ChildNode<T>)
    where
        T: FromPrimitive,
    {
        self.solver.trail.push_frame(&self.state);
        self.solver.trail.apply_assignment(
            &mut self.state,
            child.berth_index,
            child.vessel_index,
            child.new_berth_time,
            child.new_objective,
            child.start_time,
        );
        self.solver.stack.push_frame();

        self.stats.on_node_explored();
        self.stats.on_decision_applied();
        self.stats.on_depth_update(self.solver.stack.depth() as u64);
        self.monitor.on_descend(&self.state, &self.stats);

        if self.state.num_assigned_vessels() == self.model.num_vessels() {
            self.handle_complete_solution(child.new_objective);
            return;
        }

        // Node-level bound check
        if self.should_backtrack_after_expand() {
            self.stats.on_pruning_global();
            self.backtrack_step();
        }
    }

    /// Check if the given decision is structurally feasible.
    ///
    /// # Panics
    ///
    /// In debug builds, this function will panic if the vessel or berth index
    /// are out of bounds.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `decision.vessel_index()` and `decision.berth_index()`
    /// are valid indices within the model.
    ///
    /// # Note
    ///
    /// Many `DecisionBuilder` implementations will only generate
    /// structurally feasible decisions, so this check may often be redundant.
    /// However, we still need it here to ensure correctness in case
    /// a `DecisionBuilder` produces infeasible decisions. The check is
    /// very cheap and thus worth including for safety.
    #[inline(always)]
    fn is_structurally_feasible(&self, decision: &Decision) -> bool {
        let (vessel_index, berth_index) = (decision.vessel_index(), decision.berth_index());

        debug_assert!(
            vessel_index.get() < self.model.num_vessels(),
            "called `ConstraintSolverSearchSession::is_structurally_feasible` with vessel index out of bounds: the len is {} but the index is {}",
            self.model.num_vessels(),
            vessel_index.get()
        );

        debug_assert!(
            berth_index.get() < self.model.num_berths(),
            "called `ConstraintSolverSearchSession::is_structurally_feasible` with berth index out of bounds: the len is {} but the index is {}",
            self.model.num_berths(),
            berth_index.get()
        );

        unsafe {
            !self.state.is_vessel_assigned_unchecked(vessel_index)
                && self
                    .model
                    .vessel_allowed_on_berth_unchecked(vessel_index, berth_index)
        }
    }

    /// Handle a complete solution found at the current state.
    ///
    /// # Panics
    ///
    /// This function will panic if the current state cannot be converted
    /// into a valid solution. This can happen if this function is called
    /// while the solver is not at a leaf node.
    #[inline(always)]
    fn handle_complete_solution(&mut self, new_objective: T) {
        if new_objective < self.best_objective {
            if let Ok(solution) = self.state.clone().try_into() {
                self.best_objective = new_objective;
                self.stats.on_solution_found();
                self.monitor.on_solution(&solution, &self.stats);
                self.best_solution = Some(solution);
            } else {
                // In this part of te tree we have a complete assignment,
                // and should always be able to convert the state to a solution.
                panic!(
                    "called `ConstraintSolverSearchSession::handle_complete_solution` but failed to convert state to solution"
                );
            }
        }
        self.backtrack_step();
    }

    /// Determine whether to backtrack after expanding the current node.
    #[inline(always)]
    fn should_backtrack_after_expand(&mut self) -> bool
    where
        T: FromPrimitive,
    {
        let lower_bound_remaining_opt = self.evaluator.lower_bound(self.model, &self.state);
        let lower_bound_remaining = match lower_bound_remaining_opt {
            None => return true,
            Some(lower_bound) => lower_bound,
        };

        let node_lower_bound = self
            .state
            .current_objective()
            .saturating_add_val(lower_bound_remaining);

        if node_lower_bound >= self.best_objective {
            return true;
        }

        let decisions = self
            .builder
            .next_decision(self.evaluator, self.model, &self.state);
        self.solver.stack.extend(decisions);

        false
    }
}

#[cfg(test)]
mod tests {
    use super::BnbSolver;
    use crate::eval::wtft::WeightedFlowTimeEvaluator;
    use crate::monitor::composite::CompositeMonitor;
    use crate::monitor::time_limit::TimeLimitMonitor;
    use crate::{
        branching::chronological::ChronologicalExhaustiveBuilder, monitor::log::LogMonitor,
    };
    use bollard_model::{
        index::{BerthIndex, VesselIndex},
        model::ModelBuilder,
        time::ProcessingTime,
    };

    type IntegerType = i64;

    /// Build a model with the given number of berths and vessels.
    /// - Arrivals: spaced by 3 time units (0, 3, 6, ...)
    /// - Weights: cycle in [1..5]
    /// - Processing times:
    ///   - Berth 0: 10..12 depending on vessel index
    ///   - Berth 1: 7..10 depending on vessel index
    fn build_model(
        num_berths: usize,
        num_vessels: usize,
    ) -> bollard_model::model::Model<IntegerType> {
        let mut builder = ModelBuilder::<IntegerType>::new(num_berths, num_vessels);

        // Simple arrivals and weights
        for v in 0..num_vessels {
            let vi = VesselIndex::new(v);
            builder.set_vessel_arrival_time(vi, (v as IntegerType) * 3); // spaced arrivals
            builder.set_vessel_weight(vi, 1 + (v as IntegerType % 5)); // weights in [1..5]
        }

        // Feasible processing times for all berths
        for v in 0..num_vessels {
            let vi = VesselIndex::new(v);
            for b in 0..num_berths {
                let bi = BerthIndex::new(b);
                // Make berth-specific durations but always feasible
                let base = if b % 2 == 0 { 10 } else { 7 }; // alternate baselines
                let span = if b % 2 == 0 { 3 } else { 4 }; // alternate spans
                let duration = base + (v as IntegerType % span);
                builder.set_vessel_processing_time(vi, bi, ProcessingTime::some(duration));
            }
        }

        builder.build()
    }

    #[test]
    fn test_solver_with_berths_vessels() {
        let model = build_model(2, 10);

        let mut solver = BnbSolver::<IntegerType>::new();
        let mut builder = ChronologicalExhaustiveBuilder;
        let mut evaluator = WeightedFlowTimeEvaluator::<IntegerType>::preallocated(
            model.num_berths(),
            model.num_vessels(),
        );

        let composite_monitor = CompositeMonitor::default()
            .add_monitor(LogMonitor::default())
            .add_monitor(TimeLimitMonitor::<IntegerType>::new(
                std::time::Duration::from_secs(1000),
                10_000,
            ));

        // 1. Run the solver (timing is now handled internally in result.statistics)
        let result = solver.solve(&model, &mut builder, &mut evaluator, composite_monitor);

        // 2. Print the rich result (Outcome, Reason, Objective, Stats Table)
        println!("{}", result);

        // 3. Assertions
        assert!(
            result.solution.is_some(),
            "solver should find a feasible solution"
        );

        // Unwrap the solution from the result struct
        let solution = result.solution.as_ref().unwrap();

        assert_eq!(solution.num_vessels(), model.num_vessels());

        for v in 0..model.num_vessels() {
            let vi = VesselIndex::new(v);
            let bi = solution.berth_for_vessel(vi);
            assert!(
                bi.get() < model.num_berths(),
                "assigned berth must be valid"
            );
            let start = solution.start_time_for_vessel(vi);
            let arrival = model.vessel_arrival_time(vi);
            assert!(
                start >= arrival,
                "start time must be >= arrival for vessel {}",
                v
            );
        }

        assert_eq!(solution.objective_value(), 855); // Validated with Gurobi
    }
}
