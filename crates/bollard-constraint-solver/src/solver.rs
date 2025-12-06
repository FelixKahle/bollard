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
    decision::{Decision, DecisionBuilder},
    eval::ObjectiveEvaluator,
    stack::SearchStack,
    state::SearchState,
    trail::SearchTrail,
};
use bollard_core::num::{constants::MinusOne, ops::saturating_arithmetic};
use bollard_model::{index::BerthIndex, model::Model, solution::Solution};
use num_traits::{PrimInt, Signed};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IncompleteSolutionError {
    assigned_vessels: usize,
    total_vessels: usize,
}

impl std::fmt::Display for IncompleteSolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Incomplete solution: assigned {}/{} vessels",
            self.assigned_vessels, self.total_vessels
        )
    }
}

impl std::error::Error for IncompleteSolutionError {}

/// The Solver facade.
///
/// This struct is stateless and immutable regarding the search progress.
/// It acts as a factory and memory holder for the search process.
#[derive(Clone)]
pub struct Solver<T>
where
    T: PrimInt + Signed,
{
    trail: SearchTrail<T>,
    stack: SearchStack,
}

impl<T> Default for Solver<T>
where
    T: PrimInt + Signed,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Solver<T>
where
    T: PrimInt + Signed,
{
    pub fn new() -> Self {
        Self {
            trail: SearchTrail::new(),
            stack: SearchStack::new(),
        }
    }

    pub fn solve<B, E>(
        &mut self,
        model: &Model<T>,
        builder: &mut B,
        evaluator: &E,
    ) -> Option<Solution<T>>
    where
        B: DecisionBuilder<T>,
        E: ObjectiveEvaluator<T>,
        T: MinusOne
            + saturating_arithmetic::SaturatingAddVal
            + saturating_arithmetic::SaturatingMulVal,
    {
        // Delegate the active search to a transient session object.
        let mut session = SearchSession::new(self, model, builder, evaluator);
        session.run()
    }

    #[inline]
    fn reset(&mut self) {
        self.trail.reset();
        self.stack.reset();
    }
}

/// Internal session struct to encapsulate the state of a single solve attempt.
/// This implements the "Method Object" pattern to modularize the DFS logic.
struct SearchSession<'a, T, B, E>
where
    T: PrimInt + Signed,
{
    solver: &'a mut Solver<T>,
    model: &'a Model<T>,
    builder: &'a mut B,
    evaluator: &'a E,
    state: SearchState<T>,
    best_objective: T,
    best_solution: Option<Solution<T>>,
}

impl<'a, T, B, E> SearchSession<'a, T, B, E>
where
    T: PrimInt
        + Signed
        + MinusOne
        + saturating_arithmetic::SaturatingAddVal
        + saturating_arithmetic::SaturatingMulVal,
    B: DecisionBuilder<T>,
    E: ObjectiveEvaluator<T>,
{
    fn new(
        solver: &'a mut Solver<T>,
        model: &'a Model<T>,
        builder: &'a mut B,
        evaluator: &'a E,
    ) -> Self {
        let state = SearchState::new(model.num_berths(), model.num_vessels());
        Self {
            solver,
            model,
            builder,
            evaluator,
            state,
            best_objective: T::max_value(),
            best_solution: None,
        }
    }

    /// Executes the main Depth-First Search loop.
    fn run(&mut self) -> Option<Solution<T>> {
        self.initialize();

        loop {
            if self.solver.stack.is_current_level_empty() {
                if self.solver.stack.depth() <= 1 {
                    break;
                }
                self.backtrack_step();
                continue;
            }

            unsafe {
                self.process_next_decision();
            }
        }

        self.solver.reset();
        self.best_solution.take()
    }

    fn initialize(&mut self) {
        self.solver.trail.ensure_capacity(self.model.num_vessels());
        self.solver
            .stack
            .ensure_capacity(self.model.num_berths(), self.model.num_vessels());

        self.solver.stack.push_frame();
        self.solver
            .stack
            .extend(self.builder.next_decision(self.model, &self.state));
    }

    /// Reverts the last decision level.
    #[inline]
    fn backtrack_step(&mut self) {
        self.solver.trail.backtrack(&mut self.state);
        self.solver.stack.pop_frame();
    }

    /// Processes the next candidate decision from the current stack frame.
    #[inline]
    unsafe fn process_next_decision(&mut self) {
        debug_assert!(
            !self.solver.stack.is_current_level_empty(),
            "No decisions available at current level"
        );

        let decision = unsafe { self.solver.stack.pop().unwrap_unchecked() };

        if !self.is_feasible(&decision) {
            return;
        }

        if let Some(move_cost) = self.evaluate_cost(&decision) {
            let current_obj = self.state.current_objective();
            let new_objective = current_obj.saturating_add_val(move_cost);

            if new_objective > self.best_objective {
                return; // Prune: Bound exceeded
            }

            self.apply_decision(&decision, new_objective);

            if self.state.num_assigned_vessels() == self.model.num_vessels() {
                self.handle_complete_solution(new_objective);
            } else {
                self.expand_node();
            }
        }
    }

    #[inline]
    fn is_feasible(&self, decision: &Decision) -> bool {
        let (vessel_index, berth_index) = (decision.vessel_index(), decision.berth_index());
        unsafe {
            !self.state.is_vessel_assigned_unchecked(vessel_index)
                && self
                    .model
                    .allowed_on_berth_unchecked(vessel_index, berth_index)
        }
    }

    #[inline]
    fn evaluate_cost(&self, decision: &Decision) -> Option<T> {
        let (vessel_index, berth_index) = (decision.vessel_index(), decision.berth_index());
        let current_berth_time = unsafe { self.state.berth_free_time_unchecked(berth_index) };

        let cost = self.evaluator.evaluate_vessel_assignment(
            self.model,
            vessel_index,
            berth_index,
            current_berth_time,
        );

        if cost == T::max_value() {
            None
        } else {
            Some(cost)
        }
    }

    #[inline]
    fn apply_decision(&mut self, decision: &Decision, new_objective: T) {
        let (vessel_index, berth_index) = (decision.vessel_index(), decision.berth_index());
        let current_berth_time = unsafe { self.state.berth_free_time_unchecked(berth_index) };
        let processing_time = unsafe {
            self.model
                .processing_time_unchecked(vessel_index, berth_index)
                .unwrap()
        };
        let arrival = unsafe { self.model.arrival_time_unchecked(vessel_index) };

        let start_time = if arrival > current_berth_time {
            arrival
        } else {
            current_berth_time
        };
        let new_berth_time = start_time.saturating_add_val(processing_time);

        self.solver.trail.push_frame();
        self.solver.trail.apply_assignment(
            &mut self.state,
            berth_index,
            vessel_index,
            new_berth_time,
            new_objective,
            start_time,
        );
        self.solver.stack.push_frame();
    }

    fn handle_complete_solution(&mut self, new_objective: T) {
        if new_objective < self.best_objective
            && let Ok(solution) = self.reconstruct_solution(new_objective)
        {
            self.best_objective = new_objective;
            self.best_solution = Some(solution);
        }
        self.backtrack_step();
    }

    /// Reconstructs the full Solution object from the current trail of assignments.
    ///
    /// Since `SearchState` no longer stores the specific assignment details (start time),
    /// we derive them from the history in `SearchTrail`.
    ///
    /// Returns `IncompleteSolutionError` if the current state does not have all vessels assigned.
    fn reconstruct_solution(
        &self,
        objective_value: T,
    ) -> Result<Solution<T>, IncompleteSolutionError> {
        let num_vessels = self.model.num_vessels();

        if self.state.num_assigned_vessels() != num_vessels {
            return Err(IncompleteSolutionError {
                assigned_vessels: self.state.num_assigned_vessels(),
                total_vessels: num_vessels,
            });
        }

        let mut berths = vec![BerthIndex::new(0); num_vessels];
        let mut start_times = vec![T::zero(); num_vessels];

        for entry in self.solver.trail.iter_entries() {
            let vessel_index = entry.vessel_index();
            let berth_index = entry.berth_index();

            let arrival = self.model.arrival_time(vessel_index);
            let available = entry.old_berth_time();

            let start = if arrival > available {
                arrival
            } else {
                available
            };

            let vessel_index_usize = vessel_index.get();
            if vessel_index_usize < num_vessels {
                berths[vessel_index_usize] = berth_index;
                start_times[vessel_index_usize] = start;
            }
        }

        Ok(Solution::new(objective_value, berths, start_times))
    }

    fn expand_node(&mut self) {
        let lb = self.evaluator.lower_bound(self.model, &self.state);

        if lb >= self.best_objective {
            self.backtrack_step();
        } else {
            let decisions = self.builder.next_decision(self.model, &self.state);
            self.solver.stack.extend(decisions);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::WeightedFlowTimeEvaluator;
    use bollard_model::{
        index::{BerthIndex, VesselIndex},
        model::ModelBuilder,
        time::ProcessingTime,
    };
    use std::iter::FusedIterator;

    type IntegerType = i64;

    // --- 1. Chronological Iterator ---

    pub struct ChronologicalIter<'a, T>
    where
        T: PrimInt + Signed,
    {
        current_vessel_idx: VesselIndex,
        current_berth_idx: BerthIndex,
        model: &'a Model<T>,
        state: &'a SearchState<T>,
    }

    impl<'a, T> Iterator for ChronologicalIter<'a, T>
    where
        T: PrimInt + Signed + MinusOne,
    {
        type Item = Decision;

        fn next(&mut self) -> Option<Self::Item> {
            let num_vessels = self.model.num_vessels();
            let num_berths = self.model.num_berths();

            while self.current_vessel_idx.get() < num_vessels {
                let v_idx = self.current_vessel_idx;
                let b_idx = self.current_berth_idx;

                let mut next_b_val = b_idx.get() + 1;
                let mut next_v_val = v_idx.get();

                if next_b_val >= num_berths {
                    next_b_val = 0;
                    next_v_val += 1;
                }

                self.current_berth_idx = BerthIndex::new(next_b_val);
                self.current_vessel_idx = VesselIndex::new(next_v_val);

                if unsafe { self.state.is_vessel_assigned_unchecked(v_idx) } {
                    continue;
                }

                if unsafe { !self.model.allowed_on_berth_unchecked(v_idx, b_idx) } {
                    continue;
                }

                let berth_avail = unsafe { self.state.berth_free_time_unchecked(b_idx) };
                let arrival = unsafe { self.model.arrival_time_unchecked(v_idx) };

                let start_time = if arrival > berth_avail {
                    arrival
                } else {
                    berth_avail
                };

                let last_time = self.state.last_decision_time();

                if start_time < last_time {
                    continue;
                }

                if start_time == last_time {
                    let last_vessel_idx: usize = self.state.last_decision_vessel().get();
                    if v_idx.get() < last_vessel_idx {
                        continue;
                    }
                }

                return Some(Decision::new(v_idx, b_idx));
            }

            None
        }
    }

    impl<'a, T> FusedIterator for ChronologicalIter<'a, T> where T: PrimInt + Signed + MinusOne {}

    pub struct ChronologicalBuilder;

    impl<T> DecisionBuilder<T> for ChronologicalBuilder
    where
        T: PrimInt + Signed + MinusOne,
    {
        type DecisionIterator<'a>
            = ChronologicalIter<'a, T>
        where
            Self: 'a,
            T: 'a;

        fn name(&self) -> &str {
            "ChronologicalBuilder"
        }

        fn next_decision<'a>(
            &mut self,
            model: &'a Model<T>,
            state: &'a SearchState<T>,
        ) -> Self::DecisionIterator<'a> {
            ChronologicalIter {
                current_vessel_idx: VesselIndex::new(0),
                current_berth_idx: BerthIndex::new(0),
                model,
                state,
            }
        }
    }

    pub struct PermutationIter {
        current_vessel: usize,
        current_berth: usize,
        num_vessels: usize,
        num_berths: usize,
    }

    impl Iterator for PermutationIter {
        type Item = Decision;

        fn next(&mut self) -> Option<Self::Item> {
            // Loop condition: while we still have vessels to check
            if self.current_vessel < self.num_vessels {
                // Create the decision for the current state
                let decision = Decision::new(
                    VesselIndex::new(self.current_vessel),
                    BerthIndex::new(self.current_berth),
                );

                // Advance the state machine (Inner loop logic)
                self.current_berth += 1;

                // If we finished all berths for this vessel, move to the next vessel
                // and reset the berth counter.
                if self.current_berth >= self.num_berths {
                    self.current_berth = 0;
                    self.current_vessel += 1;
                }

                Some(decision)
            } else {
                // We have iterated through all vessels and all berths
                None
            }
        }
    }

    // Mark as fused so the solver is safe to call next() after None
    impl FusedIterator for PermutationIter {}

    pub struct OptimalityBuilder;

    impl<T> DecisionBuilder<T> for OptimalityBuilder
    where
        T: PrimInt + Signed,
    {
        // No Vec, no Box, just our raw struct
        type DecisionIterator<'a>
            = PermutationIter
        where
            Self: 'a,
            T: 'a;

        fn name(&self) -> &str {
            "OptimalityBuilder (Permutation)"
        }

        fn next_decision<'a>(
            &'a mut self,
            model: &'a Model<T>,
            _state: &'a SearchState<T>,
        ) -> Self::DecisionIterator<'a> {
            PermutationIter {
                current_vessel: 0,
                current_berth: 0,
                num_vessels: model.num_vessels(),
                num_berths: model.num_berths(),
            }
        }
    }

    #[test]
    fn test_chronological_builder_matches_permutation_builder_on_small_case() {
        let num_vessels = 10;
        let num_berths = 2;

        // Build a deterministic model:
        // - Every vessel is allowed on both berths
        // - Arrival time 0 for all vessels
        // - Weight 1 for all vessels
        // - Processing times: berth 0 = i+1, berth 1 = 2*(i+1)
        // This produces a clear optimum and is easy to verify.
        let mut mb = ModelBuilder::<IntegerType>::new(num_berths, num_vessels);
        for i in 0..num_vessels {
            let v = VesselIndex::new(i);
            let t0 = (i as IntegerType) + 1;
            let t1 = 2 * ((i as IntegerType) + 1);
            mb.set_processing_time(v, BerthIndex::new(0), ProcessingTime::some(t0));
            mb.set_processing_time(v, BerthIndex::new(1), ProcessingTime::some(t1));
            mb.set_vessel_weight(v, 1);
            mb.set_arrival_time(v, 0);
        }

        let model = mb.build();
        let mut solver = Solver::new();
        let evaluator = WeightedFlowTimeEvaluator::new();

        // Solve with ChronologicalBuilder
        let mut chronological_builder = ChronologicalBuilder;
        let chrono_result = solver.solve(&model, &mut chronological_builder, &evaluator);
        assert!(
            chrono_result.is_some(),
            "ChronologicalBuilder produced no solution"
        );
        let chrono_solution = chrono_result.unwrap();

        // Solve with OptimalityBuilder (permutation/exhaustive)
        solver.reset();
        let mut optimality_builder = OptimalityBuilder;
        let opt_result = solver.solve(&model, &mut optimality_builder, &evaluator);
        assert!(
            opt_result.is_some(),
            "OptimalityBuilder produced no solution"
        );
        let opt_solution = opt_result.unwrap();

        // Compare objectives: both should obtain the same optimum
        assert_eq!(
            chrono_solution.objective_value(),
            opt_solution.objective_value(),
            "ChronologicalBuilder objective differs from exhaustive OptimalityBuilder"
        );

        // Sanity checks on solution shape
        assert_eq!(chrono_solution.num_vessels(), num_vessels);
        assert_eq!(opt_solution.num_vessels(), num_vessels);

        // It's possible multiple optimal assignments exist; we only enforce objective equality.
        // Optionally, check that both solutions utilize berth 0 predominantly due to faster times.
        let chrono_berth1_count = (0..num_vessels)
            .filter(|&i| chrono_solution.berth_for_vessel(VesselIndex::new(i)).get() == 1)
            .count();
        let opt_berth1_count = (0..num_vessels)
            .filter(|&i| opt_solution.berth_for_vessel(VesselIndex::new(i)).get() == 1)
            .count();

        // Both strategies should use berth 1 at least once in a flow-time objective to reduce cumulative delay.
        assert!(
            chrono_berth1_count > 0,
            "ChronologicalBuilder should utilize berth 1 at least once for flow-time improvements"
        );
        assert!(
            opt_berth1_count > 0,
            "OptimalityBuilder should utilize berth 1 at least once for flow-time improvements"
        );
    }

    #[test]
    fn test_chronological_builder_is_faster_than_permutation_builder() {
        use std::time::Instant;

        let num_vessels = 10;
        let num_berths = 2;

        // Build a deterministic model:
        // - Every vessel is allowed on both berths
        // - Arrival time 0 for all vessels
        // - Weight 1 for all vessels
        // - Processing times: berth 0 = i+1, berth 1 = 2*(i+1)
        let mut mb = ModelBuilder::<IntegerType>::new(num_berths, num_vessels);
        for i in 0..num_vessels {
            let v = VesselIndex::new(i);
            let t0 = (i as IntegerType) + 1;
            let t1 = 2 * ((i as IntegerType) + 1);
            mb.set_processing_time(v, BerthIndex::new(0), ProcessingTime::some(t0));
            mb.set_processing_time(v, BerthIndex::new(1), ProcessingTime::some(t1));
            mb.set_vessel_weight(v, 1);
            mb.set_arrival_time(v, 0);
        }

        let model = mb.build();
        let evaluator = WeightedFlowTimeEvaluator::new();

        // Run ChronologicalBuilder
        let mut solver = Solver::new();
        let mut chrono_builder = ChronologicalBuilder;
        let start_chrono = Instant::now();
        let chrono_result = solver.solve(&model, &mut chrono_builder, &evaluator);
        let chrono_duration = start_chrono.elapsed();
        assert!(
            chrono_result.is_some(),
            "ChronologicalBuilder produced no solution"
        );
        let chrono_obj = chrono_result.unwrap().objective_value();

        // Run OptimalityBuilder (permutation/exhaustive)
        solver.reset();
        let mut opt_builder = OptimalityBuilder;
        let start_opt = Instant::now();
        let opt_result = solver.solve(&model, &mut opt_builder, &evaluator);
        let opt_duration = start_opt.elapsed();
        assert!(
            opt_result.is_some(),
            "OptimalityBuilder produced no solution"
        );
        let opt_obj = opt_result.unwrap().objective_value();

        // Sanity: both should reach the same optimum
        assert_eq!(
            chrono_obj, opt_obj,
            "Objectives differ between builders; test expects identical optimum"
        );

        // Print timings for local inspection
        println!(
            "ChronologicalBuilder: {:?}, OptimalityBuilder: {:?}",
            chrono_duration, opt_duration
        );

        // Assert the chronological approach is faster or equal.
        // Note: wall-clock timings can be noisy; if this ever flakes on CI,
        // consider relaxing or turning into an informational check.
        assert!(
            chrono_duration <= opt_duration,
            "ChronologicalBuilder should be faster or equal. Chrono: {:?}, Permutation: {:?}",
            chrono_duration,
            opt_duration
        );
    }

    #[test]
    fn test_chronological_builder_finds_global_optimum() {
        let num_vessels = 10;
        let num_berths = 2;
        let mut mb = ModelBuilder::<IntegerType>::new(num_berths, num_vessels);

        // Create a scenario where greedy choice is suboptimal
        for i in 0..num_vessels {
            let v = VesselIndex::new(i);
            let t0 = (i as IntegerType) + 1; // Fast berth processing time
            let t1 = 2 * ((i as IntegerType) + 1); // Slow berth processing time

            mb.set_processing_time(v, BerthIndex::new(0), ProcessingTime::some(t0));
            mb.set_processing_time(v, BerthIndex::new(1), ProcessingTime::some(t1));
            mb.set_vessel_weight(v, 1);
            mb.set_arrival_time(v, 0);
        }

        let model = mb.build();
        let mut solver = Solver::new();
        let mut builder = ChronologicalBuilder;
        let evaluator = WeightedFlowTimeEvaluator::new();

        let result = solver.solve(&model, &mut builder, &evaluator);

        assert!(
            result.is_some(),
            "Solver failed to find a feasible solution"
        );
        let solution = result.unwrap();
        let optimal_cost = solution.objective_value();

        // Calculate a naive Greedy Baseline (all on fastest berth 0)
        let mut greedy_cost = 0;
        let mut current_time = 0;
        for i in 0..num_vessels {
            current_time += (i as IntegerType) + 1;
            greedy_cost += current_time;
        }

        println!("Greedy Baseline (Single Berth): {}", greedy_cost);
        println!("Chronological Optimal: {}", optimal_cost);

        // The solver should utilize both berths to lower flow time, beating the single-berth greedy
        assert!(
            optimal_cost < greedy_cost,
            "ChronologicalBuilder failed to beat single-berth greedy baseline!"
        );

        // Verify that the solver actually used the second berth (Berth 1)
        let mut on_berth_1 = 0;
        for i in 0..num_vessels {
            if solution.berth_for_vessel(VesselIndex::new(i)).get() == 1 {
                on_berth_1 += 1;
            }
        }
        assert!(
            on_berth_1 > 0,
            "Optimal solution must make use of the second berth"
        );
    }

    #[test]
    fn test_chronological_builder_correctness_on_small_case() {
        let mut mb = ModelBuilder::<IntegerType>::new(1, 2);

        // V0: Arrives late (10), takes 5
        mb.set_arrival_time(VesselIndex::new(0), 10);
        mb.set_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(0),
            ProcessingTime::some(5),
        );
        mb.set_vessel_weight(VesselIndex::new(0), 1);

        // V1: Arrives early (0), takes 5
        mb.set_arrival_time(VesselIndex::new(1), 0);
        mb.set_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(0),
            ProcessingTime::some(5),
        );
        mb.set_vessel_weight(VesselIndex::new(1), 1);

        let model = mb.build();
        let mut solver = Solver::new();
        let mut builder = ChronologicalBuilder;
        let evaluator = WeightedFlowTimeEvaluator::new();

        let sol = solver.solve(&model, &mut builder, &evaluator).unwrap();

        let t0 = sol.start_time_for_vessel(VesselIndex::new(0));
        let t1 = sol.start_time_for_vessel(VesselIndex::new(1));

        // CHRONOLOGICAL LOGIC ASSERTION:
        // The builder should schedule V1 first (starts at 0).
        // It should NOT schedule V0 first, because V0 starts at 10.
        // If it scheduled V0 first (t=10), then tried to schedule V1, V1 would start at 0.
        // But 0 < 10 (previous decision), so the Iterator skips that branch.

        assert_eq!(t1, 0, "Vessel 1 must start at 0 (First in sequence)");
        assert_eq!(t0, 10, "Vessel 0 must start at 10 (Second in sequence)");

        // Objective: (0+5) + (10+5) = 5 + 15 = 20
        assert_eq!(sol.objective_value(), 20);
    }
}
