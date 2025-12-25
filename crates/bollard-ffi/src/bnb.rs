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

use crate::result::BnbSolverFFIOutcome;
use bollard_bnb::{
    bnb::BnbSolver,
    branching::{
        chronological::ChronologicalExhaustiveBuilder, decision::DecisionBuilder,
        fcfs::FcfsHeuristicBuilder, regret::RegretHeuristicBuilder, slack::SlackHeuristicBuilder,
        wspt::WsptHeuristicBuilder,
    },
    eval::{
        evaluator::ObjectiveEvaluator, hybrid::HybridEvaluator, workload::WorkloadEvaluator,
        wtft::WeightedFlowTimeEvaluator,
    },
    monitor::{
        composite::CompositeTreeSearchMonitor, log::LogTreeSearchMonitor,
        solution::SolutionLimitMonitor, time::TimeLimitMonitor,
    },
    result::BnbSolverOutcome,
};
use bollard_model::model::Model;
use std::time::Duration;

/// Creates a new Branch-and-Bound solver instance.
#[no_mangle]
pub extern "C" fn bollard_bnb_solver_new() -> *mut BnbSolver<i64> {
    let solver = BnbSolver::<i64>::new();
    Box::into_raw(Box::new(solver))
}

/// Creates a new Branch-and-Bound solver instance with preallocated memory.
#[no_mangle]
pub extern "C" fn bollard_bnb_solver_preallocated(
    num_berths: usize,
    num_vessels: usize,
) -> *mut BnbSolver<i64> {
    let solver = BnbSolver::<i64>::preallocated(num_berths, num_vessels);
    Box::into_raw(Box::new(solver))
}

/// Frees the memory allocated for the BnbSolver.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that the pointer is valid and was allocated
/// by `bollard_bnb_solver_new` or `bollard_bnb_solver_preallocated`.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_solver_free(ptr: *mut BnbSolver<i64>) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// The core generic function that handles monitor setup and solving.
/// This acts as the "template" that gets monomorphized for each type combination.
#[inline(always)]
fn solve<B, E>(
    solver: &mut BnbSolver<i64>,
    model: &Model<i64>,
    mut builder: B,
    mut evaluator: E,
    solution_limit: usize,
    time_limit_ms: i64,
    enable_log: bool,
) -> BnbSolverOutcome<i64>
where
    E: ObjectiveEvaluator<i64>,
    B: DecisionBuilder<i64, E>,
{
    let capacity =
        (solution_limit > 0) as usize + (time_limit_ms > 0) as usize + (enable_log as usize);
    let mut monitor = CompositeTreeSearchMonitor::with_capacity(capacity);

    if solution_limit > 0 {
        monitor.add_monitor(SolutionLimitMonitor::new(solution_limit as u64));
    }
    if time_limit_ms > 0 {
        let duration = Duration::from_millis(time_limit_ms as u64);
        monitor.add_monitor(TimeLimitMonitor::new(duration));
    }
    if enable_log {
        monitor.add_monitor(LogTreeSearchMonitor::default());
    }

    solver.solve(model, &mut builder, &mut evaluator, monitor)
}

macro_rules! generate_solve {
    (
        $fn_name:ident,
        $eval_ty:ty,
        $builder_ty:ty,
        $eval_init:expr,
        $builder_init:expr
    ) => {
        /// Solves the given model using the specified evaluator and decision builder types.
        ///
        /// # Panics
        ///
        /// This function will panic if called with null pointers for the solver or model.
        ///
        /// # Safety
        ///
        /// This function is unsafe because it dereferences raw pointers.
        /// The caller must ensure that the pointers are valid and were
        /// allocated by Bollard.
        #[no_mangle]
        pub unsafe extern "C" fn $fn_name(
            solver_ptr: *mut BnbSolver<i64>,
            model_ptr: *const Model<i64>,
            solution_limit: usize,
            time_limit_ms: i64,
            enable_log: bool,
        ) -> *mut BnbSolverFFIOutcome {
            assert!(
                !solver_ptr.is_null(),
                "called `{}` with null solver pointer",
                stringify!($fn_name)
            );
            assert!(
                !model_ptr.is_null(),
                "called `{}` with null model pointer",
                stringify!($fn_name)
            );

            let solver: &mut BnbSolver<i64> = &mut *solver_ptr;
            let model: &Model<i64> = &*model_ptr;

            let evaluator: $eval_ty = ($eval_init)(model);
            let builder: $builder_ty = ($builder_init)(model);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
            );

            let ffi_outcome = BnbSolverFFIOutcome::from(outcome);
            Box::into_raw(Box::new(ffi_outcome))
        }
    };
}

generate_solve!(
    bollard_bnb_solver_solve_with_hybrid_evaluator_and_chronological_exhaustive_builder,
    HybridEvaluator<i64>,
    ChronologicalExhaustiveBuilder<i64>,
    |m: &Model<i64>| HybridEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |_| ChronologicalExhaustiveBuilder::new()
);

generate_solve!(
    bollard_bnb_solver_solve_with_hybrid_evaluator_and_fcfs_heuristic_builder,
    HybridEvaluator<i64>,
    FcfsHeuristicBuilder<i64>,
    |m: &Model<i64>| HybridEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| FcfsHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

generate_solve!(
    bollard_bnb_solver_solve_with_hybrid_evaluator_and_regret_heuristic_builder,
    HybridEvaluator<i64>,
    RegretHeuristicBuilder<i64>,
    |m: &Model<i64>| HybridEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| RegretHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

generate_solve!(
    bollard_bnb_solver_solve_with_hybrid_evaluator_and_slack_heuristic_builder,
    HybridEvaluator<i64>,
    SlackHeuristicBuilder<i64>,
    |m: &Model<i64>| HybridEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| SlackHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

generate_solve!(
    bollard_bnb_solver_solve_with_hybrid_evaluator_and_wspt_heuristic_builder,
    HybridEvaluator<i64>,
    WsptHeuristicBuilder<i64>,
    |m: &Model<i64>| HybridEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| WsptHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

// --- Workload Evaluator Combinations ---

generate_solve!(
    bollard_bnb_solver_solve_with_workload_evaluator_and_chronological_exhaustive_builder,
    WorkloadEvaluator<i64>,
    ChronologicalExhaustiveBuilder<i64>,
    |m: &Model<i64>| WorkloadEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |_| ChronologicalExhaustiveBuilder::new()
);

generate_solve!(
    bollard_bnb_solver_solve_with_workload_evaluator_and_fcfs_heuristic_builder,
    WorkloadEvaluator<i64>,
    FcfsHeuristicBuilder<i64>,
    |m: &Model<i64>| WorkloadEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| FcfsHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

generate_solve!(
    bollard_bnb_solver_solve_with_workload_evaluator_and_regret_heuristic_builder,
    WorkloadEvaluator<i64>,
    RegretHeuristicBuilder<i64>,
    |m: &Model<i64>| WorkloadEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| RegretHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

generate_solve!(
    bollard_bnb_solver_solve_with_workload_evaluator_and_slack_heuristic_builder,
    WorkloadEvaluator<i64>,
    SlackHeuristicBuilder<i64>,
    |m: &Model<i64>| WorkloadEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| SlackHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

generate_solve!(
    bollard_bnb_solver_solve_with_workload_evaluator_and_wspt_heuristic_builder,
    WorkloadEvaluator<i64>,
    WsptHeuristicBuilder<i64>,
    |m: &Model<i64>| WorkloadEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| WsptHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

// --- Weighted Flow Time Evaluator Combinations ---

generate_solve!(
    bollard_bnb_solver_solve_with_wtft_evaluator_and_chronological_exhaustive_builder,
    WeightedFlowTimeEvaluator<i64>,
    ChronologicalExhaustiveBuilder<i64>,
    |m: &Model<i64>| WeightedFlowTimeEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |_| ChronologicalExhaustiveBuilder::new()
);

generate_solve!(
    bollard_bnb_solver_solve_with_wtft_evaluator_and_fcfs_heuristic_builder,
    WeightedFlowTimeEvaluator<i64>,
    FcfsHeuristicBuilder<i64>,
    |m: &Model<i64>| WeightedFlowTimeEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| FcfsHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

generate_solve!(
    bollard_bnb_solver_solve_with_wtft_evaluator_and_regret_heuristic_builder,
    WeightedFlowTimeEvaluator<i64>,
    RegretHeuristicBuilder<i64>,
    |m: &Model<i64>| WeightedFlowTimeEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| RegretHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

generate_solve!(
    bollard_bnb_solver_solve_with_wtft_evaluator_and_slack_heuristic_builder,
    WeightedFlowTimeEvaluator<i64>,
    SlackHeuristicBuilder<i64>,
    |m: &Model<i64>| WeightedFlowTimeEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| SlackHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);

generate_solve!(
    bollard_bnb_solver_solve_with_wtft_evaluator_and_wspt_heuristic_builder,
    WeightedFlowTimeEvaluator<i64>,
    WsptHeuristicBuilder<i64>,
    |m: &Model<i64>| WeightedFlowTimeEvaluator::preallocated(m.num_berths(), m.num_vessels()),
    |m: &Model<i64>| WsptHeuristicBuilder::preallocated(m.num_berths(), m.num_vessels())
);
