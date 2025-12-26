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

//! # Foreign Function Interface (FFI) for the Bollard Solver
//!
//! This module provides a stable, C-compatible ABI for interacting with the
//! Bollard Branch-and-Bound solver. It serves as the bridge between the Rust
//! core and external environments (such as C, C++, Python, Julia, or C#) while preserving
//! the solver’s performance characteristics and safety guarantees at the boundary.
//!
//! ## Overview
//!
//! The FFI is designed around opaque pointers and explicit resource management.
//! It exposes the solver's capabilities through a combinatoric set of functions
//! that allow users to select specific **Objective Evaluators** and **Search Heuristics**.
//!
//! ## Usage Lifecycle
//!
//! A typical integration flow involves the following steps:
//!
//! 1.  **Instantiation**: Create a solver instance using `bollard_bnb_solver_new`
//!     or `bollard_bnb_solver_preallocated`.
//! 2.  **Modeling**: Construct a `Model` using the sibling `model` FFI module.
//! 3.  **Solving**: Call one of the specific solve functions (see "Exported Functions" below).
//!     This step accepts constraints (time limits, solution limits) and optional
//!     fixed assignments.
//! 4.  **Inspection**: Read the results from the returned `BnbSolverFfiOutcome`.
//! 5.  **Cleanup**: Explicitly free the outcome, the model, and the solver to avoid memory leaks.
//!
//! ## Safety
//!
//! This module makes extensive use of `unsafe` code to handle raw pointers.
//! External callers **must** ensure:
//!
//! * Pointers passed to these functions are valid and allocated by this library.
//! * Pointers are not used after being passed to a `_free` function.
//! * The `fixed_len` argument accurately reflects the size of the buffer pointed to by `fixed_ptr`.
//! * `NULL` pointers are not passed unless explicitly allowed (currently, most functions panic on NULL).
//!
//! ## Exported Functions
//!
//! The following functions are available for FFI consumers.
//!
//! ### Lifecycle Management
//! * `bollard_bnb_solver_new`
//! * `bollard_bnb_solver_preallocated`
//! * `bollard_bnb_solver_free`
//!
//! ### Solve Functions (Hybrid Evaluator)
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_chronological_exhaustive_builder`
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_chronological_exhaustive_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_fcfs_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_fcfs_heuristic_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_regret_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_regret_heuristic_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_slack_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_slack_heuristic_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_wspt_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_hybrid_evaluator_and_wspt_heuristic_builder_with_fixed`
//!
//! ### Solve Functions (Workload Evaluator)
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_chronological_exhaustive_builder`
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_chronological_exhaustive_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_fcfs_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_fcfs_heuristic_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_regret_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_regret_heuristic_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_slack_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_slack_heuristic_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_wspt_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_workload_evaluator_and_wspt_heuristic_builder_with_fixed`
//!
//! ### Solve Functions (Weighted Flow Time Evaluator)
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_chronological_exhaustive_builder`
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_chronological_exhaustive_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_fcfs_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_fcfs_heuristic_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_regret_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_regret_heuristic_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_slack_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_slack_heuristic_builder_with_fixed`
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_wspt_heuristic_builder`
//! * `bollard_bnb_solver_solve_with_wtft_evaluator_and_wspt_heuristic_builder_with_fixed`

use crate::result::BnbSolverFfiOutcome;
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
    fixed::FixedAssignment,
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

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn solve<B, E>(
    solver: &mut BnbSolver<i64>,
    model: &Model<i64>,
    mut builder: B,
    mut evaluator: E,
    solution_limit: usize,
    time_limit_ms: i64,
    enable_log: bool,
    fixed_assignments: &[FixedAssignment<i64>], // Added argument
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

    // We now always use solve_with_fixed.
    // If the slice is empty, it behaves exactly like the standard solve().
    solver.solve_with_fixed(
        model,
        &mut builder,
        &mut evaluator,
        monitor,
        fixed_assignments,
    )
}

/// Macro for generating FFI solve functions for different evaluator and builder combinations.
macro_rules! generate_solve {
    (
        $fn_name:ident,
        $eval_ty:ty,
        $builder_ty:ty,
        $eval_init:expr,
        $builder_init:expr
    ) => {
        // Use the paste! macro to generate the new function name dynamically
        paste::paste! {
            /// Solves the given model.
            ///
            /// # Safety
            ///
            /// This function is unsafe because it dereferences raw pointers.
            /// The caller must ensure that the pointers are valid and were allocated
            /// by the appropriate functions.
            #[no_mangle]
            pub unsafe extern "C" fn $fn_name(
                solver_ptr: *mut BnbSolver<i64>,
                model_ptr: *const Model<i64>,
                solution_limit: usize,
                time_limit_ms: i64,
                enable_log: bool,
            ) -> *mut BnbSolverFfiOutcome {
                assert!(!solver_ptr.is_null(), "called `{}` with null solver pointer", stringify!($fn_name));
                assert!(!model_ptr.is_null(), "called `{}` with null model pointer", stringify!($fn_name));

                let solver: &mut BnbSolver<i64> = &mut *solver_ptr;
                let model: &Model<i64> = &*model_ptr;

                let evaluator: $eval_ty = ($eval_init)(model);
                let builder: $builder_ty = ($builder_init)(model);

                // Pass empty slice for standard solve
                let outcome = solve(
                    solver,
                    model,
                    builder,
                    evaluator,
                    solution_limit,
                    time_limit_ms,
                    enable_log,
                    &[],
                );

                let ffi_outcome = BnbSolverFfiOutcome::from(outcome);
                Box::into_raw(Box::new(ffi_outcome))
            }

            /// Solves the given model using fixed assignments.
            /// Expects `fixed_ptr` to point to an array of `FixedAssignment` of length `fixed_len`.
            ///
            /// # Safety
            ///
            /// This function is unsafe because it dereferences raw pointers.
            /// The caller must ensure that the pointers are valid and were allocated
            /// by the appropriate functions.
            /// Also ensures that if `fixed_len` > 0, `fixed_ptr` is not null.
            #[no_mangle]
            pub unsafe extern "C" fn [<$fn_name _with_fixed>](
                solver_ptr: *mut BnbSolver<i64>,
                model_ptr: *const Model<i64>,
                fixed_ptr: *const FixedAssignment<i64>, // Pointer to array
                fixed_len: usize,                       // Length of array
                solution_limit: usize,
                time_limit_ms: i64,
                enable_log: bool,
            ) -> *mut BnbSolverFfiOutcome {
                assert!(!solver_ptr.is_null(), "called `{}` with null solver pointer", stringify!([<$fn_name _with_fixed>]));
                assert!(!model_ptr.is_null(), "called `{}` with null model pointer", stringify!([<$fn_name _with_fixed>]));

                // Safety check for fixed array if len > 0
                if fixed_len > 0 {
                    assert!(!fixed_ptr.is_null(),
                    "called `{}` with null fixed_ptr but non-zero length",
                    stringify!([<$fn_name _with_fixed>]));
                }

                let solver: &mut BnbSolver<i64> = &mut *solver_ptr;
                let model: &Model<i64> = &*model_ptr;

                // fixed len can not be longer than number of vessels in the model
                assert!(fixed_len <= model.num_vessels(),
                    "called `{}` with fixed_len greater than number of vessels in model",
                    stringify!([<$fn_name _with_fixed>])
                );

                // Create slice from raw pointer
                let fixed_slice = if fixed_len > 0 {
                    std::slice::from_raw_parts(fixed_ptr, fixed_len)
                } else {
                    &[]
                };

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
                    fixed_slice,
                );

                let ffi_outcome = BnbSolverFfiOutcome::from(outcome);
                Box::into_raw(Box::new(ffi_outcome))
            }
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

#[cfg(test)]
mod tests {
    use crate::{
        bnb::{
            bollard_bnb_solver_free, bollard_bnb_solver_preallocated,
            bollard_bnb_solver_solve_with_hybrid_evaluator_and_regret_heuristic_builder,
        },
        model::{
            bollard_model_builder_build, bollard_model_builder_new,
            bollard_model_builder_set_arrival_time,
            bollard_model_builder_set_latest_departure_time,
            bollard_model_builder_set_processing_time, bollard_model_builder_set_vessel_weight,
            bollard_model_free,
        },
        result::{bollard_bnb_outcome_free, bollard_bnb_outcome_get_status_str},
    };

    #[test]
    fn test_solver_computes_optimal_solution() {
        unsafe {
            let builder = bollard_model_builder_new(2, 5);

            // First vessel
            bollard_model_builder_set_arrival_time(builder, 0, 0);
            bollard_model_builder_set_latest_departure_time(builder, 0, 100);
            bollard_model_builder_set_processing_time(builder, 0, 0, 4);
            bollard_model_builder_set_processing_time(builder, 0, 1, 6);
            bollard_model_builder_set_vessel_weight(builder, 0, 1);

            // Second vessel
            bollard_model_builder_set_arrival_time(builder, 1, 2);
            bollard_model_builder_set_latest_departure_time(builder, 1, 100);
            bollard_model_builder_set_processing_time(builder, 1, 0, 3);
            bollard_model_builder_set_processing_time(builder, 1, 1, 5);
            bollard_model_builder_set_vessel_weight(builder, 1, 2);

            // Third vessel
            bollard_model_builder_set_arrival_time(builder, 2, 4);
            bollard_model_builder_set_latest_departure_time(builder, 2, 100);
            bollard_model_builder_set_processing_time(builder, 2, 0, 2);
            bollard_model_builder_set_processing_time(builder, 2, 1, 4);
            bollard_model_builder_set_vessel_weight(builder, 2, 3);

            // Fourth vessel
            bollard_model_builder_set_arrival_time(builder, 3, 6);
            bollard_model_builder_set_latest_departure_time(builder, 3, 100);
            bollard_model_builder_set_processing_time(builder, 3, 0, 5);
            bollard_model_builder_set_processing_time(builder, 3, 1, 3);
            bollard_model_builder_set_vessel_weight(builder, 3, 4);

            // Fifth vessel
            bollard_model_builder_set_arrival_time(builder, 4, 8);
            bollard_model_builder_set_latest_departure_time(builder, 4, 100);
            bollard_model_builder_set_processing_time(builder, 4, 0, 4);
            bollard_model_builder_set_processing_time(builder, 4, 1, 2);
            bollard_model_builder_set_vessel_weight(builder, 4, 5);

            let model = bollard_model_builder_build(builder);
            let solver = bollard_bnb_solver_preallocated(2, 5);
            let outcome =
                bollard_bnb_solver_solve_with_hybrid_evaluator_and_regret_heuristic_builder(
                    solver, model, 0, 0, false,
                );

            assert_eq!(
                std::ffi::CStr::from_ptr(bollard_bnb_outcome_get_status_str(outcome))
                    .to_str()
                    .unwrap(),
                "Optimal"
            );

            // Clean up
            bollard_model_free(model);
            bollard_bnb_solver_free(solver);
            bollard_bnb_outcome_free(outcome);
        }
    }
}
