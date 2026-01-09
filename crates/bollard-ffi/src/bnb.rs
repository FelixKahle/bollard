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

use crate::result::BollardFfiSolverResult;
use bollard_bnb::{
    bnb::BnbSolver,
    branching::{
        chronological::ChronologicalExhaustiveBuilder, decision::DecisionBuilder,
        edf::EarliestDeadlineFirstBuilder, fcfs::FcfsHeuristicBuilder, lpt::LptHeuristicBuilder,
        regret::RegretHeuristicBuilder, slack::SlackHeuristicBuilder, spt::SptHeuristicBuilder,
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
    stats::BnbSolverStatistics,
};
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use libc::c_schar;
use num_traits::ToPrimitive;
use std::{ffi::CString, time::Duration};

// ----------------------------------------------------------------
// Termination reason
// ----------------------------------------------------------------

/// The reason for the solver's termination.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BnbSolverFfiTerminationReason {
    /// The solver found and proved optimality of a solution.
    OptimalityProven = 0,
    /// The solver proved that the problem is infeasible.
    InfeasibilityProven = 1,
    /// The solver aborted due to a search limit (time, iterations, etc.).
    /// The string contains information about the reason for abortion.
    Aborted = 2,
}

/// The termination information of the BnB solver,
/// including reason and message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BnbSolverFfiTermination {
    /// The reason for termination.
    pub reason: BnbSolverFfiTerminationReason,
    /// The termination message.
    pub message: CString,
}

impl From<bollard_bnb::result::BnbTerminationReason> for BnbSolverFfiTermination {
    fn from(value: bollard_bnb::result::BnbTerminationReason) -> Self {
        match value {
            bollard_bnb::result::BnbTerminationReason::OptimalityProven => {
                BnbSolverFfiTermination {
                    reason: BnbSolverFfiTerminationReason::OptimalityProven,
                    message: CString::new("Optimality Proven").unwrap(),
                }
            }
            bollard_bnb::result::BnbTerminationReason::InfeasibilityProven => {
                BnbSolverFfiTermination {
                    reason: BnbSolverFfiTerminationReason::InfeasibilityProven,
                    message: CString::new("Infeasibility Proven").unwrap(),
                }
            }
            bollard_bnb::result::BnbTerminationReason::Aborted(msg) => BnbSolverFfiTermination {
                reason: BnbSolverFfiTerminationReason::Aborted,
                message: CString::new(msg).expect("`CString::new` should not fail"),
            },
        }
    }
}

/// Frees a `BnbSolverFfiTermination` previously allocated by Bollard.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `BnbSolverFfiTermination`
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_termination_free(ptr: *mut BnbSolverFfiTermination) {
    if ptr.is_null() {
        return;
    }
    drop(Box::from_raw(ptr));
}

/// Returns the termination reason.
///
/// # Panics
///
/// This function will panic if `termination` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `termination` is a valid pointer to a `BnbSolverFfiTermination`.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_termination_reason(
    termination: *const BnbSolverFfiTermination,
) -> BnbSolverFfiTerminationReason {
    assert!(
        !termination.is_null(),
        "called `bollard_bnb_termination_message` with `ptr` as null pointer"
    );
    (*termination).reason
}

/// Returns the termination message.
///
/// # Panics
///
/// This function will panic if `termination` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `termination` is a valid pointer to a `BnbSolverFfiTermination`.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_termination_message(
    termination: *const BnbSolverFfiTermination,
) -> *const c_schar {
    assert!(
        !termination.is_null(),
        "called `bollard_bnb_termination_message` with `ptr` as null pointer"
    );
    (*termination).message.as_ptr()
}

// ----------------------------------------------------------------
// Solver statistics
// ----------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BnbSolverFfiStatistics {
    /// Total nodes visited.
    pub nodes_explored: u64,
    /// Total leaf nodes reached or dead-ends hit.
    pub backtracks: u64,
    /// Total distinct branching choices generated.
    pub decisions_generated: u64,
    /// The deepest level reached in the tree.
    pub max_depth: u64,
    /// Pruned because the move was structurally impossible (e.g., vessel too big).
    pub prunings_infeasible: u64,
    /// Pruned because the move cost + heuristics exceeded the bound immediately.
    /// This combines both local (node-level) and global (incumbent) pruning.
    pub prunings_bound: u64,
    /// Total solutions found during the search.
    pub solutions_found: u64,
    /// Total iterations of the main solver loop.
    pub steps: u64,
    /// Total time spent in the solver in milliseconds.
    pub time_total_ms: u64,
}

impl From<&BnbSolverStatistics> for BnbSolverFfiStatistics {
    #[inline]
    fn from(stats: &BnbSolverStatistics) -> Self {
        Self {
            nodes_explored: stats.nodes_explored,
            backtracks: stats.backtracks,
            decisions_generated: stats.decisions_generated,
            max_depth: stats.max_depth,
            prunings_infeasible: stats.prunings_infeasible,
            prunings_bound: stats.prunings_bound,
            solutions_found: stats.solutions_found,
            steps: stats.steps,
            time_total_ms: stats.time_total.as_millis().to_u64().unwrap_or(u64::MAX),
        }
    }
}

/// Creates a new `BnbSolverFfiStatistics` instance and returns a pointer to it.
///
/// # Safety
///
/// The caller is responsible for freeing the allocated memory using
/// `bollard_bnb_status_free` when it is no longer needed.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe fn bollard_bnb_status_new(
    nodes_explored: u64,
    backtracks: u64,
    decisions_generated: u64,
    max_depth: u64,
    prunings_infeasible: u64,
    prunings_bound: u64,
    solutions_found: u64,
    steps: u64,
    time_total_ms: u64,
) -> *mut BnbSolverFfiStatistics {
    let stats = BnbSolverFfiStatistics {
        nodes_explored,
        backtracks,
        decisions_generated,
        max_depth,
        prunings_infeasible,
        prunings_bound,
        solutions_found,
        steps,
        time_total_ms,
    };
    Box::into_raw(Box::new(stats))
}

/// Frees the memory allocated for `BnbSolverFfiStatistics`.
///
/// # Safety
///
/// The caller must ensure that `status` is a valid pointer to a `BnbSolverFfiStatistics`
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_status_free(status: *mut BnbSolverFfiStatistics) {
    if !status.is_null() {
        drop(Box::from_raw(status));
    }
}

/// Retrieves the number of nodes explored from the `BnbSolverFfiStatistics`.
///
/// # Panics
///
/// This function will panic if `status` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `status` is a valid pointer to a `BnbSolverFfiStatistics`.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_status_nodes_explored(
    status: *const BnbSolverFfiStatistics,
) -> u64 {
    assert!(
        !status.is_null(),
        "called `bollard_bnb_status_nodes_explored` with `status` as null pointer"
    );
    (*status).nodes_explored
}

/// Retrieves the number of backtracks from the `BnbSolverFfiStatistics`.
///
/// # Panics
///
/// This function will panic if `status` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `status` is a valid pointer to a `BnbSolverFfiStatistics`.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_status_backtracks(
    status: *const BnbSolverFfiStatistics,
) -> u64 {
    assert!(
        !status.is_null(),
        "called `bollard_bnb_status_backtracks` with `status` as null pointer",
    );
    (*status).backtracks
}

/// Retrieves the number of decisions generated from the `BnbSolverFfiStatistics`.
///
/// # Panics
///
/// This function will panic if `status` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `status` is a valid pointer to a `BnbSolverFfiStatistics`.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_status_decisions_generated(
    status: *const BnbSolverFfiStatistics,
) -> u64 {
    assert!(
        !status.is_null(),
        "called `bollard_bnb_status_decisions_generated` with `status` as null pointer",
    );
    (*status).decisions_generated
}

/// Retrieves the maximum depth reached from the `BnbSolverFfiStatistics`.
///
/// # Panics
///
/// This function will panic if `status` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `status` is a valid pointer to a `BnbSolverFfiStatistics`.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_status_max_depth(
    status: *const BnbSolverFfiStatistics,
) -> u64 {
    assert!(
        !status.is_null(),
        "called `bollard_bnb_status_max_depth` with `status` as null pointer",
    );
    (*status).max_depth
}

/// Retrieves the number of infeasible prunings from the `BnbSolverFfiStatistics`.
///
/// # Panics
///
/// This function will panic if `status` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `status` is a valid pointer to a `BnbSolverFfiStatistics`.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_status_prunings_infeasible(
    status: *const BnbSolverFfiStatistics,
) -> u64 {
    assert!(
        !status.is_null(),
        "called `bollard_bnb_status_prunings_infeasible` with `status` as null pointer",
    );
    (*status).prunings_infeasible
}

/// Retrieves the number of bound prunings from the `BnbSolverFfiStatistics`.
///
/// # Panics
///
/// This function will panic if `status` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `status` is a valid pointer to a `BnbSolverFfiStatistics`.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_status_prunings_bound(
    status: *const BnbSolverFfiStatistics,
) -> u64 {
    assert!(
        !status.is_null(),
        "called `bollard_bnb_status_prunings_bound` with `status` as null pointer",
    );
    (*status).prunings_bound
}

/// Retrieves the number of solutions found from the `BnbSolverFfiStatistics`.
///
/// # Panics
///
/// This function will panic if `status` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `status` is a valid pointer to a `BnbSolverFfiStatistics`.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_status_solutions_found(
    status: *const BnbSolverFfiStatistics,
) -> u64 {
    assert!(
        !status.is_null(),
        "called `bollard_bnb_status_solutions_found` with `status` as null pointer",
    );
    (*status).solutions_found
}

/// Retrieves the number of steps from the `BnbSolverFfiStatistics`.
///
/// # Panics
///
/// This function will panic if `status` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `status` is a valid pointer to a `BnbSolverFfiStatistics`.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_status_steps(status: *const BnbSolverFfiStatistics) -> u64 {
    assert!(
        !status.is_null(),
        "called `bollard_bnb_status_steps` with `status` as null pointer",
    );
    (*status).steps
}

/// Retrieves the total time in milliseconds from the `BnbSolverFfiStatistics`.
///
/// # Panics
///
/// This function will panic if `status` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `status` is a valid pointer to a `BnbSolverFfiStatistics`.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_status_time_total_ms(
    status: *const BnbSolverFfiStatistics,
) -> u64 {
    assert!(
        !status.is_null(),
        "called `bollard_bnb_status_time_total_ms` with `status` as null pointer",
    );
    (*status).time_total_ms
}

// ----------------------------------------------------------------
// Solver outcome
// ----------------------------------------------------------------

/// The complete outcome of the BnB solver after termination,
/// including termination reason, result, and statistics.
#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BnbSolverFfiOutcome {
    /// The termination information.
    pub termination: *mut BnbSolverFfiTermination,
    /// The solver result.
    pub result: *mut BollardFfiSolverResult,
    /// The solver statistics.
    pub statistics: *mut BnbSolverFfiStatistics,
}

impl From<bollard_bnb::result::BnbSolverOutcome<i64>> for BnbSolverFfiOutcome {
    fn from(value: bollard_bnb::result::BnbSolverOutcome<i64>) -> Self {
        let (result, termination, statistics) = value.into_inner();

        let termination_ffi = BnbSolverFfiTermination::from(termination);
        let result_ffi = BollardFfiSolverResult::from(result);
        let statistics_ffi = BnbSolverFfiStatistics::from(&statistics);

        BnbSolverFfiOutcome {
            termination: Box::into_raw(Box::new(termination_ffi)),
            result: Box::into_raw(Box::new(result_ffi)),
            statistics: Box::into_raw(Box::new(statistics_ffi)),
        }
    }
}

/// Frees the memory allocated for `BnbSolverFfiOutcome`.
///
/// # Note
///
/// This will not free the inner pointers (`termination`, `result`, `statistics`).
///
/// # Safety
///
/// The caller must ensure that `outcome` is a valid pointer to a `BnbSolverFfiOutcome`
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_free(outcome: *mut BnbSolverFfiOutcome) {
    if outcome.is_null() {
        return;
    }

    drop(Box::from_raw(outcome));
}

/// Retrieves the termination information from the `BnbSolverFfiOutcome`.
///
/// # Panics
///
/// This function will panic if `outcome` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `outcome` is a valid pointer to a `BnbSolverFfiOutcome`.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_termination(
    outcome: *const BnbSolverFfiOutcome,
) -> *mut BnbSolverFfiTermination {
    assert!(
        !outcome.is_null(),
        "called `bollard_bnb_outcome_termination` with `outcome` as null pointer",
    );
    (*outcome).termination
}

/// Retrieves the solver result from the `BnbSolverFfiOutcome`.
///
/// # Panics
///
/// This function will panic if `outcome` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `outcome` is a valid pointer to a `BnbSolverFfiOutcome`.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_result(
    outcome: *const BnbSolverFfiOutcome,
) -> *mut BollardFfiSolverResult {
    assert!(
        !outcome.is_null(),
        "called `bollard_bnb_outcome_result` with `outcome` as null pointer",
    );
    (*outcome).result
}

/// Retrieves the solver statistics from the `BnbSolverFfiOutcome`.
///
/// # Panics
///
/// This function will panic if `outcome` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `outcome` is a valid pointer to a `BnbSolverFfiOutcome`.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_outcome_statistics(
    outcome: *const BnbSolverFfiOutcome,
) -> *mut BnbSolverFfiStatistics {
    assert!(
        !outcome.is_null(),
        "called `bollard_bnb_outcome_statistics` with `outcome` as null pointer",
    );
    (*outcome).statistics
}

// ----------------------------------------------------------------
// Fixed Assignments
// ----------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BnbFfiFixedAssignment {
    /// The start time of the assignment.
    pub start_time: i64,
    /// The index of the berth.
    pub berth_index: usize,
    /// The index of the vessel.
    pub vessel_index: usize,
}

impl std::fmt::Display for BnbFfiFixedAssignment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BnbFfiFixedAssignment {{ vessel_index: {}, berth_index: {}, start_time: {} }}",
            self.vessel_index, self.berth_index, self.start_time
        )
    }
}

impl From<FixedAssignment<i64>> for BnbFfiFixedAssignment {
    #[inline]
    fn from(fixed: FixedAssignment<i64>) -> Self {
        Self {
            start_time: fixed.start_time,
            berth_index: fixed.berth_index.get(),
            vessel_index: fixed.vessel_index.get(),
        }
    }
}

impl From<BnbFfiFixedAssignment> for FixedAssignment<i64> {
    #[inline]
    fn from(val: BnbFfiFixedAssignment) -> Self {
        FixedAssignment {
            start_time: val.start_time,
            berth_index: BerthIndex::new(val.berth_index),
            vessel_index: VesselIndex::new(val.vessel_index),
        }
    }
}

// ----------------------------------------------------------------
// Parameters
// ----------------------------------------------------------------

/// The type of decision builder to use in the BnB solver.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum BnbSolverFfiDecisionBuilderType {
    ChronologicalExhaustive = 0,
    FcfsHeuristic = 1,
    RegretHeuristic = 2,
    SlackHeuristic = 3,
    #[default]
    WsptHeuristic = 4, // We default to WSPT. Its generally the best performing sorting heuristic.
    SptHeuristic = 5,
    LptHeuristic = 6,
    EarliestDeadlineFirst = 7,
}

impl std::fmt::Display for BnbSolverFfiDecisionBuilderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            BnbSolverFfiDecisionBuilderType::ChronologicalExhaustive => "ChronologicalExhaustive",
            BnbSolverFfiDecisionBuilderType::FcfsHeuristic => "FcfsHeuristic",
            BnbSolverFfiDecisionBuilderType::RegretHeuristic => "RegretHeuristic",
            BnbSolverFfiDecisionBuilderType::SlackHeuristic => "SlackHeuristic",
            BnbSolverFfiDecisionBuilderType::WsptHeuristic => "WsptHeuristic",
            BnbSolverFfiDecisionBuilderType::SptHeuristic => "SptHeuristic",
            BnbSolverFfiDecisionBuilderType::LptHeuristic => "LptHeuristic",
            BnbSolverFfiDecisionBuilderType::EarliestDeadlineFirst => "EarliestDeadlineFirst",
        };
        write!(f, "{}", name)
    }
}

/// The type of objective evaluator to use in the BnB solver.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum BnbSolverFfiObjectiveEvaluatorType {
    #[default]
    Hybrid = 0, // Basically the union of Workload and WeightedFlowTime. Just better overall.
    Workload = 1,
    WeightedFlowTime = 2,
}

impl std::fmt::Display for BnbSolverFfiObjectiveEvaluatorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            BnbSolverFfiObjectiveEvaluatorType::Hybrid => "Hybrid",
            BnbSolverFfiObjectiveEvaluatorType::Workload => "Workload",
            BnbSolverFfiObjectiveEvaluatorType::WeightedFlowTime => "WeightedFlowTime",
        };
        write!(f, "{}", name)
    }
}

// ----------------------------------------------------------------
// Solver
// ----------------------------------------------------------------

/// Creates a new Branch-and-Bound solver instance.
#[no_mangle]
pub extern "C" fn bollard_bnb_solver_new() -> *mut BnbSolver<i64> {
    let solver = BnbSolver::<i64>::new();
    Box::into_raw(Box::new(solver))
}

/// Creates a new Branch-and-Bound solver instance from the given model.
///
/// # Panics
///
/// This function will panic if `model_ptr` is null.
///
/// # Safety
///
/// The caller must ensure that `model_ptr` is a valid pointer to a `Model<i64>`.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_solver_from_model(
    model_ptr: *const Model<i64>,
) -> *mut BnbSolver<i64> {
    assert!(
        !model_ptr.is_null(),
        "called `bollard_bnb_solver_from_model` with `model_ptr` as null pointer"
    );
    let model: &Model<i64> = unsafe { &*model_ptr };

    let num_vessels = model.num_vessels();
    let num_berths = model.num_berths();

    let solver = BnbSolver::<i64>::preallocated(num_berths, num_vessels);

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
/// The caller must ensure that `ptr` is a valid pointer to a `BnbSolver`
/// allocated by Bollard.
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
    fixed_assignments: &[FixedAssignment<i64>],
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

    if fixed_assignments.is_empty() {
        solver.solve(model, &mut builder, &mut evaluator, monitor)
    } else {
        solver.solve_with_fixed(
            model,
            &mut builder,
            &mut evaluator,
            monitor,
            fixed_assignments,
        )
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn dispatch_solve(
    solver: &mut BnbSolver<i64>,
    model: &Model<i64>,
    builder_type: BnbSolverFfiDecisionBuilderType,
    evaluator_type: BnbSolverFfiObjectiveEvaluatorType,
    fixed_assignments: &[FixedAssignment<i64>], // Passed as slice
    solution_limit: usize,
    time_limit_ms: i64,
    enable_log: bool,
) -> *mut BnbSolverFfiOutcome {
    let num_vessels = model.num_vessels();
    let num_berths = model.num_berths();

    let outcome = match (builder_type, evaluator_type) {
        (
            BnbSolverFfiDecisionBuilderType::ChronologicalExhaustive,
            BnbSolverFfiObjectiveEvaluatorType::Hybrid,
        ) => {
            let builder = ChronologicalExhaustiveBuilder::preallocated(num_berths, num_vessels);
            let evaluator = HybridEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::ChronologicalExhaustive,
            BnbSolverFfiObjectiveEvaluatorType::Workload,
        ) => {
            let builder = ChronologicalExhaustiveBuilder::preallocated(num_berths, num_vessels);
            let evaluator = WorkloadEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::ChronologicalExhaustive,
            BnbSolverFfiObjectiveEvaluatorType::WeightedFlowTime,
        ) => {
            let builder = ChronologicalExhaustiveBuilder::preallocated(num_berths, num_vessels);
            let evaluator = WeightedFlowTimeEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::FcfsHeuristic,
            BnbSolverFfiObjectiveEvaluatorType::Hybrid,
        ) => {
            let builder = FcfsHeuristicBuilder::preallocated(num_berths, num_vessels);
            let evaluator = HybridEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::FcfsHeuristic,
            BnbSolverFfiObjectiveEvaluatorType::Workload,
        ) => {
            let builder = FcfsHeuristicBuilder::preallocated(num_berths, num_vessels);
            let evaluator = WorkloadEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::FcfsHeuristic,
            BnbSolverFfiObjectiveEvaluatorType::WeightedFlowTime,
        ) => {
            let builder = FcfsHeuristicBuilder::preallocated(num_berths, num_vessels);
            let evaluator = WeightedFlowTimeEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::RegretHeuristic,
            BnbSolverFfiObjectiveEvaluatorType::Hybrid,
        ) => {
            let builder = RegretHeuristicBuilder::preallocated(num_berths, num_vessels);
            let evaluator = HybridEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::RegretHeuristic,
            BnbSolverFfiObjectiveEvaluatorType::Workload,
        ) => {
            let builder = RegretHeuristicBuilder::preallocated(num_berths, num_vessels);
            let evaluator = WorkloadEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::RegretHeuristic,
            BnbSolverFfiObjectiveEvaluatorType::WeightedFlowTime,
        ) => {
            let builder = RegretHeuristicBuilder::preallocated(num_berths, num_vessels);
            let evaluator = WeightedFlowTimeEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::SlackHeuristic,
            BnbSolverFfiObjectiveEvaluatorType::Hybrid,
        ) => {
            let builder = SlackHeuristicBuilder::preallocated(num_berths, num_vessels);
            let evaluator = HybridEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::SlackHeuristic,
            BnbSolverFfiObjectiveEvaluatorType::Workload,
        ) => {
            let builder = SlackHeuristicBuilder::preallocated(num_berths, num_vessels);
            let evaluator = WorkloadEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::SlackHeuristic,
            BnbSolverFfiObjectiveEvaluatorType::WeightedFlowTime,
        ) => {
            let builder = SlackHeuristicBuilder::preallocated(num_berths, num_vessels);
            let evaluator = WeightedFlowTimeEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::WsptHeuristic,
            BnbSolverFfiObjectiveEvaluatorType::Hybrid,
        ) => {
            let builder = WsptHeuristicBuilder::preallocated(num_berths, num_vessels);
            let evaluator = HybridEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::WsptHeuristic,
            BnbSolverFfiObjectiveEvaluatorType::Workload,
        ) => {
            let builder = WsptHeuristicBuilder::preallocated(num_berths, num_vessels);
            let evaluator = WorkloadEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::WsptHeuristic,
            BnbSolverFfiObjectiveEvaluatorType::WeightedFlowTime,
        ) => {
            let builder = WsptHeuristicBuilder::preallocated(num_berths, num_vessels);
            let evaluator = WeightedFlowTimeEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::SptHeuristic,
            BnbSolverFfiObjectiveEvaluatorType::Hybrid,
        ) => {
            let builder = SptHeuristicBuilder::preallocated(num_berths, num_vessels);
            let evaluator = HybridEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::SptHeuristic,
            BnbSolverFfiObjectiveEvaluatorType::Workload,
        ) => {
            let builder = SptHeuristicBuilder::preallocated(num_berths, num_vessels);
            let evaluator = WorkloadEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::SptHeuristic,
            BnbSolverFfiObjectiveEvaluatorType::WeightedFlowTime,
        ) => {
            let builder = SptHeuristicBuilder::preallocated(num_berths, num_vessels);
            let evaluator = WeightedFlowTimeEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::LptHeuristic,
            BnbSolverFfiObjectiveEvaluatorType::Hybrid,
        ) => {
            let builder = LptHeuristicBuilder::preallocated(num_berths, num_vessels);
            let evaluator = HybridEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::LptHeuristic,
            BnbSolverFfiObjectiveEvaluatorType::Workload,
        ) => {
            let builder = LptHeuristicBuilder::preallocated(num_berths, num_vessels);
            let evaluator = WorkloadEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::LptHeuristic,
            BnbSolverFfiObjectiveEvaluatorType::WeightedFlowTime,
        ) => {
            let builder = LptHeuristicBuilder::preallocated(num_berths, num_vessels);
            let evaluator = WeightedFlowTimeEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::EarliestDeadlineFirst,
            BnbSolverFfiObjectiveEvaluatorType::Hybrid,
        ) => {
            let builder = EarliestDeadlineFirstBuilder::preallocated(num_berths, num_vessels);
            let evaluator = HybridEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::EarliestDeadlineFirst,
            BnbSolverFfiObjectiveEvaluatorType::Workload,
        ) => {
            let builder = EarliestDeadlineFirstBuilder::preallocated(num_berths, num_vessels);
            let evaluator = WorkloadEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
        (
            BnbSolverFfiDecisionBuilderType::EarliestDeadlineFirst,
            BnbSolverFfiObjectiveEvaluatorType::WeightedFlowTime,
        ) => {
            let builder = EarliestDeadlineFirstBuilder::preallocated(num_berths, num_vessels);
            let evaluator = WeightedFlowTimeEvaluator::preallocated(num_berths, num_vessels);

            let outcome = solve(
                solver,
                model,
                builder,
                evaluator,
                solution_limit,
                time_limit_ms,
                enable_log,
                fixed_assignments, // With fixed assignments
            );
            BnbSolverFfiOutcome::from(outcome)
        }
    };

    Box::into_raw(Box::new(outcome))
}

/// Solves the given model using the specified decision builder and objective evaluator types.
///
/// The caller can control the number of solutions to find, time limit, and logging options.
/// When `solution_limit` is 0, there is no limit on the number of solutions to find,
/// and when `time_limit_ms` is 0, there is no time limit.
/// Set `enable_log` to true to enable logging during the solving process.
///
/// # Panics
///
/// This function will panic if `solver_ptr` or `model_ptr` is null.
///
/// # Safety
///
/// The caller must ensure that `solver_ptr` is a valid pointer to a `BnbSolver<i64>`
/// and `model_ptr` is a valid pointer to a `Model<i64>`.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_solver_solve(
    solver_ptr: *mut BnbSolver<i64>,
    model_ptr: *const Model<i64>,
    builder_type: BnbSolverFfiDecisionBuilderType,
    evaluator_type: BnbSolverFfiObjectiveEvaluatorType,
    solution_limit: usize, // 0 means no limit
    time_limit_ms: i64,    // 0 means no limit
    enable_log: bool,      // whether to enable logging
) -> *mut BnbSolverFfiOutcome {
    assert!(
        !solver_ptr.is_null(),
        "called `bollard_bnb_solver_solve` with `solver_ptr` as null pointer"
    );
    assert!(
        !model_ptr.is_null(),
        "called `bollard_bnb_solver_solve` with `model_ptr` as null pointer"
    );

    let solver = &mut *solver_ptr;
    let model = &*model_ptr;

    dispatch_solve(
        solver,
        model,
        builder_type,
        evaluator_type,
        &[], // No fixed assignments
        solution_limit,
        time_limit_ms,
        enable_log,
    )
}

/// Solves the given model using the specified decision builder and objective evaluator types,
/// with respect to fixed assignments.
///
/// The caller can control the number of solutions to find, time limit, and logging options.
/// When `solution_limit` is 0, there is no limit on the number of solutions to find,
/// and when `time_limit_ms` is 0, there is no time limit.
/// Set `enable_log` to true to enable logging during the solving process.
///
/// # Panics
///
/// This function will panic if `solver_ptr` or `model_ptr` is null.
/// It will also panic if `fixed_assignments_ptr` is null while `num_fixed_assignments` is non-zero.
///
/// # Safety
///
/// The caller must ensure that `solver_ptr` is a valid pointer to a `BnbSolver<i64>`
/// and `model_ptr` is a valid pointer to a `Model<i64>`.
/// The caller must also ensure that `fixed_assignments_ptr` is a valid pointer to an array
/// of `BnbFfiFixedAssignment` of length `num_fixed_assignments` if `num_fixed_assignments` is greater than zero.
#[no_mangle]
pub unsafe extern "C" fn bollard_bnb_solver_solve_with_fixed_assignments(
    solver_ptr: *mut BnbSolver<i64>,
    model_ptr: *const Model<i64>,
    builder_type: BnbSolverFfiDecisionBuilderType,
    evaluator_type: BnbSolverFfiObjectiveEvaluatorType,
    solution_limit: usize, // 0 means no limit
    time_limit_ms: i64,    // 0 means no limit
    enable_log: bool,      // whether to enable logging
    fixed_assignments_ptr: *const BnbFfiFixedAssignment,
    num_fixed_assignments: usize,
) -> *mut BnbSolverFfiOutcome {
    assert!(
        !solver_ptr.is_null(),
        "called `bollard_bnb_solver_solve` with `solver_ptr` as null pointer"
    );
    assert!(
        !model_ptr.is_null(),
        "called `bollard_bnb_solver_solve` with `model_ptr` as null pointer"
    );

    if fixed_assignments_ptr.is_null() && num_fixed_assignments == 0 {
        // No fixed assignments
        return bollard_bnb_solver_solve(
            solver_ptr,
            model_ptr,
            builder_type,
            evaluator_type,
            solution_limit,
            time_limit_ms,
            enable_log,
        );
    }

    assert!(
        !fixed_assignments_ptr.is_null(), // now num_fixed_assignments > 0
        "called `bollard_bnb_solver_solve_with_fixed_assignments` with `fixed_assignments_ptr` as null pointer while `num_fixed_assignments` is non-zero"
    );

    let solver = &mut *solver_ptr;
    let model = &*model_ptr;

    let fixed_assignments_slice =
        { std::slice::from_raw_parts(fixed_assignments_ptr, num_fixed_assignments) };

    let fixed_assignments: Vec<FixedAssignment<i64>> = fixed_assignments_slice
        .iter()
        .map(|fa| (*fa).into())
        .collect();

    dispatch_solve(
        solver,
        model,
        builder_type,
        evaluator_type,
        &fixed_assignments, // With fixed assignments
        solution_limit,
        time_limit_ms,
        enable_log,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use bollard_bnb::{
        result::{BnbSolverOutcome, BnbTerminationReason},
        stats::BnbSolverStatistics,
    };
    use bollard_model::{
        index::{BerthIndex, VesselIndex},
        solution::Solution,
    };
    use std::ffi::CStr;
    use std::time::Duration;

    // Helper to interpret returned *const c_schar as CStr
    unsafe fn cstr_from_schar<'a>(ptr: *const libc::c_schar) -> &'a CStr {
        CStr::from_ptr(ptr as *const std::os::raw::c_char)
    }

    // ----------------------------
    // Termination mapping + accessors
    // ----------------------------

    #[test]
    fn test_termination_mapping_optimality_and_accessors() {
        let ffi = BnbSolverFfiTermination::from(BnbTerminationReason::OptimalityProven);
        assert_eq!(ffi.reason, BnbSolverFfiTerminationReason::OptimalityProven);

        unsafe {
            let p = &ffi as *const BnbSolverFfiTermination;
            assert_eq!(
                bollard_bnb_termination_reason(p),
                BnbSolverFfiTerminationReason::OptimalityProven
            );
            let msg = cstr_from_schar(bollard_bnb_termination_message(p))
                .to_str()
                .unwrap();
            assert_eq!(msg, "Optimality Proven");
        }
    }

    #[test]
    fn test_termination_mapping_infeasible_and_accessors() {
        let ffi = BnbSolverFfiTermination::from(BnbTerminationReason::InfeasibilityProven);
        assert_eq!(
            ffi.reason,
            BnbSolverFfiTerminationReason::InfeasibilityProven
        );

        unsafe {
            let p = &ffi as *const BnbSolverFfiTermination;
            assert_eq!(
                bollard_bnb_termination_reason(p),
                BnbSolverFfiTerminationReason::InfeasibilityProven
            );
            let msg = cstr_from_schar(bollard_bnb_termination_message(p))
                .to_str()
                .unwrap();
            assert_eq!(msg, "Infeasibility Proven");
        }
    }

    #[test]
    fn test_termination_mapping_aborted_and_accessors_with_non_ascii() {
        let reason_msg = "aborted: límite ✓";
        let ffi = BnbSolverFfiTermination::from(BnbTerminationReason::Aborted(reason_msg.into()));
        assert_eq!(ffi.reason, BnbSolverFfiTerminationReason::Aborted);

        unsafe {
            let p = &ffi as *const BnbSolverFfiTermination;
            assert_eq!(
                bollard_bnb_termination_reason(p),
                BnbSolverFfiTerminationReason::Aborted
            );
            let msg = cstr_from_schar(bollard_bnb_termination_message(p))
                .to_str()
                .unwrap();
            assert_eq!(msg, reason_msg);
        }
    }

    #[test]
    fn test_termination_free_handles_null() {
        unsafe {
            bollard_bnb_termination_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_stats_time_total_ms_saturates_on_overflow() {
        // u64::MAX seconds -> as_millis() (u128) definitely exceeds u64
        let mut s = BnbSolverStatistics::default();
        s.set_total_time(Duration::new(u64::MAX, 0));
        let ffi = BnbSolverFfiStatistics::from(&s);
        assert_eq!(ffi.time_total_ms, u64::MAX);
    }

    #[test]
    fn test_stats_ffi_constructor_getters_and_free() {
        unsafe {
            let p = bollard_bnb_status_new(
                1, // nodes_explored
                2, // backtracks
                3, // decisions_generated
                4, // max_depth
                5, // prunings_infeasible
                6, // prunings_bound
                7, // solutions_found
                8, // steps
                9, // time_total_ms
            );
            assert!(!p.is_null());
            assert_eq!(bollard_bnb_status_nodes_explored(p), 1);
            assert_eq!(bollard_bnb_status_backtracks(p), 2);
            assert_eq!(bollard_bnb_status_decisions_generated(p), 3);
            assert_eq!(bollard_bnb_status_max_depth(p), 4);
            assert_eq!(bollard_bnb_status_prunings_infeasible(p), 5);
            assert_eq!(bollard_bnb_status_prunings_bound(p), 6);
            assert_eq!(bollard_bnb_status_solutions_found(p), 7);
            assert_eq!(bollard_bnb_status_steps(p), 8);
            assert_eq!(bollard_bnb_status_time_total_ms(p), 9);

            bollard_bnb_status_free(p);
            // Null free should be a no-op
            bollard_bnb_status_free(std::ptr::null_mut());
        }
    }

    // ----------------------------
    // Outcome conversion + accessors + free
    // ----------------------------

    #[test]
    fn test_outcome_from_optimal_accessors_and_free() {
        // Build a tiny Solution<i64>
        let berths = vec![BerthIndex::new(0)];
        let starts = vec![0_i64];
        let sol = Solution::<i64>::new(123_i64, berths, starts);

        let stats = BnbSolverStatistics::default();
        let outcome = BnbSolverOutcome::<i64>::optimal(sol, stats);

        let ffi_outcome = BnbSolverFfiOutcome::from(outcome);
        assert!(!ffi_outcome.termination.is_null());
        assert!(!ffi_outcome.result.is_null());
        assert!(!ffi_outcome.statistics.is_null());

        unsafe {
            // Access via FFI functions
            let op = &ffi_outcome as *const BnbSolverFfiOutcome;
            let term_ptr = bollard_bnb_outcome_termination(op);
            let res_ptr = bollard_bnb_outcome_result(op);
            let stats_ptr = bollard_bnb_outcome_statistics(op);

            // Termination message should be "Optimality Proven"
            let term_msg = cstr_from_schar(bollard_bnb_termination_message(term_ptr))
                .to_str()
                .unwrap();
            assert_eq!(term_msg, "Optimality Proven");

            // Result status should be Optimal
            use crate::result::{
                bollard_ffi_solver_result_free, bollard_ffi_solver_result_status, BollardFfiStatus,
            };
            assert_eq!(
                bollard_ffi_solver_result_status(res_ptr),
                BollardFfiStatus::Optimal
            );

            // Stats should be present and readable
            let _ = bollard_bnb_status_nodes_explored(stats_ptr);

            // Free inner pointers first (container free does not free inners)
            bollard_bnb_termination_free(term_ptr);
            bollard_ffi_solver_result_free(res_ptr);
            bollard_bnb_status_free(stats_ptr);

            // Now free the outcome container
            let heap_outcome = Box::into_raw(Box::new(ffi_outcome));
            bollard_bnb_outcome_free(heap_outcome);
            // Null free is a no-op
            bollard_bnb_outcome_free(std::ptr::null_mut());
        }
    }

    // ----------------------------
    // Fixed assignment conversions + Display
    // ----------------------------

    #[test]
    fn test_fixed_assignment_roundtrip_and_display() {
        let ffi = BnbFfiFixedAssignment {
            start_time: 42,
            berth_index: 2,
            vessel_index: 3,
        };
        let native: FixedAssignment<i64> = ffi.into();
        assert_eq!(native.start_time, 42);
        assert_eq!(native.berth_index, BerthIndex::new(2));
        assert_eq!(native.vessel_index, VesselIndex::new(3));

        let ffi_back = BnbFfiFixedAssignment::from(native);
        assert_eq!(ffi_back, ffi);

        let s = format!("{}", ffi);
        assert_eq!(
            s,
            "BnbFfiFixedAssignment { vessel_index: 3, berth_index: 2, start_time: 42 }"
        );
    }

    // ----------------------------
    // Display on enums
    // ----------------------------

    #[test]
    fn test_decision_builder_type_display() {
        assert_eq!(
            format!(
                "{}",
                BnbSolverFfiDecisionBuilderType::ChronologicalExhaustive
            ),
            "ChronologicalExhaustive"
        );
        assert_eq!(
            format!("{}", BnbSolverFfiDecisionBuilderType::FcfsHeuristic),
            "FcfsHeuristic"
        );
        assert_eq!(
            format!("{}", BnbSolverFfiDecisionBuilderType::RegretHeuristic),
            "RegretHeuristic"
        );
        assert_eq!(
            format!("{}", BnbSolverFfiDecisionBuilderType::SlackHeuristic),
            "SlackHeuristic"
        );
        assert_eq!(
            format!("{}", BnbSolverFfiDecisionBuilderType::WsptHeuristic),
            "WsptHeuristic"
        );
        assert_eq!(
            format!("{}", BnbSolverFfiDecisionBuilderType::SptHeuristic),
            "SptHeuristic"
        );
        assert_eq!(
            format!("{}", BnbSolverFfiDecisionBuilderType::LptHeuristic),
            "LptHeuristic"
        );
        assert_eq!(
            format!("{}", BnbSolverFfiDecisionBuilderType::EarliestDeadlineFirst),
            "EarliestDeadlineFirst"
        );
    }

    #[test]
    fn test_objective_evaluator_type_display() {
        assert_eq!(
            format!("{}", BnbSolverFfiObjectiveEvaluatorType::Hybrid),
            "Hybrid"
        );
        assert_eq!(
            format!("{}", BnbSolverFfiObjectiveEvaluatorType::Workload),
            "Workload"
        );
        assert_eq!(
            format!("{}", BnbSolverFfiObjectiveEvaluatorType::WeightedFlowTime),
            "WeightedFlowTime"
        );
    }

    // ----------------------------
    // Solver handle alloc/free (smoke)
    // ----------------------------

    #[test]
    fn test_solver_new_preallocated_and_free() {
        unsafe {
            let s1 = bollard_bnb_solver_new();
            assert!(!s1.is_null());
            bollard_bnb_solver_free(s1);

            let s2 = bollard_bnb_solver_preallocated(1, 1);
            assert!(!s2.is_null());
            bollard_bnb_solver_free(s2);

            // Null free is a no-op
            bollard_bnb_solver_free(std::ptr::null_mut());
        }
    }
}
