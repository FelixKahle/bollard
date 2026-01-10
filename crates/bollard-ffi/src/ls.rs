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

// ----------------------------------------------------------------
// Termination Reason
// ----------------------------------------------------------------

use crate::solution::BollardFfiSolution;
use bollard_ls::{engine::LocalSearchEngine, stats::LocalSearchStatistics};
use num_traits::ToPrimitive;
use std::ffi::{c_char, CString};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LocalSearchFfiTerminationReason {
    LocalOptimum = 0,
    Metaheuristic = 1,
    Aborted = 2,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocalSearchFfiTermination {
    pub reason: LocalSearchFfiTerminationReason,
    pub message: CString,
}

impl From<bollard_ls::result::LocalSearchTerminationReason> for LocalSearchFfiTermination {
    fn from(value: bollard_ls::result::LocalSearchTerminationReason) -> Self {
        match value {
            bollard_ls::result::LocalSearchTerminationReason::LocalOptimum => {
                LocalSearchFfiTermination {
                    reason: LocalSearchFfiTerminationReason::LocalOptimum,
                    message: CString::new("Local Optimum Reached").unwrap(),
                }
            }
            bollard_ls::result::LocalSearchTerminationReason::Metaheuristic(msg) => {
                LocalSearchFfiTermination {
                    reason: LocalSearchFfiTerminationReason::Metaheuristic,
                    message: CString::new(msg).expect("`CString::new` should not fail"),
                }
            }
            bollard_ls::result::LocalSearchTerminationReason::Aborted(msg) => {
                LocalSearchFfiTermination {
                    reason: LocalSearchFfiTerminationReason::Aborted,
                    message: CString::new(msg).expect("`CString::new` should not fail"),
                }
            }
        }
    }
}

/// Frees a `LocalSearchFfiTermination` previously allocated by Bollard.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `LocalSearchFfiTermination`
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_termination_free(termination: *mut LocalSearchFfiTermination) {
    if !termination.is_null() {
        drop(Box::from_raw(termination));
    }
}

/// Returns the termination reason.
///
/// # Panics
///
/// This function will panic if `ptr` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `LocalSearchFfiTermination`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_termination_reason(
    ptr: *const LocalSearchFfiTermination,
) -> LocalSearchFfiTerminationReason {
    assert!(
        !ptr.is_null(),
        "called `bollard_ls_termination_reason` with `ptr` as null pointer"
    );
    unsafe { (*ptr).reason }
}

/// Returns the termination message.
///
/// # Panics
///
/// This function will panic if `ptr` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `LocalSearchFfiTermination`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_termination_message(
    ptr: *const LocalSearchFfiTermination,
) -> *const c_char {
    assert!(
        !ptr.is_null(),
        "called `bollard_ls_termination_message` with `ptr` as null pointer"
    );
    unsafe { (*ptr).message.as_ptr() }
}

// ----------------------------------------------------------------
// Solver statistics
// ----------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocalSearchFfiStatistics {
    /// Number of iterations performed by the local search.
    pub iterations: u64,

    /// Total number of solutions found during the local search.
    pub total_solutions: u64,

    /// Number of accepted solutions during the local search.
    pub accepted_solutions: u64,

    /// Total time taken by the local search in milliseconds.
    pub time_total_ms: u64,
}

impl From<&LocalSearchStatistics> for LocalSearchFfiStatistics {
    fn from(value: &LocalSearchStatistics) -> Self {
        Self {
            iterations: value.iterations,
            total_solutions: value.total_solutions,
            accepted_solutions: value.accepted_solutions,
            time_total_ms: value.time_total.as_millis().to_u64().unwrap_or(u64::MAX),
        }
    }
}

/// Creates a new `LocalSearchFfiStatistics` instance and returns a pointer to it.
///
/// # Safety
///
/// The caller is responsible for freeing the allocated memory using
/// `bollard_ls_status_free` when it is no longer needed.
#[no_mangle]
pub extern "C" fn bollard_ls_status_new(
    iterations: u64,
    total_solutions: u64,
    accepted_solutions: u64,
    time_total_ms: u64,
) -> *mut LocalSearchFfiStatistics {
    let stats = LocalSearchFfiStatistics {
        iterations,
        total_solutions,
        accepted_solutions,
        time_total_ms,
    };
    Box::into_raw(Box::new(stats))
}

/// Frees the memory allocated for `LocalSearchFfiStatistics`.
///
/// # Safety
///
/// The caller must ensure that `status` is a valid pointer to a `LocalSearchFfiStatistics`
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_status_free(stats: *mut LocalSearchFfiStatistics) {
    if !stats.is_null() {
        drop(Box::from_raw(stats));
    }
}

/// Returns the number of rejected solutions.
///
/// # Panics
///
/// This function will panic if `ptr` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `LocalSearchFfiStatistics`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_status_rejected_solutions(
    ptr: *const LocalSearchFfiStatistics,
) -> u64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_ls_status_rejected_solutions` with `ptr` as null pointer"
    );
    unsafe {
        let stats = &*ptr;
        stats
            .total_solutions
            .saturating_sub(stats.accepted_solutions)
    }
}

/// Returns the number of iterations.
///
/// # Panics
///
/// This function will panic if `ptr` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `LocalSearchFfiStatistics`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_status_iterations(ptr: *const LocalSearchFfiStatistics) -> u64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_ls_status_iterations` with `ptr` as null pointer"
    );
    unsafe { (*ptr).iterations }
}

/// Returns the total number of solutions.
///
/// # Panics
///
/// This function will panic if `ptr` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `LocalSearchFfiStatistics`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_status_total_solutions(
    ptr: *const LocalSearchFfiStatistics,
) -> u64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_ls_status_total_solutions` with `ptr` as null pointer"
    );
    unsafe { (*ptr).total_solutions }
}

/// Returns the number of accepted solutions.
///
/// # Panics
///
/// This function will panic if `ptr` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `LocalSearchFfiStatistics`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_status_accepted_solutions(
    ptr: *const LocalSearchFfiStatistics,
) -> u64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_ls_status_accepted_solutions` with `ptr` as null pointer"
    );
    unsafe { (*ptr).accepted_solutions }
}

/// Returns the number of rejected solutions.
///
/// # Panics
///
/// This function will panic if `ptr` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `LocalSearchFfiStatistics`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_rejected_solutions(
    ptr: *const LocalSearchFfiStatistics,
) -> u64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_ls_rejected_solutions` with `ptr` as null pointer"
    );
    unsafe {
        let stats = &*ptr;
        stats
            .total_solutions
            .saturating_sub(stats.accepted_solutions)
    }
}

/// Returns the search time in milliseconds.
///
/// # Panics
///
/// This function will panic if `ptr` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `LocalSearchFfiStatistics`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_status_time_total_ms(
    ptr: *const LocalSearchFfiStatistics,
) -> u64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_ls_status_time_total_ms` with `ptr` as null pointer"
    );
    unsafe { (*ptr).time_total_ms }
}

// ----------------------------------------------------------------
// Solver outcome
// ----------------------------------------------------------------

/// The complete outcome of the BnB solver after termination,
/// including termination reason, result, and statistics.
#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalSearchFfiOutcome {
    /// The termination information.
    pub termination: *mut LocalSearchFfiTermination,
    /// The solver result.
    pub solution: *mut BollardFfiSolution, // Local Search will always carry a solution, because it starts from one.
    /// The solver statistics.
    pub statistics: *mut LocalSearchFfiStatistics,
}

impl From<bollard_ls::result::LocalSearchEngineOutcome<i64>> for LocalSearchFfiOutcome {
    fn from(value: bollard_ls::result::LocalSearchEngineOutcome<i64>) -> Self {
        let (termination, solution, statistics) = value.into_inner();

        let termination_ffi = Box::new(LocalSearchFfiTermination::from(termination));
        let statistics_ffi = Box::new(LocalSearchFfiStatistics::from(&statistics));

        Self {
            termination: Box::into_raw(termination_ffi),
            solution: Box::into_raw(Box::new(BollardFfiSolution::from(solution))),
            statistics: Box::into_raw(statistics_ffi),
        }
    }
}

/// Frees the memory allocated for `LocalSearchFfiOutcome`.
///
/// # Note
///
/// This will not free the inner pointers (`termination`, `solution`, `statistics`).
///
/// # Safety
///
/// The caller must ensure that `outcome` is a valid pointer to a `LocalSearchFfiOutcome`
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_outcome_free(outcome: *mut LocalSearchFfiOutcome) {
    if outcome.is_null() {
        return;
    }

    drop(Box::from_raw(outcome));
}

/// Retrieves the termination information from the `LocalSearchFfiOutcome`.
///
/// # Panics
///
/// This function will panic if `outcome` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `outcome` is a valid pointer to a `LocalSearchFfiOutcome`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_outcome_termination(
    outcome: *const LocalSearchFfiOutcome,
) -> *mut LocalSearchFfiTermination {
    assert!(
        !outcome.is_null(),
        "called `bollard_ls_outcome_termination` with `outcome` as null pointer",
    );
    (*outcome).termination
}

/// Retrieves the solver result from the `LocalSearchFfiOutcome`.
///
/// # Panics
///
/// This function will panic if `outcome` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `outcome` is a valid pointer to a LocalSearchFfiOutcome`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_outcome_solution(
    outcome: *const LocalSearchFfiOutcome,
) -> *mut BollardFfiSolution {
    assert!(
        !outcome.is_null(),
        "called `bollard_ls_outcome_solution` with `outcome` as null pointer",
    );
    (*outcome).solution
}

/// Retrieves the solver statistics from the `LocalSearchFfiOutcome`.
///
/// # Panics
///
/// This function will panic if `outcome` is a null pointer.
///
/// # Safety
///
/// The caller must ensure that `outcome` is a valid pointer to a `LocalSearchFfiOutcome`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_outcome_statistics(
    outcome: *const LocalSearchFfiOutcome,
) -> *mut LocalSearchFfiStatistics {
    assert!(
        !outcome.is_null(),
        "called `bollard_ls_outcome_statistics` with `outcome` as null pointer",
    );
    (*outcome).statistics
}

// ----------------------------------------------------------------
// Cooling Schedule
// ----------------------------------------------------------------

/// Parameters for a geometric cooling schedule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalSearchFfiGeometricCoolingScheduleParameters {
    initial: f64,
    current: f64,
    alpha: f64,
    min_temp: f64,
}

/// Creates a new `LocalSearchFfiGeometricCoolingScheduleParameters` and returns a pointer to it.
///
/// # Safety
///
/// The caller is responsible for freeing the allocated memory using
/// `bollard_ls_geometric_cooling_schedule_parameters_free` when it is no longer needed.
#[no_mangle]
pub extern "C" fn bollard_ls_geometric_cooling_schedule_parameters_new(
    initial: f64,
    alpha: f64,
    min_temp: f64,
) -> *mut LocalSearchFfiGeometricCoolingScheduleParameters {
    let schedule = LocalSearchFfiGeometricCoolingScheduleParameters {
        initial,
        current: initial,
        alpha,
        min_temp,
    };
    Box::into_raw(Box::new(schedule))
}

/// Frees a `LocalSearchFfiGeometricCoolingScheduleParameters` previously allocated by Bollard.
///
/// # Safety
///
/// The caller must ensure that `schedule` is a valid pointer to a `LocalSearchFfiGeometricCoolingScheduleParameters`
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_geometric_cooling_schedule_parameters_free(
    schedule: *mut LocalSearchFfiGeometricCoolingScheduleParameters,
) {
    if !schedule.is_null() {
        drop(Box::from_raw(schedule));
    }
}

/// Returns the initial temperature.
///
/// # Panics
///
/// Panics if `schedule` is null.
///
/// # Safety
///
/// The caller must ensure `schedule` is valid.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_geometric_cooling_schedule_parameters_intial(
    schedule: *const LocalSearchFfiGeometricCoolingScheduleParameters,
) -> f64 {
    assert!(
        !schedule.is_null(),
        "called `bollard_ls_geometric_cooling_schedule_parameters_initial` with `schedule` as null pointer"
    );
    unsafe { (*schedule).initial }
}

/// Returns the current temperature.
///
/// # Panics
///
/// Panics if `schedule` is null.
///
/// # Safety
///
/// The caller must ensure `schedule` is valid.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_geometric_cooling_schedule_parameters_current(
    schedule: *const LocalSearchFfiGeometricCoolingScheduleParameters,
) -> f64 {
    assert!(
        !schedule.is_null(),
        "called `bollard_ls_geometric_cooling_schedule_parameters_current` with `schedule` as null pointer"
    );
    unsafe { (*schedule).current }
}

/// Returns the alpha parameter.
///
/// # Panics
///
/// Panics if `schedule` is null.
///
/// # Safety
///
/// The caller must ensure `schedule` is valid.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_geometric_cooling_schedule_parameters_alpha(
    schedule: *const LocalSearchFfiGeometricCoolingScheduleParameters,
) -> f64 {
    assert!(
        !schedule.is_null(),
        "called `bollard_ls_geometric_cooling_schedule_parameters_alpha` with `schedule` as null pointer"
    );
    unsafe { (*schedule).alpha }
}

/// Returns the minimum temperature threshold.
///
/// # Panics
///
/// Panics if `schedule` is null.
///
/// # Safety
///
/// The caller must ensure `schedule` is valid.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_geometric_cooling_schedule_parameters_min_temp(
    schedule: *const LocalSearchFfiGeometricCoolingScheduleParameters,
) -> f64 {
    assert!(
        !schedule.is_null(),
        "called `bollard_ls_geometric_cooling_schedule_parameters_min_temp` with `schedule` as null pointer"
    );
    unsafe { (*schedule).min_temp }
}

/// Sets the current temperature.
///
/// # Panics
///
/// Panics if `schedule` is null.
///
/// # Safety
///
/// The caller must ensure `schedule` is valid.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_geometric_cooling_schedule_parameters_set_current(
    schedule: *mut LocalSearchFfiGeometricCoolingScheduleParameters,
    current: f64,
) {
    assert!(
        !schedule.is_null(),
        "called `bollard_ls_geometric_cooling_schedule_parameters_set_current` with `schedule` as null pointer"
    );

    unsafe {
        (*schedule).current = current;
    }
}

/// Sets the alpha parameter.
///
/// # Panics
///
/// Panics if `schedule` is null.
///
/// # Safety
///
/// The caller must ensure `schedule` is valid.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_geometric_cooling_schedule_parameters_set_alpha(
    schedule: *mut LocalSearchFfiGeometricCoolingScheduleParameters,
    alpha: f64,
) {
    assert!(
        !schedule.is_null(),
        "called `bollard_ls_geometric_cooling_schedule_parameters_set_alpha` with `schedule` as null pointer"
    );

    unsafe {
        (*schedule).alpha = alpha;
    }
}

/// Sets the minimum temperature threshold.
///
/// # Panics
///
/// Panics if `schedule` is null.
///
/// # Safety
///
/// The caller must ensure `schedule` is valid.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_geometric_cooling_schedule_parameters_set_min_temp(
    schedule: *mut LocalSearchFfiGeometricCoolingScheduleParameters,
    min_temp: f64,
) {
    assert!(
        !schedule.is_null(),
        "called `bollard_ls_geometric_cooling_schedule_parameters_set_min_temp` with `schedule` as null pointer"
    );
    unsafe {
        (*schedule).min_temp = min_temp;
    }
}

/// Sets the initial temperature.
///
/// # Panics
///
/// Panics if `schedule` is null.
///
/// # Safety
///
/// The caller must ensure `schedule` is valid.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_geometric_cooling_schedule_parameters_set_initial(
    schedule: *mut LocalSearchFfiGeometricCoolingScheduleParameters,
    initial: f64,
) {
    assert!(
        !schedule.is_null(),
        "called `bollard_ls_geometric_cooling_schedule_parameters_set_initial` with `schedule` as null pointer"
    );
    unsafe {
        (*schedule).initial = initial;
    }
}

/// Parameters for a linear cooling schedule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalSearchFfiLinearCoolingScheduleParameters {
    initial: f64,   // The initial temperature
    current: f64,   // The current temperature
    decrement: f64, // The delta to subtract
    min_temp: f64,  // The minimum temperature threshold
}

/// Creates a new `LocalSearchFfiLinearCoolingScheduleParameters` and returns a pointer to it.
///
/// # Safety
///
/// The caller is responsible for freeing the allocated memory using
/// `bollard_ls_linear_cooling_schedule_parameters_free` when it is no longer needed.
#[no_mangle]
pub extern "C" fn bollard_ls_linear_cooling_schedule_parameters_new(
    initial: f64,
    current: f64,
    decrement: f64,
    min_temp: f64,
) -> *mut LocalSearchFfiLinearCoolingScheduleParameters {
    let params = LocalSearchFfiLinearCoolingScheduleParameters {
        initial,
        current,
        decrement,
        min_temp,
    };
    Box::into_raw(Box::new(params))
}

/// Frees a `LocalSearchFfiLinearCoolingScheduleParameters` previously allocated by Bollard.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `LocalSearchFfiLinearCoolingScheduleParameters`
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_linear_cooling_schedule_parameters_free(
    ptr: *mut LocalSearchFfiLinearCoolingScheduleParameters,
) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// Returns the initial temperature.
///
/// # Panics
///
/// Panics if `ptr` is null.
///
/// # Safety
///
/// The caller must ensure `ptr` is valid.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_linear_cooling_schedule_parameters_initial(
    ptr: *const LocalSearchFfiLinearCoolingScheduleParameters,
) -> f64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_ls_linear_cooling_schedule_parameters_initial` with `ptr` as null pointer"
    );
    (*ptr).initial
}

/// Returns the current temperature.
///
/// # Panics
///
/// Panics if `ptr` is null.
///
/// # Safety
///
/// The caller must ensure `ptr` is valid.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_linear_cooling_schedule_parameters_current(
    ptr: *const LocalSearchFfiLinearCoolingScheduleParameters,
) -> f64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_ls_linear_cooling_schedule_parameters_current` with `ptr` as null pointer"
    );
    (*ptr).current
}

/// Returns the decrement per iteration.
///
/// # Panics
///
/// Panics if `ptr` is null.
///
/// # Safety
///
/// The caller must ensure `ptr` is valid.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_linear_cooling_schedule_parameters_decrement(
    ptr: *const LocalSearchFfiLinearCoolingScheduleParameters,
) -> f64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_ls_linear_cooling_schedule_parameters_decrement` with `ptr` as null pointer"
    );
    (*ptr).decrement
}

/// Returns the minimum temperature threshold (frozen regime).
///
/// # Panics
///
/// Panics if `ptr` is null.
///
/// # Safety
///
/// The caller must ensure `ptr` is valid.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_linear_cooling_schedule_parameters_min_temp(
    ptr: *const LocalSearchFfiLinearCoolingScheduleParameters,
) -> f64 {
    assert!(
        !ptr.is_null(),
        "called `bollard_ls_linear_cooling_schedule_parameters_min_temp` with `ptr` as null pointer"
    );
    (*ptr).min_temp
}

/// Sets the current temperature.
///
/// # Panics
///
/// Panics if `ptr` is null.
///
/// # Safety
///
/// The caller must ensure `ptr` is valid.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_linear_cooling_schedule_parameters_set_current(
    ptr: *mut LocalSearchFfiLinearCoolingScheduleParameters,
    value: f64,
) {
    assert!(
        !ptr.is_null(),
        "called `bollard_ls_linear_cooling_schedule_parameters_set_current` with `ptr` as null pointer"
    );
    (*ptr).current = value;
}

/// Sets the decrement.
///
/// # Panics
///
/// Panics if `ptr` is null.
///
/// # Safety
///
/// The caller must ensure `ptr` is valid.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_linear_cooling_schedule_parameters_set_decrement(
    ptr: *mut LocalSearchFfiLinearCoolingScheduleParameters,
    value: f64,
) {
    assert!(
        !ptr.is_null(),
        "called `bollard_ls_linear_cooling_schedule_parameters_set_decrement` with `ptr` as null pointer"
    );
    (*ptr).decrement = value;
}

/// Sets the minimum temperature threshold.
///
/// # Panics
///
/// Panics if `ptr` is null.
///
/// # Safety
///
/// The caller must ensure `ptr` is valid.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_linear_cooling_schedule_parameters_set_min_temp(
    ptr: *mut LocalSearchFfiLinearCoolingScheduleParameters,
    value: f64,
) {
    assert!(
        !ptr.is_null(),
        "called `bollard_ls_linear_cooling_schedule_parameters_set_min_temp` with `ptr` as null pointer"
    );
    (*ptr).min_temp = value;
}

/// Sets the initial temperature.
///
/// # Panics
///
/// Panics if `ptr` is null.
///
/// # Safety
///
/// The caller must ensure `ptr` is valid.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_linear_cooling_schedule_parameters_set_initial(
    ptr: *mut LocalSearchFfiLinearCoolingScheduleParameters,
    value: f64,
) {
    assert!(
        !ptr.is_null(),
        "called `bollard_ls_linear_cooling_schedule_parameters_set_initial` with `ptr` as null pointer"
    );
    (*ptr).initial = value;
}

// ----------------------------------------------------------------
// Neighboorhood Config
// ----------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LocalSearchFfiNeighborhoodConfig {
    FullNeighborhood = 0, // Every ship n in N is a neighbor for every other ship m in N with n != m
    FullTopology = 1, // Only neighbors that are connected in the topology graph are considered as neighbors
}

impl std::fmt::Display for LocalSearchFfiNeighborhoodConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LocalSearchFfiNeighborhoodConfig::FullNeighborhood => {
                write!(f, "Full Neighborhood")
            }
            LocalSearchFfiNeighborhoodConfig::FullTopology => write!(f, "Full Topology"),
        }
    }
}

// ----------------------------------------------------------------
// Engine
// ----------------------------------------------------------------

/// The Local Search engine instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalSearchFfiEngine {
    inner: LocalSearchEngine<i64>,
}

impl LocalSearchFfiEngine {
    #[inline]
    fn new(engine: LocalSearchEngine<i64>) -> Self {
        Self { inner: engine }
    }
}

/// Creates a new `LocalSearchFfiEngine` instance and returns a pointer to it.
///
/// # Safety
///
/// The caller is responsible for freeing the allocated memory using
/// `bollard_ls_engine_free` when it is no longer needed.
#[no_mangle]
pub extern "C" fn bollard_ls_engine_new() -> *mut LocalSearchFfiEngine {
    let engine = LocalSearchEngine::default();
    Box::into_raw(Box::new(LocalSearchFfiEngine::new(engine)))
}

/// Creates a new `LocalSearchFfiEngine` instance with preallocated memory for the given number of vessels
/// and returns a pointer to it.
///
/// # Safety
///
/// The caller is responsible for freeing the allocated memory using
/// `bollard_ls_engine_free` when it is no longer needed.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_engine_preallocated(
    num_vessels: usize,
) -> *mut LocalSearchFfiEngine {
    let engine = LocalSearchEngine::preallocated(num_vessels);
    Box::into_raw(Box::new(LocalSearchFfiEngine::new(engine)))
}

/// Frees a `LocalSearchFfiEngine` previously allocated by Bollard.
///
/// # Safety
///
/// The caller must ensure that `engine` is a valid pointer to a `LocalSearchFfiEngine`
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_engine_free(engine: *mut LocalSearchFfiEngine) {
    if !engine.is_null() {
        drop(Box::from_raw(engine));
    }
}
