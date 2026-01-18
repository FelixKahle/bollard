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

use bollard_ls::{engine::LocalSearchEngine, stats::LocalSearchStatistics};
use bollard_model::{model::Model, solution::Solution};
use bollard_search::neighborhood::{
    dynamic::DynamicNeighborhoods, neighborhoods::FullNeighborhoods, topology::StaticTopology,
};
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
    pub solution: *mut Solution<i64>, // Local Search will always carry a solution, because it starts from one.
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
            solution: Box::into_raw(Box::new(solution)),
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
) -> *mut Solution<i64> {
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
// Neighboorhoods
// ----------------------------------------------------------------

/// Creates a new `DynamicNeighborhoods` instance using the Full Neighborhoods strategy
/// from the given model.
///
/// # Panics
///
/// This function will panic if `model` is a null pointer.
///
/// # Safety
///
/// The caller is responsible for freeing the allocated memory using
/// `bollard_ls_neighborhoods_free` when it is no longer needed.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_full_neighborhoods_new(
    model: *const Model<i64>,
) -> *mut DynamicNeighborhoods<'static> {
    assert!(
        !model.is_null(),
        "called `bollard_ls_full_neighborhoods_new` with `model` as null pointer"
    );

    let model_ref = unsafe { &*model };
    let full_neighborhoods = FullNeighborhoods::from(model_ref);
    let dynamic_neighborhoods = DynamicNeighborhoods::from_neighborhood(full_neighborhoods);
    Box::into_raw(Box::new(dynamic_neighborhoods))
}

/// Creates a new `DynamicNeighborhoods` instance using the Static Topology strategy
/// from the given model.
///
/// # Panics
///
/// This function will panic if `model` is a null pointer.
///
/// # Safety
///
/// The caller is responsible for freeing the allocated memory using
/// `bollard_ls_neighborhoods_free` when it is no longer needed.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_static_topology_neighborhoods_new(
    model: *const Model<i64>,
) -> *mut DynamicNeighborhoods<'static> {
    assert!(
        !model.is_null(),
        "called `bollard_ls_static_topology_neighborhoods_new` with `model` as null pointer"
    );

    let model_ref = unsafe { &*model };
    let static_topology = StaticTopology::from(model_ref);
    let dynamic_neighborhoods = DynamicNeighborhoods::from_neighborhood(static_topology);
    Box::into_raw(Box::new(dynamic_neighborhoods))
}

/// Frees a `DynamicNeighborhoods` previously allocated by Bollard.
///
/// # Safety
///
/// The caller must ensure that `neighborhoods` is a valid pointer to a `DynamicNeighborhoods`
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_neighborhoods_free(
    neighborhoods: *mut DynamicNeighborhoods<'static>,
) {
    if !neighborhoods.is_null() {
        drop(Box::from_raw(neighborhoods));
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
