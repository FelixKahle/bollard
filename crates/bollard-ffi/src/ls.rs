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

use bollard_ls::{
    engine::LocalSearchEngine,
    operator::{
        compound::{
            MultiArmedBanditCompoundOperator, RandomCompoundOperator, RoundRobinCompoundOperator,
        },
        dynamic::DynamicLocalSearchOperator,
        local_search_operator::LocalSearchOperator,
        scramble::ScrambleOperator,
        shift::ShiftOperator,
        swap::SwapOperator,
        two_opt::TwoOptOperator,
    },
    stats::LocalSearchStatistics,
};
use bollard_model::{model::Model, solution::Solution};
use bollard_search::neighborhood::{
    dynamic::DynamicNeighborhoods, neighborhoods::FullNeighborhoods, topology::StaticTopology,
};
use num_traits::ToPrimitive;
use rand::SeedableRng;
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
// Operators
// ----------------------------------------------------------------

// ----------------------------------------------------------------
// Operators
// ----------------------------------------------------------------

/// Unsafe helper to convert a C-style array of operator pointers into a Rust Vector of trait objects.
///
/// # Safety
/// - Assumes ownership of the pointers in the array.
/// - Does NOT free the array itself.
unsafe fn consume_operator_array(
    operators_ptr: *const *mut DynamicLocalSearchOperator<
        'static,
        i64,
        DynamicNeighborhoods<'static>,
    >,
    operators_len: usize,
) -> Vec<Box<dyn LocalSearchOperator<i64, DynamicNeighborhoods<'static>>>> {
    if operators_len == 0 {
        return Vec::new();
    }

    assert!(
        !operators_ptr.is_null(),
        "Operator array pointer is null but length > 0"
    );

    let raw_slice = std::slice::from_raw_parts(operators_ptr, operators_len);

    raw_slice
        .iter()
        .map(|&ptr| {
            assert!(!ptr.is_null(), "Found null pointer in operator array");
            // Recover concrete Box
            let concrete_box = Box::from_raw(ptr);
            // Cast to Trait Object
            concrete_box as Box<dyn LocalSearchOperator<i64, DynamicNeighborhoods<'static>>>
        })
        .collect()
}

/// Creates a new `DynamicLocalSearchOperator` using the Swap strategy.
///
/// # Safety
///
/// The caller is responsible for freeing the allocated memory using
/// `bollard_ls_free_dynamic_local_search_operator` when it is no longer needed.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_swap_operator_new(
) -> *mut DynamicLocalSearchOperator<'static, i64, DynamicNeighborhoods<'static>> {
    Box::into_raw(Box::new(DynamicLocalSearchOperator::new(Box::new(
        SwapOperator::new(),
    ))))
}

/// Creates a new `DynamicLocalSearchOperator` using the Shift strategy.
///
/// # Safety
///
/// The caller is responsible for freeing the allocated memory using
/// `bollard_ls_free_dynamic_local_search_operator` when it is no longer needed.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_shift_operator_new(
) -> *mut DynamicLocalSearchOperator<'static, i64, DynamicNeighborhoods<'static>> {
    Box::into_raw(Box::new(DynamicLocalSearchOperator::new(Box::new(
        ShiftOperator::new(),
    ))))
}

/// Creates a new `DynamicLocalSearchOperator` using the Scramble strategy with a random seed.
///
/// # Safety
///
/// The caller is responsible for freeing the allocated memory using
/// `bollard_ls_free_dynamic_local_search_operator` when it is no longer needed.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_scramble_operator_new(
) -> *mut DynamicLocalSearchOperator<'static, i64, DynamicNeighborhoods<'static>> {
    let rng = rand::rngs::StdRng::from_os_rng();
    Box::into_raw(Box::new(DynamicLocalSearchOperator::new(Box::new(
        ScrambleOperator::new(rng),
    ))))
}

/// Creates a new `DynamicLocalSearchOperator` using the Scramble strategy with a specific seed.
///
/// # Safety
///
/// The caller is responsible for freeing the allocated memory using
/// `bollard_ls_free_dynamic_local_search_operator` when it is no longer needed.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_scramble_operator_new_with_seed(
    seed: u64,
) -> *mut DynamicLocalSearchOperator<'static, i64, DynamicNeighborhoods<'static>> {
    let rng = rand::rngs::StdRng::seed_from_u64(seed);
    Box::into_raw(Box::new(DynamicLocalSearchOperator::new(Box::new(
        ScrambleOperator::new(rng),
    ))))
}

/// Creates a new `DynamicLocalSearchOperator` using the 2-Opt strategy.
///
/// # Safety
///
/// The caller is responsible for freeing the allocated memory using
/// `bollard_ls_free_dynamic_local_search_operator` when it is no longer needed.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_two_opt_operator_new(
) -> *mut DynamicLocalSearchOperator<'static, i64, DynamicNeighborhoods<'static>> {
    Box::into_raw(Box::new(DynamicLocalSearchOperator::new(Box::new(
        TwoOptOperator::new(),
    ))))
}

// ----------------------------------------------------------------
// Compound Operators
// ----------------------------------------------------------------

/// Creates a new compound operator that selects sub-operators in a Round-Robin fashion.
///
/// # Ownership Contract
///
/// This function **takes ownership** of the operators pointed to by the `operators_ptr` array.
/// The caller **must not free** the individual operators passed to this function.
/// However, the caller **must still free** the `operators_ptr` array itself.
///
/// # Panics
///
/// This function will panic if `operators_ptr` is null but `operators_len` is greater than 0,
/// or if any pointer inside the array is null.
///
/// # Safety
///
/// The caller must ensure that `operators_ptr` points to a valid array of
/// `operators_len` pointers to `DynamicLocalSearchOperator`.
/// The caller is responsible for freeing the returned pointer using
/// `bollard_ls_free_dynamic_local_search_operator`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_round_robin_operator_new(
    operators_ptr: *const *mut DynamicLocalSearchOperator<
        'static,
        i64,
        DynamicNeighborhoods<'static>,
    >,
    operators_len: usize,
) -> *mut DynamicLocalSearchOperator<'static, i64, DynamicNeighborhoods<'static>> {
    let operators = consume_operator_array(operators_ptr, operators_len);

    if operators.is_empty() {
        let op = RoundRobinCompoundOperator::empty();
        return Box::into_raw(Box::new(DynamicLocalSearchOperator::new(Box::new(op))));
    }

    let op = RoundRobinCompoundOperator::new(operators);
    Box::into_raw(Box::new(DynamicLocalSearchOperator::new(Box::new(op))))
}

/// Creates a new compound operator that selects sub-operators randomly.
///
/// # Ownership Contract
///
/// This function **takes ownership** of the operators pointed to by the `operators_ptr` array.
/// The caller **must not free** the individual operators passed to this function.
/// However, the caller **must still free** the `operators_ptr` array itself.
///
/// # Panics
///
/// This function will panic if `operators_ptr` is null but `operators_len` is greater than 0,
/// or if any pointer inside the array is null.
///
/// # Safety
///
/// The caller must ensure that `operators_ptr` points to a valid array of
/// `operators_len` pointers to `DynamicLocalSearchOperator`.
/// The caller is responsible for freeing the returned pointer using
/// `bollard_ls_free_dynamic_local_search_operator`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_random_compound_operator_new(
    operators_ptr: *const *mut DynamicLocalSearchOperator<
        'static,
        i64,
        DynamicNeighborhoods<'static>,
    >,
    operators_len: usize,
) -> *mut DynamicLocalSearchOperator<'static, i64, DynamicNeighborhoods<'static>> {
    let operators = consume_operator_array(operators_ptr, operators_len);

    if operators.is_empty() {
        let op = RandomCompoundOperator::empty();
        return Box::into_raw(Box::new(DynamicLocalSearchOperator::new(Box::new(op))));
    }

    let rng = rand::rngs::StdRng::from_os_rng();
    let op = RandomCompoundOperator::new(operators, rng);
    Box::into_raw(Box::new(DynamicLocalSearchOperator::new(Box::new(op))))
}

/// Creates a new compound operator that selects sub-operators using a Multi-Armed Bandit (MAB) strategy.
///
/// # Ownership Contract
///
/// This function **takes ownership** of the operators pointed to by the `operators_ptr` array.
/// The caller **must not free** the individual operators passed to this function.
/// However, the caller **must still free** the `operators_ptr` array itself.
///
/// # Panics
///
/// This function will panic if `operators_ptr` is null but `operators_len` is greater than 0,
/// or if any pointer inside the array is null.
///
/// # Safety
///
/// The caller must ensure that `operators_ptr` points to a valid array of
/// `operators_len` pointers to `DynamicLocalSearchOperator`.
/// The caller is responsible for freeing the returned pointer using
/// `bollard_ls_free_dynamic_local_search_operator`.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_new_multi_armed_bandit_compound_operator(
    operators_ptr: *const *mut DynamicLocalSearchOperator<
        'static,
        i64,
        DynamicNeighborhoods<'static>,
    >,
    operators_len: usize,
    memory_coeff: f64,
    exploration_coeff: f64,
) -> *mut DynamicLocalSearchOperator<'static, i64, DynamicNeighborhoods<'static>> {
    let operators = consume_operator_array(operators_ptr, operators_len);

    if operators.is_empty() {
        let op = MultiArmedBanditCompoundOperator::empty(memory_coeff, exploration_coeff);
        return Box::into_raw(Box::new(DynamicLocalSearchOperator::new(Box::new(op))));
    }

    let op = MultiArmedBanditCompoundOperator::new(operators, memory_coeff, exploration_coeff);
    Box::into_raw(Box::new(DynamicLocalSearchOperator::new(Box::new(op))))
}

/// Frees a `DynamicLocalSearchOperator` previously allocated by Bollard.
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to a `DynamicLocalSearchOperator`
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_free_dynamic_local_search_operator(
    ptr: *mut DynamicLocalSearchOperator<'static, i64, DynamicNeighborhoods<'static>>,
) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

// ----------------------------------------------------------------
// Metaheuristic
// ----------------------------------------------------------------

// ----------------------------------------------------------------
// Engine
// ----------------------------------------------------------------

/// Creates a new `LocalSearchFfiEngine` instance and returns a pointer to it.
///
/// # Safety
///
/// The caller is responsible for freeing the allocated memory using
/// `bollard_ls_engine_free` when it is no longer needed.
#[no_mangle]
pub extern "C" fn bollard_ls_engine_new() -> *mut LocalSearchEngine<i64> {
    let engine = LocalSearchEngine::default();
    Box::into_raw(Box::new(engine))
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
) -> *mut LocalSearchEngine<i64> {
    let engine = LocalSearchEngine::preallocated(num_vessels);
    Box::into_raw(Box::new(engine))
}

/// Frees a `LocalSearchFfiEngine` previously allocated by Bollard.
///
/// # Safety
///
/// The caller must ensure that `engine` is a valid pointer to a `LocalSearchFfiEngine`
/// allocated by Bollard.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_engine_free(engine: *mut LocalSearchEngine<i64>) {
    if !engine.is_null() {
        drop(Box::from_raw(engine));
    }
}
