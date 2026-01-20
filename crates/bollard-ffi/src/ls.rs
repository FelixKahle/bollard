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
    decoder::GreedyDecoder,
    engine::LocalSearchEngine,
    meta::{
        dynamic::DynamicMetaheuristic,
        greedy_descent::GreedyDescent,
        guided_local_search::GuidedLocalSearch,
        simulated_annealing::{GeometricCooling, LinearCooling, SimulatedAnnealing},
        tabu_search::TabuSearch,
    },
    monitor::{
        composite::CompositeLocalSearchMonitor, logging::LogLocalSearchMonitor,
        solution::SolutionLimitMonitor, time::TimeLimitMonitor,
    },
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
    params::LocalSearchParams,
    result::LocalSearchEngineOutcome,
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
// Metaheuristics
// ----------------------------------------------------------------

/// Creates a new greedy descent metaheuristic instance.
///
/// The returned pointer must eventually be freed on the Rust side.
#[no_mangle]
pub extern "C" fn bollard_ls_greedy_descent_metaheuristic_new(
) -> *mut DynamicMetaheuristic<'static, i64> {
    let gd = GreedyDescent::new();
    Box::into_raw(Box::new(DynamicMetaheuristic::from_metaheuristic(gd)))
}

/// Creates a new simulated annealing metaheuristic with linear cooling.
///
/// Uses a `StdRng` from the operating system as the random number generator.
#[no_mangle]
pub extern "C" fn bollard_ls_simulated_annealing_metaheuristic_with_linear_cooling_new(
    initial: f64,
    decrement: f64,
    min_temp: f64,
) -> *mut DynamicMetaheuristic<'static, i64> {
    let rng = rand::rngs::StdRng::from_os_rng();
    let cooling = LinearCooling::new(initial, decrement, min_temp);
    let sa = SimulatedAnnealing::new(cooling, rng);
    Box::into_raw(Box::new(DynamicMetaheuristic::from_metaheuristic(sa)))
}

/// Creates a new simulated annealing metaheuristic with geometric cooling.
///
/// Uses a `StdRng` from the operating system as the random number generator.
///
/// # Panics
///
/// Panics if `alpha <= 0.0` or `alpha >= 1.0`.
#[no_mangle]
pub extern "C" fn bollard_ls_simulated_annealing_metaheuristic_with_geometric_cooling_new(
    initial: f64,
    alpha: f64,
    min_temp: f64,
) -> *mut DynamicMetaheuristic<'static, i64> {
    assert!(
        alpha > 0.0 && alpha < 1.0,
        "called `bollard_ls_simulated_annealing_metaheuristic_with_geometric_cooling_new` with invalid alpha: {}. Must be in (0.0, 1.0)",
        alpha
    );

    let rng = rand::rngs::StdRng::from_os_rng();
    let cooling = GeometricCooling::new(initial, alpha, min_temp);
    let sa = SimulatedAnnealing::new(cooling, rng);
    Box::into_raw(Box::new(DynamicMetaheuristic::from_metaheuristic(sa)))
}

/// Creates a new simulated annealing metaheuristic using defaults derived
/// from an initial solution.
///
/// Uses a `StdRng` from the operating system as the random number generator.
///
/// # Safety
///
/// - `initial_solution` must be a valid, non-null pointer to a `Solution<i64>`.
/// - The solution must remain valid for the duration of the call.
/// - The pointed-to solution must have been created by this library and not
///   already freed.
///
/// # Panics
///
/// Panics if `initial_solution` is null.
#[no_mangle]
pub unsafe extern "C" fn bollard_simulated_annealing_metaheuristic_with_geometric_cooling_from_solution_new(
    initial_solution: *const Solution<i64>,
) -> *mut DynamicMetaheuristic<'static, i64> {
    assert!(
        !initial_solution.is_null(),
        "called `bollard_simulated_annealing_metaheuristic_with_geometric_cooling_from_solution_new` with `initial_solution` as null pointer"
    );

    let solution = &*initial_solution;

    let rng = rand::rngs::StdRng::from_os_rng();
    let sa = SimulatedAnnealing::with_defaults(solution, rng);
    Box::into_raw(Box::new(DynamicMetaheuristic::from_metaheuristic(sa)))
}

/// Frees a dynamic metaheuristic previously created by this module.
///
/// # Safety
///
/// - `metaheuristic` must either be null or a pointer previously returned from
///   one of the `bollard_ls_*_metaheuristic_*_new` functions.
/// - `metaheuristic` must not be used after this function returns.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_dynamic_metaheuristic_free(
    metaheuristic: *mut DynamicMetaheuristic<'static, i64>,
) {
    if !metaheuristic.is_null() {
        drop(Box::from_raw(metaheuristic));
    }
}

/// Creates a guided local search metaheuristic using defaults derived from
/// the given model and initial solution.
///
/// # Safety
///
/// - `model` must be a valid, non-null pointer to a `Model<i64>`.
/// - `initial_solution` must be a valid, non-null pointer to a `Solution<i64>`.
/// - Both pointers must remain valid for the duration of the call.
/// - The pointed-to objects must have been created by this library and not
///   already freed.
///
/// # Panics
///
/// Panics if `model` or `initial_solution` is null.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_guided_local_search_metaheuristic_with_defaults_from_model_and_solution_new(
    model: *const bollard_model::model::Model<i64>,
    initial_solution: *const Solution<i64>,
) -> *mut DynamicMetaheuristic<'static, i64> {
    assert!(
        !model.is_null(),
        "called `..._with_defaults_from_model_and_solution_new` with `model` as null pointer"
    );
    assert!(
        !initial_solution.is_null(),
        "called `..._with_defaults_from_model_and_solution_new` with `initial_solution` as null pointer"
    );

    let model_ref: &bollard_model::model::Model<i64> = &*model;
    let solution_ref: &Solution<i64> = &*initial_solution;

    let gls: GuidedLocalSearch<i64> = GuidedLocalSearch::with_defaults(model_ref, solution_ref);
    Box::into_raw(Box::new(DynamicMetaheuristic::from_metaheuristic(gls)))
}

/// Creates a tabu search metaheuristic using default parameters derived
/// from the given model.
///
/// # Safety
///
/// - `model` must be a valid, non-null pointer to a `Model<i64>`.
/// - The pointer must remain valid for the duration of the call.
/// - The pointed-to model must have been created by this library and not
///   already freed.
///
/// # Panics
///
/// Panics if `model` is null.
#[no_mangle]
pub unsafe extern "C" fn bollard_ls_tabu_search_metaheuristic_with_defaults_from_model_new(
    model: *const bollard_model::model::Model<i64>,
) -> *mut DynamicMetaheuristic<'static, i64> {
    assert!(
        !model.is_null(),
        "called `bollard_ls_tabu_search_metaheuristic_with_defaults_from_model_new` with `model` as null pointer"
    );

    let model_ref: &bollard_model::model::Model<i64> = &*model;
    let ts = TabuSearch::with_defaults(model_ref);
    Box::into_raw(Box::new(DynamicMetaheuristic::from_metaheuristic(ts)))
}

/// Creates a guided local search metaheuristic with the given penalty
/// parameter.
#[no_mangle]
pub extern "C" fn bollard_ls_guided_local_search_metaheuristic_new(
    lambda: f64,
) -> *mut DynamicMetaheuristic<'static, i64> {
    // assuming GuidedLocalSearch::new<T: SolverNumeric>(lambda: f64) -> Self
    let gls: GuidedLocalSearch<i64> = GuidedLocalSearch::new(lambda);
    Box::into_raw(Box::new(DynamicMetaheuristic::from_metaheuristic(gls)))
}

/// Creates a tabu search metaheuristic with the given tenure.
#[no_mangle]
pub extern "C" fn bollard_ls_tabu_search_metaheuristic_new(
    tenure: usize,
) -> *mut DynamicMetaheuristic<'static, i64> {
    let ts: TabuSearch<i64> = TabuSearch::new(tenure);
    Box::into_raw(Box::new(DynamicMetaheuristic::from_metaheuristic(ts)))
}

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

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn run_engine(
    engine: &mut LocalSearchEngine<i64>,
    model: &Model<i64>,
    initial: &Solution<i64>,
    neighborhood: &DynamicNeighborhoods<'static>,
    metaheuristic: &mut DynamicMetaheuristic<'static, i64>,
    operator: &mut DynamicLocalSearchOperator<'static, i64, DynamicNeighborhoods<'static>>,
    time_limit_ms: u64,
    solution_limit: u64,
    log: bool,
) -> LocalSearchEngineOutcome<i64> {
    let mut monitor = CompositeLocalSearchMonitor::with_capacity(3);

    if time_limit_ms > 0 {
        monitor.add_monitor(TimeLimitMonitor::new(std::time::Duration::from_millis(
            time_limit_ms,
        )));
    }

    if solution_limit > 0 {
        monitor.add_monitor(SolutionLimitMonitor::new(solution_limit));
    }

    if log {
        monitor.add_monitor(LogLocalSearchMonitor::new(std::time::Duration::from_secs(
            1,
        )));
    }

    let mut decoder = GreedyDecoder::preallocated(model.num_berths());
    let params = LocalSearchParams::builder(
        model,
        &mut decoder,
        neighborhood,
        operator,
        metaheuristic,
        monitor,
        initial,
    )
    .build()
    .unwrap();

    engine.run(params)
}

/// Runs the local search engine with the given components and limits.
///
/// # Panics
///
/// Panics if any of the pointer arguments is null.
///
/// # Safety
///
/// - `engine` must be a valid, non-null pointer to a `LocalSearchEngine<i64>`.
/// - `model` must be a valid, non-null pointer to a `Model<i64>`.
/// - `initial` must be a valid, non-null pointer to a `Solution<i64>`.
/// - `neighborhood` must be a valid, non-null pointer to a `DynamicNeighborhoods<'static>`.
/// - `metaheuristic` must be a valid, non-null pointer to a
///   `DynamicMetaheuristic<'static, i64>`.
/// - `operator` must be a valid, non-null pointer to a
///   `DynamicLocalSearchOperator<'static, i64, DynamicNeighborhoods<'static>>`.
/// - All pointed-to values must have been created by this library, must outlive
///   this call, and must not be aliased mutably elsewhere while this function
///   is executing.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe fn bollard_ls_engine_run(
    engine: *mut LocalSearchEngine<i64>,
    model: *const Model<i64>,
    initial: *const Solution<i64>,
    neighborhood: *const DynamicNeighborhoods<'static>,
    metaheuristic: *mut DynamicMetaheuristic<'static, i64>,
    operator: *mut DynamicLocalSearchOperator<'static, i64, DynamicNeighborhoods<'static>>,
    time_limit_ms: u64,
    solution_limit: u64,
    log: bool,
) -> *mut LocalSearchFfiOutcome {
    assert!(
        !engine.is_null(),
        "called `bollard_ls_engine_run` with `engine` as null pointer"
    );
    assert!(
        !model.is_null(),
        "called `bollard_ls_engine_run` with `model` as null pointer"
    );
    assert!(
        !initial.is_null(),
        "called `bollard_ls_engine_run` with `initial` as null pointer"
    );
    assert!(
        !neighborhood.is_null(),
        "called `bollard_ls_engine_run` with `neighborhood` as null pointer"
    );
    assert!(
        !metaheuristic.is_null(),
        "called `bollard_ls_engine_run` with `metaheuristic` as null pointer"
    );
    assert!(
        !operator.is_null(),
        "called `bollard_ls_engine_run` with `operator` as null pointer"
    );

    let engine = &mut *engine;
    let model = &*model;
    let initial = &*initial;
    let neighborhood = &*neighborhood;
    let metaheuristic = &mut *metaheuristic;
    let operator = &mut *operator;

    let outcome = run_engine(
        engine,
        model,
        initial,
        neighborhood,
        metaheuristic,
        operator,
        time_limit_ms,
        solution_limit,
        log,
    );

    Box::into_raw(Box::new(outcome.into()))
}
