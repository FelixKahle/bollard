# Bollard FFI

**Foreign Function Interface for the Bollard Scheduling Ecosystem.**

This library provides stable, C-compatible bindings for the Bollard Branch-and-Bound solver. It allows host applications written in C, C++, Python, C#, Java, or other languages to define scheduling models and solve them using Bollard's high-performance Rust core.

---

## 1. Introduction

### Usage Overview

The API is designed around **Opaque Pointers** (Handles). You cannot access struct fields directly; you must use the exported API functions.

**Safety Warning:** This library uses a "Fail-Fast" approach. Passing `NULL` pointers or invalid array indices will cause the process to **abort** (panic) immediately. Always ensure pointers are valid before passing them to the API.

### Shared Data Types

These basic structures are used across both the Model and BnB APIs.

#### `FfiOpenClosedInterval`
A struct representing a time interval `[start, end)`.

| Field | Type | Description |
| :--- | :--- | :--- |
| `start_inclusive` | `int64_t` | The start time (inclusive). |
| `end_exclusive` | `int64_t` | The end time (exclusive). |

#### `FfiFixedAssignment`
A struct used to force a specific vessel to a specific berth at a specific time during solving.

| Field | Type | Description |
| :--- | :--- | :--- |
| `start_time` | `int64_t` | The assigned time. |
| `berth_index` | `size_t` | The assigned berth index. |
| `vessel_index` | `size_t` | The vessel index. |

---

## 2. Model API

This section details the functions used to define the scheduling problem.

### Builder Lifecycle & Configuration
These functions are used to construct the problem definition.

* **`bollard_model_builder_new(size_t num_berths, size_t num_vessels)`**
    * Allocates a new mutable builder.
    * **Returns:** `*mut ModelBuilder`

* **`bollard_model_builder_free(*mut ModelBuilder ptr)`**
    * Frees the builder memory. **Do not call** if you have already called `_build`.

* **`bollard_model_builder_build(*mut ModelBuilder ptr)`**
    * Consumes the builder and returns an immutable Model.
    * **Invalidates** the passed `ptr`.
    * **Returns:** `*mut Model`

* **`bollard_model_builder_set_arrival_time(*mut ModelBuilder ptr, size_t vessel_index, int64_t time)`**
    * Sets the arrival time for a vessel.

* **`bollard_model_builder_set_latest_departure_time(*mut ModelBuilder ptr, size_t vessel_index, int64_t time)`**
    * Sets the deadline for a vessel.

* **`bollard_model_builder_set_vessel_weight(*mut ModelBuilder ptr, size_t vessel_index, int64_t weight)`**
    * Sets the priority weight for a vessel.

* **`bollard_model_builder_set_processing_time(*mut ModelBuilder ptr, size_t vessel_index, size_t berth_index, int64_t duration)`**
    * Sets how long a vessel takes to process at a specific berth.

* **`bollard_model_builder_forbid_vessel_berth_assignment(*mut ModelBuilder ptr, size_t vessel_index, size_t berth_index)`**
    * Explicitly forbids a specific vessel from using a specific berth.

* **`bollard_model_builder_add_opening_time(*mut ModelBuilder ptr, size_t berth_index, FfiOpenClosedInterval interval)`**
    * Adds an availability window to a berth.

* **`bollard_model_builder_add_closing_time(*mut ModelBuilder ptr, size_t berth_index, FfiOpenClosedInterval interval)`**
    * Adds a maintenance/unavailable window to a berth.

### Model Inspection & Lifecycle
These functions are used to query the immutable problem definition after it has been built.

* **`bollard_model_free(*mut Model ptr)`**
    * Frees the immutable Model.

* **`bollard_model_num_vessels(*const Model ptr)`**
    * **Returns:** Total number of vessels.

* **`bollard_model_num_berths(*const Model ptr)`**
    * **Returns:** Total number of berths.

* **`bollard_model_vessel_weight(*const Model ptr, size_t vessel_index)`**
    * **Returns:** Weight of the vessel.

* **`bollard_model_vessel_arrival_time(*const Model ptr, size_t vessel_index)`**
    * **Returns:** Arrival time.

* **`bollard_model_vessel_latest_departure_time(*const Model ptr, size_t vessel_index)`**
    * **Returns:** Deadline.

* **`bollard_model_processing_time(*const Model ptr, size_t vessel_index, size_t berth_index)`**
    * **Returns:** Processing time, or `-1` if the assignment is invalid/forbidden.

* **`bollard_model_num_berth_opening_times(*const Model ptr, size_t berth_index)`**
    * **Returns:** Count of opening intervals.

* **`bollard_model_num_berth_closing_times(*const Model ptr, size_t berth_index)`**
    * **Returns:** Count of closing intervals.

* **`bollard_model_berth_opening_time(*const Model ptr, size_t berth_index, size_t interval_index)`**
    * **Returns:** `FfiOpenClosedInterval` struct.

* **`bollard_model_berth_closing_time(*const Model ptr, size_t berth_index, size_t interval_index)`**
    * **Returns:** `FfiOpenClosedInterval` struct.

* **`bollard_model_model_log_complexity(*const Model ptr)`**
    * **Returns:** `double` representing the estimated log-complexity of the problem.

---

## 3. BnB API

This section details the functions for the Exact Branch-and-Bound solver.

### Solver Lifecycle

* **`bollard_bnb_solver_new()`**
    * Creates a new solver with default capacity.
    * **Returns:** `*mut BnbSolver`

* **`bollard_bnb_solver_preallocated(size_t num_berths, size_t num_vessels)`**
    * Creates a solver with internal structures pre-allocated for the given size.
    * **Returns:** `*mut BnbSolver`

* **`bollard_bnb_solver_free(*mut BnbSolver ptr)`**
    * Frees the solver instance.

### Execution Strategies
All solve functions return `*mut BnbSolverFfiOutcome` and take the following common parameters:
1.  `*mut BnbSolver` (The solver instance)
2.  `*const Model` (The immutable model)
3.  `size_t solution_limit` (Stop after finding N solutions; 0 for infinite)
4.  `int64_t time_limit_ms` (Stop after N milliseconds; 0 for infinite)
5.  `bool enable_log` (Enable stdout logging)

**Note:** Functions ending in `_with_fixed` accept two additional arguments:
6.  `*const FfiFixedAssignment fixed_ptr` (Pointer to array of fixed assignments)
7.  `size_t fixed_len` (Length of the array)

#### Hybrid Evaluator Strategies
* `bollard_bnb_solver_solve_with_hybrid_evaluator_and_chronological_exhaustive_builder`
* `bollard_bnb_solver_solve_with_hybrid_evaluator_and_chronological_exhaustive_builder_with_fixed`
* `bollard_bnb_solver_solve_with_hybrid_evaluator_and_fcfs_heuristic_builder`
* `bollard_bnb_solver_solve_with_hybrid_evaluator_and_fcfs_heuristic_builder_with_fixed`
* `bollard_bnb_solver_solve_with_hybrid_evaluator_and_regret_heuristic_builder`
* `bollard_bnb_solver_solve_with_hybrid_evaluator_and_regret_heuristic_builder_with_fixed`
* `bollard_bnb_solver_solve_with_hybrid_evaluator_and_slack_heuristic_builder`
* `bollard_bnb_solver_solve_with_hybrid_evaluator_and_slack_heuristic_builder_with_fixed`
* `bollard_bnb_solver_solve_with_hybrid_evaluator_and_wspt_heuristic_builder`
* `bollard_bnb_solver_solve_with_hybrid_evaluator_and_wspt_heuristic_builder_with_fixed`

#### Workload Evaluator Strategies
* `bollard_bnb_solver_solve_with_workload_evaluator_and_chronological_exhaustive_builder`
* `bollard_bnb_solver_solve_with_workload_evaluator_and_chronological_exhaustive_builder_with_fixed`
* `bollard_bnb_solver_solve_with_workload_evaluator_and_fcfs_heuristic_builder`
* `bollard_bnb_solver_solve_with_workload_evaluator_and_fcfs_heuristic_builder_with_fixed`
* `bollard_bnb_solver_solve_with_workload_evaluator_and_regret_heuristic_builder`
* `bollard_bnb_solver_solve_with_workload_evaluator_and_regret_heuristic_builder_with_fixed`
* `bollard_bnb_solver_solve_with_workload_evaluator_and_slack_heuristic_builder`
* `bollard_bnb_solver_solve_with_workload_evaluator_and_slack_heuristic_builder_with_fixed`
* `bollard_bnb_solver_solve_with_workload_evaluator_and_wspt_heuristic_builder`
* `bollard_bnb_solver_solve_with_workload_evaluator_and_wspt_heuristic_builder_with_fixed`

#### Weighted Flow Time (WTFT) Evaluator Strategies
* `bollard_bnb_solver_solve_with_wtft_evaluator_and_chronological_exhaustive_builder`
* `bollard_bnb_solver_solve_with_wtft_evaluator_and_chronological_exhaustive_builder_with_fixed`
* `bollard_bnb_solver_solve_with_wtft_evaluator_and_fcfs_heuristic_builder`
* `bollard_bnb_solver_solve_with_wtft_evaluator_and_fcfs_heuristic_builder_with_fixed`
* `bollard_bnb_solver_solve_with_wtft_evaluator_and_regret_heuristic_builder`
* `bollard_bnb_solver_solve_with_wtft_evaluator_and_regret_heuristic_builder_with_fixed`
* `bollard_bnb_solver_solve_with_wtft_evaluator_and_slack_heuristic_builder`
* `bollard_bnb_solver_solve_with_wtft_evaluator_and_slack_heuristic_builder_with_fixed`
* `bollard_bnb_solver_solve_with_wtft_evaluator_and_wspt_heuristic_builder`
* `bollard_bnb_solver_solve_with_wtft_evaluator_and_wspt_heuristic_builder_with_fixed`

### Outcome API
These functions are used to inspect the result of a solve operation.

#### Lifecycle
* **`bollard_bnb_outcome_free(*mut BnbSolverFfiOutcome ptr)`**
    * Frees the outcome and all contained strings/arrays.

#### Status & Metadata
* **`bollard_bnb_outcome_has_solution(*const BnbSolverFfiOutcome ptr)`**
    * **Returns:** `bool` (true if Optimal or Feasible).

* **`bollard_bnb_outcome_status(*const BnbSolverFfiOutcome ptr)`**
    * **Returns:** Enum integer (0=Optimal, 1=Feasible, 2=Infeasible, 3=Unknown).

* **`bollard_bnb_outcome_status_str(*const BnbSolverFfiOutcome ptr)`**
    * **Returns:** `const char*` (e.g., "Optimal"). Valid until outcome is freed.

* **`bollard_bnb_outcome_termination_reason(*const BnbSolverFfiOutcome ptr)`**
    * **Returns:** `const char*` description of why the solver stopped.

* **`bollard_bnb_outcome_termination_reason_enum(*const BnbSolverFfiOutcome ptr)`**
    * **Returns:** Enum integer (0=OptimalityProven, 1=InfeasibilityProven, 2=Aborted).

#### Solution Data
* **`bollard_bnb_outcome_objective(*const BnbSolverFfiOutcome ptr)`**
    * **Returns:** `int64_t` objective value. Panics if no solution.

* **`bollard_bnb_outcome_num_vessels(*const BnbSolverFfiOutcome ptr)`**
    * **Returns:** Number of vessels in the solution.

* **`bollard_bnb_outcome_berth(*const BnbSolverFfiOutcome ptr, size_t vessel_idx)`**
    * **Returns:** Assigned berth index for specific vessel.

* **`bollard_bnb_outcome_start_time(*const BnbSolverFfiOutcome ptr, size_t vessel_idx)`**
    * **Returns:** Assigned start time for specific vessel.

* **`bollard_bnb_outcome_berths(*const BnbSolverFfiOutcome ptr)`**
    * **Returns:** `const size_t*` pointer to the raw array of berth assignments.

* **`bollard_bnb_outcome_start_times(*const BnbSolverFfiOutcome ptr)`**
    * **Returns:** `const int64_t*` pointer to the raw array of start times.

* **`bollard_bnb_outcome_copy_solution(*const BnbSolverFfiOutcome ptr, size_t* out_berths, int64_t* out_start_times)`**
    * Copies solution data into user-provided buffers. Buffers must be large enough (`num_vessels`).

#### Search Statistics
All return `uint64_t`.

* `bollard_bnb_outcome_nodes_explored`
* `bollard_bnb_outcome_backtracks`
* `bollard_bnb_outcome_decisions_generated`
* `bollard_bnb_outcome_max_depth`
* `bollard_bnb_outcome_prunings_infeasible`
* `bollard_bnb_outcome_prunings_bound`
* `bollard_bnb_outcome_solutions_found`
* `bollard_bnb_outcome_steps`
* `bollard_bnb_outcome_time_total_ms`
