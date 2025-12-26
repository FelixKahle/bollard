# Bollard FFI

**Foreign Function Interface (FFI) bindings for the Bollard scheduling ecosystem.**

This crate provides a stable, C-compatible ABI (Application Binary Interface) for the Bollard project. It acts as a high-performance bridge, allowing external environments—such as C, C++, Python, C#, Julia, or Java—to leverage Bollard's scheduling solvers while maintaining strict safety guarantees.

## 🏗 Architecture

The FFI is designed around **Opaque Pointers** (the Handle pattern) and **Explicit Resource Management**. The API is modular, separating the definition of the scheduling problem from the specific algorithms used to solve it.

The library currently exports two primary modules:

1. **Model Module**: A generic, solver-agnostic interface to define scheduling problems.
2. **Branch-and-Bound (BnB) Module**: An exact solver implementation with configurable search strategies.

---

## 🔒 Safety & Memory Management

This library uses `unsafe` Rust to interact with raw pointers. Consumers must adhere to the following strict contract to prevent Undefined Behavior (UB):

1. **Opaque Pointers**: Never dereference the pointers (`BnbModel*`, `BnbSolver*`) directly in the host language. Use the provided accessor functions.
2. **Ownership Handover**:
* Functions like `bollard_model_builder_build` **consume** the input pointer. The builder is invalid immediately after this call.
* Pointers passed to `_free` functions are invalid immediately.


3. **Panic Strategy**: This crate is designed to **abort** (crash the process) on misuse (e.g., passing `NULL` where a valid pointer is required) rather than unwinding. This prevents memory corruption from crossing the FFI boundary.

---

## 📦 Module: Model

**Prefix:** `bollard_model_`

This module is responsible for constructing the immutable problem definition. It uses the **Builder Pattern**: you allocate a mutable `ModelBuilder`, configure it, and then finalize it into an immutable `Model` that can be passed to any solver.

### Data Structures

Users must define a compatible struct for time intervals in their host language:

```c
typedef struct {
    int64_t start_inclusive;
    int64_t end_exclusive;
} FfiOpenClosedInterval;

```

### 📚 API Reference

#### Lifecycle

* `bollard_model_builder_new(size_t berths, size_t vessels)`: Allocates a new builder.
* `bollard_model_builder_free(ModelBuilder* ptr)`: Frees a builder (only needed if you do *not* call build).
* `bollard_model_builder_build(ModelBuilder* ptr)`: Consumes the builder and returns an immutable `Model*`.
* `bollard_model_free(Model* ptr)`: Frees the immutable model.

#### Builder Configuration

* `bollard_model_builder_set_arrival_time(ModelBuilder* ptr, size_t v_idx, int64_t time)`
* `bollard_model_builder_set_latest_departure_time(ModelBuilder* ptr, size_t v_idx, int64_t time)`
* `bollard_model_builder_set_vessel_weight(ModelBuilder* ptr, size_t v_idx, int64_t weight)`
* `bollard_model_builder_set_processing_time(ModelBuilder* ptr, size_t v_idx, size_t b_idx, int64_t time)`
* `bollard_model_builder_forbid_vessel_berth_assignment(ModelBuilder* ptr, size_t v_idx, size_t b_idx)`
* `bollard_model_builder_add_opening_time(ModelBuilder* ptr, size_t b_idx, FfiOpenClosedInterval interval)`
* `bollard_model_builder_add_closing_time(ModelBuilder* ptr, size_t b_idx, FfiOpenClosedInterval interval)`

#### Model Inspection

* `bollard_model_num_vessels(const Model* ptr) -> size_t`
* `bollard_model_num_berths(const Model* ptr) -> size_t`
* `bollard_model_get_vessel_weight(const Model* ptr, size_t v_idx) -> int64_t`
* `bollard_model_get_vessel_arrival_time(const Model* ptr, size_t v_idx) -> int64_t`
* `bollard_model_get_vessel_latest_departure_time(const Model* ptr, size_t v_idx) -> int64_t`
* `bollard_model_get_processing_time(const Model* ptr, size_t v_idx, size_t b_idx) -> int64_t`
* `bollard_model_get_num_berth_opening_times(const Model* ptr, size_t b_idx) -> size_t`
* `bollard_model_get_berth_opening_time(const Model* ptr, size_t b_idx, size_t interval_idx) -> FfiOpenClosedInterval`
* `bollard_model_get_num_berth_closing_times(const Model* ptr, size_t b_idx) -> size_t`
* `bollard_model_get_berth_closing_time(const Model* ptr, size_t b_idx, size_t interval_idx) -> FfiOpenClosedInterval`
* `bollard_model_get_model_log_complexity(const Model* ptr) -> double`

---

## 🚀 Module: Branch-and-Bound (BnB)

**Prefix:** `bollard_bnb_`

This module provides an exact solver capable of finding optimal schedules. It supports a combinatorial set of strategies, allowing you to mix and match **Objective Evaluators** with **Search Heuristics**.

### 1. Solver API

Manages the search process.

#### Lifecycle

* `bollard_bnb_solver_new() -> BnbSolver*`: Creates a solver instance.
* `bollard_bnb_solver_preallocated(size_t berths, size_t vessels) -> BnbSolver*`: Creates a solver with pre-sized internal buffers.
* `bollard_bnb_solver_free(BnbSolver* ptr)`: Frees the solver.

#### Execution (Solve Functions)

These functions execute the search. All return a `BnbOutcome*`.
*Arguments*: `(BnbSolver* solver, const Model* model, size_t sol_limit, int64_t time_ms, bool log)`

**Hybrid Evaluator**

* `bollard_bnb_solver_solve_with_hybrid_evaluator_and_chronological_exhaustive_builder`
* `bollard_bnb_solver_solve_with_hybrid_evaluator_and_fcfs_heuristic_builder`
* `bollard_bnb_solver_solve_with_hybrid_evaluator_and_regret_heuristic_builder`
* `bollard_bnb_solver_solve_with_hybrid_evaluator_and_slack_heuristic_builder`
* `bollard_bnb_solver_solve_with_hybrid_evaluator_and_wspt_heuristic_builder`

**Workload Evaluator**

* `bollard_bnb_solver_solve_with_workload_evaluator_and_chronological_exhaustive_builder`
* `bollard_bnb_solver_solve_with_workload_evaluator_and_fcfs_heuristic_builder`
* `bollard_bnb_solver_solve_with_workload_evaluator_and_regret_heuristic_builder`
* `bollard_bnb_solver_solve_with_workload_evaluator_and_slack_heuristic_builder`
* `bollard_bnb_solver_solve_with_workload_evaluator_and_wspt_heuristic_builder`

**Weighted Flow Time (WTFT) Evaluator**

* `bollard_bnb_solver_solve_with_wtft_evaluator_and_chronological_exhaustive_builder`
* `bollard_bnb_solver_solve_with_wtft_evaluator_and_fcfs_heuristic_builder`
* `bollard_bnb_solver_solve_with_wtft_evaluator_and_regret_heuristic_builder`
* `bollard_bnb_solver_solve_with_wtft_evaluator_and_slack_heuristic_builder`
* `bollard_bnb_solver_solve_with_wtft_evaluator_and_wspt_heuristic_builder`

*Note: Each solve function also has a `_with_fixed` variant that accepts an array of fixed assignments.*

### 2. Outcome API

Manages the results returned by the solver.

#### Lifecycle

* `bollard_bnb_outcome_free(BnbOutcome* ptr)`

#### Status & Metadata

* `bollard_bnb_outcome_has_solution(const BnbOutcome* ptr) -> bool`
* `bollard_bnb_outcome_get_status(const BnbOutcome* ptr) -> int` (Enum)
* `bollard_bnb_outcome_get_status_str(const BnbOutcome* ptr) -> const char*`
* `bollard_bnb_outcome_get_termination_reason(const BnbOutcome* ptr) -> const char*`

#### Solution Data

* `bollard_bnb_outcome_get_objective(const BnbOutcome* ptr) -> int64_t`
* `bollard_bnb_outcome_get_num_vessels(const BnbOutcome* ptr) -> size_t`
* `bollard_bnb_outcome_get_berth(const BnbOutcome* ptr, size_t v_idx) -> size_t`
* `bollard_bnb_outcome_get_start_time(const BnbOutcome* ptr, size_t v_idx) -> int64_t`
* `bollard_bnb_outcome_copy_solution(const BnbOutcome* ptr, size_t* out_berths, int64_t* out_starts)`: Bulk copy for performance.

#### Statistics

* `bollard_bnb_outcome_get_nodes_explored(const BnbOutcome* ptr) -> uint64_t`
* `bollard_bnb_outcome_get_backtracks(const BnbOutcome* ptr) -> uint64_t`
* `bollard_bnb_outcome_get_max_depth(const BnbOutcome* ptr) -> uint64_t`
* `bollard_bnb_outcome_get_time_total_ms(const BnbOutcome* ptr) -> uint64_t`
