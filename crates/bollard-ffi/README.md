# Bollard FFI

**Foreign Function Interface (FFI) bindings for the Bollard scheduling suite.**

This crate provides a stable, C-compatible ABI (Application Binary Interface) for the Bollard scheduling ecosystem. It acts as a bridge, allowing external environments—such as C, C++, Python, C#, or Java—to leverage Bollard's high-performance solvers and safety guarantees while managing memory manually.

## Architecture

The FFI is designed efficiently around opaque pointers and explicit resource management. The API is modular, separating the **Problem Definition** from the **Solution Strategy**:

### 1. Common Core (Shared)

Modules shared across all solver strategies.

* **Model API**: A Builder-pattern interface to construct, configure, and finalize immutable scheduling problems.
* **Outcome API**: A unified interface to inspect results, read solution schedules, and analyze statistics.

### 2. Solvers

Specific algorithms to solve the Model.

* **Branch-and-Bound (BnB)**: An exact solver providing optimal solutions. It supports various combinations of **Objective Evaluators** and **Search Heuristics**.
* *(Future Extensions)*: The FFI is designed to accommodate additional metaheuristic solvers in the future.

---

## API Reference

Below is the complete list of all exported functions available in the library.

### 1. Model API (`bollard_model`)

These functions manage the creation and inspection of the problem definition.

**Lifecycle**

* `bollard_model_builder_new`
* `bollard_model_builder_free`
* `bollard_model_builder_build` (Consumes builder, returns `Model`)
* `bollard_model_free`

**Builder Configuration**

* `bollard_model_builder_set_arrival_time`
* `bollard_model_builder_set_latest_departure_time`
* `bollard_model_builder_set_vessel_weight`
* `bollard_model_builder_set_processing_time`
* `bollard_model_builder_forbid_vessel_berth_assignment`
* `bollard_model_builder_add_opening_time`
* `bollard_model_builder_add_closing_time`

**Model Inspection**

* `bollard_model_num_vessels`
* `bollard_model_num_berths`
* `bollard_model_get_vessel_weight`
* `bollard_model_get_vessel_arrival_time`
* `bollard_model_get_vessel_latest_departure_time`
* `bollard_model_get_processing_time`
* `bollard_model_get_num_berth_opening_times`
* `bollard_model_get_berth_opening_time`
* `bollard_model_get_num_berth_closing_times`
* `bollard_model_get_berth_closing_time`
* `bollard_model_get_model_log_complexity`

### 2. Branch-and-Bound Solver API (`bollard_bnb_solver`)

These functions instantiate the solver and execute search runs.

**Lifecycle**

* `bollard_bnb_solver_new`
* `bollard_bnb_solver_preallocated`
* `bollard_bnb_solver_free`

**Solve Functions (Hybrid Evaluator)**

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

**Solve Functions (Workload Evaluator)**

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

**Solve Functions (Weighted Flow Time Evaluator)**

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

### 3. Outcome/Result API (`bollard_bnb_outcome`)

These functions handle the results returned by the solver.

**Lifecycle**

* `bollard_bnb_outcome_free`

**Status & Metadata**

* `bollard_bnb_outcome_has_solution`
* `bollard_bnb_outcome_get_status`
* `bollard_bnb_outcome_get_status_str`
* `bollard_bnb_outcome_get_termination_reason_enum`
* `bollard_bnb_outcome_get_termination_reason`

**Solution Data**

* `bollard_bnb_outcome_get_objective`
* `bollard_bnb_outcome_get_num_vessels`
* `bollard_bnb_outcome_get_berth`
* `bollard_bnb_outcome_get_start_time`
* `bollard_bnb_outcome_get_berths` (Direct pointer access)
* `bollard_bnb_outcome_get_start_times` (Direct pointer access)
* `bollard_bnb_outcome_copy_solution` (Batch copy to user buffers)

**Statistics**

* `bollard_bnb_outcome_get_nodes_explored`
* `bollard_bnb_outcome_get_backtracks`
* `bollard_bnb_outcome_get_decisions_generated`
* `bollard_bnb_outcome_get_max_depth`
* `bollard_bnb_outcome_get_prunings_infeasible`
* `bollard_bnb_outcome_get_prunings_bound`
* `bollard_bnb_outcome_get_solutions_found`
* `bollard_bnb_outcome_get_steps`
* `bollard_bnb_outcome_get_time_total_ms`

---

## Usage Lifecycle

1. **Model Construction**:
* Allocate a `ModelBuilder`.
* Define problem parameters.
* Finalize the builder into an immutable `Model`.


2. **Solver Instantiation**:
* Choose your solver strategy (e.g., Branch-and-Bound) and create an instance.


3. **Execution**:
* Call the specific solve function for your chosen solver.
* Pass constraints such as time limits or solution limits.


4. **Inspection**:
* Check the returned `Outcome` for status.
* Extract scalar values or array data.


5. **Cleanup**:
* Explicitly free all allocated resources (`Model`, `Solver`, and `Outcome`) to prevent memory leaks.

## Integration Example

```c
#include "bollard_ffi.h"
#include <stdio.h>

int main() {
    // 1. Create & Configure Model (2 berths, 2 vessels)
    BnbModelBuilder* builder = bollard_model_builder_new(2, 2);
    bollard_model_builder_set_arrival_time(builder, 0, 10);
    bollard_model_builder_set_vessel_weight(builder, 0, 5);
    // ... configure other parameters ...
    
    // Finalize the model (consumes builder)
    BnbModel* model = bollard_model_builder_build(builder);

    // 2. Create Solver
    BnbSolver* solver = bollard_bnb_solver_new();

    // 3. Solve 
    BnbOutcome* outcome = bollard_bnb_solver_solve_with_hybrid_evaluator_and_regret_heuristic_builder(
        solver, model, 0, 1000, false
    );

    // 4. Inspect Result
    if (bollard_bnb_outcome_has_solution(outcome)) {
        int64_t obj = bollard_bnb_outcome_get_objective(outcome);
        printf("Found solution with objective: %ld\n", obj);
    } else {
        printf("No solution found. Status: %s\n", bollard_bnb_outcome_get_status_str(outcome));
    }

    // 5. Cleanup
    bollard_bnb_outcome_free(outcome);
    bollard_bnb_solver_free(solver);
    bollard_model_free(model);

    return 0;
}

```
