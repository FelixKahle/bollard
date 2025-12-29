# Bollard FFI

**Foreign Function Interface for the Bollard Scheduling Ecosystem.**

`bollard_ffi` provides stable, C-compatible bindings for the Bollard Branch-and-Bound solver. It is designed as a definitive guide for developers writing wrapper libraries (e.g., Python `ctypes`, C# P/Invoke, Java JNI) to interface with Bollard's high-performance Rust core.

## Key Features

* **Strict C ABI**: All exposed symbols follow standard C calling conventions.
* **Opaque Handles**: Uses opaque pointers to manage state, ensuring internal Rust memory layouts do not break ABI stability.
* **Fail-Fast Safety**: The library is designed to abort (panic) immediately upon receiving `NULL` pointers or out-of-bounds indices, preventing undefined behavior.
* **Zero-Copy Access**: exposes raw pointers to internal solution buffers, allowing host languages to read results without allocation overhead.

## Architecture & Data Layout

### 1. Primitives & Constants

* **Integer Types**:
* `int64_t`: Standard 64-bit signed integer (Rust `i64`).
* `uint64_t`: Standard 64-bit unsigned integer (Rust `u64`).
* `size_t`: Platform-dependent unsigned size type (Rust `usize`).


* **Booleans**:
* `bool`: Standard C boolean (1 byte). `true` = 1, `false` = 0.



### 2. Shared Data Structures (Exact Layout)

These structures are **`#[repr(C)]`**. Wrapper libraries **must** define identical structures with the exact field ordering shown below to ensure memory compatibility.

#### `FfiOpenClosedInterval`

Represents a time interval `[start, end)`.

```c
typedef struct {
    int64_t start_inclusive;
    int64_t end_exclusive;
} FfiOpenClosedInterval;

```

#### `FfiFixedAssignment`

Used to force a specific vessel to a specific berth at a specific time (Warm Start).

```c
typedef struct {
    int64_t start_time;    // The assigned time
    size_t berth_index;    // The assigned berth index
    size_t vessel_index;   // The vessel index
} FfiFixedAssignment;

```

### 3. Opaque Handles

These types are exposed only as pointers. You must never dereference them directly in the host language.

* `typedef struct ModelBuilder ModelBuilder;`
* `typedef struct Model Model;`
* `typedef struct BnbSolver BnbSolver;`
* `typedef struct BnbSolverFfiOutcome BnbSolverFfiOutcome;`

## API Reference

### Model Management (`bollard_model`)

**Lifecycle**

```c
// Allocate a new mutable builder
ModelBuilder* bollard_model_builder_new(size_t num_berths, size_t num_vessels);

// Free the builder (ONLY if _build() is never called)
void bollard_model_builder_free(ModelBuilder* ptr);

// Consumes the builder to produce an immutable Model
Model* bollard_model_builder_build(ModelBuilder* ptr);

// Free the immutable model
void bollard_model_free(Model* ptr);

```

**Configuration (Builder)**

```c
void bollard_model_builder_set_arrival_time(ModelBuilder* ptr, size_t vessel_idx, int64_t time);
void bollard_model_builder_set_latest_departure_time(ModelBuilder* ptr, size_t vessel_idx, int64_t time);
void bollard_model_builder_set_vessel_weight(ModelBuilder* ptr, size_t vessel_idx, int64_t weight);
void bollard_model_builder_set_processing_time(ModelBuilder* ptr, size_t vessel_idx, size_t berth_idx, int64_t duration);
void bollard_model_builder_forbid_vessel_berth_assignment(ModelBuilder* ptr, size_t vessel_idx, size_t berth_idx);

// Pass structs by value
void bollard_model_builder_add_opening_time(ModelBuilder* ptr, size_t berth_idx, FfiOpenClosedInterval interval);
void bollard_model_builder_add_closing_time(ModelBuilder* ptr, size_t berth_idx, FfiOpenClosedInterval interval);

```

**Inspection (Model)**

```c
size_t bollard_model_num_vessels(const Model* ptr);
size_t bollard_model_num_berths(const Model* ptr);
int64_t bollard_model_vessel_weight(const Model* ptr, size_t vessel_idx);
int64_t bollard_model_vessel_arrival_time(const Model* ptr, size_t vessel_idx);
int64_t bollard_model_vessel_latest_departure_time(const Model* ptr, size_t vessel_idx);
int64_t bollard_model_processing_time(const Model* ptr, size_t vessel_idx, size_t berth_idx);
// Returns -1 if assignment is forbidden

size_t bollard_model_num_berth_opening_times(const Model* ptr, size_t berth_idx);
size_t bollard_model_num_berth_closing_times(const Model* ptr, size_t berth_idx);

// Returns struct by value
FfiOpenClosedInterval bollard_model_berth_opening_time(const Model* ptr, size_t berth_idx, size_t interval_idx);
FfiOpenClosedInterval bollard_model_berth_closing_time(const Model* ptr, size_t berth_idx, size_t interval_idx);

double bollard_model_model_log_complexity(const Model* ptr);

```

---

### Solver Execution (`bollard_bnb`)

**Lifecycle**

```c
BnbSolver* bollard_bnb_solver_new();
BnbSolver* bollard_bnb_solver_preallocated(size_t num_berths, size_t num_vessels);
void bollard_bnb_solver_free(BnbSolver* ptr);

```

**Execution Strategies**
The API exports specific functions for every combination of **Evaluator** + **Decision Builder**.

* **Evaluators**: `hybrid`, `workload`, `wtft` (Weighted Flow Time).
* **Builders**: `chronological_exhaustive`, `fcfs_heuristic`, `regret_heuristic`, `slack_heuristic`, `wspt_heuristic`, `spt_heuristic` and `lpt_heuristic`.

*Standard Signature:*

```c
BnbSolverFfiOutcome* bollard_bnb_solver_solve_with_<EVAL>_evaluator_and_<BUILDER>_builder(
    BnbSolver* solver,
    const Model* model,
    size_t solution_limit,    // -1 = Find All
    int64_t time_limit_ms,    // -1 = Infinite
    bool enable_log
);

```

*Fixed Assignment Signature (Warm Start):*

```c
BnbSolverFfiOutcome* bollard_bnb_solver_solve_with_<EVAL>_evaluator_and_<BUILDER>_builder_with_fixed(
    BnbSolver* solver,
    const Model* model,
    size_t solution_limit,
    int64_t time_limit_ms,
    bool enable_log,
    const FfiFixedAssignment* fixed_ptr, // Pointer to array
    size_t fixed_len                     // Array length
);

```

---

### Outcome Inspection

**Lifecycle**

```c
void bollard_bnb_outcome_free(BnbSolverFfiOutcome* ptr);

```

**Status & Metadata**

```c
bool bollard_bnb_outcome_has_solution(const BnbSolverFfiOutcome* ptr);

// Status Enum: 0=Optimal, 1=Feasible, 2=Infeasible, 3=Unknown
int bollard_bnb_outcome_status(const BnbSolverFfiOutcome* ptr);
const char* bollard_bnb_outcome_status_str(const BnbSolverFfiOutcome* ptr);

// Reason Enum: 0=OptimalityProven, 1=InfeasibilityProven, 2=Aborted
int bollard_bnb_outcome_termination_reason_enum(const BnbSolverFfiOutcome* ptr);
const char* bollard_bnb_outcome_termination_reason(const BnbSolverFfiOutcome* ptr);

```

**Solution Data (Zero-Copy)**

```c
int64_t bollard_bnb_outcome_objective(const BnbSolverFfiOutcome* ptr);

size_t bollard_bnb_outcome_num_vessels(const BnbSolverFfiOutcome* ptr);

// Direct accessors
size_t bollard_bnb_outcome_berth(const BnbSolverFfiOutcome* ptr, size_t vessel_idx);
int64_t bollard_bnb_outcome_start_time(const BnbSolverFfiOutcome* ptr, size_t vessel_idx);

// Raw Pointer Access (Use with caution)
// Returns pointer to internal array of size `num_vessels`
const size_t* bollard_bnb_outcome_berths(const BnbSolverFfiOutcome* ptr);
const int64_t* bollard_bnb_outcome_start_times(const BnbSolverFfiOutcome* ptr);

// Copy Helper
// `out_berths` and `out_start_times` must be pre-allocated buffers of size `num_vessels`
void bollard_bnb_outcome_copy_solution(
    const BnbSolverFfiOutcome* ptr, 
    size_t* out_berths, 
    int64_t* out_start_times
);

```

**Statistics**
All stats functions return `uint64_t`.

* `bollard_bnb_outcome_nodes_explored`
* `bollard_bnb_outcome_backtracks`
* `bollard_bnb_outcome_decisions_generated`
* `bollard_bnb_outcome_max_depth`
* `bollard_bnb_outcome_prunings_infeasible`
* `bollard_bnb_outcome_prunings_bound`
* `bollard_bnb_outcome_solutions_found`
* `bollard_bnb_outcome_steps`
* `bollard_bnb_outcome_time_total_ms`

## Design & Implementation Notes

* **Memory Ownership**: The host application owns any pointer returned by a `_new`, `_build`, or `_solve` function and is responsible for calling the corresponding `_free` function.
* **Panic Boundaries**: The FFI boundary catches Rust panics (e.g., `catch_unwind`). However, violation of preconditions (like passing NULL) may trigger an abort to ensure safety.
* **Regularity Requirement**: The underlying solver assumes the objective function is **regular** (non-decreasing in completion times). Custom constraints defined via wrapper libraries must adhere to this to guarantee correct pruning.
