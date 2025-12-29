# Bollard Solver

**The high-level portfolio orchestrator for the Bollard scheduling ecosystem.**

`bollard_solver` is the entry point for running scheduling tasks. It runs multiple solver strategies (exact, heuristic, or local search) in parallel, manages a shared incumbent solution, and enforces global termination criteria. It is designed to maximize hardware utilization by racing different strategies against each other to find the best solution in the shortest time.

## Key Features

* **Portfolio Orchestration**: Simultaneously launches multiple strategies (implementations of `PortfolioSolver<T>`) in separate threads.
* **Shared Incumbent Integration**: Automatically manages a `SharedIncumbent<T>`, ensuring that when one thread finds a better solution, it is immediately available to all other threads for pruning or reference.
* **Ergonomic Configuration**: A fluent `SolverBuilder` to easily configure time limits, solution limits, and the specific portfolio of strategies to run.
* **Unified Outcomes**: Aggregates results from all threads into a single `SolverOutcome<T>`, consolidating statistics and determining the final termination reason (e.g., Optimality Proven, Time Limit Reached).

## Architecture

The crate provides a streamlined API surface centered around the solver lifecycle:

1. **`solver`**: The core orchestration logic.
* **`SolverBuilder<'a, T>`**: The primary configuration interface. Allows setting global constraints (time, solution count) and adding strategies.
* **`Solver<'a, T>`**: The immutable orchestrator that executes the search via `solve(&Model<T>)`.


2. **Integration**:
* It binds together `bollard_search` components (Monitors, Incumbents) and `bollard_model` (Problem definition) into a cohesive runtime.



## Quick Start

### 1. Constructing and Running a Solver

The builder pattern allows you to compose a specific set of strategies and limits.

```rust
use bollard_solver::solver::SolverBuilder;
use bollard_model::model::Model;
use std::time::Duration;

// Assume we have a strategy implementation available
// use bollard_bnb::portfolio::BnbPortfolioSolver; 

fn main() {
    // 1. Construct the problem model
    // let model: Model<i64> = ...;

    // 2. Configure the solver with specific limits and strategies
    let mut solver = SolverBuilder::<i64>::new()
        .with_solution_limit(10) // Stop after finding 10 feasible solutions
        .with_time_limit(Duration::from_secs(30)) // Or stop after 30 seconds
        // .add_solver(StrategyA::new())
        // .add_solver(StrategyB::new())
        .build();

    // 3. Execute the search
    // let outcome = solver.solve(&model);
    
    // println!("Final Status: {}", outcome.result());
}

```

### 2. Inspecting Outcomes and Statistics

The `SolverOutcome` provides a unified view of what happened during the parallel search.

```rust
use bollard_solver::solver::SolverBuilder;
use bollard_search::result::SolverResult;

fn main() {
    // ... setup and solve ...
    // let outcome = solver.solve(&model);

    // Check high-level status
    println!("Termination Reason: {}", outcome.reason());

    // Access detailed runtime statistics (aggregated across threads)
    println!("Stats:\n{}", outcome.statistics());

    // Retrieve the best solution found
    if let SolverResult::Optimal(sol) = outcome.result() {
        println!("Optimal objective: {}", sol.objective_value());
    } else if let SolverResult::Feasible(sol) = outcome.result() {
        println!("Best feasible objective: {}", sol.objective_value());
    }
}

```

## Design & Performance

* **Parallel Independence**: Each strategy in the portfolio runs in its own thread with its own monitor stack. This prevents a slow or stalled strategy from blocking the progress of others.
* **Global Optimality Signals**: The orchestrator coordinates "Optimality Proven" signals. If one thread proves optimality, the orchestrator triggers an interrupt flag to stop all other threads immediately, saving computational resources.
* **Deterministic Outcome Assembly**: While thread scheduling is non-deterministic, the result assembly is robust. The orchestrator selects the best solution from the shared incumbent and the individual thread results, ensuring the highest quality solution is always returned.
