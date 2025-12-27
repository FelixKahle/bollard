# Bollard Solver

Portfolio-based orchestration for search in the Bollard ecosystem. This crate runs multiple solver strategies in parallel, manages a shared incumbent solution, and enforces termination via pluggable monitors (time limit, solution count, external interrupt).

## What This Crate Provides

- Portfolio Orchestrator:
  - Launches multiple strategies (implementations of `PortofolioSolver<T>`) in parallel threads.
  - Builds a per-thread monitor stack (interrupt, solution-limit, optional time-limit).
  - Aggregates results and produces a unified `SolverOutcome<T>`.

- Shared Incumbent:
  - `SharedIncumbent<T>` stores the best solution found so far.
  - Atomic upper bound for fast comparisons; mutex-backed snapshot for correctness.

- Unified Outcomes & Statistics:
  - `SolverOutcome<T>` with `SolverResult<T>` and `TerminationReason`.
  - `SolverStatistics` and a fluent builder to report runtime metrics.

- Ergonomic Builder:
  - `SolverBuilder` to configure solution/time limits and add portfolio solvers.

## Module Overview

- `solver`:
  - `Solver<'a, T>`: High-level orchestrator; `solve(&Model<T>) -> SolverOutcome<T>`.
  - `SolverBuilder<'a, T>`: Builder for solver configuration.
  - Orchestration integrates `bollard-search`’s monitors and incumbent management.

## Quick Start

### 1. Build a Solver with a Portfolio

```rust
use bollard_solver::solver::{SolverBuilder};
use bollard_model::model::Model;
// bring your portfolio solver implementations
// use bollard_bnb::portfolio::BnbPortfolioSolver; // for example

fn main() {
    // Construct your model
    // let model: Model<i64> = ...;

    // Construct your portfolio strategies (implement PortofolioSolver<i64>)
    // let s1 = ...;
    // let s2 = ...;

    // Configure solver
    let mut solver = SolverBuilder::<i64>::new()
        // .with_solution_limit(10)
        // .with_time_limit(std::time::Duration::from_secs(30))
        // .add_solver(s1)
        // .add_solver(s2)
        .build();

    // Run
    // let outcome = solver.solve(&model);
    // println!("{}", outcome);
}
```

### 2. Inspect Outcome and Statistics

```rust
use bollard_solver::solver::SolverBuilder;
use bollard_search::result::SolverResult;
use std::time::Duration;

fn main() {
    // let model: Model<i64> = ...;
    let mut solver = SolverBuilder::<i64>::new()
        .with_time_limit(Duration::from_secs(10))
        .build();

    // let outcome = solver.solve(&model);
    // println!("Result: {}", outcome.result());
    // println!("Termination: {}", outcome.reason());
    // println!("Stats:\n{}", outcome.statistics());

    // if let SolverResult::Optimal(sol) = outcome.result() {
    //     println!("Optimal objective: {}", sol.objective_value());
    // }
}
```

## Design Notes

- Portfolio model: Each strategy runs independently with its own monitor stack.
- Global signals: When a thread proves optimality, the orchestrator sets an interrupt flag to stop others early.
- Minimal overhead: Monitors are designed to be efficient; atomic counters and relaxed ordering are used where safe.
- Deterministic outcome assembly: The best solution across threads and the shared incumbent is selected, with termination reasons consolidated.
