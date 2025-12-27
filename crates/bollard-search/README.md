# Bollard Search

High-level components for building exact search pipelines in the Bollard ecosystem. This crate provides pluggable monitors, a concurrent incumbent holder, portfolio solver interfaces, unified result reporting, and lightweight runtime statistics.

## What This Crate Offers

- Search Monitors:
  - Observe lifecycle events (enter/exit search, steps, solutions).
  - Enforce termination criteria (time limit, solution count, external interrupt).
  - Compose multiple monitors via a composite wrapper.

- Incumbent Management:
  - `SharedIncumbent<T>` holds the best solution discovered so far.
  - Atomic upper bound for fast filtering; mutex-protected snapshot for correctness.

- Portfolio Solvers:
  - Standardized context and result types for running multiple strategies.
  - Clean integration with shared incumbent and monitoring interfaces.

- Results & Stats:
  - Unified solver result and termination reason types.
  - Lightweight, human-readable statistics with a fluent builder.

- Numeric Bounds:
  - `SolverNumeric` collects all numeric capabilities needed by the solver (checked/saturating arithmetic, conversions, traits).

## Modules Overview

- `monitor`:
  - `search_monitor`: Trait (`SearchMonitor<T>`) and `SearchCommand` enum.
  - `composite`: Aggregate multiple monitors and forward lifecycle events.
  - `interrupt`: External stop signal via `AtomicBool`.
  - `solution`: Global solution-count limit via `AtomicU64`.
  - `time_limit`: Wall-clock time limit with step-filtered checks.
  - `index`: Strongly typed monitor indices (zero-cost wrappers).

- `incumbent`:
  - `SharedIncumbent<T>` with atomic upper bound and mutex-backed snapshot.

- `portfolio`:
  - `PortfolioSolverContext<'a, T>` and `PortofolioSolver<T>` interface.
  - `PortfolioSolverResult<T>` capturing solver result and termination reason.

- `result`:
  - `SolverResult<T>` and `TerminationReason`.
  - `SolverOutcome<T>` with `SolverStatistics`.

- `stats`:
  - `SolverStatistics` and `SolverStatisticsBuilder`.

- `num`:
  - `SolverNumeric` trait for required numeric bounds.

## Quick Examples

### 1. Building a Composite Monitor

```rust
use bollard_search::monitor::composite::CompositeMonitor;
use bollard_search::monitor::interrupt::InterruptMonitor;
use bollard_search::monitor::solution::SolutionMonitor;
use bollard_search::monitor::time_limit::TimeLimitMonitor;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::time::Duration;

fn main() {
    let stop_flag = AtomicBool::new(false);
    let global_solution_count = AtomicU64::new(0);

    let mut monitors = CompositeMonitor::<i64>::new();
    monitors.add_monitor(InterruptMonitor::new(&stop_flag));
    monitors.add_monitor(SolutionMonitor::with_limit(&global_solution_count, 5));
    monitors.add_monitor(TimeLimitMonitor::new(Duration::from_secs(30)));

    // Hook monitors into your search loop
    // monitors.on_enter_search(model);
    // loop {
    //     monitors.on_step();
    //     match monitors.search_command() {
    //         bollard_search::monitor::search_monitor::SearchCommand::Continue => { /* keep searching */ }
    //         bollard_search::monitor::search_monitor::SearchCommand::Terminate(reason) => {
    //             println!("Terminating search: {}", reason);
    //             break;
    //         }
    //     }
    // }
    // monitors.on_exit_search();
}
```

### 2. Managing the Incumbent

```rust
use bollard_search::incumbent::SharedIncumbent;
use bollard_model::solution::Solution;

fn main() {
    let incumbent: SharedIncumbent<i64> = SharedIncumbent::new();

    // Candidate solution
    let candidate = Solution::new(100i64, Vec::new(), Vec::new());

    if incumbent.try_install(&candidate) {
        println!("Installed new incumbent with objective {}", incumbent.upper_bound());
    }

    if let Some(best) = incumbent.snapshot() {
        println!("Current best objective: {}", best.objective_value());
    }
}
```

### 3. Portfolio Solver Interface

```rust
use bollard_search::portfolio::{PortfolioSolverContext, PortfolioSolverResult, PortofolioSolver};
use bollard_search::incumbent::SharedIncumbent;
use bollard_search::monitor::search_monitor::{SearchMonitor, SearchCommand, DummyMonitor};
use bollard_model::model::Model;
use num_traits::{PrimInt, Signed};

struct MyStrategy;

impl<T: PrimInt + Signed + Send + Sync> PortofolioSolver<T> for MyStrategy {
    fn invoke<'a>(&mut self, ctx: PortfolioSolverContext<'a, T>) -> PortfolioSolverResult<T> {
        ctx.monitor.on_enter_search(ctx.model);
        // ... run your strategy, propose solutions via ctx.incumbent.try_install(...)
        // ... periodically check ctx.monitor.search_command() for termination
        ctx.monitor.on_exit_search();

        // Return an outcome (here: infeasible)
        PortfolioSolverResult::infeasible()
    }

    fn name(&self) -> &str { "MyStrategy" }
}

fn main() {
    // Example: set up and invoke a portfolio strategy
    // let model: Model<i64> = ...;
    // let incumbent = SharedIncumbent::<i64>::new();
    // let mut monitor = DummyMonitor::<i64>::new();
    // let ctx = PortfolioSolverContext::new(&model, &incumbent, &mut monitor);
    // let mut strategy = MyStrategy;
    // let result = strategy.invoke(ctx);
    // println!("{}", result);
}
```

### 4. Reporting Results and Stats

```rust
use bollard_search::result::{SolverOutcome, SolverResult};
use bollard_search::stats::{SolverStatistics, SolverStatisticsBuilder};
use bollard_model::solution::Solution;
use std::time::Duration;

fn main() {
    let stats = SolverStatisticsBuilder::new()
        .solutions_found(3)
        .used_threads(8)
        .solve_duration(Duration::from_millis(1200))
        .build();

    let sol = Solution::new(42i64, Vec::new(), Vec::new());
    let outcome = SolverOutcome::optimal(sol, stats);

    println!("{}", outcome); // human-friendly multi-line report
    assert!(matches!(outcome.result(), SolverResult::Optimal(_)));
}
```

## Design Notes

- Monitors are designed with minimal overhead and frequent invocation in mind.
- `SharedIncumbent` uses relaxed atomics for the upper bound and a mutex for correctness on solution snapshots.
- Portfolio solver interfaces keep strategies modular and composable.
- Result types and statistics are simple and serializable for FFI and UI integration.
