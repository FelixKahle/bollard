# Bollard Search

**High-level components for building modular search pipelines in the Bollard ecosystem.**

`bollard_search` provides the essential infrastructure to construct robust optimization strategies. While it serves as the foundation for exact solvers (like Branch-and-Bound), its architecture is generic enough to support local search, heuristics, and hybrid meta-heuristics. It offers pluggable monitoring, concurrent incumbent management, and unified result reporting.

## Key Features

* **Pluggable Search Monitors**: A flexible observation system (`SearchMonitor`) to track lifecycle events (steps, solutions, backtracks) and enforce termination criteria (time limits, solution counts, or external interrupts).
* **Concurrent Incumbent Management**: The `SharedIncumbent<T>` type allows multiple search threads to safely share the "best so far" solution. It utilizes atomic upper bounds for rapid filtering and mutex-protected snapshots for data consistency.
* **Portfolio Interface**: Standardized traits (`PortfolioSolver`) that enable running diverse strategies (exact or heuristic) in parallel under a unified context.
* **Unified Statistics**: A lightweight, fluent statistics builder and result types that decouple the search logic from how results are reported to the user or FFI.
* **Numeric Bounds**: The `SolverNumeric` trait consolidates all required numeric capabilities (checked/saturating arithmetic, conversions) needed for a solver implementation.

## Architecture

The crate is organized into modular components that can be used independently or together:

1. **`monitor`**: Search observation and control.
* **`search_monitor`**: Defines the `SearchMonitor<T>` trait and `SearchCommand` enum.
* **`composite`**: Wrapper to aggregate multiple monitors and broadcast events.
* **`time_limit` / `solution` / `interrupt**`: Concrete implementations for common stopping criteria.


2. **`incumbent`**: Thread-safe solution storage.
* Contains `SharedIncumbent<T>`, combining atomic bounds with full solution storage.


3. **`portfolio`**: Interfaces for strategy orchestration.
* Defines `PortfolioSolverContext<'a, T>` and the `PortfolioSolver<T>` trait.


4. **`result` / `stats**`: Reporting structures.
* **`result`**: `SolverOutcome<T>`, `SolverResult<T>`, and `TerminationReason`.
* **`stats`**: `SolverStatistics` and its builder.


5. **`num`**: Numeric abstractions.
* `SolverNumeric` trait to simplify generic bounds on solver implementations.



## Quick Start

### 1. Building a Composite Monitor

Create a monitoring stack that handles time limits, solution counts, and external interrupts simultaneously.

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

    // Combine multiple monitors into one
    let mut monitors = CompositeMonitor::<i64>::new();
    monitors.add_monitor(InterruptMonitor::new(&stop_flag));
    monitors.add_monitor(SolutionMonitor::with_limit(&global_solution_count, 5));
    monitors.add_monitor(TimeLimitMonitor::new(Duration::from_secs(30)));

    // In your search loop:
    // monitors.on_step();
    // if let SearchCommand::Terminate(reason) = monitors.search_command() { ... }
}

```

### 2. Defining a Portfolio Strategy

Implement the `PortfolioSolver` trait to create a custom strategy (exact or heuristic) that integrates with the ecosystem.

```rust
use bollard_search::portfolio::{PortfolioSolverContext, PortfolioSolverResult, PortofolioSolver};
use num_traits::{PrimInt, Signed};

struct MyStrategy;

impl<T: PrimInt + Signed + Send + Sync> PortofolioSolver<T> for MyStrategy {
    fn invoke<'a>(&mut self, ctx: PortfolioSolverContext<'a, T>) -> PortfolioSolverResult<T> {
        // Notify monitors that search is starting
        ctx.monitor.on_enter_search(ctx.model);
        
        // ... execute logic (e.g., local search, constructive heuristic) ...
        // ... report new best solutions via ctx.incumbent.try_install(sol) ...
        
        ctx.monitor.on_exit_search();

        PortfolioSolverResult::infeasible() // Return final status
    }

    fn name(&self) -> &str { "MyStrategy" }
}

```

## Design & Performance

* **Generic Pipelines**: While often used for Branch-and-Bound, these components are agnostic to the search method. Local search algorithms can utilize the `SharedIncumbent` to coordinate with constructive heuristics in a parallel portfolio.
* **Low-Overhead Monitoring**: Monitors are designed for frequent invocation (e.g., every search node). Checks like time limits are often step-filtered (checked every  steps) to minimize system call overhead.
* **Concurrency**: `SharedIncumbent` uses relaxed atomic ordering for reading the upper bound, ensuring that checking "can this branch be better than the best known solution?" is extremely fast and non-blocking.
