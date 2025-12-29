# Bollard BnB

**The core Branch-and-Bound (BnB) search engine for berth allocation.**

`bollard_bnb` provides a deterministic, modular solver that separates branching logic, objective evaluation, and search orchestration. It is designed to explore vessel-berth assignment trees efficiently, using admissible lower bounds to prune subtrees and high-performance data structures for backtracking.

## Key Features

* **Modular Search Architecture**: Strategies are composed by plugging together independent components: a `DecisionBuilder` (branching), an `ObjectiveEvaluator` (bounding), and optional `SearchMonitors`.
* **Admissible Pruning**: Capable of pruning subtrees early by comparing the current lower bound against a global or local incumbent solution.
* **Efficient Backtracking**: Uses linear, cache-friendly data structures (`SearchTrail` and framed `SearchStack`) to manage the undo log, avoiding heap churn at every search node.
* **Warm Starting**: Supports `FixedAssignment` to pin specific tasks or respect partial schedules before solving begins.
* **Rich Telemetry**: Integrated with `bollard_search` monitors to provide detailed logs, step counts, and termination control (time limits, node limits).

## Architecture

The crate is organized around the primary solver loop and its pluggable extensions:

1. **`bnb`**: The core engine (`BnbSolver<T>`) that orchestrates the recursive search, state restoration, and pruning logic.
2. **`branching`**: Decision builders responsible for generating and ordering moves. Includes strategies like Chronological, FCFS, WSPT, Regret, and Slack.
3. **`eval`**: Interfaces for objective functions and lower-bound calculations (`ObjectiveEvaluator`).
4. **`fixed`**: Utilities for handling pre-assigned constraints (`FixedAssignment`).
5. **`trail` / `stack**`: High-throughput internal data structures for managing the search state and decision history.

## Design & Performance

* **Correctness & Objective Regularity**:
> **Critical Requirement:** The solver assumes the objective function is **regular**, meaning it must be **non-decreasing in completion times**.
> If a decision increases a vessel's completion time, the total cost must not decrease. This property is fundamental to the correctness of the lower-bound pruning mechanism.


* **Memory Layout**: Unlike many solver implementations that allocate nodes on the heap, `bollard_bnb` uses a flat `SearchTrail` (undo log). This ensures that backtracking is a simple pointer decrement and memory usage remains linear with search depth.
* **Deterministic Execution**: Given the same model, decision builder, and evaluator, the solver will always explore the exact same tree path. This determinism is preserved even when running within a parallel portfolio (though the *interruption* timing from other threads may vary).
