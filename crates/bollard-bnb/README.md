# Bollard‑BnB

Branch‑and‑Bound (BnB) solver for berth allocation. This crate provides a deterministic, modular search engine that separates branching, evaluation, monitoring, and incumbent handling so strategies can be swapped without touching core solver logic.

## What it does

- Explores berth–vessel assignment trees using backtracking and bounds.
- Prunes subtrees via admissible lower bounds and shared/global incumbents.
- Produces optimal or best‑known feasible schedules, along with rich statistics and a termination reason.

## Architecture at a glance

- Branching: decision generation and ordering via `branching::DecisionBuilder`.
- Evaluation: objective pricing and admissible lower bounds via `eval::ObjectiveEvaluator`.
- Monitoring: lifecycle hooks to observe/control search via `monitor::TreeSearchMonitor`.
- Incumbent: optional integration with shared incumbents (portfolio/parallel) or local only.
- State management: compact `trail` (undo log) and framed `stack` (pending decisions).

## Core components

- `bnb::BnbSolver<T>`: the engine; orchestrates search, pruning, and backtracking.
- `branching`: decision builders (e.g., chronological, FCFS, WSPT, regret, slack).
- `eval`: objective interface, lower‑bound helpers, and validation utilities.
- `monitor`: log/composite/no‑op monitors plus a wrapper for portfolio monitors.
- `portfolio`: adapter implementing `bollard_search::portfolio::PortofolioSolver<T>`.
- `result`: `BnbSolverOutcome<T>` bundles result, termination reason, and stats.
- `stats`: lightweight counters and timing (`BnbSolverStatistics`).
- `fixed`: `FixedAssignment<T>` for warm starts and pinned tasks.
- `trail`/`stack`: high‑throughput data structures for undo and pending decisions.

## Typical workflow

1. Provide a `bollard_model::Model<T>` (berths, vessels, arrivals, durations, weights).
2. Choose a `branching::DecisionBuilder` to generate ordered feasible decisions.
3. Choose an `eval::ObjectiveEvaluator` to price decisions and compute admissible bounds.
4. Optionally provide:
   - Fixed assignments (warm start and constraints).
   - An incumbent (local or shared) for early pruning.
   - Monitors (composite/log/no‑op) for telemetry and stop criteria.
5. Invoke the solver:
   - Directly via `bnb::BnbSolver::{solve, solve_with_*}`.
   - Through the `portfolio` adapter to integrate with the generic search framework.

## Pruning and correctness

- Bound pruning: if `current_objective + lower_bound >= incumbent`, prune the subtree.
- Infeasibility pruning: structural or evaluator‑imposed violations cut branches early.
- Regularity is required: objective must be non‑decreasing in completion times.
- Lower bounds must be admissible (never overestimate remaining cost).

These assumptions underpin correctness and are asserted in tests and debug builds.

## Performance notes

- Deterministic given deterministic builders and evaluators.
- Linear, cache‑friendly structures (`SearchTrail`, `SearchStack`); no per‑node heap churn.
- `preallocated(num_berths, num_vessels)` shifts allocation up‑front for large instances.
- Hot‑path helpers use unchecked variants guarded by debug assertions.

## Monitoring and statistics

- Monitors receive lifecycle callbacks (enter, step, descend, prune, backtrack, solution, exit) and can request stop conditions.
- `BnbSolverStatistics` tracks nodes explored, backtracks, prunings, steps, depth, and total time; formatted for concise console output.

## Module map

- `src/bnb.rs`: solver engine and session orchestration.
- `src/branching/`: decision builders.
- `src/eval/`: objective interface, bounds, and validation.
- `src/monitor/`: monitors (log, composite, wrappers).
- `src/portfolio.rs`: portfolio integration.
- `src/result.rs`: outcomes and termination reasons.
- `src/stats.rs`: counters and timing.
- `src/fixed.rs`: fixed assignments.
- `src/trail.rs`, `src/stack.rs`: backtracking and decision stack.

## Guarantees and limits

- Produces optimal schedules when given a regular objective and admissible bounds.
- Clean backtracking invariants: berth free times, assignments, and objective are restored exactly when unwinding.
- Not a MILP/CP solver; it is a specialized BnB tailored to berth allocation with pluggable heuristics and objectives.
