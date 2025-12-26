# Monitoring (Tree Search Monitors)

This module defines the monitoring interface and lightweight implementations used to observe and influence the branch‑and‑bound solver without touching core logic.

## Core Abstraction: `TreeSearchMonitor<T>`

A monitor receives lifecycle callbacks and may steer execution via `SearchCommand` (default: `Continue`). Callbacks are provided with context like `Model`, `SearchState`, `BnbSolverStatistics`, `Decision`, and `Solution` as appropriate.

### Lifecycle at a Glance
`enter → step → {lower‑bound | decisions → descend | prune} → backtrack → solution → exit`

Monitors can gather metrics, log progress, or request early stopping based on the current state and statistics.

## Provided Monitors

- `composite`: Fan‑out to multiple monitors. Dispatch order is insertion order. `search_command` short‑circuits on the first non‑`Continue`. Dispatch overhead is O(k) per callback for k children.
- `log`: Periodic console progress (elapsed, nodes, best objective, backtracks, pruned). Uses a step mask to reduce clock checks and overhead.
- `no_op`: Neutral monitor that ignores all callbacks and always returns `Continue`.
- `wrapper`: Adapter to `bollard_search::SearchMonitor`. Forwards generic lifecycle events and ignores tree‑specific ones.
- `time`: Tracks elapsed time and requests early stopping when a time limit is exceeded.
- `solution`: Tracks the amount of solutions found and requests early stopping when a solution limit is reached.

## Design Notes

- API is `&mut self`; monitors are assumed single‑threaded and should remain lightweight (avoid blocking I/O in hot paths).
- Monitors are typically used as boxed trait objects and combined with `composite` to layer logging, metrics, visualization, and stop criteria.
- The objective type is generic: `T: PrimInt + Signed`.
