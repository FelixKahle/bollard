# Branching (Decision Builders)

This module defines the branching layer of the solver: how feasible `(vessel, berth)` assignment options (decisions) are generated, ordered, and exposed to the search. The goal is to present a high‑quality, deterministic stream of “rich decisions” that improves cache locality, symmetry handling, and pruning for branch‑and‑bound.

## What is a Decision?

A `Decision<T>` is a fully computed candidate assignment:
- References a `vessel_index` and `berth_index`.
- Includes the scheduled `start_time` (respecting arrival and berth availability).
- Carries an immediate objective delta `cost_delta`.

Important: cost computation is injected via the active `ObjectiveEvaluator`. The default objective is weighted completion/flow‑time, but evaluators can adjust the semantics (e.g., add penalties, change aggregation, enforce deadlines). Do not hard‑code cost formulas in builders; they rely on the evaluator to price decisions consistently.

Constraints enforced:
- The selected `(vessel, berth)` must be allowed by the model topology.
- The `start_time` must fit within berth opening intervals (maintenance closures respected).
- Evaluator‑specific feasibility (e.g., finish ≤ deadline) is applied during pricing.

## What is a Decision Builder?

A `DecisionBuilder<T, E>` produces an iterator of rich decisions given:
- `Model<T>`: topology, times, weights, availability windows.
- `BerthAvailability<T>`: processed opening/closing intervals and fixed bookings.
- `SearchState<T>`: current partial schedule (berth free times, assignments).
- `ObjectiveEvaluator<E>`: injects feasibility and computes `cost_delta`.

All builders:
- Filter infeasible pairs up front (using availability, model, and evaluator rules).
- Produce decisions with computed `start_time` and injected `cost_delta`.
- Use stable, deterministic ordering (important for reproducibility).
- Return a fused iterator (once exhausted, further `next()` calls yield `None`).

## Provided Builders

- `chronological`: Row‑major traversal over `(vessel × berth)` with symmetry reduction. Canonical ordering; good baseline.
- `fcfs`: First‑Come‑First‑Served; primary key is earliest arrival, secondary tie‑break on evaluator‑injected cost.
- `wspt`: Cost‑guided; orders by increasing immediate evaluator‑injected cost (commonly weighted finish time).
- `regret`: Best‑first by regret. For per‑vessel feasible costs `{c₁ ≤ c₂ ≤ …}`, regret is `c₂ − c₁`; single‑option vessels get maximal regret.
- `slack`: Best‑first by tightest time slack. For deadline `D` and best‑case finish `F*`, slack is `S = D − F*` (smaller is more urgent).
- `spt`: Shortest Processing Time first; orders by increasing processing time `p`.
- `lpt`: Longest Processing Time first; orders by decreasing processing time `p`.
- `edf`: Earliest Deadline First; orders by increasing deadline `D`.

## Math at a Glance

These are common formulations, but the active evaluator controls the actual cost:

- Weighted completion cost (typical):
  - `C = w · F`, with `F = start_time + p` and `p` the processing time.

- Slack (deadline pressure):
  - `S = D − F*`, with `F* = min_{feasible berths} (start_time + p)`.

- Regret (sensitivity to suboptimal choice):
  - Given sorted feasible costs `{c₁ ≤ c₂ ≤ …}`, `R = c₂ − c₁`.
  - If only one feasible option exists, `R = +∞` (must‑assign‑now).

## Symmetry Reduction

Builders leverage symmetry checks to avoid exploring indistinguishable adjacent berth choices under identical state and processing conditions. This preserves canonical optimal schedules while cutting redundant branches.

## Iterator Semantics

All decision iterators implement `FusedIterator`:
- After exhaustion, further `next()` calls return `None`.

## Design Guarantees

- Feasibility is enforced before ordering (no “ghost” decisions).
- `cost_delta` is entirely injected by the evaluator; builders never compute or assume a specific objective form.
- Builders are lightweight and cache‑friendly, minimizing overhead relative to solver work.
