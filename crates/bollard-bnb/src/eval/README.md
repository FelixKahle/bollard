# Objective Evaluator Module

This module defines the interface for scoring decisions and computing lower bounds within the Berth Allocation Solver. It decouples the core search algorithms (Branch-and-Bound, A*) from the specific business logic of cost functions.

## Core Abstraction: `ObjectiveEvaluator`

The primary component is the `ObjectiveEvaluator<T>` trait. It serves two distinct purposes during the search process:

1. **Incremental Evaluation:** Calculates the immediate cost impact of assigning a specific vessel to a specific berth at a specific time.
2. **Heuristic Estimation:** Computes an optimistic lower bound on the remaining cost required to complete the partial schedule.

## Critical Requirement: Regularity

All implementations **must** represent a **Regular Objective Function**.

* **Definition:** An objective function is *regular* if it is non-decreasing with respect to completion times.
* **Implication:** Completing a task earlier or at the same time must never yield a higher cost than completing it later.
* **Prohibited:** Objectives with "earliness penalties" (e.g., Just-In-Time costs where finishing early is penalized) are incompatible with this interface.

**Why?** The solver's dominance rules assume that "earlier is better" or "earlier is equal." If this property is violated, the solver may incorrectly prune branches that lead to the optimal solution, compromising correctness.

## Context Awareness

Unlike simple cost functions, this evaluator is **context-aware**. Methods are provided with access to:

* `BerthAvailability`: To account for static constraints (e.g., maintenance windows) when estimating bounds.
* `SearchState`: To base estimates on the current depth and set of unassigned vessels.
* `Model`: For static vessel and berth parameters.

## Implementation Guide

To add a new objective (e.g., *Minimize Total Tardiness*):

1. Implement the `ObjectiveEvaluator` trait.
2. Ensure `evaluate_vessel_assignment` returns the exact cost contribution of a single move.
3. Ensure `estimate_remaining_cost` returns an **admissible** heuristic (never overestimates the true remaining cost).
