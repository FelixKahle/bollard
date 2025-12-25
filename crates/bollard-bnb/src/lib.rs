// Copyright (c) 2025 Felix Kahle.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! Bollard‑BnB: branch‑and‑bound for berth allocation
//!
//! High‑level crate that implements a deterministic, modular BnB solver for
//! berth allocation. The solver separates branching, evaluation, monitoring,
//! and incumbent handling so you can swap strategies without touching core
//! search logic.
//!
//! Core flow
//! - Provide a `bollard_model::Model<T>`.
//! - Choose a `branching::DecisionBuilder` (decision ordering).
//! - Choose an `eval::ObjectiveEvaluator` (pricing + admissible lower bounds).
//! - Optionally set fixed assignments, an incumbent, and monitors.
//! - Run `bnb::BnbSolver` directly, or integrate via `portfolio`.
//!
//! Design highlights
//! - Separation of concerns: builders generate feasible decisions; evaluators
//!   inject cost + bounds; monitors observe/control; outcomes carry stats.
//! - Tight inner loop: state is mutated in place and restored via a trail;
//! - Deterministic given deterministic builders/evaluators.
//!
//! Assumptions and guarantees
//! - Evaluators must be regular (objective non‑decreasing in completion times);
//!   pruning logic relies on this for correctness.
//! - Lower bounds must be admissible (no overestimation).
//!
//! Module map
//! - `bnb`: the solver engine and session orchestration.
//! - `branching`: decision builders (ordering/heuristics).
//! - `eval`: objective interface and helpers.
//! - `monitor`: tree‑search monitors (log, composite, wrappers).
//! - `portfolio`: adapter to `bollard_search` portfolio API.
//! - `result`: solver outcomes with termination reasons.
//! - `stats`: lightweight counters/timing.
//! - `fixed`: fixed assignments for warm starts and constraints.

mod berth_availability;
pub mod bnb;
pub mod branching;
pub mod eval;
pub mod fixed;
mod incumbent;
pub mod monitor;
pub mod portfolio;
pub mod result;
mod stack;
pub mod state;
pub mod stats;
mod trail;
