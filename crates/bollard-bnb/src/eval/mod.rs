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

//! Evaluation utilities for berth allocation
//!
//! Provides objective evaluation, lower‑bounding, and feasibility helpers
//! used by the branch‑and‑bound solver.
//!
//! Regularity requirement:
//! - Every `ObjectiveEvaluator` must be regular: the objective value must not
//!   decrease as feasible start/completion times move later (monotone
//!   non‑decreasing). The solver relies on this property for correctness and
//!   will assert it in debug builds. Use
//!   `validation::is_regular_evaluator_exhaustive` to sanity‑check custom
//!   evaluators on specific models.
//! - Non‑regular objectives (e.g., those that penalize earliness) can lead to
//!   incorrect pruning and missed optima.
//!
//! Submodules:
//! - `evaluator`: the `ObjectiveEvaluator` trait and lower‑bound helpers.
//! - `hybrid`: combinators that blend multiple evaluators.
//! - `validation`: utilities to check evaluator regularity (see
//!   `is_regular_evaluator_exhaustive`).
//! - `workload`: helpers for computing and aggregating resource load over time.
//! - `wtft`: specialized or experimental evaluation utilities.
//!
//! These components are designed to be composable across heuristics, node
//! expansion, and solution verification.

pub mod evaluator;
pub mod hybrid;
pub mod validation;
pub mod workload;
pub mod wtft;
