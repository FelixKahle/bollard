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

//! Metaheuristics for local search control.
//!
//! This module contains acceptance and control policies that steer the local
//! search engine without coupling to specific move operators. Implementations
//! conform to the `Metaheuristic` trait and interact with the engine through
//! lightweight lifecycle hooks (start, per‑iteration command, accept/reject,
//! and new‑best notifications). This separation keeps hot paths predictable
//! while allowing flexible search behavior.
//!
//! Provided heuristics:
//! - `greedy_descent`: first‑improvement hill climber that only accepts strict improvements.
//! - `guided_local_search`: augments the objective with adaptive penalties to escape local optima.
//! - `simulated_annealing`: temperature‑driven acceptance with Metropolis criterion and pluggable cooling.
//! - `tabu_search`: short‑term memory via a tabu list with aspiration to allow breakthroughs.
//!
//! All implementations use the same evaluator interface so they can be easily
//! swapped or combined in portfolios. They are designed to be deterministic
//! given fixed neighborhoods and RNG seeds (where applicable).

pub mod greedy_descent;
pub mod guided_local_search;
pub mod metaheuristic;
pub mod simulated_annealing;
pub mod tabu_search;
