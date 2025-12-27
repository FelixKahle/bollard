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

//! # Bollard Solver
//!
//! High-level orchestration for portfolio-based exact search. This crate
//! coordinates multiple solver strategies, manages shared state such as the
//! incumbent solution, and enforces termination via pluggable monitors.
//!
//! ## Modules
//!
//! - `solver`: Portfolio orchestrator with a builder, per-thread monitor stacks,
//!   shared incumbent, global counters, and unified outcome construction.
//!
//! ## Motivation
//!
//! Different strategies shine on different instances. A portfolio approach
//! improves robustness and time-to-best by running several strategies in
//! parallel and short-circuiting when global optimality is proven.
//!
//! See `solver` for detailed APIs and examples.

pub mod solver;
