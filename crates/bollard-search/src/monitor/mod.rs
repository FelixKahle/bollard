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

//! # Search Monitors
//!
//! Pluggable observers and controllers for search lifecycle events. Monitors
//! can log progress, collect metrics, enforce budgets (time, solutions), and
//! issue termination commands to guide or stop the search.
//!
//! ## Submodules
//!
//! - `search_monitor`: Core trait (`SearchMonitor<T>`) and `SearchCommand` enum,
//!   defining lifecycle hooks and control flow.
//! - `composite`: Aggregate multiple monitors into a single composite.
//! - `index`: Strongly typed monitor indices for safe addressing.
//! - `interrupt`: Atomically-driven interrupt monitor for cross-thread stops.
//! - `solution`: Solution-count monitor with global limit via `AtomicU64`.
//! - `time_limit`: Wall-clock time budget monitor with step-filtered checks.
//!
//! ## Motivation
//!
//! Exact search algorithms benefit from modular instrumentation and control.
//! These monitors enable orthogonal concerns (telemetry, UI, limits) without
//! entangling them in the core search loop.
//!
//! Refer to each submodule for detailed APIs and examples.

pub mod composite;
pub mod index;
pub mod interrupt;
pub mod search_monitor;
pub mod solution;
pub mod time_limit;
