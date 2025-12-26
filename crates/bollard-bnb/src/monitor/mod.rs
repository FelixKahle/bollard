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

//! Monitoring utilities for branch‑and‑bound tree search
//!
//! Defines the `TreeSearchMonitor` trait plus lightweight implementations to
//! observe, log, and control the solver without touching core logic.
//!
//! Components
//! - `tree_search_monitor`: the monitoring interface and prune reasons.
//! - `composite`: fan‑out monitor; short‑circuits on first non‑`Continue`.
//! - `log`: periodic console progress reporting.
//! - `no_op`: zero‑overhead placeholder.
//! - `wrapper`: adapter to `bollard_search::SearchMonitor`.
//! - `time`: time limit enforcement.
//! - `solution`: solution‑based termination.
//!
//! Notes
//! - Callbacks take `&mut self`; keep handlers fast and non‑blocking.
//! - In composites, insertion order matters for `search_command`.

pub mod composite;
pub mod log;
pub mod no_op;
pub mod solution;
pub mod time;
pub mod tree_search_monitor;
pub mod wrapper;
