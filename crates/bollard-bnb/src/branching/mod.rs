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

//! Branching strategies for berth allocation
//!
//! Defines decision types and multiple builders that generate feasible
//! `(vessel, berth)` assignments for branch‑and‑bound. Each builder produces
//! rich decisions (with start times and costs) and applies deterministic
//! ordering to improve search stability and pruning.
//!
//! Provided builders:
//! - `chronological`: row‑major traversal with symmetry reduction.
//! - `fcfs`: first‑come‑first‑served (arrival‑time priority, cost tie‑break).
//! - `wspt`: cost‑guided, akin to weighted shortest processing time.
//! - `regret`: best‑first by regret (gap between best and second‑best).
//! - `slack`: best‑first by tightest time slack (deadline pressure).
//! - `spt`: best‑first by shortest processing time.
//! - `lpt`: best‑first by longest processing time.
//!
//! All iterators are fused: once exhausted, further `next()` calls yield `None`.

pub mod chronological;
pub mod decision;
pub mod fcfs;
pub mod lpt;
pub mod regret;
pub mod slack;
pub mod spt;
pub mod wspt;
