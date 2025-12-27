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

//! # Search Monitoring Interface
//!
//! An extensible callback interface for observing and controlling the lifecycle
//! of search algorithms (e.g., branch-and-bound) in the Bollard ecosystem.
//! Implementations can collect metrics, log progress, react to solutions,
//! and issue termination commands based on external criteria.
//!
//! ## Motivation
//!
//! Long-running combinatorial searches benefit from pluggable monitoring:
//! - Capture telemetry (nodes explored, depth, time).
//! - Emit logs or UI updates.
//! - Enforce budget limits (time, steps, solutions).
//! - Integrate with external controllers or dashboards.
//!
//! ## Core Concepts
//!
////! - `SearchMonitor<T>`: Trait defining lifecycle hooks:
//!   - `on_enter_search(&mut self, model)` — initialization before search starts.
//!   - `on_exit_search(&mut self)` — cleanup after search finishes.
//!   - `on_solution_found(&mut self, solution)` — react to new solutions.
//!   - `on_step(&mut self)` — periodic heartbeat hook from the search loop.
//!   - `search_command(&self)` — return `Continue` or `Terminate(reason)`.
//! - `SearchCommand`: Control signal emitted by monitors to continue or abort.
//! - `DummyMonitor<T>`: Minimal no-op implementation useful for testing or
//!   as a template for custom monitors.
//!
//! ## Usage
//!
//! ```rust
//! use bollard_search::monitor::search_monitor::{SearchMonitor, SearchCommand, DummyMonitor};
//! use bollard_model::{model::Model, solution::Solution};
//! use num_traits::{PrimInt, Signed};
//!
//! struct MyMonitor<T: PrimInt + Signed> {
//!     steps: u64,
//!     reason: Option<String>,
//!     _phantom: std::marker::PhantomData<T>,
//! }
//!
//! impl<T: PrimInt + Signed> SearchMonitor<T> for MyMonitor<T> {
//!     fn name(&self) -> &str { "MyMonitor" }
//!     fn on_enter_search(&mut self, _model: &Model<T>) { self.steps = 0; }
//!     fn on_exit_search(&mut self) {}
//!     fn on_solution_found(&mut self, _solution: &Solution<T>) {
//!         self.reason = Some("Found good enough".into());
//!     }
//!     fn on_step(&mut self) {
//!         self.steps += 1;
//!         if self.steps > 1_000_000 {
//!             self.reason = Some("Step budget exceeded".into());
//!         }
//!     }
//!     fn search_command(&self) -> SearchCommand {
//!         self.reason
//!             .as_ref()
//!             .map(|r| SearchCommand::Terminate(r.clone()))
//!             .unwrap_or(SearchCommand::Continue)
//!     }
//! }
//! ```

use bollard_model::{model::Model, solution::Solution};
use num_traits::{PrimInt, Signed};

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub enum SearchCommand {
    #[default]
    Continue,
    Terminate(String),
}

impl std::fmt::Display for SearchCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchCommand::Continue => write!(f, "Continue"),
            SearchCommand::Terminate(reason) => write!(f, "Terminate: {}", reason),
        }
    }
}

pub trait SearchMonitor<T>
where
    T: PrimInt + Signed,
{
    fn name(&self) -> &str;
    fn on_enter_search(&mut self, model: &Model<T>);
    fn on_exit_search(&mut self);
    fn on_solution_found(&mut self, solution: &Solution<T>);
    fn on_step(&mut self);
    fn search_command(&self) -> SearchCommand;
}

impl<T> std::fmt::Debug for dyn SearchMonitor<T>
where
    T: PrimInt + Signed,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SearchMonitor({})", self.name())
    }
}

impl<T> std::fmt::Display for dyn SearchMonitor<T>
where
    T: PrimInt + Signed,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SearchMonitor({})", self.name())
    }
}

pub struct DummyMonitor<T>
where
    T: PrimInt + Signed,
{
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for DummyMonitor<T>
where
    T: PrimInt + Signed,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> DummyMonitor<T>
where
    T: PrimInt + Signed,
{
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> SearchMonitor<T> for DummyMonitor<T>
where
    T: PrimInt + Signed,
{
    fn name(&self) -> &str {
        "DummyMonitor"
    }

    fn on_enter_search(&mut self, _model: &Model<T>) {}

    fn on_exit_search(&mut self) {}

    fn on_solution_found(&mut self, _solution: &Solution<T>) {}

    fn on_step(&mut self) {}

    fn search_command(&self) -> SearchCommand {
        SearchCommand::Continue
    }
}
