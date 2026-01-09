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

//! Tree search monitoring interface
//!
//! Declares the `TreeSearchMonitor` trait and `PruneReason` for observing and
//! controlling branch‑and‑bound. Callbacks track the solver lifecycle, and a
//! monitor can influence execution via `SearchCommand` (default: Continue).
//!
//! Lifecycle highlights
//! - enter → step → {lower‑bound/prune | decisions/descend/backtrack} → solution → exit
//! - `BnbSolverStatistics` is provided to every callback for telemetry.
//!
//! Design notes
//! - Methods take `&mut self`; monitors are assumed single‑threaded.
//! - Keep callbacks lightweight; avoid blocking I/O in hot paths.
//! - Generic over `T: PrimInt + Signed` (objective type).
//!
//! Integrates with `composite`, `log`, and `no_op` monitors to mix and match
//! logging, metrics, visualization, and early stopping without touching core
//! solver logic.

use crate::{branching::decision::Decision, state::SearchState, stats::BnbSolverStatistics};
use bollard_model::{model::Model, solution::Solution};
use bollard_search::monitor::search_monitor::SearchCommand;
use num_traits::{PrimInt, Signed};

/// Reasons for pruning a search state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PruneReason {
    /// The subtree is infeasible.
    Infeasible,
    /// The subtree is dominated by the current bound.
    BoundDominated,
}

impl std::fmt::Display for PruneReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PruneReason::Infeasible => write!(f, "Infeasible"),
            PruneReason::BoundDominated => write!(f, "BoundDominated"),
        }
    }
}

/// Trait for monitoring and controlling the search process of the solver.
pub trait TreeSearchMonitor<T>
where
    T: PrimInt + Signed,
{
    /// Returns the name of the monitor.
    fn name(&self) -> &str;
    /// Called when the search starts.
    fn on_enter_search(&mut self, model: &Model<T>, statistics: &BnbSolverStatistics);
    /// Called when the search ends.
    fn on_exit_search(&mut self, statistics: &BnbSolverStatistics);
    /// Called to determine the next action of the search.
    fn search_command(
        &mut self,
        _state: &SearchState<T>,
        _statistics: &BnbSolverStatistics,
    ) -> SearchCommand {
        SearchCommand::Continue
    }
    /// Called at each step of the search.
    fn on_step(&mut self, state: &SearchState<T>, statistics: &BnbSolverStatistics);
    /// Called when a lower bound is computed for a search state.
    /// `lower_bound` is the computed lower bound,
    /// `estimated_remaining` is an estimate of the remaining cost
    /// to reach a solution from this state.
    fn on_lower_bound_computed(
        &mut self,
        state: &SearchState<T>,
        lower_bound: T,
        estimated_remaining: T,
        statistics: &BnbSolverStatistics,
    );
    /// Called when a search state is pruned.
    fn on_prune(
        &mut self,
        state: &SearchState<T>,
        reason: PruneReason,
        statistics: &BnbSolverStatistics,
    );
    /// Called when decisions are enqueued for exploration.
    fn on_decisions_enqueued(
        &mut self,
        state: &SearchState<T>,
        count: usize,
        statistics: &BnbSolverStatistics,
    );
    /// Called when descending into a child state.
    fn on_descend(
        &mut self,
        state: &SearchState<T>,
        decision: Decision<T>,
        statistics: &BnbSolverStatistics,
    );
    /// Called when backtracking to a parent state.
    fn on_backtrack(&mut self, state: &SearchState<T>, statistics: &BnbSolverStatistics);
    /// Called when a new solution is found.
    fn on_solution_found(&mut self, solution: &Solution<T>, statistics: &BnbSolverStatistics);
}

impl<T> std::fmt::Debug for dyn TreeSearchMonitor<T>
where
    T: PrimInt + Signed,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SearchMonitor({})", self.name())
    }
}

impl<T> std::fmt::Display for dyn TreeSearchMonitor<T>
where
    T: PrimInt + Signed,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SearchMonitor({})", self.name())
    }
}
