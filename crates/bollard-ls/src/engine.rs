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

//! Iterative local search driver for Bollard.
//!
//! The engine orchestrates neighborhood exploration, decoding, evaluation,
//! and acceptance under a chosen metaheuristic, while exposing hook points
//! for monitoring and statistics. It maintains a ping‑pong memory of the
//! current and candidate schedules, applies mutations via operators, and
//! leverages a fast decoder to evaluate feasibility and objective changes.
//! Termination can be requested by monitors or metaheuristics, and the final
//! outcome bundles the best solution discovered together with run statistics
//! and a clear termination reason.

use crate::{
    decoder::Decoder,
    eval::AssignmentEvaluator,
    memory::SearchMemory,
    metaheuristic::Metaheuristic,
    monitor::local_search_monitor::LocalSearchMonitor,
    neighborhood::neighborhoods::Neighborhoods,
    operator::LocalSearchOperator,
    result::{LocalSearchEngineOutcome, LocalSearchTerminationReason},
    stats::LocalSearchStatistics,
};
use bollard_model::{model::Model, solution::Solution};
use bollard_search::{monitor::search_monitor::SearchCommand, num::SolverNumeric};
use std::time::Instant;

/// Local search engine for berth scheduling.
///
/// The `LocalSearchEngine` coordinates memory management and the control flow of a local
/// search run. It keeps reusable `SearchMemory` buffers to minimize allocations across runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalSearchEngine<T>
where
    T: SolverNumeric,
{
    /// Persistent memory buffer, reused across multiple `run` calls to avoid allocation.
    memory: SearchMemory<T>,
}

impl<T> Default for LocalSearchEngine<T>
where
    T: SolverNumeric,
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> LocalSearchEngine<T>
where
    T: SolverNumeric,
{
    /// Creates a new engine with minimal initial capacity.
    #[inline]
    pub fn new() -> Self {
        Self {
            memory: SearchMemory::new(),
        }
    }

    /// Creates a new engine with pre‑allocated memory for a specific problem size.
    ///
    /// Use this to eliminate most runtime allocations in hot paths by pre‑sizing
    /// the internal `SearchMemory` buffers to `num_vessels`.
    ///
    /// Note: If a future run uses a larger problem, buffers will grow accordingly.
    #[inline]
    pub fn preallocated(num_vessels: usize) -> Self {
        Self {
            memory: SearchMemory::preallocated(num_vessels),
        }
    }

    /// Runs the local search engine to improve an initial solution.
    ///
    /// The engine performs iterative neighborhood exploration:
    /// - The operator proposes a mutation to the current schedule via the priority queue.
    /// - The decoder attempts to construct a feasible candidate schedule and objective delta.
    /// - The metaheuristic decides whether to accept the candidate.
    /// - The monitor observes progress and may request termination.
    ///
    /// On success, returns a `LocalSearchEngineOutcome` containing:
    /// - The best solution discovered (not necessarily the last accepted).
    /// - Aggregated run statistics.
    /// - A termination reason (local optimum, metaheuristic directive, or aborted).
    ///
    /// # Parameters:
    /// - `model`: Problem data (berths, vessels, opening times, processing times).
    /// - `decoder`: Responsible for building candidate schedules from the queue; must be initialized with `model`.
    /// - `neighborhood`: Defines neighborhoods explored by the operator.
    /// - `operator`: Applies mutations to the queue (genotype) with an undo log for rollback.
    /// - `metaheuristic`: Governs acceptance decisions and termination policy.
    /// - `monitor`: Observes iterations, solutions, and can request early termination.
    /// - `evaluator`: Computes per‑assignment score and objective delta.
    /// - `initial_solution`: Starting point for the search; also seeds memory/queue ordering.
    ///
    /// # Notes:
    /// - Internally reuses memory buffers across runs for performance.
    /// - Uses `decode_unchecked` for speed under the assumption that inputs are validated elsewhere.
    ///   In debug builds, assertions help catch inconsistencies early.
    #[allow(clippy::too_many_arguments)]
    pub fn run<E, N, M, D, O, SM>(
        &mut self,
        model: &Model<T>,
        decoder: &mut D,
        neighborhood: &N,
        operator: &mut O,
        metaheuristic: &mut M,
        monitor: &mut SM,
        evaluator: &E,
        initial_solution: &Solution<T>,
    ) -> LocalSearchEngineOutcome<T>
    where
        E: AssignmentEvaluator<T>,
        N: Neighborhoods,
        M: Metaheuristic<T>,
        D: Decoder<T, E>,
        O: LocalSearchOperator<T, N>,
        SM: LocalSearchMonitor<T>,
    {
        let start_time = Instant::now();
        let mut stats = LocalSearchStatistics::default();

        // Initialize memory with the initial solution
        self.memory.initialize(initial_solution);
        // Best solution found so far. Starts as the initial solution.
        let mut best_solution = self.memory.current_schedule().clone();

        debug_assert!(
            model.num_vessels() == self.memory.num_vessels()
                && self.memory.current_schedule().num_vessels() == initial_solution.num_vessels(),
            "called `LocalSearchEngine::run` with inconsistent number of vessels: model has {}, memory has {}, current schedule has {}, initial solution has {}",
            model.num_vessels(),
            self.memory.num_vessels(),
            self.memory.current_schedule().num_vessels(),
            initial_solution.num_vessels()
        );

        // Prepare the decoder
        decoder.initialize(model);

        monitor.on_start(self.memory.current_schedule());
        metaheuristic.on_start(self.memory.current_schedule());

        // Prepare for the first iteration
        operator.prepare(
            self.memory.current_schedule(),
            self.memory.queue(),
            neighborhood,
        );

        let termination_reason = loop {
            if let SearchCommand::Terminate(reason) = monitor.search_command(&stats) {
                break LocalSearchTerminationReason::Aborted(reason);
            }

            if let SearchCommand::Terminate(reason) =
                metaheuristic.search_command(stats.iterations, self.memory.current_schedule())
            {
                break LocalSearchTerminationReason::Metaheuristic(reason);
            }

            stats.on_iteration();

            let mutated = {
                let (current_sched, mut mutator) = self.memory.prepare_operator();
                operator.next_neighbor(current_sched, &mut mutator, neighborhood)
            };

            if !mutated {
                break LocalSearchTerminationReason::LocalOptimum;
            }

            let decoded = unsafe {
                let (queue, candidate) = self.memory.evaluation_target();
                decoder.decode_unchecked(model, queue, candidate, evaluator)
            };

            if !decoded {
                self.memory.finalize(false);
                continue;
            }

            debug_assert!(
                self.memory.candidate_schedule().num_vessels() == self.memory.num_vessels()
                    && self.memory.num_vessels() == model.num_vessels(),
                "called `LocalSearchEngine::run` with inconsistent number of vessels after decoding: candidate schedule has {}, memory has {}, model has {}",
                self.memory.candidate_schedule().num_vessels(),
                self.memory.num_vessels(),
                model.num_vessels()
            );

            stats.on_found_solution();

            let accept = metaheuristic.should_accept(
                self.memory.current_schedule(),
                self.memory.candidate_schedule(),
                &best_solution,
            );

            if accept {
                self.memory.accept_current();
                stats.on_accepted_solution();

                debug_assert!(
                    self.memory.current_schedule().num_vessels() == self.memory.num_vessels(),
                    "called `LocalSearchEngine::run` with inconsistent number of vessels after acceptance: current schedule has {}, memory has {}",
                    self.memory.current_schedule().num_vessels(),
                    self.memory.num_vessels()
                );

                metaheuristic.on_accept(self.memory.current_schedule());
                monitor.on_solution_accepted(self.memory.current_schedule(), &stats);

                if self.memory.current_schedule().objective_value()
                    < best_solution.objective_value()
                {
                    best_solution = self.memory.current_schedule().clone();

                    debug_assert!(
                        best_solution.objective_value()
                            <= self.memory.current_schedule().objective_value(),
                        "called `LocalSearchEngine::run` with inconsistent best solution objective value: best solution has {}, current schedule has {}",
                        best_solution.objective_value(),
                        self.memory.current_schedule().objective_value()
                    );

                    metaheuristic.on_new_best(&best_solution);
                    monitor.on_solution_found(&best_solution, &stats);
                }

                // Prepare for the next iteration
                operator.prepare(
                    self.memory.current_schedule(),
                    self.memory.queue(),
                    neighborhood,
                );
            } else {
                let queue_len_before = self.memory.num_vessels();
                self.memory.discard_candidate();

                debug_assert!(
                    self.memory.num_vessels() == queue_len_before,
                    "called `LocalSearchEngine::run` with inconsistent queue length after rejection: before was {}, now is {}",
                    queue_len_before,
                    self.memory.num_vessels()
                );

                metaheuristic.on_reject(self.memory.candidate_schedule());
                monitor.on_solution_rejected(self.memory.candidate_schedule(), &stats);
            }

            monitor.on_iteration(self.memory.current_schedule(), &stats);
        };

        stats.set_total_time(start_time.elapsed());
        monitor.on_end(&best_solution, &stats);
        let final_solution: Solution<T> = best_solution.into();

        match termination_reason {
            LocalSearchTerminationReason::LocalOptimum => {
                LocalSearchEngineOutcome::local_optimum(final_solution, stats)
            }
            LocalSearchTerminationReason::Metaheuristic(msg) => {
                LocalSearchEngineOutcome::metaheuristic(final_solution, msg, stats)
            }
            LocalSearchTerminationReason::Aborted(msg) => {
                LocalSearchEngineOutcome::aborted(final_solution, msg, stats)
            }
        }
    }
}
