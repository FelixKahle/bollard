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
    incumbent::NoSharedIncumbent,
    memory::SearchMemory,
    meta::metaheuristic::Metaheuristic,
    monitor::local_search_monitor::LocalSearchMonitor,
    operator::local_search_operator::LocalSearchOperator,
    result::{LocalSearchEngineOutcome, LocalSearchTerminationReason},
    stats::LocalSearchStatistics,
};
use bollard_model::{model::Model, solution::Solution};
use bollard_search::{
    incumbent::SharedIncumbent, monitor::search_monitor::SearchCommand,
    neighborhood::neighborhoods::Neighborhoods, num::SolverNumeric,
};

/// Parameters for a local search run.
pub struct LocalSearchParams<'a, N, H, D, O, M, T>
where
    T: SolverNumeric,
{
    pub model: &'a Model<T>,
    pub decoder: &'a mut D,
    pub neighborhood: &'a N,
    pub operator: &'a mut O,
    pub metaheuristic: &'a mut H,
    pub monitor: M,
    pub initial_solution: &'a Solution<T>,
}

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

    /// Runs with a shared incumbent: installs every new best immediately.
    ///
    /// Thin wrapper that forwards to the single internal implementation with a
    /// `SharedIncumbentAdapter`, keeping code duplication minimal.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn run_with_incumbent<N, H, D, O, M>(
        &mut self,
        params: LocalSearchParams<N, H, D, O, M, T>,
        shared: &SharedIncumbent<T>,
    ) -> LocalSearchEngineOutcome<T>
    where
        N: Neighborhoods,
        H: Metaheuristic<T>,
        D: Decoder<T, H::Evaluator>,
        O: LocalSearchOperator<T, N>,
        M: LocalSearchMonitor<T>,
    {
        let mut store = crate::incumbent::SharedIncumbentAdapter::new(shared);
        self.run_internal(params, &mut store)
    }

    /// Runs the local search engine to improve an initial solution (no incumbent).
    ///
    /// For zero-cost when no incumbent sharing is needed, this forwards into the
    /// single internal implementation using a no-op store.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn run<N, H, D, O, M>(
        &mut self,
        params: LocalSearchParams<N, H, D, O, M, T>,
    ) -> LocalSearchEngineOutcome<T>
    where
        N: Neighborhoods,
        H: Metaheuristic<T>,
        D: Decoder<T, H::Evaluator>,
        O: LocalSearchOperator<T, N>,
        M: LocalSearchMonitor<T>,
    {
        let mut store = NoSharedIncumbent::<T>::new();
        self.run_internal(params, &mut store)
    }

    /// Private internal run with incumbent abstraction.
    ///
    /// - Single source of truth for the local search loop.
    /// - Publishes every new best via `store.on_best_solution`.
    /// - Public `run` and `run_with_incumbent` forward into this method.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn run_internal<N, H, D, O, M, S>(
        &mut self,
        params: LocalSearchParams<N, H, D, O, M, T>,
        store: &mut S,
    ) -> LocalSearchEngineOutcome<T>
    where
        N: Neighborhoods,
        H: Metaheuristic<T>,
        D: Decoder<T, H::Evaluator>,
        O: LocalSearchOperator<T, N>,
        M: LocalSearchMonitor<T>,
        S: crate::incumbent::IncumbentStore<T>,
    {
        let LocalSearchParams {
            model,
            decoder,
            neighborhood,
            operator,
            metaheuristic,
            mut monitor,
            initial_solution,
        } = params;

        assert!(
            model.num_vessels() == neighborhood.num_vessels(),
            "called `LocalSearchEngine::run_internal` with inconsistent number of vessels: model has {}, neighborhood has {}",
            model.num_vessels(),
            neighborhood.num_vessels()
        );
        assert!(
            model.num_vessels() == initial_solution.num_vessels(),
            "called `LocalSearchEngine::run_internal` with inconsistent number of vessels: model has {}, initial solution has {}",
            model.num_vessels(),
            initial_solution.num_vessels()
        );
        assert!(
            neighborhood.num_vessels() == initial_solution.num_vessels(),
            "called `LocalSearchEngine::run_internal` with inconsistent number of vessels: neighborhood has {}, initial solution has {}",
            neighborhood.num_vessels(),
            initial_solution.num_vessels()
        );

        let start_time = std::time::Instant::now();
        let mut stats = LocalSearchStatistics::default();

        // Initialize memory with the initial solution
        self.memory.initialize(initial_solution);
        // Best solution found so far. Starts as the initial solution.
        let mut best_solution = self.memory.current_schedule().clone();

        debug_assert!(
            model.num_vessels() == self.memory.num_vessels()
                && self.memory.current_schedule().num_vessels() == initial_solution.num_vessels(),
            "called `LocalSearchEngine::run_internal` with inconsistent number of vessels: model has {}, memory has {}, current schedule has {}, initial solution has {}",
            model.num_vessels(),
            self.memory.num_vessels(),
            self.memory.current_schedule().num_vessels(),
            initial_solution.num_vessels()
        );

        // Prepare the decoder
        decoder.initialize(model);

        monitor.on_start(model, self.memory.current_schedule());
        metaheuristic.on_start(model, self.memory.current_schedule());

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

            if let SearchCommand::Terminate(reason) = metaheuristic.search_command(
                stats.iterations,
                model,
                self.memory.current_schedule(),
            ) {
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
                let evaluator = metaheuristic.evaluator();
                decoder.decode_unchecked(model, queue, candidate, evaluator)
            };

            if !decoded {
                self.memory.finalize(false);
                continue;
            }

            debug_assert!(
                self.memory.candidate_schedule().num_vessels() == self.memory.num_vessels()
                    && self.memory.num_vessels() == model.num_vessels(),
                "called `LocalSearchEngine::run_internal` with inconsistent number of vessels after decoding: candidate schedule has {}, memory has {}, model has {}",
                self.memory.candidate_schedule().num_vessels(),
                self.memory.num_vessels(),
                model.num_vessels()
            );

            stats.on_found_solution();
            monitor.on_solution_found(self.memory.candidate_schedule(), &stats);

            let accept = metaheuristic.should_accept(
                model,
                self.memory.current_schedule(),
                self.memory.candidate_schedule(),
                &best_solution,
            );

            if accept {
                self.memory.accept_current();
                stats.on_accepted_solution();

                debug_assert!(
                    self.memory.current_schedule().num_vessels() == self.memory.num_vessels(),
                    "called `LocalSearchEngine::run_internal` with inconsistent number of vessels after acceptance: current schedule has {}, memory has {}",
                    self.memory.current_schedule().num_vessels(),
                    self.memory.num_vessels()
                );

                metaheuristic.on_accept(model, self.memory.current_schedule());
                monitor.on_solution_accepted(self.memory.current_schedule(), &stats);

                if self.memory.current_schedule().objective_value()
                    < best_solution.objective_value()
                {
                    best_solution = self.memory.current_schedule().clone();

                    debug_assert!(
                        best_solution.objective_value()
                            <= self.memory.current_schedule().objective_value(),
                        "called `LocalSearchEngine::run_internal` with inconsistent best solution objective value: best solution has {}, current schedule has {}",
                        best_solution.objective_value(),
                        self.memory.current_schedule().objective_value()
                    );

                    metaheuristic.on_new_best(model, &best_solution);
                    monitor.on_best_solution_updated(&best_solution, &stats);

                    // Publish to incumbent store immediately (no-op for NoSharedIncumbent)
                    store.on_best_solution(&best_solution);
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
                    "called `LocalSearchEngine::run_internal` with inconsistent queue length after rejection: before was {}, now is {}",
                    queue_len_before,
                    self.memory.num_vessels()
                );

                metaheuristic.on_reject(model, self.memory.candidate_schedule());
                monitor.on_solution_rejected(self.memory.candidate_schedule(), &stats);
            }

            monitor.on_iteration(self.memory.current_schedule(), &stats);
        };

        stats.set_total_time(start_time.elapsed());
        monitor.on_end(&best_solution, &stats);
        let final_solution: Solution<T> = best_solution;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        decoder::GreedyDecoder,
        eval::DefaultAssignmentEvaluator,
        meta::greedy_descent::GreedyDescent,
        monitor::{
            composite::CompositeLocalSearchMonitor, solution::SolutionLimitMonitor,
            time::TimeLimitMonitor,
        },
        operator::swap::SwapOperator,
        result::LocalSearchTerminationReason,
    };
    use bollard_model::{
        index::{BerthIndex, VesselIndex},
        model::ModelBuilder,
        solution::Solution,
        time::ProcessingTime,
    };
    use bollard_search::{incumbent::SharedIncumbent, neighborhood::topology::StaticTopology};
    use std::time::Duration;

    fn vi(i: usize) -> VesselIndex {
        VesselIndex::new(i)
    }
    fn bi(i: usize) -> BerthIndex {
        BerthIndex::new(i)
    }

    fn build_basic_model() -> bollard_model::model::Model<i64> {
        let mut bldr = ModelBuilder::<i64>::new(1, 3);
        bldr.add_berth_closing_time(
            bi(0),
            bollard_core::math::interval::ClosedOpenInterval::new(0, 1000),
        );

        bldr.set_vessel_arrival_time(vi(0), 0)
            .set_vessel_latest_departure_time(vi(0), 1000)
            .set_vessel_weight(vi(0), 1);
        bldr.set_vessel_arrival_time(vi(1), 1)
            .set_vessel_latest_departure_time(vi(1), 1000)
            .set_vessel_weight(vi(1), 1);
        bldr.set_vessel_arrival_time(vi(2), 2)
            .set_vessel_latest_departure_time(vi(2), 1000)
            .set_vessel_weight(vi(2), 1);

        bldr.set_vessel_processing_time(vi(0), bi(0), ProcessingTime::some(5))
            .set_vessel_processing_time(vi(1), bi(0), ProcessingTime::some(7))
            .set_vessel_processing_time(vi(2), bi(0), ProcessingTime::some(3));

        bldr.build()
    }

    fn initial_solution_from_order(
        model: &bollard_model::model::Model<i64>,
        order: &[usize],
    ) -> Solution<i64> {
        let num_vessels = model.num_vessels();
        assert_eq!(num_vessels, order.len());
        let berths = vec![bi(0); num_vessels];
        let mut starts = vec![0_i64; num_vessels];
        for (pos, &vidx) in order.iter().enumerate() {
            starts[vidx] = pos as i64;
        }
        Solution::new(0, berths, starts)
    }

    #[test]
    fn test_engine_run_local_optimum_with_greedy_and_swap_static_topology() {
        let model = build_basic_model();
        let topology = StaticTopology::from_model(&model);

        let mut mh: GreedyDescent<i64> = GreedyDescent::new();
        let mut dec: GreedyDecoder<i64, DefaultAssignmentEvaluator<i64>> =
            GreedyDecoder::preallocated(model.num_berths());
        let mut op: SwapOperator<i64, StaticTopology> = SwapOperator::new();

        let init = initial_solution_from_order(&model, &[2, 1, 0]);
        let composite = CompositeLocalSearchMonitor::<i64>::new();

        let mut engine = LocalSearchEngine::<i64>::new();
        let params = LocalSearchParams {
            model: &model,
            decoder: &mut dec,
            neighborhood: &topology,
            operator: &mut op,
            metaheuristic: &mut mh,
            monitor: composite,
            initial_solution: &init,
        };
        let out = engine.run(params);

        match out.termination_reason() {
            LocalSearchTerminationReason::LocalOptimum => {}
            other => panic!("unexpected termination: {:?}", other),
        }

        let sol = out.solution();
        assert_eq!(sol.num_vessels(), model.num_vessels());
        assert_eq!(sol.berths().len(), model.num_vessels());
        assert_eq!(sol.start_times().len(), model.num_vessels());
        assert!(sol.objective_value() >= 0);
    }

    #[test]
    fn test_monitor_solution_limit_terminates_search() {
        let model = build_basic_model();
        let topology = StaticTopology::from_model(&model);
        let mut mh: GreedyDescent<i64> = GreedyDescent::new();
        let mut dec: GreedyDecoder<i64, DefaultAssignmentEvaluator<i64>> =
            GreedyDecoder::preallocated(model.num_berths());
        let mut op: SwapOperator<i64, StaticTopology> = SwapOperator::new();

        let init = initial_solution_from_order(&model, &[0, 1, 2]);

        // Terminate immediately at the first search_command check
        let limit_monitor = SolutionLimitMonitor::new(0);
        let mut composite = CompositeLocalSearchMonitor::<i64>::with_capacity(1);
        composite.add_monitor(limit_monitor);

        let mut engine = LocalSearchEngine::<i64>::new();
        let params = LocalSearchParams {
            model: &model,
            decoder: &mut dec,
            neighborhood: &topology,
            operator: &mut op,
            metaheuristic: &mut mh,
            monitor: composite,
            initial_solution: &init,
        };
        let out = engine.run(params);

        match out.termination_reason() {
            LocalSearchTerminationReason::Aborted(msg) => {
                assert!(
                    msg.contains("Solution limit"),
                    "unexpected aborted message: {}",
                    msg
                );
            }
            other => panic!(
                "expected early termination via monitor (Aborted), got {:?}",
                other
            ),
        }

        let sol = out.solution();
        assert_eq!(sol.num_vessels(), model.num_vessels());
    }

    #[test]
    fn test_time_limit_monitor_terminates_quickly_with_mask() {
        let model = build_basic_model();
        let topology = StaticTopology::from_model(&model);
        let mut mh: GreedyDescent<i64> = GreedyDescent::new();
        let mut dec: GreedyDecoder<i64, DefaultAssignmentEvaluator<i64>> =
            GreedyDecoder::preallocated(model.num_berths());
        let mut op: SwapOperator<i64, StaticTopology> = SwapOperator::new();

        let init = initial_solution_from_order(&model, &[0, 2, 1]);

        // Use a near-zero time limit and aggressive mask to check immediately
        let tlm = TimeLimitMonitor::with_mask(Duration::from_millis(0), 0);
        let mut composite = CompositeLocalSearchMonitor::<i64>::with_capacity(1);
        composite.add_monitor(tlm);

        let mut engine = LocalSearchEngine::<i64>::new();
        let params = LocalSearchParams {
            model: &model,
            decoder: &mut dec,
            neighborhood: &topology,
            operator: &mut op,
            metaheuristic: &mut mh,
            monitor: composite,
            initial_solution: &init,
        };
        let out = engine.run(params);

        match out.termination_reason() {
            LocalSearchTerminationReason::Aborted(msg) => {
                assert_eq!(msg, "time limit exceeded");
            }
            other => panic!(
                "expected Aborted(\"time limit exceeded\") from time limit monitor, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_run_with_incumbent_publishes_best() {
        let model = build_basic_model();
        let topology = StaticTopology::from_model(&model);

        let mut mh: GreedyDescent<i64> = GreedyDescent::new();
        let mut dec: GreedyDecoder<i64, DefaultAssignmentEvaluator<i64>> =
            GreedyDecoder::preallocated(model.num_berths());
        let mut op: SwapOperator<i64, StaticTopology> = SwapOperator::new();

        let init = initial_solution_from_order(&model, &[2, 1, 0]);
        let composite = CompositeLocalSearchMonitor::<i64>::new();

        let shared = SharedIncumbent::<i64>::new();

        let mut engine = LocalSearchEngine::<i64>::new();
        let params = LocalSearchParams {
            model: &model,
            decoder: &mut dec,
            neighborhood: &topology,
            operator: &mut op,
            metaheuristic: &mut mh,
            monitor: composite,
            initial_solution: &init,
        };
        let out = engine.run_with_incumbent(params, &shared);

        match out.termination_reason() {
            LocalSearchTerminationReason::LocalOptimum
            | LocalSearchTerminationReason::Metaheuristic(_)
            | LocalSearchTerminationReason::Aborted(_) => {}
        }
        let final_sol = out.solution();
        assert_eq!(final_sol.num_vessels(), model.num_vessels());
    }

    #[test]
    fn test_operator_prepare_and_iteration_integration() {
        let model = build_basic_model();
        let topology = StaticTopology::from_model(&model);

        let mut mh: GreedyDescent<i64> = GreedyDescent::new();
        let mut dec: GreedyDecoder<i64, DefaultAssignmentEvaluator<i64>> =
            GreedyDecoder::preallocated(model.num_berths());

        struct CountingSwap<T, N> {
            inner: SwapOperator<T, N>,
            prepares: usize,
        }
        impl<T, N> CountingSwap<T, N> {
            fn new() -> Self {
                Self {
                    inner: SwapOperator::new(),
                    prepares: 0,
                }
            }
            fn prepares(&self) -> usize {
                self.prepares
            }
        }
        impl<T, N> crate::operator::local_search_operator::LocalSearchOperator<T, N> for CountingSwap<T, N>
        where
            T: bollard_search::num::SolverNumeric,
            N: bollard_search::neighborhood::neighborhoods::Neighborhoods,
        {
            fn name(&self) -> &str {
                "CountingSwap"
            }
            fn prepare(
                &mut self,
                schedule: &Solution<T>,
                queue: &crate::queue::VesselPriorityQueue,
                n: &N,
            ) {
                self.prepares += 1;
                self.inner.prepare(schedule, queue, n);
            }
            fn next_neighbor(
                &mut self,
                schedule: &Solution<T>,
                mutator: &mut crate::mutator::Mutator<T>,
                n: &N,
            ) -> bool {
                self.inner.next_neighbor(schedule, mutator, n)
            }
            fn reset(&mut self) {
                self.inner.reset();
            }
        }

        let mut op = CountingSwap::<i64, StaticTopology>::new();
        let init = initial_solution_from_order(&model, &[0, 2, 1]);
        let composite = CompositeLocalSearchMonitor::<i64>::new();

        let mut engine = LocalSearchEngine::<i64>::new();
        let params = LocalSearchParams {
            model: &model,
            decoder: &mut dec,
            neighborhood: &topology,
            operator: &mut op,
            metaheuristic: &mut mh,
            monitor: composite,
            initial_solution: &init,
        };
        let out = engine.run(params);

        assert!(op.prepares() >= 1, "operator.prepare was not called");

        match out.termination_reason() {
            LocalSearchTerminationReason::LocalOptimum => {}
            other => panic!("expected local optimum termination, got {:?}", other),
        }
    }
}
