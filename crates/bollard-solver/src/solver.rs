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

//! # Portfolio-Orchestrated Solver
//!
//! A high-level orchestrator that runs multiple solver strategies in parallel,
//! manages a shared incumbent, and enforces global termination criteria via
//! pluggable monitors (time limit, solution count, external interrupt).
//!
//! ## Motivation
//!
//! Different strategies perform better on different instances. This solver
//! coordinates a portfolio of strategies, letting them compete to install the
//! best solution while respecting global limits and early-stop signals when
//! optimality is proven elsewhere.
//!
//! ## Highlights
//!
//! - Portfolio execution:
//!   - Spawn each `PortofolioSolver<T>` in a thread using `std::thread::scope`.
//!   - Build a `CompositeMonitor<T>` per thread with interrupt, solution-limit,
//!     and optional time-limit monitors.
//! - Shared state:
//!   - `SharedIncumbent<T>` stores the best solution (atomic upper bound + mutex snapshot).
//!   - Global counters (`AtomicU64`) for solutions found; `AtomicBool` stop signal.
//! - Outcome construction:
//!   - Aggregates thread results, determines best global solution,
//!     and returns `SolverOutcome<T>` with statistics and termination reason.
//! - Builder pattern:
//!   - `SolverBuilder` to configure solution/time limits and add portfolio solvers.
//!
//! ## Usage
//!
//! ```rust
//! use bollard_search::result::SolverOutcome;
//! use bollard_solver::solver::{SolverBuilder};
//! use bollard_search::result::SolverResult;
//! use bollard_model::model::Model;
//!
//! // Build a model and portfolio solvers elsewhere...
//! // let model: Model<i64> = ...;
//! // let s1 = ...; let s2 = ...; // implementors of PortofolioSolver<i64>
//!
//! let mut solver = SolverBuilder::<i64>::new()
//!     // .with_solution_limit(10)
//!     // .with_time_limit(std::time::Duration::from_secs(30))
//!     // .add_solver(s1)
////!     // .add_solver(s2)
//!     .build();
//!
//! // Run the solver
//! // let outcome = solver.solve(&model);
//! // println!("{}", outcome);
//! // assert!(matches!(outcome.result(), SolverResult::Optimal(_)));
//! ```

use bollard_model::model::Model;
use bollard_search::{
    incumbent::SharedIncumbent,
    monitor::{
        composite::CompositeMonitor, interrupt::InterruptMonitor, solution::SolutionMonitor,
        time_limit::TimeLimitMonitor,
    },
    num::SolverNumeric,
    portfolio::{PortfolioSolverContext, PortfolioSolverResult, PortofolioSolver},
    result::{SolverOutcome, SolverResult, TerminationReason},
    stats::{SolverStatistics, SolverStatisticsBuilder},
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub struct Solver<'a, T> {
    portfolio_solver: Vec<Box<dyn PortofolioSolver<T> + 'a>>,
    incumbent: SharedIncumbent<T>,
    global_solution_count: AtomicU64,
    /// Shared flag to signal all solvers to stop (e.g., when optimality is proven).
    stop_signal: AtomicBool,
    solution_limit: Option<u64>,
    time_limit: Option<std::time::Duration>,
}

impl<'a, T> Solver<'a, T>
where
    T: SolverNumeric,
{
    #[inline]
    pub fn add_solver<S>(&mut self, solver: S)
    where
        S: PortofolioSolver<T> + 'a,
    {
        self.portfolio_solver.push(Box::new(solver));
    }

    #[inline]
    pub fn add_solver_boxed(&mut self, solver: Box<dyn PortofolioSolver<T> + 'a>) {
        self.portfolio_solver.push(solver);
    }

    #[inline]
    pub fn incumbent(&self) -> &SharedIncumbent<T> {
        &self.incumbent
    }

    #[inline]
    pub fn solution_limit(&self) -> Option<u64> {
        self.solution_limit
    }

    #[inline]
    pub fn has_solution_limit(&self) -> bool {
        self.solution_limit.is_some()
    }

    #[inline]
    pub fn time_limit(&self) -> Option<std::time::Duration> {
        self.time_limit
    }

    #[inline]
    pub fn has_time_limit(&self) -> bool {
        self.time_limit.is_some()
    }

    pub fn solve(&mut self, model: &Model<T>) -> SolverOutcome<T> {
        assert!(
            !self.portfolio_solver.is_empty(),
            "called `Solver::solve` with no portfolio solvers added"
        );

        let start_time = std::time::Instant::now();

        // 1. Reset State for this run
        self.stop_signal.store(false, Ordering::Relaxed);
        self.global_solution_count.store(0, Ordering::Relaxed);

        // 2. Run Parallel Solvers
        let results = self.run_portfolio_parallel(model);

        // 3. Construct and Return Outcome
        self.construct_outcome(start_time, results)
    }

    /// Internal helper to spawn threads and collect results.
    fn run_portfolio_parallel(&mut self, model: &Model<T>) -> Vec<PortfolioSolverResult<T>> {
        // Capture references for threads
        let solution_limit = self.solution_limit;
        let time_limit = self.time_limit;
        let incumbent = &self.incumbent;
        let global_solution_count = &self.global_solution_count;
        let stop_signal = &self.stop_signal;

        let mut results = Vec::with_capacity(self.portfolio_solver.len());

        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(self.portfolio_solver.len());

            for solver in &mut self.portfolio_solver {
                let handle = scope.spawn(move || {
                    // 1. Build the monitor stack
                    let mut monitor = CompositeMonitor::<T>::new();

                    // Always add the interrupt monitor so this thread can be stopped
                    // if another thread finishes early.
                    monitor.add_monitor(InterruptMonitor::new(stop_signal));
                    monitor.add_monitor(SolutionMonitor::new(global_solution_count, solution_limit));

                    if let Some(limit) = time_limit {
                        monitor.add_monitor(TimeLimitMonitor::new(limit));
                    }

                    // 2. Run the solver
                    let ctx = PortfolioSolverContext::new(model, incumbent, &mut monitor);
                    let result = solver.invoke(ctx);

                    // 3. Signal stop if optimal
                    if matches!(result.result(), SolverResult::Optimal(_)) {
                        let name = solver.name();
                        println!("Portfolio solver '{}' found optimal solution, signaling stop to other solvers.", name);
                        stop_signal.store(true, Ordering::Relaxed);
                    }

                    result
                });
                handles.push(handle);
            }

            for handle in handles {
                results.push(handle.join().expect("portfolio solver thread panicked"));
            }
        });

        results
    }

    /// Finds the absolute best solution among all thread results and the shared incumbent.
    fn find_best_solution(
        &self,
        results: &[PortfolioSolverResult<T>],
    ) -> Option<bollard_model::solution::Solution<T>> {
        let thread_solutions = results.iter().filter_map(|r| match r.result() {
            SolverResult::Optimal(s) | SolverResult::Feasible(s) => Some(s),
            _ => None,
        });

        let incumbent_snapshot = self.incumbent.snapshot();

        thread_solutions
            .chain(incumbent_snapshot.as_ref())
            .min_by_key(|s| s.objective_value())
            .cloned()
    }

    fn build_statistics(
        &self,
        start_time: std::time::Instant,
        used_threads: usize,
    ) -> SolverStatistics {
        SolverStatisticsBuilder::new()
            .solutions_found(self.global_solution_count.load(Ordering::Relaxed))
            .used_threads(used_threads)
            .solve_duration(start_time.elapsed())
            .build()
    }

    fn construct_outcome(
        &self,
        start_time: std::time::Instant,
        results: Vec<PortfolioSolverResult<T>>,
    ) -> SolverOutcome<T> {
        let stats = self.build_statistics(start_time, results.len());

        // 1. Always identify the best solution globally first.
        let best_solution = self.find_best_solution(&results);

        // 2. Check if ANY thread mathematically proved the global optimum.
        let optimality_proven = results
            .iter()
            .any(|r| matches!(r.result(), SolverResult::Optimal(_)));

        // 3. Hierarchy: Optimality > Infeasibility > Aborted
        if let Some(sol) = best_solution {
            if optimality_proven {
                return SolverOutcome::optimal(sol, stats);
            }
            // If we have a solution but no proof, it's the best "Feasible" one.
            let reason = self.determine_abort_reason(&results);
            return SolverOutcome::feasible(sol, reason, stats);
        }

        // 4. If no solution was found, was it because it's impossible?
        if results
            .iter()
            .any(|r| matches!(r.result(), SolverResult::Infeasible))
        {
            return SolverOutcome::infeasible(stats);
        }

        // 5. Fallback: Unknown
        let reason = self.determine_abort_reason(&results);
        SolverOutcome::unknown(reason, stats)
    }

    fn determine_abort_reason(&self, results: &[PortfolioSolverResult<T>]) -> String {
        // 1. Explicit Monitor trigger (Time/Solution limit)
        if let Some(msg) = results.iter().find_map(|res| {
            if let TerminationReason::Aborted(msg) = res.termination_reason() {
                Some(msg.clone())
            } else {
                None
            }
        }) {
            return msg;
        }

        // 2. Global signal (Ctrl+C or Optimality found elsewhere)
        if self.stop_signal.load(Ordering::Relaxed) {
            return "external interrupt".to_string();
        }

        // 3. Natural Exhaustion (Heuristic finished its work)
        "search space exhausted without proof".to_string()
    }
}

pub struct SolverBuilder<'a, T> {
    portfolio_solver: Vec<Box<dyn PortofolioSolver<T> + 'a>>,
    solution_limit: Option<u64>,
    time_limit: Option<std::time::Duration>,
}

impl<'a, T> Default for SolverBuilder<'a, T>
where
    T: SolverNumeric,
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> SolverBuilder<'a, T>
where
    T: SolverNumeric,
{
    #[inline]
    pub fn new() -> Self {
        Self {
            portfolio_solver: Vec::new(),
            solution_limit: None,
            time_limit: None,
        }
    }

    #[inline]
    pub fn with_solution_limit(mut self, limit: u64) -> Self {
        self.solution_limit = Some(limit);
        self
    }

    #[inline]
    pub fn with_time_limit(mut self, limit: std::time::Duration) -> Self {
        self.time_limit = Some(limit);
        self
    }

    #[inline]
    pub fn add_solver<S>(mut self, solver: S) -> Self
    where
        S: PortofolioSolver<T> + 'a,
    {
        self.portfolio_solver.push(Box::new(solver));
        self
    }

    #[inline]
    pub fn build(self) -> Solver<'a, T> {
        Solver {
            portfolio_solver: self.portfolio_solver,
            incumbent: SharedIncumbent::new(),
            global_solution_count: AtomicU64::new(0),
            stop_signal: AtomicBool::new(false),
            solution_limit: self.solution_limit,
            time_limit: self.time_limit,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bollard_bnb::{
        branching::{
            chronological::ChronologicalExhaustiveBuilder, fcfs::FcfsHeuristicBuilder,
            regret::RegretHeuristicBuilder, wspt::WsptHeuristicBuilder,
        },
        eval::hybrid::HybridEvaluator,
        portfolio::BnbPortfolioSolver,
    };
    use bollard_model::{
        index::{BerthIndex, VesselIndex},
        model::ModelBuilder,
        time::ProcessingTime,
    };

    type IntegerType = i64;

    fn build_model(
        num_berths: usize,
        num_vessels: usize,
    ) -> bollard_model::model::Model<IntegerType> {
        let mut builder = ModelBuilder::<IntegerType>::new(num_berths, num_vessels);
        for v in 0..num_vessels {
            let vi = VesselIndex::new(v);
            builder.set_vessel_arrival_time(vi, (v as IntegerType) * 3);
            builder.set_vessel_weight(vi, 1 + (v as IntegerType % 5));
        }
        for v in 0..num_vessels {
            let vi = VesselIndex::new(v);
            for b in 0..num_berths {
                let bi = BerthIndex::new(b);
                let base = if b % 2 == 0 { 10 } else { 7 };
                let span = if b % 2 == 0 { 3 } else { 4 };
                let duration = base + (v as IntegerType % span);
                builder.set_vessel_processing_time(vi, bi, ProcessingTime::some(duration));
            }
        }
        builder.build()
    }

    #[test]
    fn test_portfolio_solver() {
        let model = build_model(2, 10);

        let first_solver = BnbPortfolioSolver::new(
            RegretHeuristicBuilder::<IntegerType>::preallocated(
                model.num_berths(),
                model.num_vessels(),
            ),
            HybridEvaluator::preallocated(model.num_berths(), model.num_vessels()),
        );

        let second_solver = BnbPortfolioSolver::new(
            WsptHeuristicBuilder::<IntegerType>::preallocated(
                model.num_berths(),
                model.num_vessels(),
            ),
            HybridEvaluator::preallocated(model.num_berths(), model.num_vessels()),
        );

        let third_solver = BnbPortfolioSolver::new(
            ChronologicalExhaustiveBuilder::new(),
            HybridEvaluator::<i64>::preallocated(model.num_berths(), model.num_vessels()),
        );

        let fourth_solver = BnbPortfolioSolver::new(
            FcfsHeuristicBuilder::<IntegerType>::preallocated(
                model.num_berths(),
                model.num_vessels(),
            ),
            HybridEvaluator::preallocated(model.num_berths(), model.num_vessels()),
        );

        let mut solver = SolverBuilder::<IntegerType>::new()
            .add_solver(first_solver)
            .add_solver(second_solver)
            .add_solver(third_solver)
            .add_solver(fourth_solver)
            .build();

        let outcome = solver.solve(&model);
        println!("Solver statistics: {}", outcome.statistics());
        println!("{}", outcome.reason());
        assert!(outcome.is_optimal());

        // Objective (Gurobi) = 855
        match outcome.result() {
            SolverResult::Infeasible => panic!("expected optimal solution, got infeasible"),
            SolverResult::Optimal(solution) => {
                assert_eq!(solution.objective_value(), 855);
            }
            SolverResult::Feasible(solution) => {
                panic!(
                    "expected optimal solution, got feasible with objective {}",
                    solution.objective_value()
                );
            }
            SolverResult::Unknown => panic!("expected optimal solution, got unknown"),
        }
    }
}
