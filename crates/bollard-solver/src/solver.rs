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

use bollard_model::model::Model;
use bollard_search::{
    incumbent::SharedIncumbent,
    monitor::{
        composite::CompositeMonitor, interrupt::InterruptMonitor,
        solution_limit::SolutionLimitMonitor, time_limit::TimeLimitMonitor,
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

                    if let Some(limit) = solution_limit {
                        monitor
                            .add_monitor(SolutionLimitMonitor::new(global_solution_count, limit));
                    }
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

    /// Aggregates results, builds statistics, and determines the final outcome.
    fn construct_outcome(
        &self,
        start_time: std::time::Instant,
        results: Vec<PortfolioSolverResult<T>>,
    ) -> SolverOutcome<T> {
        let stats = self.build_statistics(start_time, results.len());

        // Find the best solution and potentially the result wrapper that contained it
        // (so we can steal its specific abort message).
        let best_result_wrapper = self.find_best_result_wrapper(&results);
        let best_solution = self.find_best_solution(&results);

        // Optimality is proven if ANY thread returned Optimal.
        let optimality_proven = results
            .iter()
            .any(|r| matches!(r.result(), SolverResult::Optimal(_)));

        // Infeasibility is proven only if ALL threads returned Infeasible.
        let all_infeasible = !results.is_empty()
            && results
                .iter()
                .all(|r| matches!(r.result(), SolverResult::Infeasible));

        if let Some(sol) = best_solution {
            if optimality_proven {
                return SolverOutcome::optimal(sol, stats);
            }
            // If we have a solution but didn't prove optimality, it's Feasible (Aborted).
            // We pass the result wrapper to see if it has a specific abort reason.
            let reason_str = self.determine_abort_reason(best_result_wrapper);
            return SolverOutcome::feasible(sol, reason_str, stats);
        }

        if all_infeasible {
            return SolverOutcome::infeasible(stats);
        }

        // No solution found -> Unknown.
        let reason_str = self.determine_abort_reason(None);
        SolverOutcome::unknown(reason_str, stats)
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

    /// Finds the `PortfolioSolverResult` corresponding to the best solution found by threads.
    /// This is used to extract the TerminationReason string from the thread that actually found the best result.
    fn find_best_result_wrapper<'r>(
        &self,
        results: &'r [PortfolioSolverResult<T>],
    ) -> Option<&'r PortfolioSolverResult<T>> {
        results
            .iter()
            .filter(|r| {
                matches!(
                    r.result(),
                    SolverResult::Optimal(_) | SolverResult::Feasible(_)
                )
            })
            .min_by_key(|r| match r.result() {
                SolverResult::Optimal(s) | SolverResult::Feasible(s) => s.objective_value(),
                _ => unreachable!("filtered above"),
            })
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

    /// Determines the reason string for an aborted/feasible outcome.
    ///
    /// Priority:
    /// 1. The specific reason returned by the solver that found the best result (if available).
    /// 2. The Stop Signal (interrupt).
    /// 3. Global Time/Solution Limits.
    fn determine_abort_reason(&self, best_result: Option<&PortfolioSolverResult<T>>) -> String {
        let reason = best_result
            // 1. Try to extract specific message from the best thread result
            .and_then(|res| match res.termination_reason() {
                TerminationReason::Aborted(msg) => Some(msg.clone()),
                _ => None,
            })
            // 2. Check for manual interrupt signal
            .or_else(|| {
                self.stop_signal
                    .load(Ordering::Relaxed)
                    .then(|| "Interrupt signal received".to_string())
            })
            // 3. Check if a time limit was configured (and presumably hit)
            .or_else(|| {
                self.time_limit
                    .map(|limit| format!("time limit reached after {:.3}s", limit.as_secs_f64()))
            })
            // 4. Check if a solution limit was configured
            .or_else(|| {
                self.solution_limit
                    .map(|limit| format!("solution limit {} reached", limit))
            })
            // 5. Fallback default
            .unwrap_or_else(|| "portfolio finished without proving optimality".to_string());

        format!("Aborted: {}", reason)
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
        eval::workload::WorkloadEvaluator,
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
            WorkloadEvaluator::preallocated(model.num_berths(), model.num_vessels()),
        );

        let second_solver = BnbPortfolioSolver::new(
            WsptHeuristicBuilder::<IntegerType>::preallocated(
                model.num_berths(),
                model.num_vessels(),
            ),
            WorkloadEvaluator::preallocated(model.num_berths(), model.num_vessels()),
        );

        let third_solver = BnbPortfolioSolver::new(
            ChronologicalExhaustiveBuilder,
            WorkloadEvaluator::preallocated(model.num_berths(), model.num_vessels()),
        );

        let fourth_solver = BnbPortfolioSolver::new(
            FcfsHeuristicBuilder::<IntegerType>::preallocated(
                model.num_berths(),
                model.num_vessels(),
            ),
            WorkloadEvaluator::preallocated(model.num_berths(), model.num_vessels()),
        );

        let mut solver = SolverBuilder::<IntegerType>::new()
            .with_time_limit(std::time::Duration::from_secs(100))
            .add_solver(first_solver)
            .add_solver(second_solver)
            .add_solver(third_solver)
            .add_solver(fourth_solver)
            .build();

        let outcome = solver.solve(&model);
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

        println!("Solver statistics: {}", outcome.statistics());
    }
}
