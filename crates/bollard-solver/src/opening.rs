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

use bollard_bnb::{
    bnb::BnbSolver, branching::edf::EarliestDeadlineFirstBuilder, eval::hybrid::HybridEvaluator,
    monitor::solution::SolutionLimitMonitor,
};
use bollard_model::{model::Model, solution::Solution};
use bollard_search::num::SolverNumeric;

/// Finds an initial feasible solution for the given model using
/// a branch-and-bound solver with an earliest-deadline-first
/// branching strategy and a hybrid evaluator.
#[inline]
pub fn find_initial_solution<T>(model: &Model<T>) -> Option<Solution<T>>
where
    T: SolverNumeric,
{
    let num_vessels = model.num_vessels();
    let num_berths = model.num_berths();

    let mut bnb_solver = BnbSolver::preallocated(num_berths, num_vessels);
    let mut builder = EarliestDeadlineFirstBuilder::preallocated(num_berths, num_vessels);
    let mut evaluator = HybridEvaluator::preallocated(num_berths, num_vessels);
    let solution_limit_monitor = SolutionLimitMonitor::new(1);
    let outcome = bnb_solver.solve(model, &mut builder, &mut evaluator, solution_limit_monitor);

    match outcome.result() {
        bollard_search::result::SolverResult::Infeasible => None,
        bollard_search::result::SolverResult::Optimal(solution) => Some(solution.clone()),
        bollard_search::result::SolverResult::Feasible(solution) => Some(solution.clone()),
        bollard_search::result::SolverResult::Unknown => None,
    }
}
