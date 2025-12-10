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

use crate::{
    bnb::BnbSolver, branching::decision::DecisionBuilder, eval::evaluator::ObjectiveEvaluator,
    wrapper::WrapperMonitor,
};
use bollard_search::{
    num::SolverNumeric,
    portfolio::{PortfolioSolverContext, PortfolioSolverResult, PortofolioSolver},
};

#[derive(Clone)]
pub struct BnbPortfolioSolver<T, B, E>
where
    T: SolverNumeric,
    B: DecisionBuilder<T, E>,
    E: ObjectiveEvaluator<T>,
{
    inner: BnbSolver<T>,
    decision_builder: B,
    evaluator: E,
}

impl<T, B, E> BnbPortfolioSolver<T, B, E>
where
    T: SolverNumeric,
    B: DecisionBuilder<T, E>,
    E: ObjectiveEvaluator<T>,
{
    #[inline]
    pub fn new(decision_builder: B, evaluator: E) -> Self {
        Self {
            inner: BnbSolver::<T>::new(),
            decision_builder,
            evaluator,
        }
    }

    #[inline]
    pub fn preallocated(
        num_berths: usize,
        num_vessels: usize,
        decision_builder: B,
        evaluator: E,
    ) -> Self {
        Self {
            inner: BnbSolver::<T>::preallocated(num_berths, num_vessels),
            decision_builder,
            evaluator,
        }
    }

    #[inline]
    pub fn inner(&self) -> &BnbSolver<T> {
        &self.inner
    }

    #[inline]
    pub fn decision_builder(&self) -> &B {
        &self.decision_builder
    }

    #[inline]
    pub fn evaluator(&self) -> &E {
        &self.evaluator
    }
}

impl<T, B, E> PortofolioSolver<T> for BnbPortfolioSolver<T, B, E>
where
    T: SolverNumeric,
    B: DecisionBuilder<T, E>,
    E: ObjectiveEvaluator<T>,
{
    fn invoke<'a>(&mut self, context: PortfolioSolverContext<'a, T>) -> PortfolioSolverResult<T> {
        let monitor = WrapperMonitor::new(context.monitor);
        let outcome = self.inner.solve_with_incumbent(
            context.model,
            &mut self.decision_builder,
            &mut self.evaluator,
            monitor,
            context.incumbent,
        );

        outcome.into()
    }

    fn name(&self) -> &str {
        "BnbPortfolioSolver"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::branching::chronological::ChronologicalExhaustiveBuilder;
    use crate::eval::wtft::WeightedFlowTimeEvaluator;
    use bollard_model::{
        index::{BerthIndex, VesselIndex},
        model::ModelBuilder,
        time::ProcessingTime,
    };
    use bollard_search::{monitor::search_monitor::DummyMonitor, result::SolverResult};

    type IntegerType = i64;

    fn build_model(
        num_berths: usize,
        num_vessels: usize,
    ) -> bollard_model::model::Model<IntegerType> {
        let mut builder = ModelBuilder::<IntegerType>::new(num_berths, num_vessels);

        // Simple arrivals and weights
        for v in 0..num_vessels {
            let vi = VesselIndex::new(v);
            builder.set_vessel_arrival_time(vi, (v as IntegerType) * 3); // spaced arrivals
            builder.set_vessel_weight(vi, 1 + (v as IntegerType % 5)); // weights in [1..5]
        }

        // Feasible processing times for all berths
        for v in 0..num_vessels {
            let vi = VesselIndex::new(v);
            for b in 0..num_berths {
                let bi = BerthIndex::new(b);
                // Make berth-specific durations but always feasible
                let base = if b % 2 == 0 { 10 } else { 7 }; // alternate baselines
                let span = if b % 2 == 0 { 3 } else { 4 }; // alternate spans
                let duration = base + (v as IntegerType % span);
                builder.set_vessel_processing_time(vi, bi, ProcessingTime::some(duration));
            }
        }

        builder.build()
    }

    #[test]
    fn test_portfolio_bnb_solver_finds_optimal_solution() {
        // Build a non-trivial model
        let model = build_model(2, 10);

        // Construct inner solver, builder, and evaluator
        let builder = ChronologicalExhaustiveBuilder;
        let evaluator = WeightedFlowTimeEvaluator::<IntegerType>::preallocated(
            model.num_berths(),
            model.num_vessels(),
        );

        // Wrap as portfolio solver
        let mut portfolio_solver = BnbPortfolioSolver::preallocated(
            model.num_berths(),
            model.num_vessels(),
            builder,
            evaluator,
        );

        // Prepare portfolio context
        let incumbent = bollard_search::incumbent::SharedIncumbent::<IntegerType>::new();
        let mut monitor = DummyMonitor::new();
        let context = PortfolioSolverContext::new(&model, &incumbent, &mut monitor);

        // Invoke portfolio solver
        let result: PortfolioSolverResult<IntegerType> = portfolio_solver.invoke(context);

        // Validate result shape
        match result.result() {
            SolverResult::Optimal(solution) => {
                assert_eq!(solution.num_vessels(), model.num_vessels());
                // A known optimal objective for this fixture (matches the unit test in bnb.rs)
                assert_eq!(solution.objective_value(), 855);
            }
            other => panic!("expected Optimal, got {:?}", other),
        }

        // Validate basic termination is OptimalityProven
        assert!(matches!(
            result.termination_reason(),
            bollard_search::result::TerminationReason::OptimalityProven
        ));
    }
}
