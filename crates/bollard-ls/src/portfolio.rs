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

use crate::monitor::wrapper::LocalSearchMonitorWrapper;
use crate::{
    decoder::Decoder, engine::LocalSearchEngine, meta::metaheuristic::Metaheuristic,
    operator::local_search_operator::LocalSearchOperator,
};
use bollard_model::solution::Solution;
use bollard_search::{
    neighborhood::neighborhoods::Neighborhoods,
    num::SolverNumeric,
    portfolio::{PortfolioSolverContext, PortfolioSolverResult, PortofolioSolver},
};

pub struct LocalSearchPortfolioSolver<'a, T, N, H, D, O>
where
    T: SolverNumeric,
    N: Neighborhoods,
    H: Metaheuristic<T>,
    D: Decoder<T, H::Evaluator>,
    O: LocalSearchOperator<T, N>,
{
    metaheuristic: H,
    decoder: D,
    engine: LocalSearchEngine<T>,
    operator: O,
    initial_solution: &'a Solution<T>,
    _neighborhoods: std::marker::PhantomData<N>,
}

impl<'a, T, N, H, D, O> LocalSearchPortfolioSolver<'a, T, N, H, D, O>
where
    T: SolverNumeric,
    N: Neighborhoods,
    H: Metaheuristic<T>,
    D: Decoder<T, H::Evaluator>,
    O: LocalSearchOperator<T, N>,
{
    #[inline]
    pub fn new(
        metaheuristic: H,
        decoder: D,
        operator: O,
        initial_solution: &'a Solution<T>,
    ) -> Self {
        Self {
            metaheuristic,
            decoder,
            engine: LocalSearchEngine::new(),
            operator,
            initial_solution,
            _neighborhoods: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn metaheuristic(&self) -> &H {
        &self.metaheuristic
    }

    #[inline]
    pub fn decoder(&self) -> &D {
        &self.decoder
    }

    #[inline]
    pub fn operator(&self) -> &O {
        &self.operator
    }

    #[inline]
    pub fn engine(&self) -> &LocalSearchEngine<T> {
        &self.engine
    }

    #[inline]
    pub fn initial_solution(&self) -> &Solution<T> {
        self.initial_solution
    }
}

impl<'a, T, N, H, D, O> PortofolioSolver<T, N> for LocalSearchPortfolioSolver<'a, T, N, H, D, O>
where
    T: SolverNumeric,
    N: Neighborhoods + Send + Sync,
    H: Metaheuristic<T> + Send + Sync,
    D: Decoder<T, H::Evaluator> + Send + Sync,
    O: LocalSearchOperator<T, N> + Send + Sync,
{
    fn invoke<'ctx>(
        &mut self,
        context: PortfolioSolverContext<'ctx, T, N>,
    ) -> PortfolioSolverResult<T> {
        // Use only the wrapper around the provided monitor
        let ls_monitor = LocalSearchMonitorWrapper::new(context.monitor);

        // Run local search with neighborhoods from the context and the stored initial solution
        let outcome = self.engine.run_with_incumbent(
            context.model,
            &mut self.decoder,
            context.neighborhoods,
            &mut self.operator,
            &mut self.metaheuristic,
            ls_monitor,
            context.incumbent,
            self.initial_solution,
        );

        // Map local search termination to portfolio result
        use crate::result::LocalSearchTerminationReason;
        match outcome.termination_reason() {
            LocalSearchTerminationReason::LocalOptimum => PortfolioSolverResult::aborted(
                Some(outcome.solution().clone()),
                "Local optimum reached".to_string(),
            ),
            LocalSearchTerminationReason::Metaheuristic(msg) => {
                PortfolioSolverResult::aborted(Some(outcome.solution().clone()), msg.clone())
            }
            LocalSearchTerminationReason::Aborted(msg) => {
                PortfolioSolverResult::aborted(Some(outcome.solution().clone()), msg.clone())
            }
        }
    }

    fn name(&self) -> &str {
        // If you prefer including concrete types like the BnB adapter, we can build a string
        // via type_name::<...>(). For brevity, return a static name here.
        "LocalSearchPortfolioSolver"
    }
}
