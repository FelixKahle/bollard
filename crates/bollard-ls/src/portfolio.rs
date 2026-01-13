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

use bollard_search::{
    neighborhood::neighborhoods::Neighborhoods,
    num::SolverNumeric,
    portfolio::{PortfolioSolverContext, PortfolioSolverResult, PortofolioSolver},
};

use crate::{
    decoder::Decoder, engine::LocalSearchEngine, meta::metaheuristic::Metaheuristic,
    monitor::local_search_monitor::LocalSearchMonitor,
    operator::local_search_operator::LocalSearchOperator,
};

pub struct LocalSearchPortfolio<T, N, H, D, O, M>
where
    T: SolverNumeric,
    N: Neighborhoods,
    H: Metaheuristic<T>,
    D: Decoder<T, H::Evaluator>,
    O: LocalSearchOperator<T, N>,
    M: LocalSearchMonitor<T>,
{
    metaheuristic: H,
    decoder: D,
    engine: LocalSearchEngine<T>,
    operator: O,
    monitor: M,
    name: String,
    _neighborhoods: std::marker::PhantomData<N>,
}

impl<T, N, H, D, O, M> LocalSearchPortfolio<T, N, H, D, O, M>
where
    T: SolverNumeric,
    N: Neighborhoods,
    H: Metaheuristic<T>,
    D: Decoder<T, H::Evaluator>,
    O: LocalSearchOperator<T, N>,
    M: LocalSearchMonitor<T>,
{
    #[inline]
    pub fn new(metaheuristic: H, decoder: D, operator: O, monitor: M) -> Self {
        let name = format!(
            "LocalSearchPortfolioSolver<{}, {}, {}, {}>",
            metaheuristic.name(),
            decoder.name(),
            operator.name(),
            monitor.name(),
        );
        Self {
            metaheuristic,
            decoder,
            engine: LocalSearchEngine::new(),
            operator,
            monitor,
            name,
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
    pub fn monitor(&self) -> &M {
        &self.monitor
    }

    #[inline]
    pub fn engine(&self) -> &LocalSearchEngine<T> {
        &self.engine
    }
}

impl<T, N, H, D, O, M> PortofolioSolver<T, N> for LocalSearchPortfolio<T, N, H, D, O, M>
where
    T: SolverNumeric,
    N: Neighborhoods,
    H: Metaheuristic<T>,
    D: Decoder<T, H::Evaluator>,
    O: LocalSearchOperator<T, N>,
    M: LocalSearchMonitor<T>,
{
    fn invoke<'a>(
        &mut self,
        context: PortfolioSolverContext<'a, T, N>,
    ) -> PortfolioSolverResult<T> {
        todo!()
    }

    fn name(&self) -> &str {
        todo!()
    }
}
