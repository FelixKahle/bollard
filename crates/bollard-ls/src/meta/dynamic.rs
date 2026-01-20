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

use crate::meta::metaheuristic::{Evaluation, Metaheuristic};
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
    solution::Solution,
};
use bollard_search::{monitor::search_monitor::SearchCommand, num::SolverNumeric};

pub struct DynamicMetaheuristic<'a, T> {
    inner: Box<dyn Metaheuristic<T> + 'a>,
}

impl<'a, T> DynamicMetaheuristic<'a, T>
where
    T: SolverNumeric,
{
    pub fn new(inner: Box<dyn Metaheuristic<T> + 'a>) -> Self {
        Self { inner }
    }

    pub fn from_metaheuristic<M>(metaheuristic: M) -> Self
    where
        M: Metaheuristic<T> + 'a,
    {
        Self::new(Box::new(metaheuristic))
    }
}

impl<'a, T> Metaheuristic<T> for DynamicMetaheuristic<'a, T>
where
    T: SolverNumeric,
{
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn on_start(&mut self, model: &Model<T>, initial_solution: &Solution<T>) {
        self.inner.on_start(model, initial_solution);
    }

    fn on_end(&mut self, model: &Model<T>, final_solution: &Solution<T>) {
        self.inner.on_end(model, final_solution);
    }

    fn search_command(
        &mut self,
        iteration: u64,
        model: &Model<T>,
        current_solution: &Solution<T>,
        best_solution: &Solution<T>,
    ) -> SearchCommand {
        self.inner
            .search_command(iteration, model, current_solution, best_solution)
    }

    fn should_accept(
        &mut self,
        model: &Model<T>,
        current: &Solution<T>,
        candidate: &Solution<T>,
        best: &Solution<T>,
    ) -> bool {
        self.inner.should_accept(model, current, candidate, best)
    }

    fn on_accept(&mut self, model: &Model<T>, new_current: &Solution<T>) {
        self.inner.on_accept(model, new_current);
    }

    fn on_reject(&mut self, model: &Model<T>, rejected_candidate: &Solution<T>) {
        self.inner.on_reject(model, rejected_candidate);
    }

    fn on_new_best(&mut self, model: &Model<T>, new_best: &Solution<T>) {
        self.inner.on_new_best(model, new_best);
    }

    fn evaluate_assignment(
        &self,
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
    ) -> Option<Evaluation<T>> {
        self.inner
            .evaluate_assignment(model, vessel_index, berth_index, start_time)
    }

    unsafe fn evaluate_assignment_unchecked(
        &self,
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
    ) -> Option<Evaluation<T>> {
        unsafe {
            self.inner
                .evaluate_assignment_unchecked(model, vessel_index, berth_index, start_time)
        }
    }

    fn on_neighbourhood_exhausted(
        &mut self,
        _model: &Model<T>,
        _current: &Solution<T>,
        _best: &Solution<T>,
    ) -> bool {
        self.inner
            .on_neighbourhood_exhausted(_model, _current, _best)
    }
}
