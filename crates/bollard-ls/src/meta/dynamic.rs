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

//! Dynamic metaheuristics using trait objects.
//!
//! This module provides `DynamicMetaheuristic`, a type-erasing wrapper that allows metaheuristic
//! implementations to be selected, composed, and exchanged at runtime without exposing their
//! concrete types. It stores the inner metaheuristic as a boxed trait object and forwards all
//! orchestration and search decisions to that implementation while preserving a readable name for
//! logging and diagnostics. The wrapper integrates with `DynamicAssignmentEvaluator`, ensuring a
//! cohesive dynamic pipeline across evaluation and search. This design is useful for configuration-
//! driven solver setups, plugin-style architectures, and for crossing crate boundaries where
//! stability and ABI-neutral interfaces are preferred.

use crate::{eval::dynamic::DynamicAssignmentEvaluator, meta::metaheuristic::Metaheuristic};
use bollard_model::{model::Model, solution::Solution};
use bollard_search::{monitor::search_monitor::SearchCommand, num::SolverNumeric};

/// A type-erasing wrapper around any `Metaheuristic` that operates with a
/// `DynamicAssignmentEvaluator`.
///
/// This enables selecting and composing metaheuristics
/// at runtime without exposing their concrete types, while preserving a human-readable
/// name for logging and diagnostics. The wrapper owns the metaheuristic as a boxed
/// trait object and forwards all orchestration and search logic to the inner
/// implementation. The lifetime `'a` ties the wrapper to the lifetime of the inner
/// metaheuristic and `T` denotes the solver’s numeric type.
pub struct DynamicMetaheuristic<'a, T>
where
    T: SolverNumeric,
{
    inner: Box<dyn Metaheuristic<T, Evaluator = DynamicAssignmentEvaluator<'a, T>> + 'a>,
    name: String,
}

impl<'a, T> DynamicMetaheuristic<'a, T>
where
    T: SolverNumeric,
{
    /// Creates a new DynamicMetaheuristic from a boxed metaheuristic.
    #[inline]
    pub fn new(
        inner: Box<dyn Metaheuristic<T, Evaluator = DynamicAssignmentEvaluator<'a, T>> + 'a>,
    ) -> Self {
        let name = format!("DynamicMetaheuristic({})", inner.name());

        Self { inner, name }
    }

    /// Creates a new DynamicMetaheuristic from a metaheuristic.
    #[inline]
    pub fn from_metaheuristic<M>(metaheuristic: M) -> Self
    where
        M: Metaheuristic<T, Evaluator = DynamicAssignmentEvaluator<'a, T>> + 'a,
    {
        Self::new(Box::new(metaheuristic))
    }

    /// Returns a reference to the inner metaheuristic.
    #[inline]
    pub fn inner(&self) -> &dyn Metaheuristic<T, Evaluator = DynamicAssignmentEvaluator<'a, T>> {
        self.inner.as_ref()
    }
}

impl<'a, T> From<Box<dyn Metaheuristic<T, Evaluator = DynamicAssignmentEvaluator<'a, T>> + 'a>>
    for DynamicMetaheuristic<'a, T>
where
    T: SolverNumeric,
{
    fn from(
        value: Box<dyn Metaheuristic<T, Evaluator = DynamicAssignmentEvaluator<'a, T>> + 'a>,
    ) -> Self {
        Self::new(value)
    }
}

impl<'a, T> std::fmt::Debug for DynamicMetaheuristic<'a, T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicMetaheuristic")
            .field("inner", &self.inner.name())
            .finish()
    }
}

impl<'a, T> std::fmt::Display for DynamicMetaheuristic<'a, T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DynamicMetaheuristic({})", self.inner.name())
    }
}

impl<'a, T> Metaheuristic<T> for DynamicMetaheuristic<'a, T>
where
    T: SolverNumeric,
{
    type Evaluator = DynamicAssignmentEvaluator<'a, T>;

    fn name(&self) -> &str {
        &self.name
    }

    fn evaluator(&self) -> &Self::Evaluator {
        self.inner.evaluator()
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
        best_solution: &Solution<T>,
    ) -> SearchCommand {
        self.inner.search_command(iteration, model, best_solution)
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
}
