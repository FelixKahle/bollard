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

use bollard_model::solution::Solution;
use bollard_search::{incumbent::SharedIncumbent, num::SolverNumeric};
use std::marker::PhantomData;

/// Trait for managing incumbent solutions in a branch-and-bound solver.
/// This trait defines methods for initializing, synchronizing, and updating
/// the incumbent solution during the solving process.
/// This is particularly useful in parallel or distributed solving scenarios,
/// where multiple solver instances may need to share and update the best-known solution
/// and its lower bound.
pub trait IncumbentStore<T>
where
    T: SolverNumeric,
{
    /// Returns the initial upper bound for the incumbent solution.
    fn initial_upper_bound(&self) -> T;
    /// Synchronizes the current local best solution with the shared incumbent.
    fn tighten(&self, current_local_best: T) -> T;
    /// Notifies the backing that a new solution has been found.
    fn on_solution_found(&self, solution: &Solution<T>);
}

/// An `IncumbentBacking` implementation that does not share the incumbent
/// solution between different solver instances. Use this for
/// single-threaded or isolated solving scenarios.
#[repr(transparent)]
pub struct NoSharedIncumbent<T>(PhantomData<T>);

impl<T> Default for NoSharedIncumbent<T>
where
    T: SolverNumeric,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> NoSharedIncumbent<T>
where
    T: SolverNumeric,
{
    /// Creates a new `NoSharedIncumbent` instance.
    #[inline(always)]
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T> IncumbentStore<T> for NoSharedIncumbent<T>
where
    T: SolverNumeric,
{
    #[inline(always)]
    fn initial_upper_bound(&self) -> T {
        T::max_value()
    }

    #[inline(always)]
    fn tighten(&self, current_local_best: T) -> T {
        current_local_best
    }

    #[inline(always)]
    fn on_solution_found(&self, _: &Solution<T>) {}
}

/// An `IncumbentBacking` implementation that shares the incumbent
/// solution between different solver instances using a `SharedIncumbent`.
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct SharedIncumbentAdapter<'a, T> {
    inner: &'a SharedIncumbent<T>,
}

impl<'a, T> SharedIncumbentAdapter<'a, T> {
    /// Creates a new `SharedIncumbentAdapter` that wraps the given
    /// `SharedIncumbent`.
    #[inline(always)]
    pub fn new(inner: &'a SharedIncumbent<T>) -> Self {
        Self { inner }
    }
}

impl<'a, T> IncumbentStore<T> for SharedIncumbentAdapter<'a, T>
where
    T: SolverNumeric,
{
    #[inline(always)]
    fn initial_upper_bound(&self) -> T {
        self.inner.upper_bound().into()
    }

    #[inline(always)]
    fn tighten(&self, current_local_best: T) -> T {
        let shared: T = self.inner.upper_bound().into();
        shared.min(current_local_best)
    }

    #[inline(always)]
    fn on_solution_found(&self, solution: &Solution<T>) {
        self.inner.try_install(solution);
    }
}
