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

pub trait IncumbentStore<T>
where
    T: SolverNumeric,
{
    fn on_best_solution(&mut self, solution: &Solution<T>);
}

#[repr(transparent)]
pub struct NoSharedIncumbent<T>(PhantomData<T>);

impl<T> IncumbentStore<T> for NoSharedIncumbent<T>
where
    T: SolverNumeric,
{
    #[inline(always)]
    fn on_best_solution(&mut self, _solution: &Solution<T>) {
        // No-op
    }
}

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
    fn on_best_solution(&mut self, solution: &Solution<T>) {
        // Delegate to the inner SharedIncumbent.
        self.inner.try_install(solution);
    }
}
