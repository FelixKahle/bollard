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

//! Dynamic local search operators using trait objects.
//!
//! This module provides `DynamicLocalSearchOperator`, a thin, type-erasing wrapper around any
//! implementation of the `LocalSearchOperator` trait. It enables selecting and exchanging operator
//! implementations at runtime without exposing concrete types, which is useful for configuration-
//! driven solver setups, plugin-style registration, and crossing crate boundaries. The wrapper
//! stores a boxed trait object and forwards all trait methods to the inner operator. The display
//! name is preserved and incorporated into the wrapper’s own `name` so downstream logging and
//! diagnostics remain informative. Construction is ergonomic via `new` with a boxed trait object,
//! `from_operator` for concrete implementors, or `From<Box<dyn LocalSearchOperator<..>>>` to reduce
//! boilerplate when lifting existing operators into a dynamic form.

use crate::{
    mutator::Mutator, operator::local_search_operator::LocalSearchOperator,
    queue::VesselPriorityQueue,
};
use bollard_model::solution::Solution;
use bollard_search::{neighborhood::neighborhoods::Neighborhoods, num::SolverNumeric};

/// A type-erasing wrapper around any `LocalSearchOperator`, enabling runtime selection and
/// composition of operators without exposing their concrete types. The wrapper owns a boxed
/// trait object, forwards all operator callbacks to the inner implementation, and preserves
/// the inner operator’s name (prefixed) so logs and diagnostics remain informative. Use this
/// when operators are chosen by configuration, loaded via plugins, or passed across crate
/// boundaries. The lifetime `'a` ties the wrapper to the inner operator’s lifetime, `T` is
/// the solver numeric type, and `N` is the `Neighborhoods` implementation the operator
/// interacts with.
pub struct DynamicLocalSearchOperator<'a, T, N>
where
    T: SolverNumeric,
    N: Neighborhoods,
{
    inner: Box<dyn LocalSearchOperator<T, N> + 'a>,
    name: String,
}

impl<'a, T, N> DynamicLocalSearchOperator<'a, T, N>
where
    T: SolverNumeric,
    N: Neighborhoods,
{
    /// Creates a new `DynamicLocalSearchOperator` instance from a boxed `LocalSearchOperator` implementation.
    #[inline]
    pub fn new(inner: Box<dyn LocalSearchOperator<T, N> + 'a>) -> Self {
        let name = format!("DynamicLocalSearchOperator({})", inner.name());
        Self { inner, name }
    }

    /// Creates a new `DynamicLocalSearchOperator` instance from any type that implements the `LocalSearchOperator` trait.
    #[inline]
    pub fn from_operator<O>(operator: O) -> Self
    where
        O: LocalSearchOperator<T, N> + 'a,
    {
        Self::new(Box::new(operator))
    }

    /// Returns a reference to the inner `LocalSearchOperator` implementation.
    #[inline]
    pub fn inner(&self) -> &dyn LocalSearchOperator<T, N> {
        self.inner.as_ref()
    }
}

impl<'a, T, N> From<Box<dyn LocalSearchOperator<T, N> + 'a>>
    for DynamicLocalSearchOperator<'a, T, N>
where
    T: SolverNumeric,
    N: Neighborhoods,
{
    #[inline]
    fn from(inner: Box<dyn LocalSearchOperator<T, N> + 'a>) -> Self {
        Self::new(inner)
    }
}

impl<'a, T, N> std::fmt::Debug for DynamicLocalSearchOperator<'a, T, N>
where
    T: SolverNumeric,
    N: Neighborhoods,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner_name = self.inner.name();
        f.debug_struct("DynamicLocalSearchOperator")
            .field("inner", &inner_name)
            .finish()
    }
}

impl<'a, T, N> std::fmt::Display for DynamicLocalSearchOperator<'a, T, N>
where
    T: SolverNumeric,
    N: Neighborhoods,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DynamicLocalSearchOperator({})", self.inner.name())
    }
}

impl<'a, T, N> LocalSearchOperator<T, N> for DynamicLocalSearchOperator<'a, T, N>
where
    T: SolverNumeric,
    N: Neighborhoods,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn prepare(&mut self, schedule: &Solution<T>, queue: &VesselPriorityQueue, neighborhoods: &N) {
        self.inner.prepare(schedule, queue, neighborhoods)
    }

    fn next_neighbor(&mut self, schedule: &Solution<T>, mutator: &mut Mutator<T>, n: &N) -> bool {
        self.inner.next_neighbor(schedule, mutator, n)
    }

    fn reset(&mut self) {
        self.inner.reset()
    }
}
