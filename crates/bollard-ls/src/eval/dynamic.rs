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

use crate::eval::evaluator::{AssignmentEvaluator, Evaluation};
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use bollard_search::num::SolverNumeric;

/// A type-erasing wrapper around any `AssignmentEvaluator`, enabling runtime selection and
/// substitution of evaluators without exposing their concrete types.
///
/// The wrapper owns a boxed trait object and forwards all calls to the inner implementation,
/// preserving the evaluator’s reported name for debugging and logging.
/// This is useful when the evaluator is chosen by configuration, provided by plugins,
/// or passed across crate boundaries.
/// The lifetime `'a` ties the wrapper to the lifetime of the inner evaluator and `T` denotes the solver’s numeric type.
pub struct DynamicAssignmentEvaluator<'a, T>
where
    T: SolverNumeric,
{
    inner: Box<dyn AssignmentEvaluator<T> + 'a>,
}

impl<'a, T> DynamicAssignmentEvaluator<'a, T>
where
    T: SolverNumeric,
{
    /// Creates a new `DynamicAssignmentEvaluator` instance from a boxed `AssignmentEvaluator` implementation.
    #[inline]
    pub fn new(inner: Box<dyn AssignmentEvaluator<T> + 'a>) -> Self {
        Self { inner }
    }

    /// Creates a new `DynamicAssignmentEvaluator` instance from any type that implements the `AssignmentEvaluator` trait.
    #[inline]
    pub fn from_evaluator<E>(evaluator: E) -> Self
    where
        E: AssignmentEvaluator<T> + 'a,
    {
        Self {
            inner: Box::new(evaluator),
        }
    }

    /// Returns a reference to the inner `AssignmentEvaluator` implementation.
    #[inline]
    pub fn inner(&self) -> &dyn AssignmentEvaluator<T> {
        self.inner.as_ref()
    }
}

impl<'a, T> From<Box<dyn AssignmentEvaluator<T> + 'a>> for DynamicAssignmentEvaluator<'a, T>
where
    T: SolverNumeric,
{
    #[inline]
    fn from(inner: Box<dyn AssignmentEvaluator<T> + 'a>) -> Self {
        Self::new(inner)
    }
}

impl<'a, T> std::fmt::Debug for DynamicAssignmentEvaluator<'a, T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner_name = self.inner.name();
        f.debug_struct("DynamicAssignmentEvaluator")
            .field("inner", &inner_name)
            .finish()
    }
}

impl<'a, T> std::fmt::Display for DynamicAssignmentEvaluator<'a, T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DynamicAssignmentEvaluator")
    }
}

impl<'a, T> AssignmentEvaluator<T> for DynamicAssignmentEvaluator<'a, T>
where
    T: SolverNumeric,
{
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn evaluate(
        &self,
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
    ) -> Option<Evaluation<T>> {
        self.inner
            .evaluate(model, vessel_index, berth_index, start_time)
    }

    unsafe fn evaluate_unchecked(
        &self,
        model: &Model<T>,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        start_time: T,
    ) -> Option<Evaluation<T>> {
        unsafe {
            self.inner
                .evaluate_unchecked(model, vessel_index, berth_index, start_time)
        }
    }
}
