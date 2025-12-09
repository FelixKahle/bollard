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
    incumbent::SharedIncumbent, monitor::search_monitor::SearchMonitor, result::SolverResult,
};
use bollard_model::model::Model;
use num_traits::{PrimInt, Signed};
use std::sync::atomic::AtomicBool;

pub struct PortfolioSolverContext<'a, T>
where
    T: PrimInt + Signed,
{
    pub model: &'a Model<T>,
    pub incumbent: &'a SharedIncumbent<T>,
    pub monitor: &'a mut dyn SearchMonitor<T>,
    pub stop: &'a AtomicBool,
}

pub enum PortfolioSolverTerminationReason {
    /// The solver found and proved optimality of a solution.
    OptimalityProven,
    /// The solver proved that the problem is infeasible.
    InfeasibilityProven,
    /// The solver aborted due to a search limit (time, iterations, etc.).
    /// The string contains information about the reason for abortion.
    Aborted(String),
    /// The solver was interrupted by the stop flag.
    Interrupted,
}

pub struct PortfolioSolverResult<T>
where
    T: PrimInt + Signed,
{
    pub result: SolverResult<T>,
    pub termination_reason: PortfolioSolverTerminationReason,
}

pub trait PortofolioSolver<T>
where
    T: PrimInt + Signed,
{
    fn solve<'a>(&mut self, context: PortfolioSolverContext<'a, T>) -> PortfolioSolverResult<T>;
    fn name(&self) -> &str;
}
