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
use bollard_core::num::constants::MinusOne;
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

impl<'a, T> PortfolioSolverContext<'a, T>
where
    T: PrimInt + Signed,
{
    #[inline(always)]
    pub fn new(
        model: &'a Model<T>,
        incumbent: &'a SharedIncumbent<T>,
        monitor: &'a mut dyn SearchMonitor<T>,
        stop: &'a AtomicBool,
    ) -> Self {
        Self {
            model,
            incumbent,
            monitor,
            stop,
        }
    }
}

impl<'a, T> std::fmt::Debug for PortfolioSolverContext<'a, T>
where
    T: PrimInt + Signed + Copy + MinusOne + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PortfolioSolverContext")
            .field("model", &self.model)
            .field("incumbent", &self.incumbent)
            .field("monitor", &self.monitor.name())
            .field("stop", &self.stop)
            .finish()
    }
}

impl<'a, T> std::fmt::Display for PortfolioSolverContext<'a, T>
where
    T: PrimInt + Signed + Copy + MinusOne + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PortfolioSolverContext(model: {}, monitor: {})",
            self.model,
            self.monitor.name(),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

impl std::fmt::Display for PortfolioSolverTerminationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PortfolioSolverTerminationReason::OptimalityProven => {
                write!(f, "Optimality Proven")
            }
            PortfolioSolverTerminationReason::InfeasibilityProven => {
                write!(f, "Infeasibility Proven")
            }
            PortfolioSolverTerminationReason::Aborted(reason) => {
                write!(f, "Aborted: {}", reason)
            }
            PortfolioSolverTerminationReason::Interrupted => write!(f, "Interrupted"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortfolioSolverResult<T>
where
    T: PrimInt + Signed,
{
    pub result: SolverResult<T>,
    pub termination_reason: PortfolioSolverTerminationReason,
}

impl<T> PortfolioSolverResult<T>
where
    T: PrimInt + Signed,
{
    #[inline(always)]
    pub fn new(
        result: SolverResult<T>,
        termination_reason: PortfolioSolverTerminationReason,
    ) -> Self {
        Self {
            result,
            termination_reason,
        }
    }
}

impl<T> std::fmt::Display for PortfolioSolverResult<T>
where
    T: PrimInt + Signed + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PortfolioSolverResult(result: {}, termination_reason: {})",
            self.result, self.termination_reason
        )
    }
}

pub trait PortofolioSolver<T>
where
    T: PrimInt + Signed,
{
    fn solve<'a>(&mut self, context: PortfolioSolverContext<'a, T>) -> PortfolioSolverResult<T>;
    fn name(&self) -> &str;
}
