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

use crate::fixed::FixedAssignment;
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
    solution::Solution,
};
use bollard_search::num::SolverNumeric;

/// Error: a fixed vessel index is not present in a solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MissingVesselInSolutionError {
    pub vessel_index: VesselIndex,
    pub num_vessels_in_solution: usize,
}

impl std::fmt::Display for MissingVesselInSolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "fixed assignment refers to vessel index {} but solution only contains {} vessel(s)",
            self.vessel_index, self.num_vessels_in_solution
        )
    }
}

impl std::error::Error for MissingVesselInSolutionError {}

/// Error: the berth in a solution does not match the fixed berth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BerthMismatchError {
    pub vessel_index: VesselIndex,
    pub expected_berth: BerthIndex,
    pub actual_berth: BerthIndex,
}

impl std::fmt::Display for BerthMismatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "fixed assignment for vessel {} requires berth {}, but solution uses berth {}",
            self.vessel_index, self.expected_berth, self.actual_berth
        )
    }
}

impl std::error::Error for BerthMismatchError {}

/// Error: the start time in a solution does not match the fixed start time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StartTimeMismatchError<T> {
    pub vessel_index: VesselIndex,
    pub expected_start_time: T,
    pub actual_start_time: T,
}

impl<T> std::fmt::Display for StartTimeMismatchError<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "fixed assignment for vessel {} requires start time {}, but solution uses start time {}",
            self.vessel_index, self.expected_start_time, self.actual_start_time
        )
    }
}

impl<T> std::error::Error for StartTimeMismatchError<T> where T: std::fmt::Display + std::fmt::Debug {}

/// Public error enum describing why an initial solution is incompatible with
/// a set of fixed assignments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedSolutionError<T> {
    MissingVesselInSolution(MissingVesselInSolutionError),
    BerthMismatch(BerthMismatchError),
    StartTimeMismatch(StartTimeMismatchError<T>),
}

impl<T> std::fmt::Display for FixedSolutionError<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FixedSolutionError::MissingVesselInSolution(e) => e.fmt(f),
            FixedSolutionError::BerthMismatch(e) => e.fmt(f),
            FixedSolutionError::StartTimeMismatch(e) => e.fmt(f),
        }
    }
}

impl<T> std::error::Error for FixedSolutionError<T> where T: std::fmt::Display + std::fmt::Debug {}

/// Validate that a concrete `solution` satisfies all `fixed` assignments.
///
/// This is a pure, allocation-free check.
pub fn validate_fixed_solution<T>(
    fixed: &[FixedAssignment<T>],
    solution: &Solution<T>,
) -> Result<(), FixedSolutionError<T>>
where
    T: SolverNumeric,
{
    let num_vessels = solution.num_vessels();

    for fix in fixed {
        let v = fix.vessel_index;

        // Vessel index must be present in the solution.
        let idx = v.get();
        if idx >= num_vessels {
            return Err(FixedSolutionError::MissingVesselInSolution(
                MissingVesselInSolutionError {
                    vessel_index: v,
                    num_vessels_in_solution: num_vessels,
                },
            ));
        }

        // Berth must match.
        let sol_berth = solution.berth_for_vessel(v);
        if sol_berth != fix.berth_index {
            return Err(FixedSolutionError::BerthMismatch(BerthMismatchError {
                vessel_index: v,
                expected_berth: fix.berth_index,
                actual_berth: sol_berth,
            }));
        }

        // Start time must match.
        let sol_start = solution.start_time_for_vessel(v);
        if sol_start != fix.start_time {
            return Err(FixedSolutionError::StartTimeMismatch(
                StartTimeMismatchError {
                    vessel_index: v,
                    expected_start_time: fix.start_time,
                    actual_start_time: sol_start,
                },
            ));
        }
    }

    Ok(())
}

/// Parameters for a Branch-and-Bound solver run.
///
/// # Construction
///
/// This struct cannot be constructed directly. Use [`BnbSearchParams::builder`] to
/// create a [`BnbSearchParamsBuilder`], configure your parameters, and call `.build()`.
///
/// The `.build()` method ensures that the configuration is valid (e.g., that
/// the initial solution respects fixed assignments). If validation fails, it
/// returns an [`Err`] with a descriptive error explaining the problem.
pub struct BnbSearchParams<'a, T, B, E, S>
where
    T: SolverNumeric,
{
    model: &'a Model<T>,
    builder: &'a mut B,
    evaluator: &'a mut E,
    monitor: S,
    fixed: Option<&'a [FixedAssignment<T>]>,
    initial_solution: Option<&'a Solution<T>>,
}

/// A builder for [`BnbSearchParams`].
///
/// This struct allows optional configuration of fixed assignments and initial solutions
/// before constructing the final parameter object.
pub struct BnbSearchParamsBuilder<'a, T, B, E, S>
where
    T: SolverNumeric,
{
    model: &'a Model<T>,
    builder: &'a mut B,
    evaluator: &'a mut E,
    monitor: S,
    fixed: Option<&'a [FixedAssignment<T>]>,
    initial_solution: Option<&'a Solution<T>>,
}

impl<'a, T, B, E, S> BnbSearchParamsBuilder<'a, T, B, E, S>
where
    T: SolverNumeric,
{
    /// Sets the fixed assignments for the solver.
    #[inline]
    pub fn with_fixed_assignments(mut self, fixed: &'a [FixedAssignment<T>]) -> Self {
        self.fixed = Some(fixed);
        self
    }

    /// Attach an initial solution to warm-start the solver.
    #[inline]
    pub fn with_initial_solution(mut self, initial: &'a Solution<T>) -> Self {
        self.initial_solution = Some(initial);
        self
    }

    /// Consumes the builder and produces a validated [`BnbSearchParams`] object.
    pub fn build(self) -> Result<BnbSearchParams<'a, T, B, E, S>, FixedSolutionError<T>> {
        if let (Some(f), Some(s)) = (self.fixed, self.initial_solution) {
            validate_fixed_solution(f, s)?;
        }

        Ok(BnbSearchParams {
            model: self.model,
            builder: self.builder,
            evaluator: self.evaluator,
            monitor: self.monitor,
            fixed: self.fixed,
            initial_solution: self.initial_solution,
        })
    }

    pub fn build_unchecked(self) -> BnbSearchParams<'a, T, B, E, S> {
        BnbSearchParams {
            model: self.model,
            builder: self.builder,
            evaluator: self.evaluator,
            monitor: self.monitor,
            fixed: self.fixed,
            initial_solution: self.initial_solution,
        }
    }
}

impl<'a, T, B, E, S> BnbSearchParams<'a, T, B, E, S>
where
    T: SolverNumeric,
{
    /// Creates a new builder for search parameters.
    ///
    /// Requires the mandatory dependencies (model, decision builder, evaluator, monitor).
    /// Optional parameters (fixed assignments, initial solution) can be added via the returned builder.
    #[inline]
    pub fn builder(
        model: &'a Model<T>,
        builder: &'a mut B,
        evaluator: &'a mut E,
        monitor: S,
    ) -> BnbSearchParamsBuilder<'a, T, B, E, S> {
        BnbSearchParamsBuilder {
            model,
            builder,
            evaluator,
            monitor,
            fixed: None,
            initial_solution: None,
        }
    }

    #[inline]
    pub fn model(&self) -> &'a Model<T> {
        self.model
    }

    /// Returns a mutable reference to the decision builder.
    #[inline]
    pub fn builder_mut(&mut self) -> &mut B {
        self.builder
    }

    /// Returns a mutable reference to the evaluator.
    #[inline]
    pub fn evaluator_mut(&mut self) -> &mut E {
        self.evaluator
    }

    /// Returns a mutable reference to the monitor.
    #[inline]
    pub fn monitor_mut(&mut self) -> &mut S {
        &mut self.monitor
    }

    /// Consumes the params and returns the monitor.
    #[inline]
    pub fn into_monitor(self) -> S {
        self.monitor
    }

    #[inline]
    pub fn fixed_assignments(&self) -> Option<&'a [FixedAssignment<T>]> {
        self.fixed
    }

    #[inline]
    pub fn initial_solution(&self) -> Option<&'a Solution<T>> {
        self.initial_solution
    }

    #[inline]
    pub fn has_fixed_assignments(&self) -> bool {
        self.fixed.is_some()
    }

    #[inline]
    pub fn has_initial_solution(&self) -> bool {
        self.initial_solution.is_some()
    }

    #[allow(clippy::type_complexity)]
    pub fn into_inner(
        self,
    ) -> (
        &'a Model<T>,
        &'a mut B,
        &'a mut E,
        S,
        Option<&'a [FixedAssignment<T>]>,
        Option<&'a Solution<T>>,
    ) {
        (
            self.model,
            self.builder,
            self.evaluator,
            self.monitor,
            self.fixed,
            self.initial_solution,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bollard_model::index::{BerthIndex, VesselIndex};
    use bollard_model::model::ModelBuilder;

    type Int = i64;

    fn bi(i: usize) -> BerthIndex {
        BerthIndex::new(i)
    }

    fn vi(i: usize) -> VesselIndex {
        VesselIndex::new(i)
    }

    fn fixed(start: Int, berth: usize, vessel: usize) -> FixedAssignment<Int> {
        FixedAssignment::new(start, bi(berth), vi(vessel))
    }

    #[test]
    fn test_validate_fixed_solution_ok() {
        let sol = Solution::new(0, vec![bi(0), bi(1)], vec![10, 20]);
        let f = vec![fixed(10, 0, 0), fixed(20, 1, 1)];

        let res = validate_fixed_solution(&f, &sol);
        assert!(res.is_ok());
    }

    #[test]
    fn test_validate_fixed_solution_missing_vessel() {
        let sol = Solution::new(0, vec![bi(0)], vec![10]);
        let f = vec![fixed(10, 0, 1)];

        let res = validate_fixed_solution(&f, &sol);
        match res {
            Err(FixedSolutionError::MissingVesselInSolution(e)) => {
                assert_eq!(e.vessel_index, vi(1));
                assert_eq!(e.num_vessels_in_solution, 1);
                // smoke test Display
                let msg = e.to_string();
                assert!(msg.contains("vessel index"));
            }
            _ => panic!("expected MissingVesselInSolution error"),
        }
    }

    #[test]
    fn test_validate_fixed_solution_berth_mismatch() {
        let sol = Solution::new(0, vec![bi(0)], vec![10]);
        let f = vec![fixed(10, 1, 0)];

        let res = validate_fixed_solution(&f, &sol);
        match res {
            Err(FixedSolutionError::BerthMismatch(e)) => {
                assert_eq!(e.vessel_index, vi(0));
                assert_eq!(e.expected_berth, bi(1));
                assert_eq!(e.actual_berth, bi(0));
                let msg = e.to_string();
                assert!(msg.contains("requires berth"));
            }
            _ => panic!("expected BerthMismatch error"),
        }
    }

    #[test]
    fn test_validate_fixed_solution_start_time_mismatch() {
        let sol = Solution::new(0, vec![bi(0)], vec![10]);
        let f = vec![fixed(20, 0, 0)];

        let res = validate_fixed_solution(&f, &sol);
        match res {
            Err(FixedSolutionError::StartTimeMismatch(e)) => {
                assert_eq!(e.vessel_index, vi(0));
                assert_eq!(e.expected_start_time, 20);
                assert_eq!(e.actual_start_time, 10);
                let msg = e.to_string();
                assert!(msg.contains("requires start time"));
            }
            _ => panic!("expected StartTimeMismatch error"),
        }
    }

    #[test]
    fn test_bnb_search_params_builder() {
        struct DummyBuilder;
        struct DummyEvaluator;
        struct DummyMonitor;

        let mb = ModelBuilder::<Int>::new(0, 0);
        let model = mb.build();

        let mut builder = DummyBuilder;
        let mut evaluator = DummyEvaluator;
        let monitor = DummyMonitor;

        // 1. Basic build
        let params = BnbSearchParams::builder(&model, &mut builder, &mut evaluator, monitor)
            .build()
            .unwrap();
        assert!(!params.has_fixed_assignments());
        assert!(!params.has_initial_solution());

        // 2. With fixed only
        let mut builder2 = DummyBuilder;
        let mut evaluator2 = DummyEvaluator;
        let monitor2 = DummyMonitor;
        let fixed_assignments = [fixed(10, 0, 0)];

        let params2 = BnbSearchParams::builder(&model, &mut builder2, &mut evaluator2, monitor2)
            .with_fixed_assignments(&fixed_assignments)
            .build()
            .unwrap();

        assert!(params2.has_fixed_assignments());
        assert!(!params2.has_initial_solution());

        // 3. With valid initial solution
        let mut builder3 = DummyBuilder;
        let mut evaluator3 = DummyEvaluator;
        let monitor3 = DummyMonitor;
        let valid_sol = Solution::new(0, vec![bi(0)], vec![10]);

        let params3 = BnbSearchParams::builder(&model, &mut builder3, &mut evaluator3, monitor3)
            .with_fixed_assignments(&fixed_assignments)
            .with_initial_solution(&valid_sol)
            .build()
            .unwrap();

        assert!(params3.has_fixed_assignments());
        assert!(params3.has_initial_solution());

        // 4. Validation Failure (Result)
        let mut builder4 = DummyBuilder;
        let mut evaluator4 = DummyEvaluator;
        let monitor4 = DummyMonitor;
        let invalid_sol = Solution::new(0, vec![bi(0)], vec![999]); // Mismatch: start time 10 vs 999

        let result = BnbSearchParams::builder(&model, &mut builder4, &mut evaluator4, monitor4)
            .with_fixed_assignments(&fixed_assignments)
            .with_initial_solution(&invalid_sol)
            .build();

        assert!(
            result.is_err(),
            "build() should return Err when initial solution conflicts with fixed assignments"
        );
    }
}
