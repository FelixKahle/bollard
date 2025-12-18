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

//! Validation utilities for objective evaluators. This module provides a small,
//! availability‑aware harness for checking that an `ObjectiveEvaluator` behaves
//! as a regular objective in the sense required by the solver: costs must not
//! decrease as feasible start times move later. The check samples candidate
//! start times between each vessel’s arrival and latest feasible start, uses the
//! evaluator to score those assignments, and verifies that the sequence of
//! feasible costs is non‑decreasing. Infeasible times are ignored for the
//! monotonicity test so periods without a valid assignment do not invalidate an
//! otherwise regular objective.
//!
//! The sampling density is controlled per vessel. When the number of iterations
//! is at least the time span between arrival and deadline, every integer start
//! time is examined; otherwise a step size is chosen to spread the samples
//! evenly across the interval, and the final sample always aligns with the
//! vessel’s latest start. Arithmetic is saturating to guard against overflow.
//! These routines are intended for diagnostics during development and testing;
//! they do not alter solver state and do not attempt to prove properties beyond
//! the sampled grid. Use them to catch non‑regular implementations early and
//! to document objective behavior with concrete instances.

use crate::{berth_availability::BerthAvailability, eval::evaluator::ObjectiveEvaluator};
use bollard_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};
use bollard_search::num::SolverNumeric;

/// Iterates over candidate start times from an arrival up to and including the
/// latest permissible start. Progress is made in fixed steps chosen by the
/// caller; the sequence is clamped so the final value is exactly the latest
/// start even when the step does not divide the span. Saturating arithmetic
/// prevents wraparound near numeric limits, and no items are yielded if the
/// arrival is after the deadline. This iterator is used to probe an evaluator
/// at representative points without constructing full decisions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TimePointIterator<T> {
    current: T,
    latest_start: T,
    step_size: T,
    done: bool,
}

impl<T> Iterator for TimePointIterator<T>
where
    T: SolverNumeric,
{
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let yield_val = self.current;

        if self.current >= self.latest_start {
            self.done = true;
        } else {
            let next_val = self.current.saturating_add_val(self.step_size);

            if next_val > self.latest_start || next_val < self.current {
                self.current = self.latest_start;
            } else {
                self.current = next_val;
            }

            if self.current == yield_val && self.current < self.latest_start {
                self.current = self.latest_start;
            }
        }

        Some(yield_val)
    }
}

/// Checks whether an evaluator is regular by sampling start times for every
/// vessel–berth pair permitted by the model and verifying that feasible costs
/// are non‑decreasing as starts move later. The berth availability structure is
/// initialized from the model so feasibility and timing respect closures and
/// static constraints.
///
/// The `max_iterations_per_vessel` controls the sampling density. If it is at
/// least the arrival‑to‑deadline span, the check is exhaustive over integer
/// start times; otherwise the interval is probed with an even step and the last
/// sample is forced to the latest start. Infeasible scores (`None`) are skipped
/// in the monotonicity relation, so only transitions between feasible points are
/// compared. The function returns `true` when no violation is detected, and
/// `false` at the first decrease observed.
pub fn is_regular_evaluator_exhaustive<T, E>(
    evaluator: &mut E,
    model: &Model<T>,
    max_iterations_per_vessel: usize,
) -> bool
where
    T: SolverNumeric,
    E: ObjectiveEvaluator<T>,
{
    let mut berth_availability = BerthAvailability::<T>::new();
    berth_availability.initialize(model, &[]);

    if model.num_vessels() == 0 || model.num_berths() == 0 {
        return true;
    }

    // SAFETY: We control the indices, which never extend the model bounds.
    // Therefore all unchecked calls here are safe.

    for vessel_index in 0..model.num_vessels() {
        let vessel = VesselIndex::new(vessel_index);
        let arrival = unsafe { model.vessel_arrival_time_unchecked(vessel) };
        let deadline = unsafe { model.vessel_latest_departure_time_unchecked(vessel) };

        let span = deadline.saturating_sub_val(arrival);
        let divisor = T::from_usize(max_iterations_per_vessel).unwrap_or(T::one());

        let step_size = if span > divisor {
            let s = span / divisor;
            if s == T::zero() { T::one() } else { s }
        } else {
            T::one()
        };

        for berth_index in 0..model.num_berths() {
            let berth = BerthIndex::new(berth_index);
            if unsafe { !model.vessel_allowed_on_berth_unchecked(vessel, berth) } {
                continue;
            }

            let mut last_valid_cost: Option<T> = None;

            let points = TimePointIterator {
                current: arrival,
                latest_start: deadline,
                step_size,
                done: arrival > deadline,
            };

            for t in points {
                let score = unsafe {
                    evaluator.evaluate_vessel_assignment_unchecked(
                        model,
                        &berth_availability,
                        vessel,
                        berth,
                        t,
                    )
                };

                if !check_monotonicity(last_valid_cost, score) {
                    return false;
                }

                if score.is_some() {
                    last_valid_cost = score;
                }
            }
        }
    }
    true
}

/// Returns whether the sampled cost sequence remains non-decreasing while
/// treating infeasible samples as neutral.
///
/// This helper is used when validating objective regularity over sampled
/// start times. It compares only when both values are `Some`; any transition
/// that involves `None` is considered monotonic (returns `true`). This lets you
/// skip infeasible points (represented by `None`) without breaking the
/// non-decreasing check.
///
/// Behavior:
/// - If both `last_some` and `current` are `Some`, returns `current >= last_some`.
/// - If either is `None`, returns `true`.
#[inline(always)]
fn check_monotonicity<T: SolverNumeric>(last_some: Option<T>, current: Option<T>) -> bool {
    match (last_some, current) {
        (Some(prev), Some(curr)) => curr >= prev,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::{TimePointIterator, check_monotonicity};
    use bollard_search::num::SolverNumeric;

    // Helper to construct an iterator with numeric generics
    fn make_iter<T: SolverNumeric>(arrival: T, deadline: T, step_size: T) -> TimePointIterator<T> {
        TimePointIterator {
            current: arrival,
            latest_start: deadline,
            step_size,
            done: arrival > deadline,
        }
    }

    #[test]
    fn test_time_point_iterator_basic_progression() {
        // arrival=0, deadline=10, step=3 -> expect: 0, 3, 6, 9, 10
        let iter = make_iter(0, 10, 3);
        let points: Vec<i64> = iter.collect();
        assert_eq!(points, vec![0, 3, 6, 9, 10]);
    }

    #[test]
    fn test_time_point_iterator_exact_step_end() {
        // arrival=0, deadline=9, step=3 -> expect: 0, 3, 6, 9
        let iter = make_iter(0, 9, 3);
        let points: Vec<i64> = iter.collect();
        assert_eq!(points, vec![0, 3, 6, 9]);
    }

    #[test]
    fn test_time_point_iterator_handles_large_step() {
        // arrival=5, deadline=7, step=10 -> should yield: 5, then deadline clamp to 7 (not exceed), so 5,7
        let iter = make_iter(5, 7, 10);
        let points: Vec<i64> = iter.collect();
        assert_eq!(points, vec![5, 7]);
    }

    #[test]
    fn test_time_point_iterator_arrival_after_deadline() {
        // arrival > deadline -> iterator is done immediately
        let iter = make_iter(10, 5, 1);
        let points: Vec<i64> = iter.collect();
        assert!(points.is_empty());
    }

    #[test]
    fn test_time_point_iterator_saturating_add_wrap_guard() {
        // Simulate near-maximum value to ensure saturating_add_val + wrap guard clamps to latest_start.
        // Using i64::MAX-2 to keep step increase from overflowing immediately.
        let arrival = i64::MAX - 2;
        let deadline = i64::MAX;
        let step = 5; // arrival + step would overflow; iterator should clamp to deadline
        let iter = make_iter(arrival, deadline, step);
        let points: Vec<i64> = iter.collect();
        assert_eq!(points, vec![i64::MAX - 2, i64::MAX]);
    }

    #[test]
    fn test_check_monotonicity_allows_none_transitions() {
        // Any None involved should be treated as monotonic (true)
        assert!(check_monotonicity::<i64>(None, None));
        assert!(check_monotonicity::<i64>(None, Some(5)));
        assert!(check_monotonicity::<i64>(Some(5), None));
    }

    #[test]
    fn test_check_monotonicity_non_decreasing() {
        assert!(check_monotonicity::<i64>(Some(3), Some(3)));
        assert!(check_monotonicity::<i64>(Some(3), Some(4)));
        assert!(!check_monotonicity::<i64>(Some(4), Some(3)));
    }
}
