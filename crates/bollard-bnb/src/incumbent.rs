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

//! Incumbent management for branch‑and‑bound
//!
//! Declares `IncumbentStore<T>`, a minimal interface to read/update the best
//! known objective (upper bound) and publish new solutions during search.
//! This abstracts over local (single‑threaded) and shared (multi‑solver) use.
//!
//! Implementations
//! - `NoSharedIncumbent<T>`: local only. `initial_upper_bound = T::MAX`,
//!   `tighten(x) = x`, and `on_solution_found` is a no‑op.
//! - `SharedIncumbentAdapter<'a, T>`: wraps `bollard_search::incumbent::SharedIncumbent<T>`;
//!   `initial_upper_bound()` mirrors the shared value, `tighten(x)` returns
//!   `min(shared, x)`, and `on_solution_found` attempts installation.
//!
//! Notes
//! - `SharedIncumbentAdapter` holds a borrowed handle and is lifetime‑bound.
//! - Use shared incumbents to coordinate bounds across parallel/portfolio runs.
//! - Constrained by `T: SolverNumeric`.

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

#[cfg(test)]
mod tests {
    use super::*;
    use bollard_model::{
        index::{BerthIndex, VesselIndex},
        model::ModelBuilder,
        solution::Solution,
        time::ProcessingTime,
    };
    use bollard_search::incumbent::SharedIncumbent;

    type IntegerType = i64;

    fn bi(i: usize) -> BerthIndex {
        BerthIndex::new(i)
    }
    fn vi(i: usize) -> VesselIndex {
        VesselIndex::new(i)
    }

    fn make_solution(objective: IntegerType, n: usize) -> Solution<IntegerType> {
        let berths = (0..n).map(bi).collect::<Vec<_>>();
        let start_times = (0..n).map(|i| i as IntegerType).collect::<Vec<_>>();
        Solution::new(objective, berths, start_times)
    }

    #[test]
    fn no_shared_incumbent_initial_upper_bound_is_max() {
        let store: NoSharedIncumbent<IntegerType> = NoSharedIncumbent::new();
        assert_eq!(store.initial_upper_bound(), IntegerType::MAX);
    }

    #[test]
    fn no_shared_incumbent_tighten_is_passthrough() {
        let store: NoSharedIncumbent<IntegerType> = NoSharedIncumbent::new();
        let values = [0, 1, 42, IntegerType::MAX - 1];
        for &val in &values {
            assert_eq!(store.tighten(val), val);
        }
    }

    #[test]
    fn shared_incumbent_adapter_initial_upper_bound_reads_shared() {
        let shared = SharedIncumbent::<IntegerType>::new();
        let adapter = SharedIncumbentAdapter::new(&shared);

        // Fresh SharedIncumbent starts with i64::MAX
        assert_eq!(shared.upper_bound(), i64::MAX);
        // Adapter should convert to Int::max_value()
        assert_eq!(adapter.initial_upper_bound(), IntegerType::MAX);

        // Install a better solution to update shared upper bound
        let s = make_solution(120, 3);
        assert!(shared.try_install(&s));
        assert_eq!(shared.upper_bound(), 120);

        // Adapter now reflects the new bound
        assert_eq!(adapter.initial_upper_bound(), 120);
    }

    #[test]
    fn shared_incumbent_adapter_tighten_returns_min_with_shared_bound() {
        let shared = SharedIncumbent::<IntegerType>::new();
        let adapter = SharedIncumbentAdapter::new(&shared);

        // Install current best in the shared incumbent
        let best = make_solution(200, 2);
        assert!(shared.try_install(&best));
        assert_eq!(shared.upper_bound(), 200);

        // Local best is worse -> tighten returns shared (min)
        let local_worse = 350;
        assert_eq!(adapter.tighten(local_worse), 200);

        // Local best is better -> returns local (min)
        let local_better = 150;
        assert_eq!(adapter.tighten(local_better), 150);
    }

    #[test]
    fn shared_incumbent_adapter_on_solution_found_installs_in_shared() {
        let shared = SharedIncumbent::<IntegerType>::new();
        let adapter = SharedIncumbentAdapter::new(&shared);

        // Build a simple solution and report it via adapter
        let s = make_solution(95, 4);
        adapter.on_solution_found(&s);

        // Shared should now contain the incumbent
        assert_eq!(shared.upper_bound(), 95);
        let snap = shared.snapshot().expect("snapshot should be Some");
        assert_eq!(snap.objective_value(), 95);
        assert_eq!(snap.num_vessels(), 4);
        assert_eq!(snap.berth_for_vessel(vi(3)).get(), 3);
        assert_eq!(snap.start_time_for_vessel(vi(1)), 1);
    }

    #[test]
    fn integration_with_model_builder_constructs_realistic_solution() {
        // Use the real ModelBuilder to construct a simple instance.
        let mut mb = ModelBuilder::<IntegerType>::new(2, 3);

        // Set arrivals, departures, weights, and processing times.
        for v in 0..3 {
            mb.set_vessel_arrival_time(vi(v), v as IntegerType);
            mb.set_vessel_latest_departure_time(vi(v), 1000);
            mb.set_vessel_weight(vi(v), 1);
        }
        // Allow processing on berth 0 with simple times, disallow on berth 1 for vessel 2.
        mb.set_vessel_processing_time(vi(0), bi(0), ProcessingTime::some(5));
        mb.set_vessel_processing_time(vi(1), bi(0), ProcessingTime::some(7));
        mb.set_vessel_processing_time(vi(2), bi(0), ProcessingTime::some(3));
        mb.set_vessel_processing_time(vi(0), bi(1), ProcessingTime::some(6));
        mb.set_vessel_processing_time(vi(1), bi(1), ProcessingTime::some(8));
        mb.set_vessel_processing_time(vi(2), bi(1), ProcessingTime::none());

        let model = mb.build();

        // Construct a feasible-looking solution for testing incumbent install paths.
        // Map v0 -> b0, v1 -> b1, v2 -> b0 with start times aligned to arrivals.
        let berths = vec![bi(0), bi(1), bi(0)];
        let starts = vec![0, 1, 2];
        let sol = Solution::new(150, berths, starts);

        let shared = SharedIncumbent::<IntegerType>::new();
        let adapter = SharedIncumbentAdapter::new(&shared);

        adapter.on_solution_found(&sol);
        assert_eq!(shared.upper_bound(), 150);

        let snap = shared.snapshot().expect("snapshot present");
        assert_eq!(snap.objective_value(), 150);
        assert_eq!(snap.num_vessels(), model.num_vessels());
        // sanity checks vs model dimensions
        assert_eq!(model.num_vessels(), 3);
        assert_eq!(model.num_berths(), 2);
    }
}
