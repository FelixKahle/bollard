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

//! # Shared Incumbent (Best Solution Holder)
//!
//! A concurrent container for the best solution discovered so far during search.
//! It exposes a fast, lock-free upper bound via an atomic and stores the actual
//! `Solution<T>` behind a `Mutex` as the source of truth. Designed for exact
//! search pipelines where multiple threads propose improvements.
//!
//! ## Motivation
//!
//! - Fast heuristic checks: a cheap atomic upper bound short-circuits attempts
//!   to install obviously worse candidates without locking.
//! - Correctness by locking: the authoritative incumbent is protected by a
//!   `Mutex`, ensuring consistent updates even under contention.
//! - Simple sentinel: `upper_bound` starts at `i64::MAX` meaning "no incumbent yet."
//!
//! ## Highlights
//!
//! - `try_install(&Solution<T>) -> bool` installs strictly better candidates,
//!   updating both the snapshot and the atomic upper bound.
//! - `snapshot() -> Option<Solution<T>>` returns a cloned snapshot of the current
//!   incumbent (if any).
//! - `upper_bound() -> i64` and `upper_bound_as::<T>() -> Result<T, _>` for
//!   quick reads and typed conversions.
//! - Concurrency: atomic reads/writes use `Ordering::Relaxed` for performance,
//!   while the mutex ensures correctness of the stored solution.
//!
//! ## Usage
//!
//! ```rust
//! use bollard_search::incumbent::SharedIncumbent;
//! use bollard_model::solution::Solution;
//!
//! let inc: SharedIncumbent<i64> = SharedIncumbent::new();
//! let candidate = Solution::new(100, Vec::new(), Vec::new());
//!
//! if inc.try_install(&candidate) {
//!     // Installed as new best
//! }
//!
//! let ub = inc.upper_bound();           // fast atomic read
//! let snap = inc.snapshot();            // optional cloned solution
//! ```

use bollard_model::solution::Solution;
use num_traits::{PrimInt, Signed};
use std::sync::{Mutex, atomic::AtomicI64};

/// A concurrent holder for the best (incumbent) solution found during search.
///
/// This structure maintains:
/// - an `AtomicI64` upper bound (objective value) for fast, lock-free reads, and
/// - a `Mutex<Option<Solution<T>>>` for the actual solution, which is the source of truth.
///
/// Concurrency and memory ordering:
/// - The upper bound is loaded/stored with `Ordering::Relaxed`. This is sufficient because
///   it serves as a heuristic to short-circuit work (e.g., avoid locking when a candidate
///   is obviously worse). All correctness-sensitive state (the solution and its objective)
///   is synchronized via the `Mutex`.
///
/// Sentinel initialization:
/// - `upper_bound` is initialized to `i64::MAX` to represent “no solution installed yet.”
///   Using an `Option<AtomicI64>` would introduce additional branching and complexity
///   with negligible benefit in this use case. Since we minimize the objective and cannot
///   represent values greater than `i64::MAX`, the sentinel is both simple and effective.
#[derive(Debug)]
pub struct SharedIncumbent<T> {
    /// Objective of the incumbent solution stored as `i64` for atomic access.
    ///
    /// When Rust gains support for generic atomics (e.g., `Atomic<T>`), consider
    /// migrating to a type that matches the objective’s representation.
    ///
    /// See the tracking issue:
    /// - Generic atomics: [rust-lang/rust#130539](https://github.com/rust-lang/rust/issues/130539)
    upper_bound: AtomicI64,

    /// The incumbent solution, protected by a mutex for safe concurrent access.
    ///
    /// # Note
    ///
    /// We did not run any tests whether using `parking_lot::Mutex` would yield better performance
    /// than the standard library `Mutex`. The decision was made after an Medium article
    /// [Inside Rusts std and parking lot mutexes who wins](https://medium.com/@cuongleqq/inside-rusts-std-and-parking-lot-mutexes-who-wins-fb0ae1fb3041)
    solution: Mutex<Option<Solution<T>>>,
}

impl<T> Default for SharedIncumbent<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> std::fmt::Display for SharedIncumbent<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let upper_bound = self.upper_bound();
        write!(f, "Incumbent(upper_bound: {})", upper_bound)
    }
}

impl<T> SharedIncumbent<T> {
    /// Creates a new shared incumbent with no solution installed.
    /// The initial upper bound is set to i64::MAX.
    #[inline]
    pub fn new() -> Self {
        SharedIncumbent {
            upper_bound: AtomicI64::new(i64::MAX),
            solution: Mutex::new(None),
        }
    }

    /// Returns the current upper bound.
    #[inline]
    pub fn upper_bound(&self) -> i64 {
        self.upper_bound.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns the current upper bound converted to type T.
    #[inline]
    pub fn upper_bound_as(&self) -> Result<T, <T as std::convert::TryFrom<i64>>::Error>
    where
        T: TryFrom<i64>,
    {
        let val = self.upper_bound.load(std::sync::atomic::Ordering::Relaxed);
        T::try_from(val)
    }

    /// Returns a snapshot of the current incumbent solution, if any.
    #[inline]
    pub fn snapshot(&self) -> Option<Solution<T>>
    where
        T: Clone,
    {
        let guard = self.solution.lock().unwrap();
        guard.clone()
    }

    /// Attempts to install the given candidate solution as the new incumbent.
    /// Returns `true` if the candidate was installed, `false` otherwise.
    #[inline]
    pub fn try_install(&self, candidate: &Solution<T>) -> bool
    where
        T: PrimInt + Signed + Into<i64>,
    {
        let candidate_objective: i64 = candidate.objective_value().into();
        let current_upper_bound = self.upper_bound();

        // We are minimizing, so lower is better.
        if candidate_objective >= current_upper_bound {
            return false;
        }

        let mut guard = self.solution.lock().unwrap();
        // Another thread might have updated the solution while we were waiting for the lock.
        // We must compare against the *actual* solution in the Mutex, not the atomic hint we read earlier.
        if let Some(current_solution) = guard.as_ref() {
            let current_objective: i64 = current_solution.objective_value().into();
            if candidate_objective >= current_objective {
                return false;
            }
        }

        // Install the new incumbent.
        *guard = Some(candidate.clone());
        // Update the upper bound atomically.
        self.upper_bound
            .store(candidate_objective, std::sync::atomic::Ordering::Relaxed);

        true
    }
}

#[cfg(test)]
mod tests {
    use super::SharedIncumbent;
    use bollard_model::index::{BerthIndex, VesselIndex};
    use bollard_model::solution::Solution;
    use std::sync::Arc;
    use std::thread;

    fn bi(i: usize) -> BerthIndex {
        BerthIndex::new(i)
    }

    fn vi(i: usize) -> VesselIndex {
        VesselIndex::new(i)
    }

    fn make_solution(objective: i64, n: usize) -> Solution<i64> {
        // Build a simple valid solution with n vessels, mapping vessel i -> berth i, start time i
        let berths = (0..n).map(bi).collect::<Vec<_>>();
        let start_times = (0..n).map(|i| i as i64).collect::<Vec<_>>();
        Solution::new(objective, berths, start_times)
    }

    #[test]
    fn test_initial_state() {
        let inc: SharedIncumbent<i64> = SharedIncumbent::new();
        assert_eq!(inc.upper_bound(), i64::MAX);
        assert!(inc.snapshot().is_none());
    }

    #[test]
    fn test_install_better_solution_updates_upper_bound_and_snapshot() {
        let inc: SharedIncumbent<i64> = SharedIncumbent::new();
        let s = make_solution(100, 3);

        let installed = inc.try_install(&s);
        assert!(installed);
        assert_eq!(inc.upper_bound(), 100);

        let snap = inc.snapshot().expect("snapshot should be Some");
        assert_eq!(snap.objective_value(), 100);
        assert_eq!(snap.num_vessels(), 3);
        // sanity check mappings
        assert_eq!(snap.berth_for_vessel(vi(0)).get(), 0);
        assert_eq!(snap.start_time_for_vessel(vi(2)), 2);
    }

    #[test]
    fn test_reject_worse_or_equal_candidates() {
        let inc: SharedIncumbent<i64> = SharedIncumbent::new();

        let best = make_solution(100, 2);
        assert!(inc.try_install(&best));
        assert_eq!(inc.upper_bound(), 100);

        let worse = make_solution(150, 2);
        assert!(!inc.try_install(&worse));
        assert_eq!(inc.upper_bound(), 100);

        let equal = make_solution(100, 2);
        assert!(!inc.try_install(&equal));
        assert_eq!(inc.upper_bound(), 100);

        // Snapshot remains the original best
        let snap = inc.snapshot().unwrap();
        assert_eq!(snap.objective_value(), 100);
    }

    #[test]
    fn test_reject_worse_after_mutex_check() {
        // Ensure the path that compares against the actual mutex-held solution is exercised.
        let inc: SharedIncumbent<i64> = SharedIncumbent::new();

        // Install a best solution
        let s1 = make_solution(80, 2);
        assert!(inc.try_install(&s1));
        assert_eq!(inc.upper_bound(), 80);

        // Now attempt to install a candidate that is better than the atomic upper_bound hint
        // if it were stale, but worse than the actual mutex-held solution.
        let s2 = make_solution(90, 2);
        assert!(!inc.try_install(&s2));
        assert_eq!(inc.upper_bound(), 80);

        let snap = inc.snapshot().unwrap();
        assert_eq!(snap.objective_value(), 80);
    }

    #[test]
    fn test_concurrent_installs_minimum_wins() {
        let inc = Arc::new(SharedIncumbent::<i64>::new());
        let objectives = vec![300, 200, 400, 50, 120, 75, 500, 60, 90];

        let mut handles = Vec::new();
        for obj in objectives.iter().cloned() {
            let inc_cloned = Arc::clone(&inc);
            handles.push(thread::spawn(move || {
                let s = make_solution(obj, 4);
                inc_cloned.try_install(&s)
            }));
        }

        // Join threads and collect install outcomes
        let results = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>();
        assert!(
            results.iter().any(|&r| r),
            "at least one install should succeed"
        );

        // The final upper bound should be the minimum objective
        let min_obj = *objectives.iter().min().unwrap();
        assert_eq!(inc.upper_bound(), min_obj);

        // Snapshot should exist and reflect the minimum objective
        let snap = inc
            .snapshot()
            .expect("snapshot should be Some after installs");
        assert_eq!(snap.objective_value(), min_obj);

        // Sanity: solution shape is consistent
        assert_eq!(snap.num_vessels(), 4);
    }

    #[test]
    fn test_incumbent_with_i16() {
        // Use i16 as the objective type
        let inc: SharedIncumbent<i16> = SharedIncumbent::new();

        // Build solutions with i16 objective values
        let berths = vec![bi(0), bi(1), bi(2)];
        let starts_i16 = vec![0i16, 1i16, 2i16];

        let best = Solution::new(50i16, berths.clone(), starts_i16.clone());
        let worse = Solution::new(120i16, berths.clone(), starts_i16.clone());
        let equal = Solution::new(50i16, berths.clone(), starts_i16.clone());

        // First install should succeed
        assert!(inc.try_install(&best));
        // Upper bound is i64, should reflect the i16 objective via Into<i64>
        assert_eq!(inc.upper_bound(), 50i64);

        // Worse candidate should be rejected
        assert!(!inc.try_install(&worse));
        assert_eq!(inc.upper_bound(), 50i64);

        // Equal candidate should be rejected
        assert!(!inc.try_install(&equal));
        assert_eq!(inc.upper_bound(), 50i64);

        // Snapshot should be Some and carry i16 objective/start times
        let snap = inc.snapshot().expect("snapshot should be Some");
        assert_eq!(snap.objective_value(), 50i16);
        assert_eq!(snap.num_vessels(), 3);
        assert_eq!(snap.start_time_for_vessel(vi(2)), 2i16);
        assert_eq!(snap.berth_for_vessel(vi(1)).get(), 1);
    }
}
