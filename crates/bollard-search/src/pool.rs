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

use crate::num::SolverNumeric;
use bollard_model::solution::Solution;

/// A pool of solutions, ordered by objective value (ascending: best first).
///
/// Internally, `solutions` stores all solutions contiguously,
/// and `order` stores indices into `solutions`, sorted by objective.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolutionPool<T>
where
    T: SolverNumeric,
{
    solutions: Vec<Solution<T>>, // the solutions
    order: Vec<usize>,           // permutation of [0..solutions.len()), sorted by objective
    pool_size: usize,            // maximum number of solutions to keep
}

impl<T> SolutionPool<T>
where
    T: SolverNumeric,
{
    /// Creates an empty pool with the given maximum size.
    pub fn new(pool_size: usize) -> Self {
        Self {
            solutions: Vec::new(),
            order: Vec::new(),
            pool_size,
        }
    }

    /// Returns the number of solutions currently stored in the pool.
    ///
    /// This is always `<= pool_size()`.
    #[inline]
    pub fn len(&self) -> usize {
        self.solutions.len()
    }

    /// Returns `true` if the pool does not contain any solutions.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    /// Removes all solutions from the pool.
    ///
    /// After calling this, `len() == 0`, and `best()` / `worst()` return `None`,
    /// but the `pool_size` remains unchanged and the pool can be reused.
    #[inline]
    pub fn clear(&mut self) {
        self.solutions.clear();
        self.order.clear();
    }

    /// Returns a reference to the best solution (lowest objective value), if any.
    ///
    /// If the pool is empty, returns `None`.
    #[inline]
    pub fn best(&self) -> Option<&Solution<T>> {
        self.order
            .first()
            .map(|&idx| unsafe { self.solutions.get_unchecked(idx) })
    }

    /// Returns a reference to the worst solution (highest objective value), if any.
    ///
    /// If the pool is empty, returns `None`.
    #[inline]
    pub fn worst(&self) -> Option<&Solution<T>> {
        self.order
            .last()
            .map(|&idx| unsafe { self.solutions.get_unchecked(idx) })
    }

    /// Returns all solutions in the pool ordered from best to worst.
    ///
    /// This allocates a new `Vec<&Solution<T>>` of length `len()`, but preserves
    /// the internal contiguous layout of `solutions`.
    #[inline]
    pub fn ordered_solutions(&self) -> Vec<&Solution<T>> {
        let mut v = Vec::with_capacity(self.order.len());
        for &idx in &self.order {
            unsafe {
                v.push(self.solutions.get_unchecked(idx));
            }
        }
        v
    }

    /// Inserts a solution into the pool, keeping solutions ordered and capped at `pool_size`.
    ///
    /// The solution is inserted according to its `objective_value()`, such that
    /// `best()` always returns the solution with the smallest objective and
    /// `worst()` the one with the largest objective.
    ///
    /// If inserting the new solution causes the pool to exceed its capacity,
    /// the current worst solution is evicted.
    pub fn insert(&mut self, solution: Solution<T>) {
        let obj = solution.objective_value();

        // Index in solutions.
        let new_index = self.solutions.len();
        self.solutions.push(solution);

        // Insert into order by objective.
        let pos = self.order.binary_search_by(|&idx| {
            let s = unsafe { self.solutions.get_unchecked(idx) };
            s.objective_value()
                .partial_cmp(&obj)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let insert_at = match pos {
            Ok(p) | Err(p) => p,
        };
        self.order.insert(insert_at, new_index);

        // If we exceed capacity, remove the worst.
        if self.solutions.len() > self.pool_size {
            self.evict_worst();
        }
    }

    /// Evicts the worst solution (largest objective value) from the pool.
    ///
    /// This is an internal helper used to enforce `pool_size` after insertions
    /// or size changes. It updates both the `solutions` buffer and the
    /// corresponding indices in `order`.
    fn evict_worst(&mut self) {
        if let Some(worst_pos_in_order) = self.order.pop() {
            let worst_idx = worst_pos_in_order;

            // Physically remove worst solution.
            self.solutions.remove(worst_idx);

            // All indices > worst_idx now shift left by 1.
            for idx in &mut self.order {
                if *idx > worst_idx {
                    *idx -= 1;
                }
            }
        }
    }

    /// Returns an iterator over the solutions from best to worst.
    ///
    /// The iterator yields `&Solution<T>` in non-decreasing objective value order.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Solution<T>> {
        self.order
            .iter()
            .map(move |&idx| unsafe { self.solutions.get_unchecked(idx) })
    }

    /// Returns an iterator over the solutions from worst to best.
    ///
    /// The iterator yields `&Solution<T>` in non-increasing objective value order.
    #[inline]
    pub fn iter_rev(&self) -> impl Iterator<Item = &Solution<T>> {
        self.order
            .iter()
            .rev()
            .map(move |&idx| unsafe { self.solutions.get_unchecked(idx) })
    }

    /// Returns the current maximum number of solutions the pool will keep.
    #[inline]
    pub fn pool_size(&self) -> usize {
        self.pool_size
    }

    /// Changes the maximum pool size and evicts worst solutions if needed.
    ///
    /// If `new_size` is smaller than the current `len()`, the worst solutions
    /// are repeatedly evicted until `len() <= new_size`. If `new_size` is
    /// larger than the current capacity, no solutions are added or removed.
    #[inline]
    pub fn set_pool_size(&mut self, new_size: usize) {
        self.pool_size = new_size;
        while self.solutions.len() > self.pool_size {
            self.evict_worst();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bollard_model::index::BerthIndex;

    fn bi(i: usize) -> BerthIndex {
        BerthIndex::new(i)
    }

    /// Helper to build a `Solution<i64>` with given objective and trivial data.
    fn make_solution(obj: i64, n: usize) -> Solution<i64> {
        let berths = (0..n).map(bi).collect::<Vec<_>>();
        let start_times = (0..n as i64).collect::<Vec<_>>();
        Solution::new(obj, berths, start_times)
    }

    #[test]
    fn test_new_pool_is_empty() {
        let pool: SolutionPool<i64> = SolutionPool::new(5);
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
        assert_eq!(pool.pool_size(), 5);
        assert!(pool.best().is_none());
        assert!(pool.worst().is_none());
    }

    #[test]
    fn test_insert_orders_by_objective_and_respects_best_and_worst() {
        let mut pool = SolutionPool::new(10);

        pool.insert(make_solution(30, 1));
        pool.insert(make_solution(10, 1));
        pool.insert(make_solution(20, 1));

        assert_eq!(pool.len(), 3);

        let best = pool.best().unwrap();
        let worst = pool.worst().unwrap();
        assert_eq!(best.objective_value(), 10);
        assert_eq!(worst.objective_value(), 30);

        let ordered: Vec<i64> = pool
            .ordered_solutions()
            .iter()
            .map(|s| s.objective_value())
            .collect();
        assert_eq!(ordered, vec![10, 20, 30]);
    }

    #[test]
    fn test_capacity_enforced_and_worst_evicts() {
        // Pool can only hold 3 solutions.
        let mut pool = SolutionPool::new(3);

        pool.insert(make_solution(50, 1));
        pool.insert(make_solution(10, 1));
        pool.insert(make_solution(30, 1));
        assert_eq!(pool.len(), 3);

        // Insert a better solution -> should evict the current worst (50).
        pool.insert(make_solution(5, 1));

        assert_eq!(pool.len(), 3);

        let ordered: Vec<i64> = pool.iter().map(|s| s.objective_value()).collect();
        // The worst (50) must have been evicted, remaining should be 5,10,30.
        assert_eq!(ordered, vec![5, 10, 30]);

        let best = pool.best().unwrap();
        let worst = pool.worst().unwrap();
        assert_eq!(best.objective_value(), 5);
        assert_eq!(worst.objective_value(), 30);
    }

    #[test]
    fn test_worse_than_all_is_dropped_immediately() {
        let mut pool = SolutionPool::new(2);

        pool.insert(make_solution(10, 1));
        pool.insert(make_solution(20, 1));
        assert_eq!(pool.len(), 2);

        // This is worse than all existing, so once capacity is enforced,
        // it should be the one evicted.
        pool.insert(make_solution(30, 1));

        assert_eq!(pool.len(), 2);
        let ordered: Vec<i64> = pool.iter().map(|s| s.objective_value()).collect();
        assert_eq!(ordered, vec![10, 20]);
    }

    #[test]
    fn test_set_pool_size_can_shrink_and_expand() {
        let mut pool = SolutionPool::new(5);

        pool.insert(make_solution(10, 1));
        pool.insert(make_solution(20, 1));
        pool.insert(make_solution(30, 1));
        pool.insert(make_solution(40, 1));
        pool.insert(make_solution(50, 1));
        assert_eq!(pool.len(), 5);

        // Shrink to 3 -> should keep the 3 best: 10,20,30.
        pool.set_pool_size(3);
        assert_eq!(pool.pool_size(), 3);
        assert_eq!(pool.len(), 3);

        let ordered_after_shrink: Vec<i64> = pool.iter().map(|s| s.objective_value()).collect();
        assert_eq!(ordered_after_shrink, vec![10, 20, 30]);

        // Expanding capacity does not change current contents.
        pool.set_pool_size(10);
        assert_eq!(pool.pool_size(), 10);
        assert_eq!(pool.len(), 3);

        let ordered_after_expand: Vec<i64> = pool.iter().map(|s| s.objective_value()).collect();
        assert_eq!(ordered_after_expand, vec![10, 20, 30]);

        // Insert more solutions now that capacity is larger.
        pool.insert(make_solution(25, 1));
        pool.insert(make_solution(5, 1));

        let ordered_final: Vec<i64> = pool.iter().map(|s| s.objective_value()).collect();
        assert_eq!(ordered_final, vec![5, 10, 20, 25, 30]);
    }

    #[test]
    fn test_clear_resets_pool() {
        let mut pool = SolutionPool::new(3);
        pool.insert(make_solution(10, 1));
        pool.insert(make_solution(20, 1));
        assert_eq!(pool.len(), 2);
        assert!(!pool.is_empty());

        pool.clear();

        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
        assert!(pool.best().is_none());
        assert!(pool.worst().is_none());

        // Still usable after clear.
        pool.insert(make_solution(5, 1));
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.best().unwrap().objective_value(), 5);
    }
}
