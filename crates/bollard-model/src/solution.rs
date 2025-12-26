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

use crate::index::{BerthIndex, VesselIndex};
use num_traits::{PrimInt, Signed};

/// The final solution to the Berth Allocation Problem.
///
/// This struct uses a Structure of Arrays (SoA) layout.
/// Data is indexed directly by `VesselIndex` (i.e., index `i` corresponds to vessel `i`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Solution<T> {
    /// The total objective cost of this solution.
    objective_value: T,

    /// The assigned berth for each vessel.
    /// `berths[v]` is the berth assigned to vessel `v`.
    berths: Vec<BerthIndex>,

    /// The assigned start time for each vessel.
    /// `start_times[v]` is the start time for vessel `v`.
    start_times: Vec<T>,
}

impl<T> Solution<T>
where
    T: PrimInt + Signed + Copy,
{
    /// Constructs a new `Solution`.
    ///
    /// # Panics
    ///
    /// Panics if `berths` and `start_times` have different lengths.
    pub fn new(objective_value: T, berths: Vec<BerthIndex>, start_times: Vec<T>) -> Self {
        assert_eq!(
            berths.len(),
            start_times.len(),
            "called Solution::new with inconsistent vector lengths: berths.len() = {}, start_times.len() = {}",
            berths.len(),
            start_times.len()
        );

        Self {
            objective_value,
            berths,
            start_times,
        }
    }

    /// Returns the assigned berth for a specific vessel.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is out of bounds.
    #[inline]
    pub fn berth_for_vessel(&self, vessel_index: VesselIndex) -> BerthIndex {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels(),
            "called `Solution::berth_for_vessel` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        self.berths[index]
    }

    /// Returns the assigned start time for a specific vessel.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is out of bounds.
    #[inline]
    pub fn start_time_for_vessel(&self, vessel_index: VesselIndex) -> T {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels(),
            "called `Solution::start_time_for_vessel` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        self.start_times[vessel_index.get()]
    }

    /// Returns the number of vessels in this solution.
    #[inline]
    pub fn num_vessels(&self) -> usize {
        self.berths.len()
    }

    /// Returns the total objective value of this solution.
    #[inline]
    pub fn objective_value(&self) -> T {
        self.objective_value
    }

    /// Returns a slice of assigned berths for all vessels.
    #[inline]
    pub fn berths(&self) -> &[BerthIndex] {
        &self.berths
    }

    /// Returns a slice of assigned start times for all vessels.
    #[inline]
    pub fn start_times(&self) -> &[T] {
        &self.start_times
    }
}

impl<T> std::fmt::Display for Solution<T>
where
    T: PrimInt + Signed + Copy + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Solution Summary")?;
        writeln!(f, "   Objective Value: {}", self.objective_value)?;
        writeln!(f)?;

        if self.num_vessels() == 0 {
            writeln!(f, "   (No vessels assigned)")?;
            return Ok(());
        }

        writeln!(
            f,
            "   {:<10} | {:<10} | {:<12}",
            "Vessel", "Berth", "Start Time"
        )?;
        writeln!(f, "   {:-<10}-+-{:-<10}-+-{:-<12}", "", "", "")?;
        for i in 0..self.num_vessels() {
            let berth_id = self.berths[i].get();
            let start_time = self.start_times[i];
            writeln!(f, "   {:<10} | {:<10} | {:<12}", i, berth_id, start_time)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{BerthIndex, VesselIndex};

    fn bi(i: usize) -> BerthIndex {
        BerthIndex::new(i)
    }

    fn vi(i: usize) -> VesselIndex {
        VesselIndex::new(i)
    }

    #[test]
    fn test_new_and_basic_accessors() {
        // Objective value and consistent vectors
        let objective = 123i64;
        let berths = vec![bi(0), bi(2), bi(1)];
        let start_times = vec![10i64, 25i64, 17i64];

        let sol = Solution::new(objective, berths.clone(), start_times.clone());

        // Objective
        assert_eq!(sol.objective_value(), objective);

        // Num vessels
        assert_eq!(sol.num_vessels(), 3);

        // Slices
        assert_eq!(sol.berths(), &berths[..]);
        assert_eq!(sol.start_times(), &start_times[..]);

        // Per-vessel getters
        assert_eq!(sol.berth_for_vessel(vi(0)).get(), 0);
        assert_eq!(sol.berth_for_vessel(vi(1)).get(), 2);
        assert_eq!(sol.berth_for_vessel(vi(2)).get(), 1);

        assert_eq!(sol.start_time_for_vessel(vi(0)), 10);
        assert_eq!(sol.start_time_for_vessel(vi(1)), 25);
        assert_eq!(sol.start_time_for_vessel(vi(2)), 17);
    }

    #[test]
    #[should_panic(expected = "called Solution::new with inconsistent vector lengths")]
    fn test_new_panics_on_length_mismatch() {
        // berths.len() != start_times.len() should panic
        let _ = Solution::new(0i64, vec![bi(0), bi(1)], vec![10i64]);
    }

    #[test]
    fn test_empty_solution_is_valid() {
        let sol = Solution::new(0i64, Vec::new(), Vec::new());
        assert_eq!(sol.objective_value(), 0);
        assert_eq!(sol.num_vessels(), 0);
        assert_eq!(sol.berths(), &[]);
        assert_eq!(sol.start_times(), &[]);
    }

    #[test]
    fn test_clone_eq_and_debug() {
        let sol = Solution::new(42i64, vec![bi(0), bi(1)], vec![5i64, 7i64]);
        let sol2 = sol.clone();
        assert_eq!(sol, sol2);

        // Debug should include field names
        let dbg = format!("{:?}", sol);
        assert!(dbg.contains("Solution"));
        assert!(dbg.contains("objective_value"));
        assert!(dbg.contains("berths"));
        assert!(dbg.contains("start_times"));
    }

    #[test]
    fn test_display_formatting_example() {
        let sol = Solution::new(100i64, vec![bi(0), bi(1)], vec![10i64, 20i64]);

        let displayed = format!("{}", sol);

        let mut expected = String::new();
        expected.push_str("Solution Summary\n");
        expected.push_str("   Objective Value: 100\n");
        expected.push('\n');
        expected.push_str("   Vessel     | Berth      | Start Time  \n");
        expected.push_str("   -----------+------------+-------------\n");
        expected.push_str("   0          | 0          | 10          \n");
        expected.push_str("   1          | 1          | 20          \n");

        assert_eq!(displayed, expected);
    }
}
