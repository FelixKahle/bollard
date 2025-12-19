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

//! Fixed assignments for berth allocation
//!
//! `FixedAssignment<T>` is a compact value object that pins a vessel to a berth
//! at a concrete start time. It’s used to express pre‑placed work, seeds/warm
//! starts, or to serialize schedules from solver output.
//!
//! Ordering
//! - Total order: by `start_time`, then `vessel_index`, then `berth_index`.

use bollard_model::index::{BerthIndex, VesselIndex};

/// A fixed assignment of a vessel to a berth at a specific start time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FixedAssignment<T> {
    // We optimize for T = i64,
    // minimizing the size of FixedAssignment by
    // minimizing padding.
    /// The start time of the assignment.
    pub start_time: T,

    /// The index of the berth.
    pub berth_index: BerthIndex,

    /// The index of the vessel.
    pub vessel_index: VesselIndex,
}

impl<T> FixedAssignment<T> {
    #[inline]
    pub fn new(
        start_time: T,
        berth_index: BerthIndex,
        vessel_index: VesselIndex,
    ) -> FixedAssignment<T> {
        Self {
            start_time,
            berth_index,
            vessel_index,
        }
    }
}

impl<T> PartialOrd for FixedAssignment<T>
where
    T: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for FixedAssignment<T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.start_time
            .cmp(&other.start_time)
            .then(self.vessel_index.cmp(&other.vessel_index))
            .then(self.berth_index.cmp(&other.berth_index))
    }
}

impl<T> std::fmt::Display for FixedAssignment<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FixedAssigned(vessel: {}, berth: {}, start_time: {})",
            self.vessel_index, self.berth_index, self.start_time
        )
    }
}
