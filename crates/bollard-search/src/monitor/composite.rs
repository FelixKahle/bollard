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

use crate::monitor::{
    index::MonitorIndex,
    search_monitor::{SearchCommand, SearchMonitor},
};
use bollard_model::solution::Solution;
use num_traits::{PrimInt, Signed};

/// A composite monitor that aggregates multiple monitors and forwards events to all of them.
pub struct CompositeMonitor<'a, T> {
    monitors: Vec<Box<dyn SearchMonitor<T> + 'a>>,
}

impl<'a, T> std::fmt::Debug for CompositeMonitor<'a, T>
where
    T: PrimInt + Signed,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let monitors_str = self
            .monitors
            .iter()
            .map(|m| m.name())
            .collect::<Vec<&str>>()
            .join(", ");

        f.debug_struct("CompositeMonitor")
            .field("monitors", &monitors_str)
            .finish()
    }
}

impl<'a, T> std::fmt::Display for CompositeMonitor<'a, T>
where
    T: PrimInt + Signed + Copy,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let monitors_str = self
            .monitors
            .iter()
            .map(|m| m.name())
            .collect::<Vec<&str>>()
            .join(", ");

        write!(f, "CompositeMonitor([{}])", monitors_str)
    }
}

impl<'a, T> Default for CompositeMonitor<'a, T>
where
    T: PrimInt + Signed,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> CompositeMonitor<'a, T>
where
    T: PrimInt + Signed,
{
    /// Creates a new empty `CompositeMonitor`.
    #[inline]
    pub fn new() -> CompositeMonitor<'a, T> {
        CompositeMonitor {
            monitors: Vec::new(),
        }
    }

    /// Creates a new `CompositeMonitor` with the specified capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> CompositeMonitor<'a, T> {
        CompositeMonitor {
            monitors: Vec::with_capacity(capacity),
        }
    }

    /// Creates a new `CompositeMonitor` from a vector of boxed monitors.
    #[inline]
    pub fn from_vec(monitors: Vec<Box<dyn SearchMonitor<T> + 'a>>) -> CompositeMonitor<'a, T> {
        CompositeMonitor { monitors }
    }

    /// Adds a new monitor to the composite monitor.
    #[inline]
    pub fn add_monitor<M>(&mut self, monitor: M)
    where
        M: SearchMonitor<T> + 'a,
    {
        self.monitors.push(Box::new(monitor));
    }

    /// Adds a new boxed monitor to the composite monitor.
    #[inline]
    pub fn add_monitor_boxed(&mut self, monitor: Box<dyn SearchMonitor<T> + 'a>) {
        self.monitors.push(monitor);
    }

    /// Returns the number of monitors in the composite monitor.
    #[inline]
    pub fn len(&self) -> usize {
        self.monitors.len()
    }

    /// Returns `true` if the composite monitor contains no monitors.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.monitors.is_empty()
    }

    /// Returns a reference to the monitor at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if `monitor_index` is out of bounds.
    #[inline]
    pub fn monitor(&'a self, monitor_index: MonitorIndex) -> &'a dyn SearchMonitor<T> {
        let index = monitor_index.get();
        debug_assert!(
            index < self.monitors.len(),
            "called `CompositeMonitor::monitor` with monitor index out of bounds: the len is {} but the index is {}",
            index,
            self.monitors.len()
        );

        self.monitors[index].as_ref()
    }

    /// Returns a mutable reference to the monitor at the specified index.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `monitor_index` is out of bounds.
    ///
    /// # Safety
    ///
    /// The caller mus ensure tha `monitor_index` is within bounds
    /// `0..self.len()`.
    #[inline]
    pub unsafe fn monitor_unchecked(
        &'a mut self,
        monitor_index: MonitorIndex,
    ) -> &'a dyn SearchMonitor<T> {
        let index = monitor_index.get();
        debug_assert!(
            index < self.monitors.len(),
            "called `CompositeMonitor::monitor_unchecked` with monitor index out of bounds: the len is {} but the index is {}",
            index,
            self.monitors.len()
        );

        unsafe { self.monitors.get_unchecked(index).as_ref() }
    }
}

impl<'a, T> FromIterator<Box<dyn SearchMonitor<T> + 'a>> for CompositeMonitor<'a, T>
where
    T: PrimInt + Signed,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Box<dyn SearchMonitor<T> + 'a>>,
    {
        let monitors: Vec<Box<dyn SearchMonitor<T> + 'a>> = iter.into_iter().collect();
        CompositeMonitor { monitors }
    }
}

impl<'a, T> SearchMonitor<T> for CompositeMonitor<'a, T>
where
    T: PrimInt + Signed,
{
    fn name(&self) -> &str {
        "CompositeMonitor"
    }

    fn on_solution_found(&mut self, solution: &Solution<T>) {
        for monitor in &mut self.monitors {
            monitor.on_solution_found(solution);
        }
    }

    fn on_step(&mut self) {
        for monitor in &mut self.monitors {
            monitor.on_step();
        }
    }

    fn search_command(&self) -> SearchCommand {
        // We could have used `Iterator::find_map` here, but that would require
        // creating an iterator, an more importantly, an `Option` on each iteration.
        // We can safe those few cycles by using a simple for loop.
        // Seems overengineered, but in high-performance scenarios every cycle counts,
        // and this line here is suspected to be called incredibly often.
        for monitor in &self.monitors {
            if let SearchCommand::Terminate(reason) = monitor.search_command() {
                return SearchCommand::Terminate(reason);
            }
        }
        SearchCommand::Continue
    }

    fn on_enter_search(&mut self, model: &bollard_model::model::Model<T>) {
        for monitor in &mut self.monitors {
            monitor.on_enter_search(model);
        }
    }

    fn on_exit_search(&mut self) {
        for monitor in &mut self.monitors {
            monitor.on_exit_search();
        }
    }
}
