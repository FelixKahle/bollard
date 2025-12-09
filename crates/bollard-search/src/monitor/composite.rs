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
    #[inline]
    pub fn new() -> CompositeMonitor<'a, T> {
        CompositeMonitor {
            monitors: Vec::new(),
        }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> CompositeMonitor<'a, T> {
        CompositeMonitor {
            monitors: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn from_vec(monitors: Vec<Box<dyn SearchMonitor<T>>>) -> CompositeMonitor<'a, T> {
        CompositeMonitor { monitors }
    }

    #[inline]
    pub fn add_monitor<M>(&mut self, monitor: M)
    where
        M: SearchMonitor<T> + 'a,
    {
        self.monitors.push(Box::new(monitor));
    }

    #[inline]
    pub fn add_monitor_boxed(&mut self, monitor: Box<dyn SearchMonitor<T> + 'a>) {
        self.monitors.push(monitor);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.monitors.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.monitors.is_empty()
    }

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

        unsafe { self.monitors.get_unchecked_mut(index).as_mut() }
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
        for monitor in &self.monitors {
            let command = monitor.search_command();
            if let SearchCommand::Terminate(reason) = command {
                return SearchCommand::Terminate(reason);
            }
        }
        SearchCommand::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bollard_model::solution::Solution;

    #[derive(Default)]
    struct TestMonitor<T> {
        name: &'static str,
        steps: usize,
        solutions_seen: usize,
        last_objective: Option<T>,
        command: SearchCommand,
    }

    impl<T> TestMonitor<T> {
        fn new(name: &'static str) -> Self {
            Self {
                name,
                steps: 0,
                solutions_seen: 0,
                last_objective: None,
                command: SearchCommand::Continue,
            }
        }

        fn with_command(name: &'static str, command: SearchCommand) -> Self {
            Self {
                name,
                steps: 0,
                solutions_seen: 0,
                last_objective: None,
                command,
            }
        }
    }

    impl<T> SearchMonitor<T> for TestMonitor<T>
    where
        T: PrimInt + Signed,
    {
        fn name(&self) -> &str {
            self.name
        }

        fn on_solution_found(&mut self, solution: &Solution<T>) {
            self.solutions_seen += 1;
            self.last_objective = Some(solution.objective_value());
        }

        fn on_step(&mut self) {
            self.steps += 1;
        }

        fn search_command(&self) -> SearchCommand {
            self.command.clone()
        }
    }

    #[test]
    fn test_new_and_capacity() {
        let cm: CompositeMonitor<i64> = CompositeMonitor::new();
        assert!(cm.is_empty());
        assert_eq!(cm.len(), 0);

        let cm2: CompositeMonitor<i64> = CompositeMonitor::with_capacity(10);
        assert!(cm2.is_empty());
        assert_eq!(cm2.len(), 0);
    }

    #[test]
    fn test_add_monitor_and_len() {
        let mut cm: CompositeMonitor<i64> = CompositeMonitor::new();
        cm.add_monitor(TestMonitor::<i64>::new("m1"));
        cm.add_monitor(TestMonitor::<i64>::new("m2"));
        assert_eq!(cm.len(), 2);
        assert!(!cm.is_empty());
    }

    #[test]
    fn test_add_monitor_boxed() {
        let mut cm: CompositeMonitor<i64> = CompositeMonitor::new();
        let m = Box::new(TestMonitor::<i64>::new("boxed"));
        cm.add_monitor_boxed(m);
        assert_eq!(cm.len(), 1);
        assert_eq!(cm.monitor(MonitorIndex::new(0)).name(), "boxed");
    }

    #[test]
    fn test_from_vec() {
        let v: Vec<Box<dyn SearchMonitor<i64>>> = vec![
            Box::new(TestMonitor::<i64>::new("a")),
            Box::new(TestMonitor::<i64>::new("b")),
        ];
        let cm = CompositeMonitor::from_vec(v);
        assert_eq!(cm.len(), 2);
        assert_eq!(cm.monitor(MonitorIndex::new(0)).name(), "a");
        assert_eq!(cm.monitor(MonitorIndex::new(1)).name(), "b");
    }

    #[test]
    fn test_monitor_access() {
        let mut cm: CompositeMonitor<i64> = CompositeMonitor::new();
        cm.add_monitor(TestMonitor::<i64>::new("first"));
        cm.add_monitor(TestMonitor::<i64>::new("second"));

        let m0 = cm.monitor(MonitorIndex::new(0));
        let m1 = cm.monitor(MonitorIndex::new(1));
        assert_eq!(m0.name(), "first");
        assert_eq!(m1.name(), "second");
    }

    #[test]
    fn test_search_command_continue_and_terminate_propagation() {
        let mut cm: CompositeMonitor<i64> = CompositeMonitor::new();
        cm.add_monitor(TestMonitor::<i64>::with_command(
            "c1",
            SearchCommand::Continue,
        ));
        cm.add_monitor(TestMonitor::<i64>::with_command(
            "t1",
            SearchCommand::Terminate("stop now".into()),
        ));
        cm.add_monitor(TestMonitor::<i64>::with_command(
            "c2",
            SearchCommand::Continue,
        ));

        // The composite should surface the first Terminate it encounters
        match cm.search_command() {
            SearchCommand::Continue => panic!("expected termination"),
            SearchCommand::Terminate(reason) => assert_eq!(reason, "stop now"),
        }
    }

    #[test]
    fn test_display_and_debug() {
        let mut cm: CompositeMonitor<i64> = CompositeMonitor::new();
        cm.add_monitor(TestMonitor::<i64>::new("alpha"));
        cm.add_monitor(TestMonitor::<i64>::new("beta"));

        let disp = format!("{}", cm);
        assert!(disp.contains("CompositeMonitor([alpha, beta])"));

        let dbg = format!("{:?}", cm);
        assert!(dbg.contains("CompositeMonitor"));
        assert!(dbg.contains("alpha"));
        assert!(dbg.contains("beta"));
    }
}
