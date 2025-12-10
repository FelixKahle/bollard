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

use crate::monitor::search_monitor::{SearchCommand, SearchMonitor};
use bollard_model::{model::Model, solution::Solution};
use num_traits::{PrimInt, Signed};
use std::sync::atomic::AtomicBool;

/// A search monitor that checks an atomic boolean flag to determine
/// whether the search should be interrupted.
#[derive(Debug, Clone)]
pub struct InterruptMonitor<'a, T> {
    stop_flag: &'a AtomicBool,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T> InterruptMonitor<'a, T> {
    /// Creates a new `InterruptMonitor` that monitors the given atomic boolean flag.
    /// The search will be terminated if the flag is set to `true`.
    #[inline(always)]
    pub fn new(stop_flag: &'a AtomicBool) -> Self {
        Self {
            stop_flag,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T> SearchMonitor<T> for InterruptMonitor<'a, T>
where
    T: PrimInt + Signed,
{
    fn name(&self) -> &str {
        "InterruptMonitor"
    }

    fn on_enter_search(&mut self, _model: &Model<T>) {}
    fn on_exit_search(&mut self) {}
    fn on_solution_found(&mut self, _solution: &Solution<T>) {}
    fn on_step(&mut self) {}

    fn search_command(&self) -> super::search_monitor::SearchCommand {
        if self.stop_flag.load(std::sync::atomic::Ordering::Relaxed) {
            SearchCommand::Terminate("Interrupt signal received".to_string())
        } else {
            SearchCommand::Continue
        }
    }
}

#[cfg(test)]
mod tests {
    use super::InterruptMonitor;
    use crate::monitor::search_monitor::{SearchCommand, SearchMonitor};
    use std::sync::atomic::{AtomicBool, Ordering};

    type IntegerType = i64;

    #[test]
    fn test_interrupt_monitor_continues_when_flag_is_clear() {
        let flag = AtomicBool::new(false);
        let monitor = InterruptMonitor::<IntegerType>::new(&flag);

        // No need to call lifecycle hooks; we test the command directly
        match monitor.search_command() {
            SearchCommand::Continue => {}
            other => panic!("expected Continue, got {:?}", other),
        }
    }

    #[test]
    fn test_interrupt_monitor_terminates_when_flag_is_set() {
        let flag = AtomicBool::new(false);
        let monitor = InterruptMonitor::<IntegerType>::new(&flag);

        // Set the flag
        flag.store(true, Ordering::Relaxed);

        match monitor.search_command() {
            SearchCommand::Terminate(reason) => {
                assert_eq!(reason, "Interrupt signal received");
            }
            other => panic!("expected Terminate, got {:?}", other),
        }
    }
}
