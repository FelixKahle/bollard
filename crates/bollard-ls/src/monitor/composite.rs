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

use crate::{
    memory::Schedule, monitor::local_search_monitor::LocalSearchMonitor,
    stats::LocalSearchStatistics,
};
use bollard_search::{monitor::search_monitor::SearchCommand, num::SolverNumeric};

#[derive(Default)]
pub struct CompositeLocalSearchMonitor<'a, T>
where
    T: SolverNumeric,
{
    monitors: Vec<Box<dyn LocalSearchMonitor<T> + 'a>>,
}

impl<'a, T> CompositeLocalSearchMonitor<'a, T>
where
    T: SolverNumeric,
{
    #[inline]
    pub fn new() -> Self {
        Self {
            monitors: Vec::new(),
        }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            monitors: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn add_monitor<M>(&mut self, monitor: M)
    where
        M: LocalSearchMonitor<T> + 'a,
    {
        self.monitors.push(Box::new(monitor));
    }

    #[inline]
    pub fn add_boxed_monitor(&mut self, monitor: Box<dyn LocalSearchMonitor<T> + 'a>) {
        self.monitors.push(monitor);
    }

    #[inline]
    pub fn add_boxed_monitors<I>(&mut self, monitors: I)
    where
        I: IntoIterator<Item = Box<dyn LocalSearchMonitor<T> + 'a>>,
    {
        self.monitors.extend(monitors);
    }

    #[inline]
    pub fn monitors(&self) -> &[Box<dyn LocalSearchMonitor<T> + 'a>] {
        &self.monitors
    }
}

impl<'a, T> LocalSearchMonitor<T> for CompositeLocalSearchMonitor<'a, T>
where
    T: SolverNumeric,
{
    fn name(&self) -> &str {
        "CompositeLocalSearchMonitor"
    }

    fn on_start(&mut self, initial_solution: &Schedule<T>) {
        for m in &mut self.monitors {
            m.on_start(initial_solution);
        }
    }

    fn on_end(&mut self, best_solution: &Schedule<T>, statistics: &LocalSearchStatistics) {
        for m in &mut self.monitors {
            m.on_end(best_solution, statistics);
        }
    }

    fn on_iteration(&mut self, current_solution: &Schedule<T>, statistics: &LocalSearchStatistics) {
        for m in &mut self.monitors {
            m.on_iteration(current_solution, statistics);
        }
    }

    fn on_solution_found(&mut self, solution: &Schedule<T>, statistics: &LocalSearchStatistics) {
        for m in &mut self.monitors {
            m.on_solution_found(solution, statistics);
        }
    }

    fn on_solution_accepted(&mut self, solution: &Schedule<T>, statistics: &LocalSearchStatistics) {
        for m in &mut self.monitors {
            m.on_solution_accepted(solution, statistics);
        }
    }

    fn on_solution_rejected(&mut self, solution: &Schedule<T>, statistics: &LocalSearchStatistics) {
        for m in &mut self.monitors {
            m.on_solution_rejected(solution, statistics);
        }
    }

    fn on_best_solution_updated(
        &mut self,
        solution: &Schedule<T>,
        statistics: &LocalSearchStatistics,
    ) {
        for m in &mut self.monitors {
            m.on_best_solution_updated(solution, statistics);
        }
    }

    fn search_command(&mut self, statistics: &LocalSearchStatistics) -> SearchCommand {
        for m in &mut self.monitors {
            match m.search_command(statistics) {
                SearchCommand::Continue => continue,
                // Return the first terminate request to keep ordering deterministic
                SearchCommand::Terminate(msg) => return SearchCommand::Terminate(msg),
            }
        }
        SearchCommand::Continue
    }
}
