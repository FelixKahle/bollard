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
    monitor::search_monitor::{SearchCommand, SearchMonitor},
    num::SolverNumeric,
};
use bollard_model::{model::Model, solution::Solution};

pub struct DynamicSearchMonitor<'a, T> {
    monitor: Box<dyn SearchMonitor<T> + 'a>,
}

impl<'a, T> DynamicSearchMonitor<'a, T>
where
    T: SolverNumeric,
{
    #[inline]
    pub fn new<M>(monitor: M) -> Self
    where
        M: SearchMonitor<T> + 'a,
    {
        Self {
            monitor: Box::new(monitor),
        }
    }

    #[inline]
    pub fn monitor(&self) -> &dyn SearchMonitor<T> {
        self.monitor.as_ref()
    }
}

impl<'a, T> SearchMonitor<T> for DynamicSearchMonitor<'a, T>
where
    T: SolverNumeric,
{
    fn name(&self) -> &str {
        self.monitor.name()
    }

    fn on_enter_search(&mut self, model: &Model<T>) {
        self.monitor.on_enter_search(model)
    }

    fn on_exit_search(&mut self) {
        self.monitor.on_exit_search()
    }

    fn on_solution_found(&mut self, solution: &Solution<T>) {
        self.monitor.on_solution_found(solution)
    }

    fn on_step(&mut self) {
        self.monitor.on_step()
    }

    fn search_command(&self) -> SearchCommand {
        self.monitor.search_command()
    }
}
