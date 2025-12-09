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

use bollard_model::{model::Model, solution::Solution};
use num_traits::{PrimInt, Signed};

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub enum SearchCommand {
    #[default]
    Continue,
    Terminate(String),
}

impl std::fmt::Display for SearchCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchCommand::Continue => write!(f, "Continue"),
            SearchCommand::Terminate(reason) => write!(f, "Terminate: {}", reason),
        }
    }
}

pub trait SearchMonitor<T>
where
    T: PrimInt + Signed,
{
    fn name(&self) -> &str;
    fn on_enter_search(&mut self, model: &Model<T>);
    fn on_exit_search(&mut self);
    fn on_solution_found(&mut self, solution: &Solution<T>);
    fn on_step(&mut self);
    fn search_command(&self) -> SearchCommand;
}

impl<T> std::fmt::Debug for dyn SearchMonitor<T>
where
    T: PrimInt + Signed,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SearchMonitor({})", self.name())
    }
}

impl<T> std::fmt::Display for dyn SearchMonitor<T>
where
    T: PrimInt + Signed,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SearchMonitor({})", self.name())
    }
}
