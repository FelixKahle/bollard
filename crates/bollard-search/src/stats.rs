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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverStatistics {
    pub solutions_found: u64,
    pub used_threads: usize,
    pub max_memory_bytes: usize,
    pub solve_duration: std::time::Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverStatisticsBuilder {
    solutions_found: u64,
    used_threads: usize,
    max_memory_bytes: usize,
    solve_duration: std::time::Duration,
}

impl SolverStatisticsBuilder {
    pub fn new() -> Self {
        Self {
            solutions_found: 0,
            used_threads: 1,
            max_memory_bytes: 0,
            solve_duration: std::time::Duration::ZERO,
        }
    }

    pub fn solutions_found(mut self, solutions_found: u64) -> Self {
        self.solutions_found = solutions_found;
        self
    }

    pub fn used_threads(mut self, used_threads: usize) -> Self {
        self.used_threads = used_threads;
        self
    }

    pub fn max_memory_bytes(mut self, max_memory_bytes: usize) -> Self {
        self.max_memory_bytes = max_memory_bytes;
        self
    }

    pub fn solve_duration(mut self, solve_duration: std::time::Duration) -> Self {
        self.solve_duration = solve_duration;
        self
    }

    pub fn build(self) -> SolverStatistics {
        SolverStatistics {
            solutions_found: self.solutions_found,
            used_threads: self.used_threads,
            max_memory_bytes: self.max_memory_bytes,
            solve_duration: self.solve_duration,
        }
    }
}
