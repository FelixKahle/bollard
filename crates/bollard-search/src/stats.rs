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

/// Statistics collected during the solving process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverStatistics {
    /// Number of solutions found during the solving process.
    pub solutions_found: u64,
    /// Number of threads used during the solving process.
    pub used_threads: usize,
    /// Maximum memory used (in bytes) during the solving process.
    pub max_memory_bytes: usize,
    /// Total duration of the solving process.
    pub solve_duration: std::time::Duration,
}

impl std::fmt::Display for SolverStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Solver Statistics:")?;
        writeln!(f, "  Solutions Found: {}", self.solutions_found)?;
        writeln!(f, "  Used Threads: {}", self.used_threads)?;
        writeln!(f, "  Max Memory Used (bytes): {}", self.max_memory_bytes)?;
        writeln!(
            f,
            "  Solve Duration (secs): {:.3}",
            self.solve_duration.as_secs_f64()
        )
    }
}

/// Builder for `SolverStatistics`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverStatisticsBuilder {
    solutions_found: u64,
    used_threads: usize,
    max_memory_bytes: usize,
    solve_duration: std::time::Duration,
}

impl Default for SolverStatisticsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SolverStatisticsBuilder {
    /// Creates a new `SolverStatisticsBuilder` with default values.
    #[inline]
    pub fn new() -> Self {
        Self {
            solutions_found: 0,
            used_threads: 1,
            max_memory_bytes: 0,
            solve_duration: std::time::Duration::ZERO,
        }
    }

    /// Sets the number of solutions found.
    #[inline]
    pub fn solutions_found(mut self, solutions_found: u64) -> Self {
        self.solutions_found = solutions_found;
        self
    }

    /// Sets the number of threads used.
    #[inline]
    pub fn used_threads(mut self, used_threads: usize) -> Self {
        self.used_threads = used_threads;
        self
    }

    /// Sets the maximum memory used in bytes.
    #[inline]
    pub fn max_memory_bytes(mut self, max_memory_bytes: usize) -> Self {
        self.max_memory_bytes = max_memory_bytes;
        self
    }

    /// Sets the total solve duration.
    #[inline]
    pub fn solve_duration(mut self, solve_duration: std::time::Duration) -> Self {
        self.solve_duration = solve_duration;
        self
    }

    /// Builds the `SolverStatistics` instance.
    #[inline]
    pub fn build(self) -> SolverStatistics {
        SolverStatistics {
            solutions_found: self.solutions_found,
            used_threads: self.used_threads,
            max_memory_bytes: self.max_memory_bytes,
            solve_duration: self.solve_duration,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SolverStatistics;
    use super::SolverStatisticsBuilder;
    use std::time::Duration;

    #[test]
    fn builder_constructs_expected_struct() {
        let stats = SolverStatisticsBuilder::new()
            .solutions_found(3)
            .used_threads(8)
            .max_memory_bytes(1_048_576)
            .solve_duration(Duration::from_millis(1234))
            .build();

        assert_eq!(stats.solutions_found, 3);
        assert_eq!(stats.used_threads, 8);
        assert_eq!(stats.max_memory_bytes, 1_048_576);
        assert_eq!(stats.solve_duration, Duration::from_millis(1234));
    }

    #[test]
    fn test_display_formats_all_fields() {
        let stats = SolverStatistics {
            solutions_found: 2,
            used_threads: 4,
            max_memory_bytes: 2_000_000,
            solve_duration: Duration::from_millis(1234),
        };

        let rendered = format!("{}", stats);

        // Header line
        assert!(rendered.contains("Solver Statistics:"), "missing header");

        // Fields
        assert!(
            rendered.contains("Solutions Found: 2"),
            "missing solutions_found"
        );
        assert!(rendered.contains("Used Threads: 4"), "missing used_threads");
        assert!(
            rendered.contains("Max Memory Used (bytes): 2000000"),
            "missing max_memory_bytes"
        );

        // Duration line should be formatted to three decimals
        // 1.2345 rounded to 1.235
        assert!(
            rendered.contains("Solve Duration (secs): 1.234"),
            "duration not formatted to 3 decimals"
        );
    }

    #[test]
    fn test_display_handles_zero_values() {
        let stats = SolverStatistics {
            solutions_found: 0,
            used_threads: 1,
            max_memory_bytes: 0,
            solve_duration: Duration::ZERO,
        };

        let rendered = format!("{}", stats);

        assert!(rendered.contains("Solutions Found: 0"));
        assert!(rendered.contains("Used Threads: 1"));
        assert!(rendered.contains("Max Memory Used (bytes): 0"));
        assert!(rendered.contains("Solve Duration (secs): 0.000"));
    }
}
