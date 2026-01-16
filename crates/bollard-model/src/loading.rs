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

//! Problem instance loader for the Berth Allocation domain.
//!
//! This module turns whitespace-delimited text streams into a validated `Model`,
//! mapping arrivals, opening and closing windows, processing times, and deadlines
//! into the compact layout consumed by solvers and decoders.
//!
//! The `ProblemLoader` emphasizes clarity and robustness. Inputs are read in a
//! straightforward order and converted into typed indices with explicit bounds,
//! while processing times can be selectively treated as disconnections using a
//! threshold to accommodate formats that encode “Infinity” as a large integer.
//! Optional feasibility checks allow early rejection of instances where a vessel
//! has no admissible berth, producing descriptive errors that point directly at
//! the offending index.
//!
//! The parser accepts any `BufRead`, file path, raw reader, or string slice,
//! making it convenient to integrate with benchmarks, tests, and tooling. Lines
//! may contain comments introduced by `#`, which are ignored during tokenization,
//! and debug assertions document internal invariants without interfering with
//! the primary error reporting path.

use crate::{
    index::{BerthIndex, VesselIndex},
    model::{Model, ModelBuilder},
    time::ProcessingTime,
};
use bollard_core::{math::interval::ClosedOpenInterval, num::constants::MinusOne};
use num_traits::{PrimInt, Signed};
use std::{
    fmt::{Debug, Display},
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
    str::FromStr,
};

/// The error type for the problem loading process.
#[derive(Debug)]
pub enum ProblemLoaderError {
    /// An I/O error occurred while reading the input stream.
    Io(std::io::Error),
    /// The input stream ended unexpectedly (e.g., missing tokens).
    UnexpectedEof,
    /// A token could not be parsed into the expected numeric type.
    Parse(ParseTokenError),
    /// The problem dimensions (N or M) are invalid (must be > 0).
    InvalidDimensions,
    /// The model is logically infeasible based on the loader configuration.
    Feasibility(FeasibilityError),
}

/// Details about a failed token parsing attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseTokenError {
    /// The string token that failed to parse.
    pub token: String,
    /// The name of the type we tried to parse into (e.g., "i64").
    pub type_name: &'static str,
}

impl std::fmt::Display for ParseTokenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Could not parse token '{}' as type {}",
            self.token, self.type_name
        )
    }
}

impl std::error::Error for ParseTokenError {}

/// Details about a logical feasibility violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeasibilityError {
    /// The index of the vessel that could not be assigned to any berth.
    pub vessel_index: VesselIndex,
}

impl std::fmt::Display for FeasibilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Vessel {} has no valid berth assignments",
            self.vessel_index.get()
        )
    }
}

impl std::error::Error for FeasibilityError {}

impl Display for ProblemLoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::UnexpectedEof => write!(f, "Unexpected end of file while parsing instance"),
            Self::Parse(e) => write!(f, "Parse error: {}", e),
            Self::InvalidDimensions => {
                write!(f, "Problem dimensions (N and M) must be positive integers")
            }
            Self::Feasibility(e) => write!(f, "Feasibility error: {}", e),
        }
    }
}

impl std::error::Error for ProblemLoaderError {}

impl From<std::io::Error> for ProblemLoaderError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<ParseTokenError> for ProblemLoaderError {
    fn from(e: ParseTokenError) -> Self {
        Self::Parse(e)
    }
}

impl From<FeasibilityError> for ProblemLoaderError {
    fn from(e: FeasibilityError) -> Self {
        Self::Feasibility(e)
    }
}

/// A configurable loader for BAP problem instances.
///
/// The format this parser expects is as follows (whitespace-separated tokens):
///
/// ```raw
/// N // number of ships
/// M // number of berths
/// ta_1 ... ta_|N| (expected arrival of the vessels)
/// s_1 ... s_[M] (expected opening time of berths)
/// h_1_1 ... h_1_|M| (handling time of p_vessel_1_berth_1, p_vessel_1_berth_2, ...)
/// ...
/// h_|N|_1 ... h_|N|_|M| (processing time of p_vessel_|N|_berth_1, p_vessel_|N|_berth_2, ...)
/// e_1 ... e_|M| (expected ending time of berths)
/// t'_1 ... t'_|N| (maximum departure time of vessels)
/// ```
///
/// # Configuration
/// * `forbid_at_least`: Any processing time $\ge$ this value is treated as `None` (disconnected).
///   Useful for handling formats where "Infinity" is represented by a large integer.
/// * `fail_on_unassignable`: If true, the loader returns an error if any vessel cannot dock anywhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProblemLoader<T> {
    forbid_at_least: Option<T>,
    fail_on_unassignable: bool,
}

impl<T> Default for ProblemLoader<T> {
    fn default() -> Self {
        Self {
            forbid_at_least: None,
            fail_on_unassignable: true,
        }
    }
}

impl<T> ProblemLoader<T>
where
    T: PrimInt + Signed + MinusOne + FromStr + Display + Debug,
{
    /// Creates a new `ProblemLoader` with default settings.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets a threshold value. Any processing time read from the input that is greater than
    /// or equal to `v` will be treated as impossible (`ProcessingTime::None`).
    #[inline]
    pub fn forbid_at_least(mut self, v: T) -> Self {
        self.forbid_at_least = Some(v);
        self
    }

    /// Configures whether to return an error if a vessel ends up with no valid berth options.
    #[inline]
    pub fn fail_on_unassignable(mut self, yes: bool) -> Self {
        self.fail_on_unassignable = yes;
        self
    }

    /// Loads a problem from a type implementing `BufRead`.
    pub fn from_bufread<R: BufRead>(&self, rdr: R) -> Result<Model<T>, ProblemLoaderError> {
        let mut sc = Scanner::new(rdr);

        // Read Dimensions
        let n_val: T = sc.next()?;
        let m_val: T = sc.next()?;

        let n = n_val
            .to_usize()
            .ok_or(ProblemLoaderError::InvalidDimensions)?;
        let m = m_val
            .to_usize()
            .ok_or(ProblemLoaderError::InvalidDimensions)?;

        if n == 0 || m == 0 {
            return Err(ProblemLoaderError::InvalidDimensions);
        }

        let mut builder = ModelBuilder::new(m, n);

        // Read Arrival Times (ta)
        for i in 0..n {
            let val = sc.next()?;
            builder.set_vessel_arrival_time(VesselIndex::new(i), val);
        }

        // Read Opening Times (s)
        // Format: s_j is when berth j opens.
        // Logic: Close [0, s_j).
        for j in 0..m {
            let s_val = sc.next()?;
            if s_val > T::zero() {
                builder.add_berth_closing_time(
                    BerthIndex::new(j),
                    ClosedOpenInterval::new(T::zero(), s_val),
                );
            }
        }

        // Read Processing Matrix (h)
        // Format: Rows are vessels, Columns are berths.
        for i in 0..n {
            let v_idx = VesselIndex::new(i);
            let mut feasible_found = false;

            for j in 0..m {
                let h_val = sc.next()?;
                let b_idx = BerthIndex::new(j);

                // Check forbid threshold
                let is_forbidden = self.forbid_at_least.is_some_and(|limit| h_val >= limit);

                // Assuming negative numbers in input might signify impossible assignments
                // (though usually huge numbers are used). We ensure non-negative logic.
                if !is_forbidden && h_val >= T::zero() {
                    builder.set_vessel_processing_time(v_idx, b_idx, ProcessingTime::some(h_val));
                    feasible_found = true;
                } else {
                    builder.set_vessel_processing_time(v_idx, b_idx, ProcessingTime::none());
                }
            }

            if self.fail_on_unassignable && !feasible_found {
                return Err(ProblemLoaderError::Feasibility(FeasibilityError {
                    vessel_index: v_idx,
                }));
            }
        }

        // Read Closing Times (e)
        // Format: e_j is when berth j closes.
        // Logic: Close [e_j, MAX).
        for j in 0..m {
            let e_val = sc.next()?;
            // Only apply constraint if it's less than MAX to avoid redundant checks
            if e_val < T::max_value() {
                builder.add_berth_closing_time(
                    BerthIndex::new(j),
                    ClosedOpenInterval::new(e_val, T::max_value()),
                );
            }
        }

        // Read Deadlines (t_max)
        for i in 0..n {
            let val = sc.next()?;
            builder.set_vessel_latest_departure_time(VesselIndex::new(i), val);
        }

        Ok(builder.build())
    }

    /// Loads a problem from a file path.
    #[inline]
    pub fn from_path<P: AsRef<Path>>(&self, path: P) -> Result<Model<T>, ProblemLoaderError> {
        let file = File::open(path)?;
        self.from_bufread(BufReader::new(file))
    }

    /// Loads a problem from a generic reader.
    #[inline]
    pub fn from_reader<R: Read>(&self, r: R) -> Result<Model<T>, ProblemLoaderError> {
        self.from_bufread(BufReader::new(r))
    }

    /// Loads a problem from a string slice.
    #[inline]
    pub fn from_str(&self, s: &str) -> Result<Model<T>, ProblemLoaderError> {
        self.from_reader(s.as_bytes())
    }
}

/// A helper to read whitespace-delimited tokens from a generic reader.
struct Scanner<R> {
    rdr: R,
    buf: String,
    pos: usize,
}

impl<R: BufRead> Scanner<R> {
    /// Creates a new `Scanner` wrapping the given reader.
    #[inline]
    fn new(rdr: R) -> Self {
        Self {
            rdr,
            buf: String::new(),
            pos: 0,
        }
    }

    /// Refills the internal line buffer. Returns `Ok(true)` if data read, `Ok(false)` on EOF.
    #[inline]
    fn fill_line(&mut self) -> Result<bool, ProblemLoaderError> {
        self.buf.clear();
        self.pos = 0;
        let n = self
            .rdr
            .read_line(&mut self.buf)
            .map_err(ProblemLoaderError::Io)?;
        Ok(n > 0)
    }

    /// Reads the next token and parses it into `T`.
    /// Automatically skips whitespace and comments starting with '#'.
    fn next<T>(&mut self) -> Result<T, ProblemLoaderError>
    where
        T: FromStr,
    {
        loop {
            // Refill buffer if empty or consumed
            if self.pos >= self.buf.len() && !self.fill_line()? {
                return Err(ProblemLoaderError::UnexpectedEof);
            }

            // Skip whitespace and comments
            while self.pos < self.buf.len() {
                let remainder = &self.buf[self.pos..];

                // Found a comment? Skip to end of line immediately.
                if remainder.starts_with('#') {
                    self.pos = self.buf.len();
                    break;
                }

                let c = remainder.chars().next().unwrap();
                if !c.is_whitespace() {
                    break; // Found start of a token
                }

                self.pos += c.len_utf8();
            }

            // If we consumed the whole line (whitespace/comments), loop to get next line
            if self.pos >= self.buf.len() {
                continue;
            }

            // Find end of token
            let mut end = self.pos;
            while end < self.buf.len() {
                let remainder = &self.buf[end..];

                // Token ends at whitespace or start of a comment
                if remainder.starts_with('#') {
                    break;
                }

                let c = remainder.chars().next().unwrap();
                if c.is_whitespace() {
                    break;
                }
                end += c.len_utf8();
            }

            let token = &self.buf[self.pos..end];
            self.pos = end;

            if token.is_empty() {
                continue;
            }

            return token.parse::<T>().map_err(|_| {
                ProblemLoaderError::Parse(ParseTokenError {
                    token: token.to_owned(),
                    type_name: std::any::type_name::<T>(),
                })
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMALL_INSTANCE: &str = r#"
        2 2         # N=2 Vessels, M=2 Berths
        10 20       # Arrivals
        0 5         # Opening
        5 1000      # Durations V0
        999 6       # Durations V1
        100 200     # Closing
        50 60       # Deadlines
    "#;

    #[test]
    fn test_loads_and_maps_correctly() {
        let loader = ProblemLoader::new().forbid_at_least(900);
        let model: Model<i64> = loader.from_str(SMALL_INSTANCE).expect("Failed to load");

        assert_eq!(model.num_vessels(), 2);
        assert_eq!(model.num_berths(), 2);

        // V0 on B1 (1000) should be forbidden (None)
        let pt = model.vessel_processing_time(VesselIndex::new(0), BerthIndex::new(1));
        assert!(Option::<i64>::from(pt).is_none());

        // V0 on B0 (5) should be allowed
        let pt = model.vessel_processing_time(VesselIndex::new(0), BerthIndex::new(0));
        assert_eq!(Option::<i64>::from(pt), Some(5));
    }

    #[test]
    fn test_fail_on_unassignable() {
        let data = "1 1  0  0  1000  100  50"; // Cost 1000, forbid 900 -> Unassignable
        let loader = ProblemLoader::new()
            .forbid_at_least(900)
            .fail_on_unassignable(true);
        let res: Result<Model<i64>, _> = loader.from_str(data);

        match res {
            Err(ProblemLoaderError::Feasibility(FeasibilityError { vessel_index })) => {
                assert_eq!(vessel_index.get(), 0);
            }
            _ => panic!("Expected FeasibilityError"),
        }
    }

    #[test]
    fn test_parse_error_structure() {
        let data = "2 2 garbage";
        let loader = ProblemLoader::<i64>::new();
        let res = loader.from_str(data);

        match res {
            Err(ProblemLoaderError::Parse(e)) => {
                assert_eq!(e.token, "garbage");
                assert!(e.type_name.contains("i64"));
            }
            _ => panic!("Expected Parse error with context"),
        }
    }
}
