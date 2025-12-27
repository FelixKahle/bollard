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

//! # Bollard Model
//!
//! **The Core Domain Model for the Bollard Berth Allocation Solver.**
//!
//! This crate defines the fundamental data structures used to represent the **Berth Allocation Problem (BAP)**.
//! It serves as the data interchange layer between the problem definition (user input) and the
//! solving engine (`bollard_bnb`).
//!
//! ## Architecture
//!
//! The crate is designed around a strict separation of concerns between **construction** and **solving**:
//!
//! * **`index`**: Provides strongly-typed wrappers (`VesselIndex`, `BerthIndex`) to prevent logical indexing errors.
//! * **`model`**: Contains the `Model` (immutable, optimized for solving) and `ModelBuilder` (mutable, optimized for configuration).
//! * **`solution`**: Defines the output format, including objective values and specific assignments.
//! * **`time`**: Low-level time primitives, including sentinel-based optimizations for performance-critical loops.
//!
//! ## Design Philosophy
//!
//! 1.  **Type Safety**: Indices are distinct types. You cannot accidentally use a `VesselIndex` to access a `Berth`.
//! 2.  **Memory Layout**: Data is stored in **Structure of Arrays (SoA)** format (flattened vectors) rather than Arrays of Structures (AoS) to maximize cache locality during the branch-and-bound search.
//! 3.  **Fail-Fast**: Builders and constructors validate inputs eagerly to ensure the solver never encounters an invalid state.

pub mod index;
pub mod model;
pub mod solution;
pub mod time;
