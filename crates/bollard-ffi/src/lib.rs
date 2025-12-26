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

//! # Bollard FFI
//!
//! **Foreign Function Interface (FFI) bindings for the Bollard scheduling ecosystem.**
//!
//! This crate provides a stable, C-compatible Application Binary Interface (ABI) for
//! interacting with the Bollard project. It acts as a high-performance bridge, allowing
//! external environments—such as C, C++, Python, Julia, C#, or Java—to leverage Bollard's
//! scheduling solvers while maintaining strict safety guarantees and managing memory manually.
//!
//! ## Architecture
//!
//! The library is designed around **Opaque Pointers** (the Handle pattern) and **Explicit
//! Resource Management**. It is structured into modular components to separate problem
//! definition from solution strategy:
//!
//! 1.  **Problem Definition (`model`)**:
//!     A generic, solver-agnostic interface to define scheduling problems. Users construct
//!     an immutable `Model` containing vessels, berths, processing times, and constraints.
//!     This model is the common input format for all solvers.
//!
//! 2.  **Solvers**:
//!     Implementations of specific algorithms to solve the defined `Model`.
//!     * **`bnb`**: An exact **Branch-and-Bound** solver capable of finding optimal schedules
//!       using configurable objective evaluators and search heuristics.
//!
//! ## Usage Pattern
//!
//! A typical integration follows this linear workflow:
//!
//! 1.  **Define**: Use the `model` module to build a `Model`.
//! 2.  **Solve**: Pass that `Model` to a specific solver in the `bnb` module.
//! 3.  **Inspect**: Analyze the returned `Outcome` structure.
//! 4.  **Free**: Explicitly release memory for all objects created.
//!
//! ## Safety Strategy
//!
//! This crate prioritizes safety at the boundary. It is designed to **abort** (terminate the process)
//! immediately upon misuse (such as passing `NULL` pointers) rather than unwinding. This
//! prevents undefined behavior or memory corruption from crossing the FFI boundary into the
//! host application.

pub mod bnb;
pub mod model;
