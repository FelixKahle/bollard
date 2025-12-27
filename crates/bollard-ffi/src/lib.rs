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
//! **High-Performance C-Compatible Bindings for the Bollard Scheduling Solver.**
//!
//! This crate serves as the bridge between the high-performance Rust core of Bollard and
//! external environments such as C, C++, Python, C#, and Java. It exposes a stable,
//! ABI-compliant interface designed around **Opaque Pointers** (Handles) and strict
//! **resource management**.
//!
//! ## Core Design Principles
//!
//! 1.  **Opaque Handles**: All complex Rust structures (`Model`, `ModelBuilder`, `BnbSolver`, `Outcome`)
//!     are hidden behind raw pointers. The host application never accesses struct fields directly;
//!     it uses the provided accessor functions.
//! 2.  **Explicit Lifecycle**: Memory is manually managed. Every `_new` call must have a corresponding
//!     `_free` call. Failing to do so will result in memory leaks.
//! 3.  **Fail-Fast Safety**: To protect the integrity of the host application, this FFI layer
//!     adopts a "Fail-Fast" strategy. Passing `NULL` pointers or invalid indices results in an
//!     immediate process abort (panic) rather than undefined behavior or stack unwinding.

pub mod bnb;
pub mod model;
