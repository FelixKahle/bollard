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

//! Local search components for Bollard.
//!
//! This crate provides the building blocks to run high‑performance local search
//! on scheduling models: a decoder that maps priority queues to concrete schedules,
//! a stateful operator interface for neighborhood exploration, metaheuristics for
//! acceptance and termination control, and lightweight monitoring and statistics.
//! The design emphasizes tight inner loops with clear safety contracts, cache‑friendly
//! data structures, and reuse of preallocated memory to minimize overhead during
//! iterative improvement.

pub mod decoder;
pub mod engine;
mod incumbent;
pub mod memory;
pub mod meta;
pub mod monitor;
pub mod mutator;
pub mod operator;
pub mod portfolio;
pub mod queue;
pub mod result;
pub mod stats;
pub mod undo;
