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

use bollard_bap::{column::Column, simplex::SimplexOptimizer};
use bollard_model::index::{BerthIndex, VesselIndex};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

/// Helper to generate a "regular" column for benchmarking.
///
/// Creates a column that covers a specific subset of vessels to simulate
/// realistic sparsity/density.
fn create_test_column(
    berth_id: usize,
    vessels: &[usize],
    cost: f64,
    total_vessels: usize,
) -> Column {
    let assignments: Vec<(VesselIndex, f64)> = vessels
        .iter()
        .map(|&v| (VesselIndex::new(v), 0.0))
        .collect();

    Column::new_regular(BerthIndex::new(berth_id), assignments, cost, total_vessels)
}

/// Benchmark 1: Initialization Overhead
///
/// Measures the cost of allocating the optimizer, including the
/// pre-calculated scratch memory buffers for `faer`.
fn bench_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("Simplex Initialization");

    // Benchmark for different problem sizes (N vessels)
    for size in [10, 100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &n| {
            b.iter(|| {
                // We perform a black_box to prevent the compiler from optimizing away the allocation
                black_box(SimplexOptimizer::new(n));
            });
        });
    }
    group.finish();
}

/// Benchmark 2: State Recomputation (The Hot Loop)
///
/// This is the most critical benchmark. It measures the time taken to:
/// 1. Rebuild the dense basis matrix.
/// 2. Perform in-place LU decomposition.
/// 3. Solve for Primal and Dual values.
fn bench_recompute_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("Recompute State (LU + Solve)");

    for size in [10, 100, 500].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &n| {
            // Setup: Create an optimizer and force a state change so it's not just pure Identity.
            // In a real scenario, the basis is rarely Identity after the first few iterations.
            let mut opt = SimplexOptimizer::new(n);

            // We pivot in one "dense" column to make the factorization non-trivial
            // covering every other vessel.
            let vessels: Vec<usize> = (0..n).step_by(2).collect();
            let dense_col = create_test_column(1, &vessels, 10.0, n);

            // Update state initially
            opt.recompute_state();
            // Perform one pivot to dirty the basis matrix
            opt.perform_pivot(dense_col);

            b.iter(|| {
                // This is the function we want to optimize
                opt.recompute_state();
            });
        });
    }
    group.finish();
}

/// Benchmark 3: Pivoting Operation
///
/// Measures the cost of the `perform_pivot` function, which involves:
/// 1. Solving B * d = A_j (System solve)
/// 2. The Ratio Test (Iterating over constraints)
/// 3. Vector updates
fn bench_perform_pivot(c: &mut Criterion) {
    let mut group = c.benchmark_group("Perform Pivot");

    // Fix N=100 for this detailed breakdown
    let n = 200;

    group.bench_function("Pivot N=200", |b| {
        // We use iter_batched because perform_pivot MUTATES the optimizer state.
        // We need a fresh optimizer for every single iteration of the benchmark.
        b.iter_batched(
            || {
                let mut opt = SimplexOptimizer::new(n);
                opt.recompute_state();

                // Create a candidate column to enter the basis
                // Covers the first 10 vessels
                let vessels: Vec<usize> = (0..10).collect();
                let col = create_test_column(1, &vessels, 5.0, n);
                (opt, col)
            },
            |(mut opt, col)| {
                // Benchmark the pivot
                let success = opt.perform_pivot(col);
                black_box(success);
            },
            // BatchSize::SmallInput is usually fine here, but if memory copying
            // becomes a bottleneck in the benchmark harness, try LargeInput.
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_initialization,
    bench_recompute_state,
    bench_perform_pivot
);
criterion_main!(benches);
