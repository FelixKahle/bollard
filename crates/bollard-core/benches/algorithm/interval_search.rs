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

use bollard_core::math::interval::ClosedOpenInterval;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

/// Benchmark lower bound search algorithms
/// for disjoint sorted intervals.
fn bench_lower_bound(c: &mut Criterion) {
    let mut group = c.benchmark_group("Lower Bound Search");
    let sizes = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256];

    for size in sizes {
        // 1. Generate Data: Disjoint sorted intervals [0, 10), [10, 20), ...
        // We use i64 as the coordinate type.
        let intervals: Vec<ClosedOpenInterval<i64>> = (0..size)
            .map(|i| {
                let start = (i as i64) * 10;
                ClosedOpenInterval::new(start, start + 10)
            })
            .collect();

        // 2. Define a search key.
        // We search for a key located roughly at ~75% of the array.
        // This prevents "best case" scenarios (finding it immediately at index 0)
        // and simulates a realistic lookup depth.
        let target_key = (size as i64 * 10) * 3 / 4;

        // 3. Benchmark Linear Search
        group.bench_with_input(BenchmarkId::new("Linear", size), &size, |b, &_| {
            b.iter(|| {
                bollard_core::algorithm::interval::lower_bound_start_linear(
                    black_box(&intervals),
                    black_box(target_key),
                )
            })
        });

        // 4. Benchmark Binary Search
        group.bench_with_input(BenchmarkId::new("Binary", size), &size, |b, &_| {
            b.iter(|| {
                bollard_core::algorithm::interval::lower_bound_start_binary(
                    black_box(&intervals),
                    black_box(target_key),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_lower_bound);
criterion_main!(benches);
