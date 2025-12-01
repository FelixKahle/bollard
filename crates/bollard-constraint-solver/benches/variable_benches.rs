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

use bollard_constraint_solver::variable::{ClosedInterval, IntegerDomain};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

/// The size of the SmallVec used in IntegerDomain for these benchmarks.
/// We keep it small to force heap allocations for larger domains,
/// also; a small number is also used in the bollard_constraint_solver codebase.
const INTEGER_DOMAIN_SMALLVECTOR_SIZE: usize = 1;

/// The sizes of IntegerDomains to benchmark.
const INTEGER_DOMAIN_SIZES: [usize; 6] = [1, 4, 16, 64, 256, 1024];

/// Create an IntegerDomain consisting of `num` intervals, each of `width`,
/// separated by `gap`, starting at `offset`.
fn make_domain<const N: usize>(
    num: usize,
    width: i32,
    gap: i32,
    offset: i32,
) -> IntegerDomain<i32, N> {
    let mut intervals = Vec::with_capacity(num);
    let stride = width + gap;

    for i in 0..num {
        let lower = offset + (i as i32) * stride;
        let upper = lower + width - 1;
        intervals.push(ClosedInterval::new(lower, upper));
    }

    IntegerDomain::from_sorted_vec(intervals)
}

fn bench_intersects(c: &mut Criterion) {
    let mut group = c.benchmark_group("integer_domain_intersects");

    for n in INTEGER_DOMAIN_SIZES {
        let a: IntegerDomain<i32, INTEGER_DOMAIN_SMALLVECTOR_SIZE> =
            make_domain::<INTEGER_DOMAIN_SMALLVECTOR_SIZE>(n, 10, 10, 0);
        let b: IntegerDomain<i32, INTEGER_DOMAIN_SMALLVECTOR_SIZE> =
            make_domain::<INTEGER_DOMAIN_SMALLVECTOR_SIZE>(n, 10, 10, 5);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, &_n| {
            bencher.iter(|| black_box(a.intersects(black_box(&b))));
        });
    }

    group.finish();
}

fn bench_contains_value(c: &mut Criterion) {
    let mut group = c.benchmark_group("integer_domain_contains_value");

    for n in INTEGER_DOMAIN_SIZES {
        let d: IntegerDomain<i32, INTEGER_DOMAIN_SMALLVECTOR_SIZE> =
            make_domain::<INTEGER_DOMAIN_SMALLVECTOR_SIZE>(n, 10, 10, 0);
        let mut queries = Vec::with_capacity(2 * n);
        for i in 0..n {
            let base = (i as i32) * 20;
            queries.push(base + 3);
            queries.push(base + 15);
        }

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, &_n| {
            bencher.iter(|| {
                let mut count = 0_usize;
                for &q in &queries {
                    if d.contains_value(black_box(q)) {
                        count += 1;
                    }
                }
                black_box(count)
            });
        });
    }

    group.finish();
}

fn bench_contains_interval(c: &mut Criterion) {
    let mut group = c.benchmark_group("integer_domain_contains_interval");

    for &n in &[1_usize, 4, 16, 64, 256, 1024] {
        let d: IntegerDomain<i32, INTEGER_DOMAIN_SMALLVECTOR_SIZE> =
            make_domain::<INTEGER_DOMAIN_SMALLVECTOR_SIZE>(n, 10, 10, 0);
        let mut contained = Vec::with_capacity(n);
        let mut not_contained = Vec::with_capacity(n);

        for i in 0..n {
            let base = (i as i32) * 20;
            contained.push(ClosedInterval::new(base + 2, base + 7));
            not_contained.push(ClosedInterval::new(base + 8, base + 12));
        }

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, &_n| {
            bencher.iter(|| {
                let mut count = 0_usize;

                for &iv in &contained {
                    if d.contains_interval(black_box(iv)) {
                        count += 1;
                    }
                }
                for &iv in &not_contained {
                    if d.contains_interval(black_box(iv)) {
                        count += 1;
                    }
                }

                black_box(count)
            });
        });
    }

    group.finish();
}

fn bench_intersection(c: &mut Criterion) {
    let mut group = c.benchmark_group("integer_domain_intersection");

    // Overlapping domains: offset by 5 so each interval overlaps by 5 values.
    for n in INTEGER_DOMAIN_SIZES {
        let a: IntegerDomain<i32, INTEGER_DOMAIN_SMALLVECTOR_SIZE> =
            make_domain::<INTEGER_DOMAIN_SMALLVECTOR_SIZE>(n, 10, 10, 0);
        let b: IntegerDomain<i32, INTEGER_DOMAIN_SMALLVECTOR_SIZE> =
            make_domain::<INTEGER_DOMAIN_SMALLVECTOR_SIZE>(n, 10, 10, 5);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, &_n| {
            bencher.iter(|| {
                // Prevent the optimizer from eliding the intersection by consuming the size.
                let res = black_box(a.intersection(black_box(&b)));
                match res {
                    Some(ref dom) => black_box(dom.size()),
                    None => black_box(0i32),
                }
            });
        });
    }

    group.finish();
}

fn criterion_benches(c: &mut Criterion) {
    bench_intersects(c);
    bench_contains_value(c);
    bench_contains_interval(c);
    bench_intersection(c);
}

criterion_group!(benches, criterion_benches);
criterion_main!(benches);
