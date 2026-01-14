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

use bollard_ls::{queue::VesselPriorityQueue, undo::UndoLog};
use bollard_model::index::VesselIndex;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn vi(i: usize) -> VesselIndex {
    VesselIndex::new(i)
}

/// Sets up a dummy queue with N vessels.
fn setup_queue(n: usize) -> VesselPriorityQueue {
    let mut q = VesselPriorityQueue::preallocated(n);
    // Fill with 0..N
    q.extend((0..n).map(vi));
    q
}

/// Benchmarks the "Hot Path": Mutate -> Record -> Rollback.
/// This simulates checking a neighbor and rejecting it.
fn bench_undo_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("undo_log_operations");

    // We use a reasonably sized queue.
    // Small enough to fit in L1 cache (to measure logic overhead, not RAM latency),
    // but large enough to be realistic.
    let num_vessels = 1000;

    // 1. Benchmark: Swap Round-Trip
    // This is the most common operation in Local Search (2-opt, Swap).
    group.throughput(Throughput::Elements(1));
    group.bench_function("swap_round_trip", |b| {
        let mut q = setup_queue(num_vessels);
        // Preallocate log to ensure we strictly measure push/pop overhead, not realloc
        let mut log = UndoLog::preallocated(num_vessels);

        // Indices to swap
        let idx_a = 10;
        let idx_b = 50;

        b.iter(|| {
            // 1. Forward Mutation
            q.buffer_mut().swap(idx_a, idx_b);

            // 2. Record
            log.push_swap(black_box(idx_a), black_box(idx_b));

            // 3. Rollback (Restores state)
            log.apply_rollback(black_box(&mut q));
        });
    });

    // 2. Benchmark: Set Round-Trip
    // Used for single-value changes.
    group.bench_function("set_round_trip", |b| {
        let mut q = setup_queue(num_vessels);
        let mut log = UndoLog::preallocated(num_vessels);

        let target_idx = 42;
        let new_val = vi(9999);

        b.iter(|| {
            // 1. Capture Old
            let old_val = q.get(target_idx).unwrap();

            // 2. Record
            log.push_set(black_box(target_idx), black_box(old_val));

            // 3. Forward Mutation
            q.set(target_idx, new_val);

            // 4. Rollback
            log.apply_rollback(black_box(&mut q));
        });
    });

    // 3. Benchmark: Shift (Inverse) Round-Trip
    // Used for Insert/Relocate operators.
    group.bench_function("shift_round_trip", |b| {
        let mut q = setup_queue(num_vessels);
        let mut log = UndoLog::preallocated(num_vessels);

        // Move element at 10 to 20
        let from = 10;
        let to = 20;

        b.iter(|| {
            // 1. Forward Mutation (Rotate)
            // Note: Actual logic depends on if from < to or >.
            // Here from < to, so we rotate right.
            q.buffer_mut()[from..=to].rotate_right(1);

            // 2. Record Inverse
            log.push_shift_inverse(black_box(from), black_box(to));

            // 3. Rollback
            log.apply_rollback(black_box(&mut q));
        });
    });

    // 4. Benchmark: Range Backup Round-Trip
    // This benchmarks the `memcpy` speed of backing up chunks.
    // We test a small range (e.g., swapping two blocks of 5).
    let range_len = 16; // Small SIMD-friendly size
    group.bench_with_input(
        BenchmarkId::new("range_backup_round_trip", range_len),
        &range_len,
        |b, &len| {
            let mut q = setup_queue(num_vessels);
            let mut log = UndoLog::preallocated(num_vessels);
            let start = 100;

            b.iter(|| {
                // 1. Record (Backup Data)
                // We must slice carefully to mimic the logic
                let slice = &q.buffer()[start..start + len];
                log.push_range_backup(black_box(start), black_box(slice));

                // 2. Forward Mutation (Simulate writing garbage)
                // In a real scenario, we'd copy new data here.
                // We define a dummy mutation to force the CPU to acknowledge the change.
                q.buffer_mut()[start] = vi(999);

                // 3. Rollback
                log.apply_rollback(black_box(&mut q));
            });
        },
    );

    group.finish();
}

/// Benchmarks "Stack Depth" scaling.
/// Does the log get slower as we push more operations before rolling back?
/// (It shouldn't, due to L1 locality, but good to verify).
fn bench_stack_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("undo_log_scaling");
    let num_vessels = 5000;

    for depth in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*depth as u64));
        group.bench_with_input(BenchmarkId::new("swap_chain", depth), depth, |b, &d| {
            let mut q = setup_queue(num_vessels);
            let mut log = UndoLog::preallocated(num_vessels);

            // We will execute `d` swaps and then rollback `d` swaps.
            b.iter(|| {
                for i in 0..d {
                    // Swap i and i+1
                    q.buffer_mut().swap(i, i + 1);
                    log.push_swap(i, i + 1);
                }

                log.apply_rollback(black_box(&mut q));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_undo_operations, bench_stack_depth);
criterion_main!(benches);
