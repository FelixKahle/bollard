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

use bollard_bnb::bnb::BnbSolver;
use bollard_bnb::branching::edf::EarliestDeadlineFirstBuilder;
use bollard_bnb::eval::hybrid::HybridEvaluator;
use bollard_bnb::monitor::solution::SolutionLimitMonitor;
use bollard_ls::decoder::{Decoder, GreedyDecoder};
use bollard_ls::eval::WeightedFlowTimeEvaluator;
use bollard_ls::memory::SearchMemory;
use bollard_ls::queue::VesselPriorityQueue;
use bollard_model::index::{BerthIndex, VesselIndex};
use bollard_model::loading::ProblemLoader;
use bollard_model::model::Model;
use bollard_model::solution::Solution;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use regex::Regex;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};

fn vi(i: usize) -> VesselIndex {
    VesselIndex::new(i)
}

fn bi(i: usize) -> BerthIndex {
    BerthIndex::new(i)
}

fn find_instances_dir() -> Option<PathBuf> {
    let mut cur: Option<&Path> = Some(Path::new(env!("CARGO_MANIFEST_DIR")));
    while let Some(p) = cur {
        let cand = p.join("data");
        if cand.is_dir() {
            return Some(cand);
        }
        cur = p.parent();
    }
    None
}

/// Helper to gather all instance files matching the regex "^f\d+x\d+-\d+\.txt$".
fn get_instance_files() -> Vec<PathBuf> {
    let dir = find_instances_dir().expect("Could not find 'data/' directory");

    let re = Regex::new(r"^f\d+x\d+-\d+\.txt$").unwrap();

    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .expect("Failed to read data directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|s| re.is_match(s))
                .unwrap_or(false)
        })
        .collect();

    // Sort for deterministic benchmark order
    files.sort();
    files
}

fn find_feasible_solution(model: &Model<i64>) -> Solution<i64> {
    let num_vessels = model.num_vessels();
    let num_berths = model.num_berths();

    let mut bnb_solver = BnbSolver::preallocated(num_berths, num_vessels);
    let mut builder = EarliestDeadlineFirstBuilder::preallocated(num_berths, num_vessels);
    let mut evaluator = HybridEvaluator::preallocated(num_berths, num_vessels);
    let solution_limit_monitor = SolutionLimitMonitor::new(1);
    let outcome = bnb_solver.solve(model, &mut builder, &mut evaluator, solution_limit_monitor);

    let res = outcome.result().unwrap_feasible();
    res.clone()
}

fn construct_queue(solution: &Solution<i64>) -> VesselPriorityQueue {
    let n = solution.num_vessels();
    let mut indices: Vec<VesselIndex> = (0..n).map(vi).collect();
    indices.sort_by_key(|&v| solution.start_time_for_vessel(v));
    let mut q = VesselPriorityQueue::preallocated(n);
    q.extend(indices);
    q
}

fn bench_real_instances(c: &mut Criterion) {
    let files = get_instance_files();
    if files.is_empty() {
        eprintln!("No instance files found in data/ matching pattern. Skipping benchmark.");
        return;
    }

    let loader = ProblemLoader::<i64>::new();
    let mut evaluator = WeightedFlowTimeEvaluator::<i64>::default();
    let group_re = Regex::new(r"f(\d+x\d+)").unwrap();
    let mut group = c.benchmark_group("decoder_benchmark");

    for path in files {
        let file_name = path.file_name().unwrap().to_string_lossy();
        let model = loader
            .from_path(&path)
            .unwrap_or_else(|e| panic!("Failed to load {}: {}", file_name, e));

        let num_vessels = model.num_vessels();
        let num_berths = model.num_berths();

        // Pre-Solve to get a valid order
        // This ensures our benchmark actually measures the full decoding process,
        // rather than failing early due to a bad random queue.
        let feasible_sol = find_feasible_solution(&model);
        let queue = construct_queue(&feasible_sol);

        let mut memory = SearchMemory::<i64>::preallocated(num_vessels);
        let placeholder = Solution::new(0_i64, vec![bi(0); num_vessels], vec![0_i64; num_vessels]);
        memory.initialize(&placeholder);

        let mut decoder = GreedyDecoder::new(num_berths);

        let size_label = group_re
            .captures(&file_name)
            .map(|caps| caps[1].to_string())
            .unwrap_or_else(|| "unknown".to_string());

        group.throughput(Throughput::Elements(num_vessels as u64));

        group.bench_with_input(
            BenchmarkId::new(&size_label, &file_name),
            &path,
            |b, _path| {
                b.iter(|| {
                    let ok = unsafe {

                    decoder.decode_unchecked(
                        black_box(&model),
                        black_box(&queue),
                        black_box(memory.candidate_schedule_mut()),
                        black_box(&mut evaluator),
                    ) };

                    if !ok {
                        panic!("Benchmark configuration error: GreedyDecoder failed on a known-feasible queue.");
                    }
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_real_instances);
criterion_main!(benches);
