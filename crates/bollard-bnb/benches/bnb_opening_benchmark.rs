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

use bollard_bnb::{
    bnb::BnbSolver, branching::edf::EarliestDeadlineFirstBuilder,
    eval::wtft::WeightedFlowTimeEvaluator, monitor::solution::SolutionLimitMonitor,
    params::BnbSearchParams,
};
use bollard_model::{loading::ProblemLoader, model::Model};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use regex::Regex;
use std::path::{Path, PathBuf};

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

/// Helper to gather all instance files matching the regex "^f\\d+x\\d+-\\d+\\.txt$".
fn get_instance_files() -> Vec<PathBuf> {
    let dir = find_instances_dir().expect("Could not find 'data/' directory");

    let re = Regex::new(r"^f\d+x\d+-\d+\.txt$").unwrap();

    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
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

fn load_model(path: &Path) -> Model<i64> {
    let loader = ProblemLoader::<i64>::new();
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<unknown>");

    loader
        .from_path(path)
        .unwrap_or_else(|e| panic!("Failed to load {}: {}", file_name, e))
}

fn benchmark_edf(c: &mut Criterion) {
    let files = get_instance_files();
    if files.is_empty() {
        eprintln!("No instance files found in data/ matching pattern. Skipping benchmark.");
        return;
    }

    let mut group = c.benchmark_group("bnb_opening_edf");

    for path in files {
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("<unknown>");

        let model = load_model(&path);

        group.bench_with_input(BenchmarkId::new("edf", file_name), &path, |b, _| {
            b.iter(|| {
                let mut solver: BnbSolver<i64> = BnbSolver::new();
                let mut builder = EarliestDeadlineFirstBuilder::<i64>::new();
                let mut evaluator = WeightedFlowTimeEvaluator::<i64>::new();
                let monitor = SolutionLimitMonitor::<i64>::new(1);

                let params =
                    BnbSearchParams::builder(&model, &mut builder, &mut evaluator, monitor)
                        .build()
                        .unwrap();

                let _outcome = solver.solve(params);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_edf);
criterion_main!(benches);
