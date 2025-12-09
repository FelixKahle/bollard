# bollard

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Rust 2024](https://img.shields.io/badge/Rust-Edition%202024-orange.svg)](https://doc.rust-lang.org/edition-guide/rust-2024)

---

## 🚢 Overview

`bollard` is a research project developed for modeling and solving the Berth Allocation Problem (BAP), a classical NP-hard problem in maritime logistics. It provides data structures and utilities to represent vessels, berths, time windows, and processing durations, enabling experimentation with algorithms and heuristics. The project aims to combine exact and heuristic methods into a single high-performance search engine for the BAP.

---

## Problem Description

The Berth Allocation Problem involves assigning vessels to berth positions over time, taking into account:

- Vessel arrival windows and latest departure times
- Berth availability via opening/closing intervals
- Processing times per vessel and berth
- Optional vessel weights/priorities

Solving this efficiently reduces congestion and improves throughput at container terminals.

---

## Status

Actively evolving research code. Expect frequent changes and occasional breaking changes.

## License

This project is licensed under the MIT License. See [LICENSE.txt](LICENSE.txt) for details.
