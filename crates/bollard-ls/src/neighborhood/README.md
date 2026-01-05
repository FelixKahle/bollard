# Neighborhoods in `bollard-ls`

This module provides core abstractions and data structures for expressing vessel connectivity in local search. It is designed to be **model-agnostic**, focusing on hot-loop performance and predictable memory layout. Implementations expose contiguous slices for efficient iteration, suitable for rejection sampling and other inner-loop heuristics.

### Architecture

The trait-centric design allows solvers to query whether two entities can meaningfully interact and to iterate over all potential partners without incurring per-access overhead. By returning slices (`&[VesselIndex]`), the solver ensures that neighborhood data is always contiguous, maximizing CPU cache performance during linear scans.

### The CSR Implementation

The primary implementation, `StaticTopology`, uses a **Compressed Sparse Row (CSR)** layout. It stores all neighbor lists in a single flattened vector with a sentinel offsets array. Graph-wise, it represents a simple, undirected, loop-free vessel contention graph:
* **Vertices**: Vessels.
* **Edges**: An edge exists between two vessels if they share at least one berth where both are allowed to dock and their operating time windows overlap (closed-open intervals [start, end)).
* **Symmetry**: Adjacency is symmetric; if A is a neighbor of B, then B is a neighbor of A.
* **No Self-Loops**: Vessels are never neighbors with themselves.
* **Sparsity**: The graph is typically sparse due to berth compatibility and time-window pruning.

This provides:
* **Slice Access**: Retrieving a vessel's neighbors is just two pointer lookups.
* **Cache Locality**: Iterating through neighbors is a linear scan over a dense integer array.
* **Minimal RAM Overhead**: No pointer chasing or small-vector headers.
