# Bollard Model

**The Core Domain Model for the Bollard Berth Allocation Solver.**

`bollard_model` provides the fundamental data structures, validation logic, and builder patterns required to define the **Berth Allocation Problem (BAP)**. It is designed for high-performance combinatorial optimization, utilizing memory-efficient layouts (Structure of Arrays) and strict type safety to prevent common logical errors before the solving phase begins.

## Key Features

* **Two-Phase Construction**: Implements the **Builder Pattern** to separate the mutable configuration phase from the immutable, high-speed solving phase, allowing for rigorous validation during the build step.
* **Type-Safe Indexing**: Distinct `VesselIndex` and `BerthIndex` types ensure that ship IDs and dock IDs are never accidentally swapped, eliminating a common class of bugs in scheduling logic.
* **Memory Optimization**: Internal data structures use **Structure of Arrays (SoA)** and flattened vectors to maximize cache locality during hot loops in the solver.
* **Sentinel-Based Optionals**: Custom `ProcessingTime` types utilize sentinel values to avoid the storage overhead of standard Rust `Option<T>` wrappers.
* **Complexity Analysis**: Includes built-in tools to calculate the logarithmic search space size, helping to estimate solver difficulty and memory requirements upfront.

## Architecture

The crate is organized into four main modules:

1. **`index`**: Strongly typed indices (`VesselIndex`, `BerthIndex`) used throughout the ecosystem.
2. **`model`**: The core definitions.
* `ModelBuilder`: The mutable interface for setting up constraints.
* `Model`: The immutable, validated struct optimized for read-heavy solver access.


3. **`solution`**: Structures representing the final assignment of vessels to berths and times (`Solution`).
4. **`time`**: Low-level time representations, including the sentinel-optimized `ProcessingTime`.

## Quick Start

### Defining a Problem Instance

The lifecycle involves initializing a builder, defining constraints, and "freezing" it into an immutable Model.

```rust
use bollard_model::model::ModelBuilder;
use bollard_model::index::{VesselIndex, BerthIndex};
use bollard_model::time::ProcessingTime;
use bollard_core::math::interval::ClosedOpenInterval;

fn main() {
    // 1. Initialize a builder for 2 berths and 2 vessels
    let mut builder = ModelBuilder::<i64>::new(2, 2);

    // 2. Define Vessel 0 properties
    let v0 = VesselIndex::new(0);
    builder.set_vessel_arrival_time(v0, 10)
           .set_vessel_weight(v0, 5)
           .set_vessel_latest_departure_time(v0, 100);

    // 3. Define topology (Processing Times)
    // Vessel 0 takes 50 units of time on Berth 0
    builder.set_vessel_processing_time(
        v0,
        BerthIndex::new(0),
        ProcessingTime::some(50)
    );

    // 4. Define Berth Constraints (e.g., maintenance window)
    builder.add_berth_closing_time(
        BerthIndex::new(1),
        ClosedOpenInterval::new(0, 20) // Closed for first 20 units
    );

    // 5. Build the immutable model
    let model = builder.build();

    println!("Model Complexity: {}", model.complexity());
}

```

## Design & Performance

* **Immutable by Default**: Once `builder.build()` is called, the resulting `Model` is immutable. This allows the solver to access problem data concurrently without locking synchronization primitives.
* **Cache Locality**: By flattening vessel and berth properties into parallel arrays (Structure of Arrays), the CPU prefetcher can load relevant data more effectively than with an array of structs (AoS), significantly speeding up the evaluation of objective functions.
* **Constraint Validation**: The builder enforces logical consistency (e.g., ensuring arrival times are non-negative, intervals are well-formed) before the potentially expensive search process begins.
