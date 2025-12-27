# Bollard Model

**The Core Domain Model for the Bollard Berth Allocation Solver.**

`bollard_model` provides the fundamental data structures, types, and builders required to define, validate, and represent the **Berth Allocation Problem (BAP)**. It is designed for high-performance combinatorial optimization, utilizing memory-efficient layouts and strict type safety to prevent common logic errors during the solving process.

## Key Features

* **Type-Safe Indexing**: Distinct `VesselIndex` and `BerthIndex` types ensure that ship IDs and dock IDs are never accidentally swapped.
* **Two-Phase Construction**: Implements the **Builder Pattern** to separate the mutable configuration phase from the immutable, high-speed solving phase.
* **Memory Optimization**: Uses **Structure of Arrays (SoA)** and flattened vectors for cache locality.
* **Sentinel-Based Optionals**: Custom `ProcessingTime` types avoid `Option<T>` overhead in hot loops.
* **Complexity Analysis**: Built-in tools to calculate the logarithmic search space size to estimate solver difficulty.

## Usage Example

The lifecycle of a problem definition involves creating a `ModelBuilder`, adding constraints, and building the immutable `Model`.

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

## Architecture

The crate is organized into four main modules:

1. **`index`**: Strongly typed indices (`VesselIndex`, `BerthIndex`).
2. **`model`**: The core `Model` struct (immutable) and `ModelBuilder` (mutable). Contains logic for validation and complexity estimation.
3. **`solution`**: The `Solution` struct representing the final assignment of berths and times.
4. **`time`**: Low-level time representations, including the sentinel-optimized `ProcessingTime`.
