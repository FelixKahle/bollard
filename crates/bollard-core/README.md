# Bollard Core

**Low-level utilities, numerics, and math primitives for the Bollard scheduling ecosystem.**

`bollard_core` provides the foundational building blocks shared by higher-level crates such as `bollard_model` and the solver engines. It prioritizes correctness, type safety, and ergonomic APIs, offering specialized primitives for numeric operations, typed indices, and interval arithmetic that are critical for scheduling algorithms.

## Key Features

* **Robust Interval Math**: specialized `[start, end)` closed-open interval types designed for scheduling. Includes efficient intersection, union, difference, and double-ended iteration support.
* **By-Value Numerics**: Arithmetic traits (`CheckedAddVal`, `SaturatingAddVal`, etc.) that mirror Rust’s intrinsic operators but work by value, enabling cleaner numeric pipelines without reference clutter.
* **Type-Safe Indexing**: Zero-cost `TypedIndex<T>` wrappers that use phantom tags to prevent accidental mixing of different domain indices (e.g., vessels vs. berths) at compile time.
* **Iterator Composition**: Lightweight utilities like `MaybeIter`, which provides a unified `ExactSizeIterator` and `DoubleEndedIterator` interface over optional iterators.
* **Nominal Typing**: A `Brand` type mechanism to create nominal types, effectively preventing type aliasing issues in complex domain models.

## Architecture

The crate is organized into three primary modules:

1. **`math`**: Mathematical primitives.
* **`interval`**: Contains `ClosedOpenInterval<T>` for handling time ranges and `ClosedOpenIntervalIterator<T>` for stepping through them.


2. **`num`**: Numeric traits and implementations.
* **`constants`**: Traits like `MinusOne`, `PlusOne`, and `Zero` implemented for integer primitives.
* **`ops`**: Contains `checked_arithmetic` and `saturating_arithmetic` modules defining by-value trait variants for all standard operators.


3. **`utils`**: General-purpose helpers.
* **`index`**: The `TypedIndex<T>` and `TypedIndexTag` machinery.
* **`iter`**: Iterator wrappers like `MaybeIter<I>`.
* **`marker`**: Marker types such as `Brand<'x>`.



## Quick Start

### Interval Math

Handling time windows with robust set operations.

```rust
use bollard_core::math::interval::ClosedOpenInterval;

fn main() {
    // Construct a closed-open interval [10, 20)
    let a = ClosedOpenInterval::new(10i64, 20i64);
    let b = ClosedOpenInterval::new(15i64, 25i64);

    // Intersection: [15, 20)
    let i = a.intersection(b);
    assert_eq!(i.start(), 15);
    assert_eq!(i.end(), 20);

    // Union (non-overlapping handled internally)
    let u = a.union(b);
    assert_eq!(u.start(), 10);
    assert_eq!(u.end(), 25);

    // Difference: subtract b from a -> [10, 15)
    let d = a.difference(b);
    assert!(d.is_some());
    let left = d.unwrap();
    assert_eq!(left.start(), 10);
    assert_eq!(left.end(), 15);
}

```

### Type-Safe Indices

Preventing logic errors by distinguishing integer IDs at the type level.

```rust
use bollard_core::utils::index::{TypedIndex, TypedIndexTag};

#[derive(Clone)]
struct VesselTag;

impl TypedIndexTag for VesselTag {
    const NAME: &'static str = "VesselIndex";
}

type VesselIndex = TypedIndex<VesselTag>;

fn main() {
    let v0 = VesselIndex::new(0);
    let v1 = VesselIndex::new(1);

    // Compile-time guarantee: cannot compare or mix with other index types
    assert_eq!(v0.get(), 0);
    
    // Arithmetic operations remain available
    let v2 = v1 + 1;
    assert_eq!(v2.get(), 2);
    assert_eq!(format!("{}", v2), "VesselIndex(2)");
}

```

### By-Value Arithmetic

Performing checked arithmetic cleanly without references.

```rust
use bollard_core::num::ops::checked_arithmetic::{CheckedAddVal, CheckedMulVal};

fn main() {
    let a: u8 = 200;
    
    // Standard overflow check returns None
    assert_eq!(a.checked_add_val(100u8), None);
    
    // Valid addition
    assert_eq!(a.checked_add_val(20u8), Some(220));
}

```

## Design & Performance

* **Zero-Cost Abstractions**: Utilities like `TypedIndex` and `MaybeIter` are designed to compile down to the same machine code as their primitive counterparts, ensuring no runtime penalty for the added safety.
* **Defensive Math**: The `ClosedOpenInterval` implementation includes comprehensive unit tests to handle edge cases in intersection and difference operations, which are common sources of bugs in scheduling logic.
* **Trait Alignment**: The numeric traits are deliberately designed to accept arguments **by value** (copy). This aligns with Rust's intrinsic integer methods and avoids the syntactic noise of referencing/dereferencing small integer types.
