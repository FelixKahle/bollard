# Bollard Core

Low-level utilities, numerics, and math primitives used across the Bollard scheduling ecosystem.

`bollard_core` provides foundational building blocks that are shared by higher-level crates such as `bollard_model` and the solver crates. It focuses on correctness, performance, and ergonomic APIs in areas like numeric operations, typed indices, and interval math.

## Key Features

- Interval Math:
  - Efficient, type-safe closed-open intervals (`[start, end)`) with intersection/union/difference operations.
  - Iteration support (including double-ended and fused iterators) and conversion to/from Rust `Range`.
- Numerics:
  - By-value checked arithmetic traits (e.g., `CheckedAddVal`, `CheckedMulVal`) that mirror Rust’s intrinsic checked ops.
  - By-value saturating arithmetic traits (e.g., `SaturatingAddVal`, `SaturatingNegVal`) for robust numeric pipelines.
  - Constants traits (`MinusOne`, `PlusOne`, `Zero`) implemented for all core integer types.
- Utilities:
  - Strongly typed indices via phantom tags to prevent accidental index mixing in large systems.
  - Optional iterator wrapper (`MaybeIter`) with `ExactSizeIterator`, `DoubleEndedIterator`, and `FusedIterator` support.
  - `Brand` type for creating nominal types and preventing type aliasing issues.

## Module Overview

- `math`:
  - `interval`: `ClosedOpenInterval<T>` and `ClosedOpenIntervalIterator<T>` for `[start, end)` interval operations.
- `num`:
  - `constants`: `MinusOne`, `PlusOne`, `Zero` traits and implementations for integer primitives.
  - `ops::checked_arithmetic`: Checked arithmetic traits by value for add/sub/mul/div/rem/neg/shifts.
  - `ops::saturating_arithmetic`: Saturating arithmetic traits by value for add/sub/mul/neg.
- `utils`:
  - `index`: `TypedIndex<T>` and `TypedIndexTag` for type-safe indexing.
  - `iter`: `MaybeIter<I>` for optional iterator composition.
  - `marker`: `Brand<'x>` for nominal typing.

## Usage Examples

### Interval Math

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

    // Iterate across [start, end) step-by-step
    let values: Vec<i64> = a.iter().collect();
    assert_eq!(values, (10..20).collect::<Vec<_>>());
}
```

### Type-Safe Indices

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

    // Prevents mixing with other index types at compile time
    assert_eq!(v0.get(), 0);
    assert!(!v1.is_zero());

    // Arithmetic operations on indices
    let v2 = v1 + 1;
    assert_eq!(v2.get(), 2);
    assert_eq!(format!("{}", v2), "VesselIndex(2)");
}
```

### Checked Arithmetic (By Value)

```rust
use bollard_core::num::ops::checked_arithmetic::{CheckedAddVal, CheckedMulVal};

fn main() {
    let a: u8 = 200;
    let b: u8 = 100;

    // Overflow returns None
    assert_eq!(a.checked_add_val(b), None);

    let c: u8 = 20;
    assert_eq!(a.checked_add_val(c), Some(220));

    // Checked multiplication
    let x: u8 = 20;
    let y: u8 = 20;
    assert_eq!(x.checked_mul_val(y), None); // 400 overflows u8
}
```

### Saturating Arithmetic (By Value)

```rust
use bollard_core::num::ops::saturating_arithmetic::{SaturatingAddVal, SaturatingNegVal};

fn main() {
    let a: u8 = 250;
    let b: u8 = 10;
    assert_eq!(a.saturating_add_val(b), 255); // clamps at u8::MAX

    let m: i8 = -128;
    assert_eq!(m.saturating_neg_val(), 127); // clamps to i8::MAX
}
```

### Optional Iterator Composition

```rust
use bollard_core::utils::iter::MaybeIter;

fn main() {
    let some = MaybeIter::new(Some(vec![1, 2, 3].into_iter()));
    let none: MaybeIter<std::vec::IntoIter<i32>> = MaybeIter::new(None);

    assert_eq!(some.collect::<Vec<_>>(), vec![1, 2, 3]);
    assert_eq!(none.collect::<Vec<_>>(), Vec::<i32>::new());

    // Works with adaptor chains
    let doubled: Vec<i32> = MaybeIter::new(Some(vec![1, 2, 3].into_iter()))
        .map(|x| x * 2)
        .collect();
    assert_eq!(doubled, vec![2, 4, 6]);
}
```

## Design Notes

- By-value traits: Many numeric traits here are "by value" to align with Rust’s intrinsic methods and reduce ambiguity with reference-based trait APIs that are common in external crates.
- Composition-first utilities: Types like `TypedIndex` and `MaybeIter` are deliberately minimal but compose cleanly with standard library traits and iterators.
- Defensive math: `ClosedOpenInterval` is designed for robust interval operations and comes with thorough unit tests, double-ended iteration, and conversions to standard ranges.
