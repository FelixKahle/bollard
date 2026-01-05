# Local Search Operators & Adaptive Selection

This module provides the core abstractions for performing Local Search (LS) on a `VesselPriorityQueue`. It introduces a stateful operator interface and an intelligent "Meta-Operator" that uses Reinforcement Learning (specifically Multi-Armed Bandit strategies) to automatically select the best search heuristic during runtime.

## 1. The Core Abstraction: `LocalSearchOperator`

In high-performance local search, operators cannot simply be stateless functions. They often need to pre-calculate moves, maintain a cursor in a neighborhood, or cache expensive analysis.

The `LocalSearchOperator` trait models a specific neighborhood (e.g., "Swap Moves", "Shift Moves") as a stateful iterator. It relies on the `Neighborhoods` trait to respect the connectivity and constraints of the problem topology.

### Lifecycle

The search engine interacts with operators using a strict three-phase lifecycle:

1. **`prepare(schedule, queue, neighborhoods)`**:
   * **When:** Called once whenever the search accepts a new "best" solution.
   * **Purpose:** The operator analyzes the new solution. It performs expensive calculations here, such as identifying bottleneck vessels or generating a list of valid swap pairs based on the provided `neighborhoods` topology.
   * **Cost:** High (performed rarely).

2. **`next_neighbor(schedule, mutator, neighborhoods)`**:
   * **When:** Called repeatedly in the inner loop of the search.
   * **Purpose:** Applies *exactly one* mutation to the `queue` using the `mutator`. It typically uses `neighborhoods` for rejection sampling (skipping unconnected vessels) to ensure only valid or promising moves are attempted.
   * **Returns:** `true` if a move was applied, `false` if the neighborhood is exhausted.
   * **Cost:** Low (performed frequently).

3. **`reset()`**:
   * **When:** Called during restarts or re-intensification.
   * **Purpose:** Resets internal cursors to the start of the *current* neighborhood without re-running the expensive `prepare` analysis.

---

## 2. Adaptive Selection: The Multi-Armed Bandit

The `MultiArmedBanditCompoundOperator` is a wrapper that manages multiple sub-operators. Instead of running them in a fixed order (Round-Robin), it treats operator selection as a **Non-Stationary Multi-Armed Bandit (MAB)** problem.

It balances **Exploration** (trying operators we haven't used much) with **Exploitation** (using the operator that currently yields the biggest improvements).

### The Math: UCB1 with Dynamic Normalization

Standard UCB1 algorithms fail in optimization contexts because objective improvements (rewards) are unscaled. An improvement of `500.0` would dwarf the exploration bonus, causing the algorithm to get stuck.

We solve this using **Dynamic Max-Normalization**.

The score for operator $i$ is calculated as:

$$Score_i = \underbrace{\frac{\bar{X}_i}{X_{max}}}_{\text{Quality}} + \underbrace{C \cdot \sqrt{\frac{2 \ln N}{n_i}}}_{\text{Uncertainty}}$$

Where:
* $\bar{X}_i$: The exponential moving average of improvements yielded by operator $i$.
* $X_{max}$: The **global maximum improvement** seen by *any* operator so far (used for normalization).
* $N$: Total number of moves tried across all operators.
* $n_i$: Number of times operator $i$ has been selected.
* $C$: The exploration coefficient (typically $\sqrt{2} \approx 1.414$).

### Handling Non-Stationarity (Forgetting)

Local search landscapes change rapidly. An operator that is effective at the start (coarse-grained moves) is often useless at the end (fine-grained tuning).

We use an **Exponential Moving Average (EMA)** to update $\bar{X}_i$, allowing the bandit to "forget" old successes:

$$\bar{X}_{new} \leftarrow \bar{X}_{old} + \alpha \cdot (Reward - \bar{X}_{old})$$

* $\alpha$ (**`memory_coeff`**): Controls the forgetting rate. A value of `0.2` implies a "memory" of roughly the last 9-10 events.

### Configuration & Defaults

#### `MultiArmedBanditCompoundOperator::with_defaults`

Initializes the adaptive operator with mathematically robust defaults for general vehicle routing/scheduling problems.

| Parameter | Value | Reason |
| :--- | :--- | :--- |
| **`memory_coeff`** | `0.2` | High reactivity. Allows the search to quickly discard operators that have stopped working (e.g., swapping is no longer effective, switch to shifting). |
| **`exploration_coeff`** | `1.414` | The theoretical optimum ($\sqrt{2}$) for UCB1. Valid because we normalize rewards to $[0, 1]$. |

#### Manual Configuration

```rust
let mab = MultiArmedBanditCompoundOperator::new(
    operators,
    0.1,  // Slower learning (more stable history)
    2.0   // High exploration (try unused operators more aggressively)
);

```

---

## 3. Alternative Compound Operators

### `RoundRobinCompoundOperator`

A deterministic "Round Robin" strategy. It iterates through its list of sub-operators in a fixed sequential order.

* **Logic:** It runs Operator A until it is exhausted (`next_neighbor` returns `false`), then immediately switches to Operator B, and so on.
* **Exhaustion:** The compound operator itself only returns `false` (exhausted) when **all** sub-operators have been exhausted for the current solution.
* **Use Case:** Best when you want to guarantee that every possible neighborhood is fully explored in a predictable order.

### `RandomOperatorCompoundOperator`

A stochastic strategy that selects one random sub-operator for each descent step.

* **Logic:** When `prepare` is called (at the start of a new best solution), it randomly selects one sub-operator. It sticks with this operator until the next `prepare` call.
* **Use Case:** Useful for escaping local optima or adding diversity to the search path. It prevents the search from getting stuck in deterministic loops that might occur with fixed-order operators.

---

## 4. API Reference

### Structs

* **`LocalSearchOperatorIterator`**: An adapter that turns any `LocalSearchOperator` into a standard Rust `Iterator`. Useful for testing or simple linear sweeps. It holds references to the `schedule`, `mutator`, and `neighborhoods`.
* **`BanditStats`**: Internal struct managing the `total_samples`, `avg_improvements`, and `global_max_improvement`.

### Safety

The `BanditStats` implementation uses `unsafe` blocks for array access (`get_unchecked`) inside the hot path of the search loop. This is rigorously guarded by `debug_assert!` checks to ensure indices remain within bounds.
