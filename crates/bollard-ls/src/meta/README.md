# Metaheuristic Strategies

This module provides the high-level control logic that guides the Local Search engine. It defines the `Metaheuristic` trait and implements several standard strategies ranging from simple greedy descent to advanced memory-based and temperature-driven methods.

## 1. The Core Abstraction: `Metaheuristic`

A metaheuristic is the "brain" of the search process. It does not generate moves itself (that is the job of the `Operator`) nor does it calculate costs (that is the job of the `Evaluator`). Instead, it decides **whether to accept or reject** a candidate solution proposed by the engine.

The `Metaheuristic` trait provides a set of lifecycle hooks that allow the strategy to maintain internal state (like temperature or tabu lists) and influence the search trajectory.

### Lifecycle

The engine interacts with the metaheuristic through a strict event loop:

1.  **`on_start(model, initial_solution)`**:
    * **When:** Called once before the search loop begins.
    * **Purpose:** Initialize internal state (e.g., set initial temperature, clear tabu lists, size penalty matrices).

2.  **`search_command(iteration, model, best_solution)`**:
    * **When:** Called at the start of every iteration.
    * **Purpose:** Allows the strategy to terminate the search early (e.g., if a target objective is reached or stagnation limit exceeded).
    * **Returns:** `SearchCommand::Continue` or `SearchCommand::Terminate(reason)`.

3.  **`should_accept(model, current, candidate, best)`**:
    * **When:** Called after a valid candidate solution has been generated and evaluated.
    * **Purpose:** The core decision point. Determines if `candidate` should replace `current`.
    * **Logic:** Can be deterministic (Greedy) or stochastic (Simulated Annealing).

4.  **`on_accept / on_reject`**:
    * **When:** Called immediately after the move is committed or discarded.
    * **Purpose:** Update internal state (e.g., cool down temperature, increment stagnation counters, update tabu lists).

---

## 2. Implemented Strategies

### `GreedyDescent` (Hill Climbing)

A deterministic, first-improvement strategy. It serves as a fast baseline and acts as the "frozen" state for more complex algorithms.

* **Logic:** Accepts a candidate $S'$ if and only if its objective $f(S')$ is strictly better than the current solution $S$.
    $$f(S') < f(S)$$
* **Behavior:** Moves rapidly to the nearest local optimum and then stops (or continues searching without accepting). It effectively performs a "First Improvement" descent.
* **Use Case:** Polishing solutions at the end of a genetic algorithm or as a high-speed initial pass.

---

### `SimulatedAnnealing`

A stochastic strategy inspired by thermodynamics. It escapes local optima by occasionally accepting worsening moves, with a probability that decreases over time.

* **Logic:**
    * If $f(S') < f(S)$ (improvement), always accept.
    * If $f(S') \ge f(S)$ (worsening), accept with probability $P$:
        $$P(\text{accept}) = \exp\left(\frac{-(f(S') - f(S))}{T}\right)$$
        Where $T$ is the current **Temperature**.

* **Cooling Schedules:**
    The module supports pluggable cooling schedules via the `CoolingSchedule` trait:
    * **Geometric:** $T_{k+1} = T_k \times \alpha$ (where $0 < \alpha < 1$).
    * **Linear:** $T_{k+1} = \max(0, T_k - \delta)$.

* **Numerical Robustness:**
    The implementation explicitly guards against numerical instability (e.g., $T \approx 0$ or infinite costs) to ensure safe, deterministic behavior in edge cases.

---

### `TabuSearch`

A deterministic, memory-based strategy that prevents cycling by forbidding recently visited solutions.



* **Logic:**
    It maintains a FIFO queue ("Tabu List") of solution signatures (hashes). A move to $S'$ is rejected if $S'$ is in the list, unless it satisfies the **Aspiration Criterion**.
* **Aspiration:**
    A Tabu move is accepted if it is strictly better than the global best solution found so far ($S_{best}$).
    $$\text{Accept if } (S' \notin \text{TabuList}) \lor (f(S') < f(S_{best}))$$
* **Hash-Based Memory:**
    Instead of recording move attributes (which couples the metaheuristic to specific operators), this implementation hashes the **solution state** (objective + structure). This makes it operator-agnostic.

---

### `GuidedLocalSearch` (GLS)

A deterministic strategy that dynamically modifies the objective function to escape local optima. It "fills in" valleys by penalizing specific solution features.



* **Logic:**
    When the search stagnates (rejects $k$ consecutive moves), GLS identifies "bad features" in the current solution and penalizes them.
* **Augmented Objective:**
    The search minimizes a modified cost function $h(S)$:
    $$h(S) = f(S) + \lambda \sum_{(v,b) \in S} p_{v,b}$$
    Where:
    * $f(S)$: The original objective (Weighted Flow Time).
    * $\lambda$: The penalty scaling factor.
    * $p_{v,b}$: The number of times assigning vessel $v$ to berth $b$ has been penalized.

* **Utility Function:**
    Penalties are applied to features maximizing the utility:
    $$\text{Utility}(v, b) = \frac{\text{Cost}(v, b)}{1 + p_{v,b}}$$
    For Weighted Flow Time, the feature cost is $\text{Weight}_v \times \text{CompletionTime}_v$. This correctly targets heavy vessels that are scheduled late.

---

## 3. Architecture & Safety

### Evaluator Separation
The metaheuristic owns a `GuidedEvaluator` (or standard `WeightedFlowTimeEvaluator`). This separation ensures that the "Search Logic" (decision making) is decoupled from the "Physics Logic" (cost calculation).
* **Base Cost:** Calculated by the inner evaluator.
* **Augmented Cost:** Calculated by the metaheuristic wrapper (e.g., adding penalties).

### Numeric Safety
All metaheuristics are generic over `T: SolverNumeric`. However, probability calculations (SA) and utility ratios (GLS) require floating-point math.
* **Conversion:** The code uses `num_traits` to robustly convert between `T` (e.g., `i64`) and `f64`.
* **Guards:** Explicit checks for `is_finite()` and division-by-zero ensure that invalid states (like integer overflow or NaN) result in a safe rejection rather than undefined behavior or panics.

### Memory Efficiency
* **Penalty Matrix:** Uses a flattened `Vec<u32>` instead of `Vec<Vec<T>>` to maximize cache locality.
* **Tabu Hashing:** Uses a lightweight hashing strategy on solution components rather than deep cloning solution objects.
