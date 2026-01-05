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

//! Simulated Annealing metaheuristic.
//!
//! This module implements a temperature‑driven search that occasionally accepts
//! worsening moves to escape local optima. The acceptance probability follows the
//! Metropolis criterion and decays over time according to a cooling schedule,
//! shifting the search from exploratory to exploitative behavior as the temperature
//! decreases. The design separates the acceptance logic from temperature management,
//! allowing different cooling strategies to be plugged in without changing the core
//! algorithm.
//!
//! The type `SimulatedAnnealing<T, R, C>` provides the control flow required by
//! the `Metaheuristic` trait. It consults a `CoolingSchedule` to read and update
//! the temperature. When the schedule indicates a frozen state, the method reduces
//! to strict descent and rejects non‑improving moves. Otherwise, a candidate with
//! higher cost than the current solution may still be accepted with a probability
//! that decreases as the temperature drops and as the cost increase grows.
//!
//! Cooling schedules such as geometric and linear decay are included. Geometric
//! cooling multiplies the temperature by a factor in `(0, 1)` each update, while
//! linear cooling subtracts a fixed decrement and clamps at zero. Both schedules
//! expose a minimum temperature threshold that defines the frozen regime in which
//! worsening moves are no longer considered.
//!
//! Numerical robustness is prioritized when converting solver‑specific numeric types
//! to `f64` for probability calculations. If a conversion fails or yields a non‑finite
//! value, the candidate is rejected. Extremely low temperatures are treated as frozen
//! to avoid unstable acceptance probabilities. The evaluator used for objective
//! comparison is `WeightedFlowTimeEvaluator`, keeping scoring consistent with other
//! metaheuristics in this crate.

use crate::meta::metaheuristic::Metaheuristic;
use crate::{eval::WeightedFlowTimeEvaluator, memory::Schedule};
use bollard_model::model::Model;
use bollard_search::{monitor::search_monitor::SearchCommand, num::SolverNumeric};
use rand::Rng;

/// Defines the thermodynamics of the annealing process.
///
/// Implementors control the initial temperature, the decay function, and the
/// "frozen" condition where the search reverts to a greedy descent.
pub trait CoolingSchedule: Send + Sync + std::fmt::Debug {
    /// Resets the temperature to its initial state.
    /// Called once at the start of the search.
    fn on_start(&mut self);

    /// Updates the temperature for the next iteration.
    /// Called after every move (accepted or rejected).
    fn update(&mut self);

    /// Returns the current temperature ($T$).
    fn current(&self) -> f64;

    /// Returns `true` if the temperature is low enough to stop accepting worsening moves.
    ///
    /// When frozen, the metaheuristic effectively behaves like a Hill Climber,
    /// avoiding expensive floating-point math for probability calculations.
    fn is_frozen(&self) -> bool;
}

/// A standard geometric cooling schedule: $T_{k+1} = T_k \times \alpha$.
///
/// This is the most common schedule in literature. It cools rapidly at first
/// and then slows down as it approaches zero, allowing fine-grained settling
/// in the final phases.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometricCooling {
    initial: f64,
    current: f64,
    alpha: f64, // The decay rate (e.g., 0.995)
    min_temp: f64,
}

impl GeometricCooling {
    /// Creates a new geometric cooling schedule.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is not strictly between `0.0` and `1.0`.
    #[inline]
    pub fn new(initial: f64, alpha: f64, min_temp: f64) -> Self {
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "called `GeometricCooling::new()` with invalid alpha: {}. Must be in (0.0, 1.0)",
            alpha
        );
        Self {
            initial,
            current: initial,
            alpha,
            min_temp,
        }
    }
}

impl CoolingSchedule for GeometricCooling {
    #[inline]
    fn on_start(&mut self) {
        self.current = self.initial;
    }

    #[inline]
    fn update(&mut self) {
        // Simple multiplication is numerically stable enough for this purpose
        self.current *= self.alpha;
    }

    #[inline]
    fn current(&self) -> f64 {
        self.current
    }

    #[inline]
    fn is_frozen(&self) -> bool {
        self.current <= self.min_temp
    }
}

/// A linear cooling schedule: $T_{k+1} = \max(0, T_k - \delta)$.
///
/// This reduces the temperature by a fixed amount at every step. It is useful
/// when you have a fixed budget of iterations and want to ensure the temperature
/// hits zero exactly at the deadline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearCooling {
    initial: f64,   // The initial temperature
    current: f64,   // The current temperature
    decrement: f64, // The delta to subtract
    min_temp: f64,  // The minimum temperature threshold
}

impl LinearCooling {
    /// Creates a new linear cooling schedule.
    #[inline]
    pub fn new(initial: f64, decrement: f64, min_temp: f64) -> Self {
        Self {
            initial,
            current: initial,
            decrement,
            min_temp,
        }
    }
}

impl CoolingSchedule for LinearCooling {
    /// Resets the temperature to its initial value.
    #[inline]
    fn on_start(&mut self) {
        self.current = self.initial;
    }

    /// Decreases the temperature by the fixed decrement, clamped at zero.
    #[inline]
    fn update(&mut self) {
        self.current = (self.current - self.decrement).max(0.0);
    }

    /// Returns the current temperature.
    #[inline]
    fn current(&self) -> f64 {
        self.current
    }

    /// Returns `true` if the temperature is below the minimum threshold.
    #[inline]
    fn is_frozen(&self) -> bool {
        self.current <= self.min_temp
    }
}

/// A Simulated Annealing metaheuristic powered by a pluggable `CoolingSchedule`.
///
/// It uses the **Metropolis Criterion** to decide whether to accept a move:
/// $$ P(\text{accept}) = \exp\left(\frac{-(C_{\text{new}} - C_{\text{old}})}{T}\right) $$
#[derive(Debug, Clone)]
pub struct SimulatedAnnealing<T, R, C>
where
    T: SolverNumeric,
{
    cooling_schedule: C, // The cooling schedule managing temperature decay
    rng: R,              // Random number generator for stochastic acceptance
    evaluator: WeightedFlowTimeEvaluator<T>, // Evaluator for objective calculation
}

impl<T, R, C> SimulatedAnnealing<T, R, C>
where
    T: SolverNumeric,
    R: Rng,
    C: CoolingSchedule,
{
    /// Creates a new Simulated Annealing instance.
    ///
    /// # Arguments
    /// * `cooling_schedule`: The strategy for temperature decay (e.g., `GeometricCooling`).
    /// * `rng`: The random number generator used for the probabilistic acceptance check.
    #[inline]
    pub fn new(cooling_schedule: C, rng: R) -> Self {
        Self {
            cooling_schedule,
            rng,
            evaluator: WeightedFlowTimeEvaluator::new(),
        }
    }
}

impl<T, R, C> Metaheuristic<T> for SimulatedAnnealing<T, R, C>
where
    T: SolverNumeric,
    R: Rng + Send + Sync,
    C: CoolingSchedule,
{
    type Evaluator = WeightedFlowTimeEvaluator<T>;

    fn name(&self) -> &str {
        "SimulatedAnnealing"
    }

    fn on_start(&mut self, _model: &Model<T>, _initial_solution: &Schedule<T>) {
        self.cooling_schedule.on_start();
    }

    fn search_command(
        &mut self,
        _iteration: u64,
        _model: &Model<T>,
        _best_solution: &Schedule<T>,
    ) -> SearchCommand {
        SearchCommand::Continue
    }

    fn should_accept(
        &mut self,
        _model: &Model<T>,
        current: &Schedule<T>,
        candidate: &Schedule<T>,
        _best: &Schedule<T>,
    ) -> bool {
        let current_cost = current.objective_value();
        let candidate_cost = candidate.objective_value();

        // Strict improvement: always accept
        if candidate_cost < current_cost {
            return true;
        }

        // Frozen state: only accept improvements (greedy descent)
        if self.cooling_schedule.is_frozen() {
            return false;
        }

        // Worsening move: Metropolis criterion
        // cost_difference = candidate - current. Since we minimize, a worse solution has a higher objective.
        let (candidate_cost_value, current_cost_value) =
            match (candidate_cost.to_f64(), current_cost.to_f64()) {
                (Some(cand), Some(curr)) => (cand, curr),
                // If conversion fails, we cannot compute a reliable probability.
                // Reject the move.
                _ => return false,
            };

        let cost_difference = candidate_cost_value - current_cost_value;

        // Safety: If cost_difference is negative (better), we should have caught it above.
        // If cost_difference is 0 (equal), exp(0) = 1.0 -> 100% acceptance (random walk on plateau).
        let temperature = self.cooling_schedule.current();
        if temperature <= 1e-9 {
            return false; // Prevent division by zero / extreme probabilities
        }

        let acceptance_probability = (-cost_difference / temperature).exp();

        // Stochastic acceptance check
        self.rng.random_bool(acceptance_probability)
    }

    fn on_accept(&mut self, _model: &Model<T>, _new_current: &Schedule<T>) {
        self.cooling_schedule.update();
    }

    fn on_reject(&mut self, _model: &Model<T>, _rejected_candidate: &Schedule<T>) {
        // Temperature decays regardless of outcome ("time passes")
        self.cooling_schedule.update();
    }

    fn on_new_best(&mut self, _model: &Model<T>, _new_best: &Schedule<T>) {
        // Standard SA does not react specially to new bests.
        // Adaptive variants could implement reheating here.
    }

    fn evaluator(&self) -> &WeightedFlowTimeEvaluator<T> {
        &self.evaluator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Schedule;
    use bollard_model::index::BerthIndex;
    use bollard_model::model::ModelBuilder;
    use bollard_model::solution::Solution;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn sched(obj: i64, berths: Vec<BerthIndex>, starts: Vec<i64>) -> Schedule<i64> {
        Schedule::from(Solution::new(obj, berths, starts))
    }

    fn rng_send_sync() -> ChaCha8Rng {
        // ChaCha8Rng is Send + Sync and deterministic with a fixed seed
        ChaCha8Rng::seed_from_u64(42)
    }

    #[test]
    fn test_name_and_evaluator_access() {
        let cooling = GeometricCooling::new(1.0, 0.99, 0.0);
        let rng = rng_send_sync();
        let sa: SimulatedAnnealing<i64, _, _> = SimulatedAnnealing::new(cooling, rng);

        assert_eq!(sa.name(), "SimulatedAnnealing");
        let _eval = sa.evaluator();
    }

    #[test]
    fn test_lifecycle_hooks_do_not_panic() {
        let mut sa: SimulatedAnnealing<i64, _, _> =
            SimulatedAnnealing::new(GeometricCooling::new(1.0, 0.99, 0.01), rng_send_sync());

        // Empty model and schedules are fine for SA
        let model = ModelBuilder::<i64>::new(0, 0).build();
        let current = sched(10, vec![], vec![]);
        let candidate = sched(9, vec![], vec![]);

        // Lifecycle
        sa.on_start(&model, &current);
        sa.on_reject(&model, &candidate);
        sa.on_accept(&model, &candidate);
        sa.on_new_best(&model, &candidate);
    }

    #[test]
    fn test_should_accept_strict_improvement() {
        // Strict improvements must always be accepted regardless of temperature
        let mut sa: SimulatedAnnealing<i64, _, _> =
            SimulatedAnnealing::new(GeometricCooling::new(1.0, 0.95, 0.0), rng_send_sync());

        let model = ModelBuilder::<i64>::new(0, 0).build();

        let current = sched(100, vec![BerthIndex::new(0)], vec![10]);
        let better = sched(99, vec![BerthIndex::new(0)], vec![10]);
        let best = current.clone();

        sa.on_start(&model, &current);

        assert!(
            sa.should_accept(&model, &current, &better, &best),
            "SA must accept strictly better candidate"
        );
    }

    #[test]
    fn test_should_accept_frozen_rejects_equal_and_worse() {
        // With min_temp >= initial, schedule is frozen from the start
        let mut sa: SimulatedAnnealing<i64, _, _> =
            SimulatedAnnealing::new(GeometricCooling::new(1.0, 0.99, 1.0), rng_send_sync());
        let model = ModelBuilder::<i64>::new(0, 0).build();

        let current = sched(100, vec![BerthIndex::new(0)], vec![10]);
        let equal = sched(100, vec![BerthIndex::new(0)], vec![10]);
        let worse = sched(101, vec![BerthIndex::new(0)], vec![10]);
        let best = current.clone();

        sa.on_start(&model, &current);

        assert!(
            !sa.should_accept(&model, &current, &equal, &best),
            "Frozen SA must reject equal-cost candidate"
        );
        assert!(
            !sa.should_accept(&model, &current, &worse, &best),
            "Frozen SA must reject worse candidate"
        );
    }

    #[test]
    fn test_should_accept_equal_cost_non_frozen_is_deterministic() {
        // Non-frozen temperature, equal cost => acceptance_probability = exp(0) = 1.0
        // This should accept deterministically regardless of RNG.
        let mut sa: SimulatedAnnealing<i64, _, _> =
            SimulatedAnnealing::new(LinearCooling::new(100.0, 0.0, 0.0), rng_send_sync());
        let model = ModelBuilder::<i64>::new(0, 0).build();

        let current = sched(100, vec![BerthIndex::new(0)], vec![10]);
        let equal = sched(100, vec![BerthIndex::new(0)], vec![10]);
        let best = current.clone();

        sa.on_start(&model, &current);

        assert!(
            sa.should_accept(&model, &current, &equal, &best),
            "With non-frozen temperature, equal-cost candidate should be accepted with probability 1.0"
        );
    }

    #[test]
    fn test_should_accept_low_temperature_guard_rejects_worse() {
        // Temperature effectively zero should force rejection of non-improving moves
        let mut sa: SimulatedAnnealing<i64, _, _> =
            SimulatedAnnealing::new(LinearCooling::new(0.0, 0.0, 0.0), rng_send_sync());
        let model = ModelBuilder::<i64>::new(0, 0).build();

        let current = sched(100, vec![BerthIndex::new(0)], vec![10]);
        let worse = sched(120, vec![BerthIndex::new(0)], vec![10]);
        let best = current.clone();

        sa.on_start(&model, &current);
        assert!(
            !sa.should_accept(&model, &current, &worse, &best),
            "Near-zero temperature must reject worse candidate to avoid unstable probabilities"
        );
    }
}
