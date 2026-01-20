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

use bollard_model::{model::Model, solution::Solution};
use bollard_search::{neighborhood::neighborhoods::Neighborhoods, num::SolverNumeric};
use std::fmt;

/// Error indicating a mismatch between the Model and the Neighborhood vessel counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelNeighborhoodMismatchError {
    pub model_count: usize,
    pub neighborhood_count: usize,
}

impl fmt::Display for ModelNeighborhoodMismatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "vessel count mismatch: model has {}, but neighborhood has {}",
            self.model_count, self.neighborhood_count
        )
    }
}

impl std::error::Error for ModelNeighborhoodMismatchError {}

/// Error indicating a mismatch between the Model and the Initial Solution vessel counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSolutionMismatchError {
    pub model_count: usize,
    pub solution_count: usize,
}

impl fmt::Display for ModelSolutionMismatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "vessel count mismatch: model has {}, but initial solution has {}",
            self.model_count, self.solution_count
        )
    }
}

impl std::error::Error for ModelSolutionMismatchError {}

/// Error indicating a mismatch between the Neighborhood and the Initial Solution vessel counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeighborhoodSolutionMismatchError {
    pub neighborhood_count: usize,
    pub solution_count: usize,
}

impl fmt::Display for NeighborhoodSolutionMismatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "vessel count mismatch: neighborhood has {}, but initial solution has {}",
            self.neighborhood_count, self.solution_count
        )
    }
}

impl std::error::Error for NeighborhoodSolutionMismatchError {}

/// Errors that can occur when building `LocalSearchParams`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalSearchBuilderError {
    ModelNeighborhoodMismatch(ModelNeighborhoodMismatchError),
    ModelSolutionMismatch(ModelSolutionMismatchError),
    NeighborhoodSolutionMismatch(NeighborhoodSolutionMismatchError),
}

impl fmt::Display for LocalSearchBuilderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelNeighborhoodMismatch(e) => write!(f, "builder error: {}", e),
            Self::ModelSolutionMismatch(e) => write!(f, "builder error: {}", e),
            Self::NeighborhoodSolutionMismatch(e) => write!(f, "builder error: {}", e),
        }
    }
}

impl std::error::Error for LocalSearchBuilderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ModelNeighborhoodMismatch(e) => Some(e),
            Self::ModelSolutionMismatch(e) => Some(e),
            Self::NeighborhoodSolutionMismatch(e) => Some(e),
        }
    }
}

/// The validated parameters required to run the local search engine.
pub struct LocalSearchParams<'a, T, N, H, D, O, M>
where
    T: SolverNumeric,
{
    model: &'a Model<T>,
    decoder: &'a mut D,
    neighborhood: &'a N,
    operator: &'a mut O,
    metaheuristic: &'a mut H,
    monitor: M,
    initial_solution: &'a Solution<T>,
}

pub(crate) struct SearchParams<'a, T, N, H, D, O, M>
where
    T: SolverNumeric,
{
    pub(crate) model: &'a Model<T>,
    pub(crate) decoder: &'a mut D,
    pub(crate) neighborhood: &'a N,
    pub(crate) operator: &'a mut O,
    pub(crate) metaheuristic: &'a mut H,
    pub(crate) monitor: M,
    pub(crate) initial_solution: &'a Solution<T>,
}

impl<'a, T, N, H, D, O, M> LocalSearchParams<'a, T, N, H, D, O, M>
where
    T: SolverNumeric,
{
    #[inline]
    pub fn model(&'a self) -> &'a Model<T> {
        self.model
    }

    #[inline]
    pub fn decoder(&'a self) -> &'a D {
        self.decoder
    }

    #[inline]
    pub fn neighborhood(&'a self) -> &'a N {
        self.neighborhood
    }

    #[inline]
    pub fn operator(&'a self) -> &'a O {
        self.operator
    }

    #[inline]
    pub fn metaheuristic(&'a self) -> &'a H {
        self.metaheuristic
    }

    #[inline]
    pub fn monitor(&self) -> &M {
        &self.monitor
    }

    #[inline]
    pub fn initial_solution(&'a self) -> &'a Solution<T> {
        self.initial_solution
    }

    #[inline]
    pub(crate) fn into_inner(self) -> SearchParams<'a, T, N, H, D, O, M> {
        SearchParams {
            model: self.model,
            decoder: self.decoder,
            neighborhood: self.neighborhood,
            operator: self.operator,
            metaheuristic: self.metaheuristic,
            monitor: self.monitor,
            initial_solution: self.initial_solution,
        }
    }
}

impl<'a, T, N, H, D, O, M> LocalSearchParams<'a, T, N, H, D, O, M>
where
    T: SolverNumeric,
{
    pub fn builder(
        model: &'a Model<T>,
        decoder: &'a mut D,
        neighborhood: &'a N,
        operator: &'a mut O,
        metaheuristic: &'a mut H,
        monitor: M,
        initial_solution: &'a Solution<T>,
    ) -> LocalSearchParamsBuilder<'a, T, N, H, D, O, M> {
        LocalSearchParamsBuilder {
            model,
            decoder,
            neighborhood,
            operator,
            metaheuristic,
            monitor,
            initial_solution,
        }
    }
}

pub struct LocalSearchParamsBuilder<'a, T, N, H, D, O, M>
where
    T: SolverNumeric,
{
    model: &'a Model<T>,
    decoder: &'a mut D,
    neighborhood: &'a N,
    operator: &'a mut O,
    metaheuristic: &'a mut H,
    monitor: M,
    initial_solution: &'a Solution<T>,
}

impl<'a, T, N, H, D, O, M> LocalSearchParamsBuilder<'a, T, N, H, D, O, M>
where
    T: SolverNumeric,
    N: Neighborhoods,
{
    pub fn new(
        model: &'a Model<T>,
        decoder: &'a mut D,
        neighborhood: &'a N,
        operator: &'a mut O,
        metaheuristic: &'a mut H,
        monitor: M,
        initial_solution: &'a Solution<T>,
    ) -> Self {
        Self {
            model,
            decoder,
            neighborhood,
            operator,
            metaheuristic,
            monitor,
            initial_solution,
        }
    }

    pub fn build(self) -> Result<LocalSearchParams<'a, T, N, H, D, O, M>, LocalSearchBuilderError> {
        let model_vessels = self.model.num_vessels();
        let neighbor_vessels = self.neighborhood.num_vessels();
        let solution_vessels = self.initial_solution.num_vessels();

        if model_vessels != neighbor_vessels {
            return Err(LocalSearchBuilderError::ModelNeighborhoodMismatch(
                ModelNeighborhoodMismatchError {
                    model_count: model_vessels,
                    neighborhood_count: neighbor_vessels,
                },
            ));
        }

        if model_vessels != solution_vessels {
            return Err(LocalSearchBuilderError::ModelSolutionMismatch(
                ModelSolutionMismatchError {
                    model_count: model_vessels,
                    solution_count: solution_vessels,
                },
            ));
        }

        if neighbor_vessels != solution_vessels {
            return Err(LocalSearchBuilderError::NeighborhoodSolutionMismatch(
                NeighborhoodSolutionMismatchError {
                    neighborhood_count: neighbor_vessels,
                    solution_count: solution_vessels,
                },
            ));
        }

        Ok(LocalSearchParams {
            model: self.model,
            decoder: self.decoder,
            neighborhood: self.neighborhood,
            operator: self.operator,
            metaheuristic: self.metaheuristic,
            monitor: self.monitor,
            initial_solution: self.initial_solution,
        })
    }
}
