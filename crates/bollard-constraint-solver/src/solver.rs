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

use crate::{state::SearchState, trail::SearchTrail};
use bollard_core::num::constants::MinusOne;
use bollard_model::model::Model;
use num_traits::{PrimInt, Signed};

#[derive(Clone)]
pub struct Solver<'m, T>
where
    T: PrimInt + Signed,
{
    model: &'m Model<T>,
    state: SearchState<T>,
    trail: SearchTrail<T>,
}

impl<'m, T> std::fmt::Debug for Solver<'m, T>
where
    T: PrimInt + Signed + MinusOne + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Solver")
            .field("model", &self.model)
            .field("state", &self.state)
            .field("trail", &self.trail)
            .finish()
    }
}

impl<'m, T> std::fmt::Display for Solver<'m, T>
where
    T: PrimInt + Signed + MinusOne + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Solver(model: {}, state: {}, trail: {})",
            self.model, self.state, self.trail
        )
    }
}
