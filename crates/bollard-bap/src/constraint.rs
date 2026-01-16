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

use bollard_model::index::{BerthIndex, VesselIndex};
use std::fmt;

/// Represents a branching decision imposed on a node in the Branch-and-Price tree.
///
/// In the context of the Berth Allocation Problem (BAP), branching usually occurs
/// on the assignment variables $x_{ij}$, where $x_{ij} = 1$ if vessel $i$ is assigned to berth $j$.
///
/// When the Column Generation (CG) solution is fractional (e.g., Vessel 1 is 50% on Berth 1
/// and 50% on Berth 2), the engine picks a branching candidate and creates two child nodes:
/// 1. **Force Branch:** $x_{ij} = 1$ (Vessel $i$ MUST use Berth $j$).
/// 2. **Forbid Branch:** $x_{ij} = 0$ (Vessel $i$ must NOT use Berth $j$).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BranchConstraint {
    /// Forces a specific vessel to be serviced by a specific berth.
    ///
    /// # Mathematical Implication
    /// $x_{vb} = 1$.
    ///
    /// # Effect on Pricing Oracle
    /// The Oracle must prune any path/schedule for Vessel $v$ that does *not* use Berth $b$.
    /// Effectively, this isolates Vessel $v$ to only consider Berth $b$.
    ForceAssignment {
        vessel: VesselIndex,
        berth: BerthIndex,
    },

    /// Forbids a specific vessel from being serviced by a specific berth.
    ///
    /// # Mathematical Implication
    /// $x_{vb} = 0$.
    ///
    /// # Effect on Pricing Oracle
    /// The Oracle must remove the edge corresponding to Berth $b$ from Vessel $v$'s graph.
    /// Any column generated in this subtree cannot map Vessel $v$ to Berth $b$.
    ForbidAssignment {
        vessel: VesselIndex,
        berth: BerthIndex,
    },
}

impl BranchConstraint {
    /// Returns the logical negation of this constraint.
    ///
    /// This is used to generate the "sibling" node in the search tree.
    ///
    /// # Example
    /// * Negation of `Force(V1, B1)` is `Forbid(V1, B1)`.
    /// * Negation of `Forbid(V1, B1)` is `Force(V1, B1)`.
    #[inline]
    pub fn negate(&self) -> Self {
        match *self {
            Self::ForceAssignment { vessel, berth } => Self::ForbidAssignment { vessel, berth },
            Self::ForbidAssignment { vessel, berth } => Self::ForceAssignment { vessel, berth },
        }
    }

    /// Returns the vessel involved in this constraint.
    #[inline]
    pub fn vessel(&self) -> VesselIndex {
        match *self {
            Self::ForceAssignment { vessel, .. } => vessel,
            Self::ForbidAssignment { vessel, .. } => vessel,
        }
    }

    /// Returns the berth involved in this constraint.
    #[inline]
    pub fn berth(&self) -> BerthIndex {
        match *self {
            Self::ForceAssignment { berth, .. } => berth,
            Self::ForbidAssignment { berth, .. } => berth,
        }
    }
}

impl fmt::Display for BranchConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ForceAssignment { vessel, berth } => {
                write!(f, "Force(V{} -> B{})", vessel.get(), berth.get())
            }
            Self::ForbidAssignment { vessel, berth } => {
                write!(f, "Forbid(V{} -> B{})", vessel.get(), berth.get())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_negation_symmetry() {
        let v = VesselIndex::new(1);
        let b = BerthIndex::new(2);

        let force = BranchConstraint::ForceAssignment {
            vessel: v,
            berth: b,
        };
        let forbid = BranchConstraint::ForbidAssignment {
            vessel: v,
            berth: b,
        };

        assert_eq!(force.negate(), forbid);
        assert_eq!(forbid.negate(), force);
        assert_eq!(force.negate().negate(), force);
    }

    #[test]
    fn test_accessors() {
        let v = VesselIndex::new(5);
        let b = BerthIndex::new(3);
        let constraint = BranchConstraint::ForceAssignment {
            vessel: v,
            berth: b,
        };

        assert_eq!(constraint.vessel(), v);
        assert_eq!(constraint.berth(), b);
    }

    #[test]
    fn test_formatting() {
        let v = VesselIndex::new(0);
        let b = BerthIndex::new(1);
        let c = BranchConstraint::ForceAssignment {
            vessel: v,
            berth: b,
        };

        assert_eq!(format!("{}", c), "Force(V0 -> B1)");
    }
}
