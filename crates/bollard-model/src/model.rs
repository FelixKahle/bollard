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

use crate::{
    index::{BerthIndex, VesselIndex},
    time::ProcessingTime,
};
use bollard_core::{math::interval::ClosedOpenInterval, num::constants::MinusOne};
use num_traits::{PrimInt, Signed};

#[inline(always)]
fn flatten_index(num_berths: usize, vessel_index: VesselIndex, berth_index: BerthIndex) -> usize {
    vessel_index.get() * num_berths + berth_index.get()
}

/// A compact representation of vessel data, pre-sorted for chronological scanning.
#[derive(Clone, Copy, Debug)]
pub struct ChronologicalVesselData<T> {
    /// The original index of the vessel (to link back to the State).
    index: VesselIndex,
    /// The arrival time.
    arrival_time: T,
    /// The shortest possible processing time across ANY berth (pre-calculated).
    min_processing_time: T,
    /// The weight/priority of the vessel.
    weight: T,
}

impl<T> ChronologicalVesselData<T>
where
    T: PrimInt + Signed,
{
    /// Creates a new `ChronologicalVesselData` instance.
    pub fn new(index: VesselIndex, arrival_time: T, min_processing_time: T, weight: T) -> Self {
        Self {
            index,
            arrival_time,
            min_processing_time,
            weight,
        }
    }

    /// Returns the original index of the vessel.
    pub fn index(&self) -> VesselIndex {
        self.index
    }

    /// Returns the arrival time of the vessel.
    pub fn arrival_time(&self) -> T {
        self.arrival_time
    }

    /// Returns the shortest possible processing time across any berth.
    pub fn min_processing_time(&self) -> T {
        self.min_processing_time
    }

    /// Returns the weight/priority of the vessel.
    pub fn weight(&self) -> T {
        self.weight
    }
}

impl<T> std::fmt::Display for ChronologicalVesselData<T>
where
    T: PrimInt + Signed + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VesselIndex: {}, ArrivalTime: {}, MinProcessingTime: {}, Weight: {}",
            self.index.get(),
            self.arrival_time,
            self.min_processing_time,
            self.weight
        )
    }
}

/// The immutable data model describing vessels, berths, opening times, and processing times.
///
/// This struct holds all pre-validated, queryable data:
/// - `arrival_times[vessel]`: the arrival time for each vessel.
/// - `latest_departure_times[vessel]`: the latest allowed departure for each vessel.
/// - `processing_times[vessel * num_berths + berth]`: per-(vessel, berth) processing time,
///   encoded via `ProcessingTime<T>` (sentinel-based option).
/// - `opening_times[berth]`: a sorted, disjoint, non-empty list of `ClosedOpenInterval<T>`
///   describing when the berth is open.
/// - `shortest_processing_times[vessel]`: the minimum `Some(time)` across berths for a given vessel,
///   or `None` if no berth is available.
///
/// Construction:
/// - Use `ModelBuilder` and call `ModelBuilder::build` to obtain a validated `Model`.
#[derive(Clone)]
pub struct Model<T>
where
    T: PrimInt + Signed,
{
    arrival_times: Vec<T>,                                   // len = num_vessels
    latest_departure_times: Vec<T>,                          // len = num_vessels
    vessel_weights: Vec<T>,                                  // len = num_vessels
    processing_times: Vec<ProcessingTime<T>>,                // len = num_vessels * num_berths
    opening_times: Vec<Vec<ClosedOpenInterval<T>>>,          // len = num_berths.
    shortest_processing_times: Vec<ProcessingTime<T>>,       // len = num_vessels
    arrival_sorted_vessels: Vec<ChronologicalVesselData<T>>, // len = num_vessels
}

impl<T> Model<T>
where
    T: PrimInt + Signed + Copy,
{
    /// Returns the number of vessels in the model.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let builder = ModelBuilder::<i64>::new(3, 5);
    /// let model = builder.build();
    /// assert_eq!(model.num_vessels(), 5);
    /// ```
    #[inline]
    pub fn num_vessels(&self) -> usize {
        self.arrival_times.len()
    }

    /// Returns the number of berths in the model.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let builder = ModelBuilder::<i64>::new(4, 2);
    /// let model = builder.build();
    /// assert_eq!(model.num_berths(), 4);
    /// ```
    #[inline]
    pub fn num_berths(&self) -> usize {
        self.opening_times.len()
    }

    /// Returns a slice of all arrival times.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let builder = ModelBuilder::<i64>::new(2, 3);
    /// let model = builder.build();
    /// let arrival_times = model.vessel_arrival_times();
    /// assert_eq!(arrival_times.len(), 3);
    /// ```
    #[inline]
    pub fn vessel_arrival_times(&self) -> &[T] {
        &self.arrival_times
    }

    /// Returns a slice of all vessel weights.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let builder = ModelBuilder::<i64>::new(2, 4);
    /// let model = builder.build();
    /// let vessel_weights = model.vessel_weights();
    /// assert_eq!(vessel_weights.len(), 4);
    /// ```
    #[inline]
    pub fn vessel_weights(&self) -> &[T] {
        &self.vessel_weights
    }

    /// Returns a slice of all latest departure times.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let builder = ModelBuilder::<i64>::new(3, 2);
    /// let model = builder.build();
    /// let latest_departure_times = model.vessel_latest_departure_times();
    /// assert_eq!(latest_departure_times.len(), 2);
    /// ```
    #[inline]
    pub fn vessel_latest_departure_times(&self) -> &[T] {
        &self.latest_departure_times
    }

    /// Returns a slice of all processing times.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let builder = ModelBuilder::<i64>::new(2, 3);
    /// let model = builder.build();
    /// let processing_times = model.vessel_processing_times();
    /// assert_eq!(processing_times.len(), 6); // 2 berths * 3 vessels
    /// ```
    #[inline]
    pub fn vessel_processing_times(&self) -> &[ProcessingTime<T>] {
        &self.processing_times
    }

    /// Returns a slice of all opening times.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let builder = ModelBuilder::<i64>::new(4, 2);
    /// let model = builder.build();
    /// let opening_times = model.vessel_opening_times();
    /// assert_eq!(opening_times.len(), 4);
    /// ```
    #[inline]
    pub fn vessel_opening_times(&self) -> &[Vec<ClosedOpenInterval<T>>] {
        &self.opening_times
    }

    /// Returns a slice of all shortest processing times.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let builder = ModelBuilder::<i64>::new(2, 3);
    /// let model = builder.build();
    /// let shortest_processing_times = model.vessel_shortest_processing_times();
    /// assert_eq!(shortest_processing_times.len(), 3);
    /// ```
    #[inline]
    pub fn vessel_shortest_processing_times(&self) -> &[ProcessingTime<T>] {
        &self.shortest_processing_times
    }

    /// Returns the arrival time for the specified vessel.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is not in `0..num_vessels()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.set_vessel_arrival_time(bollard_model::index::VesselIndex::new(0), 10);
    /// builder.set_vessel_arrival_time(bollard_model::index::VesselIndex::new(1), 20);
    /// let model = builder.build();
    /// assert_eq!(model.vessel_arrival_time(bollard_model::index::VesselIndex::new(0)), 10);
    /// assert_eq!(model.vessel_arrival_time(bollard_model::index::VesselIndex::new(1)), 20);
    /// ```
    #[inline]
    pub fn vessel_arrival_time(&self, vessel_index: VesselIndex) -> T {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels(),
            "called `Model::vessel_arrival_time` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        self.arrival_times[index]
    }

    /// Returns the arrival time for the specified vessel without bounds checking.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not perform bounds checking on `vessel_index`.
    /// The caller must ensure that `vessel_index` is in `0..num_vessels()`. Undefined behavior
    /// may occur if this precondition is violated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.set_vessel_arrival_time(bollard_model::index::VesselIndex::new(0), 10);
    /// builder.set_vessel_arrival_time(bollard_model::index::VesselIndex::new(1), 20);
    /// let model = builder.build();
    /// unsafe {
    ///     assert_eq!(model.vessel_arrival_time_unchecked(bollard_model::index::VesselIndex::new(0)), 10);
    ///     assert_eq!(model.vessel_arrival_time_unchecked(bollard_model::index::VesselIndex::new(1)), 20);
    /// }
    /// ```
    #[inline]
    pub unsafe fn vessel_arrival_time_unchecked(&self, vessel_index: VesselIndex) -> T {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels(),
            "called `Model::vessel_arrival_time_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        unsafe { *self.arrival_times.get_unchecked(index) }
    }

    /// Returns the latest departure time for the specified vessel.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is not in `0..num_vessels()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.set_vessel_latest_departure_time(bollard_model::index::VesselIndex::new(0), 100);
    /// builder.set_vessel_latest_departure_time(bollard_model::index::VesselIndex::new(1), 200);
    /// let model = builder.build();
    /// assert_eq!(model.vessel_latest_departure_time(bollard_model::index::VesselIndex::new(0)), 100);
    /// assert_eq!(model.vessel_latest_departure_time(bollard_model::index::VesselIndex::new(1)), 200);
    /// ```
    #[inline]
    pub fn vessel_latest_departure_time(&self, vessel_index: VesselIndex) -> T {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels(),
            "called `Model::vessel_latest_departure_time` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        self.latest_departure_times[index]
    }

    /// Returns the latest departure time for the specified vessel without bounds checking.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not perform bounds checking on `vessel_index`.
    /// The caller must ensure that `vessel_index` is in `0..num_vessels()`. Undefined behavior
    /// may occur if this precondition is violated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.set_vessel_latest_departure_time(bollard_model::index::VesselIndex::new(0), 100);
    /// builder.set_vessel_latest_departure_time(bollard_model::index::VesselIndex::new(1), 200);
    /// let model = builder.build();
    /// unsafe {
    ///     assert_eq!(model.vessel_latest_departure_time_unchecked(bollard_model::index::VesselIndex::new(0)), 100);
    ///     assert_eq!(model.vessel_latest_departure_time_unchecked(bollard_model::index::VesselIndex::new(1)), 200);
    /// }
    #[inline]
    pub unsafe fn vessel_latest_departure_time_unchecked(&self, vessel_index: VesselIndex) -> T {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels(),
            "called `Model::vessel_latest_departure_time_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        unsafe { *self.latest_departure_times.get_unchecked(index) }
    }

    /// Returns the weight for the specified vessel.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is not in `0..num_vessels()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.set_vessel_weight(bollard_model::index::VesselIndex::new(0), 5);
    /// builder.set_vessel_weight(bollard_model::index::VesselIndex::new(1), 7);
    /// let model = builder.build();
    /// assert_eq!(model.vessel_weight(bollard_model::index::VesselIndex::new(0)), 5);
    /// assert_eq!(model.vessel_weight(bollard_model::index::VesselIndex::new(1)), 7);
    /// ```
    #[inline]
    pub fn vessel_weight(&self, vessel_index: VesselIndex) -> T {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels(),
            "called `Model::vessel_weight` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        self.vessel_weights[index]
    }

    /// Returns the weight for the specified vessel without bounds checking.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not perform bounds checking on `vessel_index`.
    /// The caller must ensure that `vessel_index` is in `0..num_vessels()`. Undefined behavior
    /// may occur if this precondition is violated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.set_vessel_weight(bollard_model::index::VesselIndex::new(0), 5);
    /// builder.set_vessel_weight(bollard_model::index::VesselIndex::new(1), 7);
    /// let model = builder.build();
    /// unsafe {
    ///     assert_eq!(model.vessel_weight_unchecked(bollard_model::index::VesselIndex::new(0)), 5);
    ///     assert_eq!(model.vessel_weight_unchecked(bollard_model::index::VesselIndex::new(1)), 7);
    /// }
    /// ```
    #[inline]
    pub unsafe fn vessel_weight_unchecked(&self, vessel_index: VesselIndex) -> T {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels(),
            "called `Model::vessel_weight_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        unsafe { *self.vessel_weights.get_unchecked(index) }
    }

    /// Returns the processing time for the specified (vessel, berth) pair.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is not in `0..num_vessels()` or
    /// if `berth_index` is not in `0..num_berths()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.set_vessel_processing_time(
    ///     bollard_model::index::VesselIndex::new(0),
    ///     bollard_model::index::BerthIndex::new(0),
    ///     bollard_model::time::ProcessingTime::from_option(Some(50)),
    /// );
    /// builder.set_vessel_processing_time(
    ///     bollard_model::index::VesselIndex::new(0),
    ///     bollard_model::index::BerthIndex::new(1),
    ///     bollard_model::time::ProcessingTime::none(),
    /// );
    /// let model = builder.build();
    /// assert_eq!(
    ///     model.vessel_processing_time(
    ///         bollard_model::index::VesselIndex::new(0),
    ///         bollard_model::index::BerthIndex::new(0)
    ///     ),
    ///     bollard_model::time::ProcessingTime::from_option(Some(50))
    /// );
    /// assert_eq!(
    ///     model.vessel_processing_time(
    ///         bollard_model::index::VesselIndex::new(0),
    ///         bollard_model::index::BerthIndex::new(1)
    ///     ),
    ///     bollard_model::time::ProcessingTime::none()
    /// );
    /// ```
    #[inline]
    pub fn vessel_processing_time(
        &self,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
    ) -> ProcessingTime<T> {
        debug_assert!(
            vessel_index.get() < self.num_vessels(),
            "called `Model::vessel_processing_time` with vessel index out of bounds: the len is {} but the index is {}",
            vessel_index.get(),
            self.num_vessels()
        );

        debug_assert!(
            berth_index.get() < self.num_berths(),
            "called `Model::vessel_processing_time` with berth index out of bounds: the len is {} but the index is {}",
            berth_index.get(),
            self.num_berths()
        );

        let flat_index = flatten_index(self.num_berths(), vessel_index, berth_index);
        debug_assert!(
            flat_index < self.processing_times.len(),
            "called `Model::vessel_processing_time_unchecked` with flat index out of bounds: the len is {} but the index is {}",
            self.processing_times.len(),
            flat_index
        );

        self.processing_times[flat_index]
    }

    /// Returns the processing time for the specified (vessel, berth) pair without bounds checking.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not perform bounds checking on `vessel_index` and `berth_index`.
    /// The caller must ensure that `vessel_index` is in `0..num_vessels()` and
    /// `berth_index` is in `0..num_berths()`. Undefined behavior
    /// may occur if this precondition is violated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.set_vessel_processing_time(
    ///     bollard_model::index::VesselIndex::new(0),
    ///     bollard_model::index::BerthIndex::new(0),
    ///     bollard_model::time::ProcessingTime::from_option(Some(50)),
    /// );
    /// builder.set_vessel_processing_time(
    ///     bollard_model::index::VesselIndex::new(0),
    ///     bollard_model::index::BerthIndex::new(1),
    ///     bollard_model::time::ProcessingTime::none(),
    /// );
    /// let model = builder.build();
    /// unsafe {
    ///     assert_eq!(
    ///         model.vessel_processing_time_unchecked(
    ///             bollard_model::index::VesselIndex::new(0),
    ///             bollard_model::index::BerthIndex::new(0)
    ///         ),
    ///         bollard_model::time::ProcessingTime::from_option(Some(50))
    ///     );
    ///     assert_eq!(
    ///         model.vessel_processing_time_unchecked(
    ///             bollard_model::index::VesselIndex::new(0),
    ///             bollard_model::index::BerthIndex::new(1)
    ///         ),
    ///         bollard_model::time::ProcessingTime::none()
    ///     );
    /// }
    /// ```
    #[inline]
    pub unsafe fn vessel_processing_time_unchecked(
        &self,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
    ) -> ProcessingTime<T> {
        debug_assert!(
            vessel_index.get() < self.num_vessels(),
            "called `Model::vessel_processing_time_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            vessel_index.get(),
            self.num_vessels()
        );

        debug_assert!(
            berth_index.get() < self.num_berths(),
            "called `Model::vessel_processing_time_unchecked` with berth index out of bounds: the len is {} but the index is {}",
            berth_index.get(),
            self.num_berths()
        );

        let flat_index = flatten_index(self.num_berths(), vessel_index, berth_index);
        debug_assert!(
            flat_index < self.processing_times.len(),
            "called `Model::vessel_processing_time_unchecked` with flat index out of bounds: the len is {} but the index is {}",
            self.processing_times.len(),
            flat_index
        );

        unsafe { *self.processing_times.get_unchecked(flat_index) }
    }

    /// Returns `true` if the specified vessel is allowed to dock at the specified berth.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is not in `0..num_vessels()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.set_vessel_processing_time(
    ///     bollard_model::index::VesselIndex::new(0),
    ///     bollard_model::index::BerthIndex::new(0),
    ///     bollard_model::time::ProcessingTime::from_option(Some(50)),
    /// );
    /// builder.set_vessel_processing_time(
    ///     bollard_model::index::VesselIndex::new(0),
    ///     bollard_model::index::BerthIndex::new(1),
    ///     bollard_model::time::ProcessingTime::none(),
    /// );
    /// let model = builder.build();
    /// assert!(model.vessel_allowed_on_berth(
    ///     bollard_model::index::VesselIndex::new(0),
    ///     bollard_model::index::BerthIndex::new(0)
    /// ));
    /// assert!(!model.vessel_allowed_on_berth(
    ///     bollard_model::index::VesselIndex::new(0),
    ///     bollard_model::index::BerthIndex::new(1)
    /// ));
    /// ```
    #[inline]
    pub fn vessel_allowed_on_berth(
        &self,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
    ) -> bool
    where
        T: MinusOne,
    {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels(),
            "called `Model::vessel_allowed_on_berth` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        self.vessel_processing_time(vessel_index, berth_index)
            .is_some()
    }

    /// Returns `true` if the specified vessel is allowed to dock at the specified berth without bounds checking.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not perform bounds checking on `vessel_index`.
    /// The caller must ensure that `vessel_index` is in `0..num_vessels()`. Undefined behavior
    /// may occur if this precondition is violated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.set_vessel_processing_time(
    ///     bollard_model::index::VesselIndex::new(0),
    ///     bollard_model::index::BerthIndex::new(0),
    ///     bollard_model::time::ProcessingTime::from_option(Some(50)),
    /// );
    /// builder.set_vessel_processing_time(
    ///     bollard_model::index::VesselIndex::new(0),
    ///     bollard_model::index::BerthIndex::new(1),
    ///     bollard_model::time::ProcessingTime::none(),
    /// );
    /// let model = builder.build();
    /// unsafe {
    ///     assert!(model.vessel_allowed_on_berth_unchecked(
    ///         bollard_model::index::VesselIndex::new(0),
    ///         bollard_model::index::BerthIndex::new(0)
    ///     ));
    ///     assert!(!model.vessel_allowed_on_berth_unchecked(
    ///         bollard_model::index::VesselIndex::new(0),
    ///         bollard_model::index::BerthIndex::new(1)
    ///     ));
    /// }
    /// ```
    #[inline]
    pub unsafe fn vessel_allowed_on_berth_unchecked(
        &self,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
    ) -> bool
    where
        T: MinusOne,
    {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels(),
            "called `Model::vessel_allowed_on_berth_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        unsafe { self.vessel_processing_time_unchecked(vessel_index, berth_index) }.is_some()
    }

    /// Returns a slice of all vessels sorted by arrival time.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.set_vessel_arrival_time(bollard_model::index::VesselIndex::new(0), 20);
    /// builder.set_vessel_arrival_time(bollard_model::index::VesselIndex::new(1), 10);
    /// // Ensure both vessels have processing times so they are included in the sorted list
    /// builder
    ///     .set_vessel_processing_time(
    ///         bollard_model::index::VesselIndex::new(0),
    ///         bollard_model::index::BerthIndex::new(0),
    ///         bollard_model::time::ProcessingTime::some(5),
    ///     )
    ///     .set_vessel_processing_time(
    ///         bollard_model::index::VesselIndex::new(1),
    ///         bollard_model::index::BerthIndex::new(0),
    ///         bollard_model::time::ProcessingTime::some(3),
    ///     );
    /// let model = builder.build();
    /// let sorted_vessels = model.vessel_arrival_sorted_vessels();
    /// assert_eq!(sorted_vessels[0].index().get(), 1);
    /// assert_eq!(sorted_vessels[1].index().get(), 0);
    /// ```
    #[inline]
    pub fn vessel_arrival_sorted_vessels(&self) -> &[ChronologicalVesselData<T>] {
        &self.arrival_sorted_vessels
    }

    /// Panics if `sorted_index` is not in `0..num_vessels()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.set_vessel_arrival_time(bollard_model::index::VesselIndex::new(0), 20);
    /// builder.set_vessel_arrival_time(bollard_model::index::VesselIndex::new(1), 10);
    /// // Ensure vessels are included in the sorted list
    /// builder
    ///     .set_vessel_processing_time(
    ///         bollard_model::index::VesselIndex::new(0),
    ///         bollard_model::index::BerthIndex::new(0),
    ///         bollard_model::time::ProcessingTime::some(5),
    ///     )
    ///     .set_vessel_processing_time(
    ///         bollard_model::index::VesselIndex::new(1),
    ///         bollard_model::index::BerthIndex::new(0),
    ///         bollard_model::time::ProcessingTime::some(3),
    ///     );
    /// let model = builder.build();
    ///
    /// assert_eq!(model.vessel_arrival_time_sorted_vessel(0), 10);
    /// assert_eq!(model.vessel_arrival_time_sorted_vessel(1), 20);
    /// ```
    #[inline]
    pub fn vessel_arrival_time_sorted_vessel(&self, sorted_index: usize) -> T {
        debug_assert!(
            sorted_index < self.arrival_sorted_vessels.len(),
            "called `Model::vessel_arrival_time_sorted_vessel` with sorted index out of bounds: the len is {} but the index is {}",
            sorted_index,
            self.arrival_sorted_vessels.len()
        );

        self.arrival_sorted_vessels[sorted_index].arrival_time()
    }

    /// # Examples
    ///
    /// # Panics
    ///
    /// In debug builds, this function will panic if `sorted_index` is not in `0..num_vessels()`.
    /// In release builds, no bounds checking is performed.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not perform bounds checking on `sorted_index`.
    /// The caller must ensure that `sorted_index` is in `0..num_vessels()`. Undefined behavior
    /// may occur if this precondition is violated.
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.set_vessel_arrival_time(bollard_model::index::VesselIndex::new(0), 20);
    /// builder.set_vessel_arrival_time(bollard_model::index::VesselIndex::new(1), 10);
    /// // Ensure vessels are included in the sorted list
    /// builder
    ///     .set_vessel_processing_time(
    ///         bollard_model::index::VesselIndex::new(0),
    ///         bollard_model::index::BerthIndex::new(0),
    ///         bollard_model::time::ProcessingTime::some(5),
    ///     )
    ///     .set_vessel_processing_time(
    ///         bollard_model::index::VesselIndex::new(1),
    ///         bollard_model::index::BerthIndex::new(0),
    ///         bollard_model::time::ProcessingTime::some(3),
    ///     );
    /// let model = builder.build();
    ///
    /// unsafe {
    ///     assert_eq!(model.vessel_arrival_time_sorted_vessel_unchecked(0), 10);
    ///     assert_eq!(model.vessel_arrival_time_sorted_vessel_unchecked(1), 20);
    /// }
    /// ```
    #[inline]
    pub unsafe fn vessel_arrival_time_sorted_vessel_unchecked(&self, sorted_index: usize) -> T {
        debug_assert!(
            sorted_index < self.arrival_sorted_vessels.len(),
            "called `Model::vessel_arrival_time_sorted_vessel_unchecked` with sorted index out of bounds: the len is {} but the index is {}",
            sorted_index,
            self.arrival_sorted_vessels.len()
        );

        unsafe { self.arrival_sorted_vessels.get_unchecked(sorted_index) }.arrival_time()
    }

    /// Returns the opening times for the specified berth.
    ///
    /// # Panics
    ///
    /// Panics if `berth_index` is not in `0..num_berths()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.add_berth_closing_time(
    ///     bollard_model::index::BerthIndex::new(0),
    ///     bollard_core::math::interval::ClosedOpenInterval::new(50, 100),
    /// );
    /// let model = builder.build();
    /// let opening_times = model.berth_opening_times(bollard_model::index::BerthIndex::new(0));
    /// assert_eq!(
    ///     opening_times,
    ///     &[bollard_core::math::interval::ClosedOpenInterval::new(0, 50), bollard_core::math::interval::ClosedOpenInterval::new(100, i64::MAX)]
    /// );
    /// ```
    #[inline]
    pub fn berth_opening_times(&self, berth_index: BerthIndex) -> &[ClosedOpenInterval<T>] {
        let index = berth_index.get();
        debug_assert!(
            index < self.num_berths(),
            "called `Model::berth_opening_times` with berth index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        &self.opening_times[index]
    }

    /// Returns the opening times for the specified berth without bounds checking.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not perform bounds checking on `berth_index`.
    /// The caller must ensure that `berth_index` is in `0..num_berths()`. Undefined behavior
    /// may occur if this precondition is violated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.add_berth_closing_time(
    ///     bollard_model::index::BerthIndex::new(0),
    ///     bollard_core::math::interval::ClosedOpenInterval::new(50, 100),
    /// );
    /// let model = builder.build();
    /// let opening_times = unsafe { model.berth_opening_times_unchecked(bollard_model::index::BerthIndex::new(0)) };
    /// assert_eq!(
    ///     opening_times,
    ///     &[bollard_core::math::interval::ClosedOpenInterval::new(0, 50), bollard_core::math::interval::ClosedOpenInterval::new(100, i64::MAX)]
    /// );
    /// ```
    #[inline]
    pub unsafe fn berth_opening_times_unchecked(
        &self,
        berth_index: BerthIndex,
    ) -> &[ClosedOpenInterval<T>] {
        let index = berth_index.get();
        debug_assert!(
            index < self.num_berths(),
            "called `Model::berth_opening_times_unchecked` with berth index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        unsafe { self.opening_times.get_unchecked(index) }
    }

    /// Returns the shortest processing time for the specified vessel.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is not in `0..num_vessels()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.set_vessel_processing_time(
    ///     bollard_model::index::VesselIndex::new(0),
    ///     bollard_model::index::BerthIndex::new(0),
    ///     bollard_model::time::ProcessingTime::from_option(Some(50)),
    /// );
    /// builder.set_vessel_processing_time(
    ///     bollard_model::index::VesselIndex::new(0),
    ///     bollard_model::index::BerthIndex::new(1),
    ///     bollard_model::time::ProcessingTime::from_option(Some(30)),
    /// );
    ///
    /// let model = builder.build();
    /// assert_eq!(
    ///     model.vessel_shortest_processing_time(bollard_model::index::VesselIndex::new(0)),
    ///     bollard_model::time::ProcessingTime::from_option(Some(30))
    /// );
    /// ```
    #[inline]
    pub fn vessel_shortest_processing_time(&self, vessel_index: VesselIndex) -> ProcessingTime<T> {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels(),
            "called `Model::vessel_shortest_processing_time` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        self.shortest_processing_times[index]
    }

    /// Returns the shortest processing time for the specified vessel without bounds checking.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not perform bounds checking on `vessel_index`.
    /// The caller must ensure that `vessel_index` is in `0..num_vessels()`. Undefined behavior
    /// may occur if this precondition is violated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.set_vessel_processing_time(
    ///     bollard_model::index::VesselIndex::new(0),
    ///     bollard_model::index::BerthIndex::new(0),
    ///     bollard_model::time::ProcessingTime::from_option(Some(50)),
    /// );
    /// builder.set_vessel_processing_time(
    ///     bollard_model::index::VesselIndex::new(0),
    ///     bollard_model::index::BerthIndex::new(1),
    ///     bollard_model::time::ProcessingTime::from_option(Some(30)),
    /// );
    ///
    /// let model = builder.build();
    /// assert_eq!(
    ///     unsafe { model.vessel_shortest_processing_time_unchecked(bollard_model::index::VesselIndex::new(0)) },
    ///     bollard_model::time::ProcessingTime::from_option(Some(30))
    /// );
    /// ```
    #[inline]
    pub unsafe fn vessel_shortest_processing_time_unchecked(
        &self,
        vessel_index: VesselIndex,
    ) -> ProcessingTime<T> {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels(),
            "called `Model::vessel_shortest_processing_time_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        unsafe { *self.shortest_processing_times.get_unchecked(index) }
    }
}

impl<T> std::fmt::Debug for Model<T>
where
    T: PrimInt + Signed + Copy + MinusOne + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("arrival_times", &self.arrival_times)
            .field("latest_departure_times", &self.latest_departure_times)
            .field("vessel_weights", &self.vessel_weights)
            .field("processing_times", &self.processing_times)
            .field("opening_times", &self.opening_times)
            .field("shortest_processing_times", &self.shortest_processing_times)
            .finish()
    }
}

impl<T> std::fmt::Display for Model<T>
where
    T: PrimInt + Signed + Copy + MinusOne + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Model(num_vessels: {}, num_berths: {})",
            self.num_vessels(),
            self.num_berths()
        )
    }
}

#[derive(Clone)]
pub struct ModelBuilder<T>
where
    T: PrimInt + Signed,
{
    num_berths: usize,
    num_vessels: usize,
    opening_times: Vec<rangemap::RangeSet<T>>,
    arrival_times: Vec<T>,
    latest_departure_times: Vec<T>,
    vessel_weights: Vec<T>,
    processing_times: Vec<ProcessingTime<T>>,
}

#[inline(always)]
fn unconstrained_berth<T: PrimInt + Copy>() -> rangemap::RangeSet<T> {
    let mut r = rangemap::RangeSet::new();
    r.insert(T::zero()..T::max_value());
    r
}

impl<T> ModelBuilder<T>
where
    T: PrimInt + Signed + MinusOne,
{
    /// Creates a new `ModelBuilder` initialized with **permissive bounds** and an **empty topology**.
    ///
    /// This constructor follows the standard "Solver Philosophy" where nothing is assumed to
    /// exist until defined:
    ///
    /// 1.  **Permissive Bounds:** Time windows (`0` to `MAX`) and availability (`Open 24/7`) are relaxed
    ///     to their widest possible values. Constraints are added by *reducing* these bounds.
    /// 2.  **Empty Topology:** The graph starts disconnected. Connections between Vessels and Berths
    ///     (Processing Times) do not exist (`None`).
    ///
    /// # Note on Feasibility
    ///
    /// **The model is INFEASIBLE by default.**
    /// Because the topology is empty (processing times are `None`), vessels cannot dock anywhere.
    /// You must explicitly add valid connections via `set_vessel_processing_time` to make the problem solvable.
    /// This design choice prevents "Ghost Edges" (impossible connections being used silently).
    ///
    /// # Defaults
    ///
    /// The builder initializes with the following state:
    ///
    /// | Domain | Field | Default Value | Semantics |
    /// | :--- | :--- | :--- | :--- |
    /// | **Resource** | `opening_times` | `[0, T::MAX)` | Berths are **Open 24/7**. You must explicitly *remove* time for closing. |
    /// | **Time** | `arrival_times` | `0` | Vessels are ready at the very start of the planning horizon. |
    /// | **Time** | `latest_departure` | `T::MAX` | Vessels have **no deadline** (infinite time). |
    /// | **Objective**| `vessel_weights` | `1` | All vessels have standard, equal priority. |
    /// | **Graph** | `processing_times` | `None` | **Disconnected.** The vessel *cannot* dock. You must explicitly allow connections. |
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let builder = ModelBuilder::<i64>::new(3, 5);
    /// let model = builder.build();
    /// assert_eq!(model.num_berths(), 3);
    /// assert_eq!(model.num_vessels(), 5);
    /// ```
    pub fn new(num_berths: usize, num_vessels: usize) -> Self {
        ModelBuilder {
            num_berths,
            num_vessels,
            opening_times: vec![unconstrained_berth(); num_berths],
            arrival_times: vec![T::zero(); num_vessels],
            latest_departure_times: vec![T::max_value(); num_vessels],
            vessel_weights: vec![T::one(); num_vessels],
            processing_times: vec![ProcessingTime::none(); num_vessels * num_berths],
        }
    }

    /// Returns the number of berths in the model.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let builder = ModelBuilder::<i64>::new(4, 2);
    /// assert_eq!(builder.num_berths(), 4);
    /// ```
    #[inline]
    pub fn num_berths(&self) -> usize {
        self.num_berths
    }

    /// Returns the number of vessels in the model.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let builder = ModelBuilder::<i64>::new(4, 2);
    /// assert_eq!(builder.num_vessels(), 2);
    /// ```
    #[inline]
    pub fn num_vessels(&self) -> usize {
        self.num_vessels
    }

    /// Adds a closing time interval to the specified berth.
    ///
    /// # Panics
    ///
    /// Panics if `berth_index` is not in `0..num_berths()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.add_berth_closing_time(
    ///     bollard_model::index::BerthIndex::new(0),
    ///     bollard_core::math::interval::ClosedOpenInterval::new(50, 100),
    /// );
    /// let model = builder.build();
    /// let opening_times = model.berth_opening_times(bollard_model::index::BerthIndex::new(0));
    /// assert_eq!(
    ///     opening_times,
    ///     &[bollard_core::math::interval::ClosedOpenInterval::new(0, 50), bollard_core::math::interval::ClosedOpenInterval::new(100, i64::MAX)]
    /// );
    /// ```
    #[inline]
    pub fn add_berth_closing_time(
        &mut self,
        berth_index: BerthIndex,
        closing_interval: ClosedOpenInterval<T>,
    ) -> &mut Self {
        let index = berth_index.get();
        debug_assert!(
            index < self.num_berths(),
            "called `ModelBuilder::add_berth_closing_time` with berth index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        self.opening_times[index].remove(closing_interval.into());
        self
    }

    /// Adds multiple closing time intervals to the specified berth.
    ///
    /// # Panics
    ///
    /// Panics if `berth_index` is not in `0..num_berths()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.add_berth_closing_times(
    ///     bollard_model::index::BerthIndex::new(0),
    ///     vec![
    ///         bollard_core::math::interval::ClosedOpenInterval::new(50, 100),
    ///         bollard_core::math::interval::ClosedOpenInterval::new(150, 200),
    ///     ],
    /// );
    /// let model = builder.build();
    /// let opening_times = model.berth_opening_times(bollard_model::index::BerthIndex::new(0));
    /// assert_eq!(
    ///     opening_times,
    ///     &[
    ///         bollard_core::math::interval::ClosedOpenInterval::new(0, 50),
    ///         bollard_core::math::interval::ClosedOpenInterval::new(100, 150),
    ///         bollard_core::math::interval::ClosedOpenInterval::new(200, i64::MAX)
    ///     ]
    /// );
    /// ```
    pub fn add_berth_closing_times<I>(
        &mut self,
        berth_index: BerthIndex,
        closing_intervals: I,
    ) -> &mut Self
    where
        I: IntoIterator<Item = ClosedOpenInterval<T>>,
    {
        let index = berth_index.get();
        debug_assert!(
            index < self.num_berths(),
            "called `ModelBuilder::add_berth_closing_times` with berth index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        for interval in closing_intervals {
            self.opening_times[index].remove(interval.into());
        }
        self
    }

    /// Adds an opening time interval to the specified berth.
    ///
    /// # Panics
    ///
    /// Panics if `berth_index` is not in `0..num_berths()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.add_berth_opening_time(
    ///     bollard_model::index::BerthIndex::new(0),
    ///     bollard_core::math::interval::ClosedOpenInterval::new(50, 100),
    /// );
    /// let model = builder.build();
    /// let opening_times = model.berth_opening_times(bollard_model::index::BerthIndex::new(0));
    /// assert_eq!(
    ///     opening_times,
    ///     &[bollard_core::math::interval::ClosedOpenInterval::new(0, i64::MAX)]
    /// );
    /// ```
    #[inline]
    pub fn add_berth_opening_time(
        &mut self,
        berth_index: BerthIndex,
        opening_interval: ClosedOpenInterval<T>,
    ) -> &mut Self {
        let index = berth_index.get();
        debug_assert!(
            index < self.num_berths(),
            "called `ModelBuilder::add_berth_opening_time` with berth index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        self.opening_times[index].insert(opening_interval.into());
        self
    }

    /// Adds multiple opening time intervals to the specified berth.
    ///
    /// # Panics
    ///
    /// Panics if `berth_index` is not in `0..num_berths()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.add_berth_opening_times(
    ///     bollard_model::index::BerthIndex::new(0),
    ///     vec!
    ///     [
    ///         bollard_core::math::interval::ClosedOpenInterval::new(50, 100),
    ///         bollard_core::math::interval::ClosedOpenInterval::new(150, 200),
    ///     ],
    /// );
    /// let model = builder.build();
    ///
    /// let opening_times = model.berth_opening_times(bollard_model::index::BerthIndex::new(0));
    /// assert_eq!(
    ///     opening_times,
    ///     &[
    ///         bollard_core::math::interval::ClosedOpenInterval::new(0, i64::MAX)
    ///     ]
    /// );
    pub fn add_berth_opening_times<I>(
        &mut self,
        berth_index: BerthIndex,
        opening_intervals: I,
    ) -> &mut Self
    where
        I: IntoIterator<Item = ClosedOpenInterval<T>>,
    {
        let index = berth_index.get();
        debug_assert!(
            index < self.num_berths(),
            "called `ModelBuilder::add_berth_opening_times` with berth index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        for interval in opening_intervals {
            self.opening_times[index].insert(interval.into());
        }
        self
    }

    /// Sets the arrival time for the specified vessel.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is not in `0..num_vessels()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.set_vessel_arrival_time(bollard_model::index::VesselIndex::new(0), 100);
    /// let model = builder.build();
    /// assert_eq!(model.vessel_arrival_time(bollard_model::index::VesselIndex::new(0)), 100);
    /// ```
    #[inline]
    pub fn set_vessel_arrival_time(
        &mut self,
        vessel_index: VesselIndex,
        vessel_arrival_time: T,
    ) -> &mut Self {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels(),
            "called `ModelBuilder::set_vessel_arrival_time` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        self.arrival_times[index] = vessel_arrival_time;
        self
    }

    /// Sets the latest departure time for the specified vessel.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is not in `0..num_vessels()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.set_vessel_latest_departure_time(bollard_model::index::VesselIndex::new(0), 500);
    /// let model = builder.build();
    /// assert_eq!(model.vessel_latest_departure_time(bollard_model::index::VesselIndex::new(0)), 500);
    /// ```
    #[inline]
    pub fn set_vessel_latest_departure_time(
        &mut self,
        vessel_index: VesselIndex,
        latest_departure_time: T,
    ) -> &mut Self {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels(),
            "called `ModelBuilder::set_vessel_latest_departure_time` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        self.latest_departure_times[index] = latest_departure_time;
        self
    }

    /// Sets the weight for the specified vessel.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is not in `0..num_vessels()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.set_vessel_weight(bollard_model::index::VesselIndex::new(0), 10);
    /// let model = builder.build();
    /// assert_eq!(model.vessel_weight(bollard_model::index::VesselIndex::new(0)), 10);
    /// ```
    #[inline]
    pub fn set_vessel_weight(&mut self, vessel_index: VesselIndex, weight: T) -> &mut Self {
        let index = vessel_index.get();
        debug_assert!(
            index < self.num_vessels(),
            "called `ModelBuilder::set_vessel_weight` with vessel index out of bounds: the len is {} but the index is {}",
            index,
            self.num_vessels()
        );

        self.vessel_weights[index] = weight;
        self
    }

    /// Sets the processing time for the specified (vessel, berth) pair.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is not in `0..num_vessels()` or
    /// if `berth_index` is not in `0..num_berths()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// builder.set_vessel_processing_time(
    ///     bollard_model::index::VesselIndex::new(0),
    ///     bollard_model::index::BerthIndex::new(0),
    ///     bollard_model::time::ProcessingTime::from_option(Some(50)),
    /// );
    /// let model = builder.build();
    ///
    /// assert_eq!(
    ///     model.vessel_processing_time(
    ///         bollard_model::index::VesselIndex::new(0),
    ///         bollard_model::index::BerthIndex::new(0)
    ///     ),
    ///     bollard_model::time::ProcessingTime::from_option(Some(50))
    /// );
    /// ```
    #[inline]
    pub fn set_vessel_processing_time(
        &mut self,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        processing_time: ProcessingTime<T>,
    ) -> &mut Self {
        debug_assert!(
            vessel_index.get() < self.num_vessels(),
            "called `ModelBuilder::set_vessel_processing_time` with vessel index out of bounds: the len is {} but the index is {}",
            vessel_index.get(),
            self.num_vessels()
        );

        debug_assert!(
            berth_index.get() < self.num_berths(),
            "called `ModelBuilder::vessel_processing_time_unchecked` with berth index out of bounds: the len is {} but the index is {}",
            berth_index.get(),
            self.num_berths()
        );

        let flat_index = flatten_index(self.num_berths, vessel_index, berth_index);
        debug_assert!(
            flat_index < self.processing_times.len(),
            "called `Model::vessel_processing_time_unchecked` with flat index out of bounds: the len is {} but the index is {}",
            self.processing_times.len(),
            flat_index
        );

        self.processing_times[flat_index] = processing_time;
        self
    }

    /// Builds the `Model` from the current state of the `ModelBuilder`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::model::ModelBuilder;
    ///
    /// let mut builder = ModelBuilder::<i64>::new(2, 2);
    /// let model = builder.build();
    /// assert_eq!(model.num_berths(), 2);
    /// assert_eq!(model.num_vessels(), 2);
    /// ```
    pub fn build(self) -> Model<T> {
        let opening_times_model: Vec<Vec<ClosedOpenInterval<T>>> = self
            .opening_times
            .into_iter()
            .map(|range_set| range_set.into_iter().map(|r| r.into()).collect())
            .collect();

        let shortest_processing_times: Vec<ProcessingTime<T>> =
            match (self.num_berths, self.num_vessels) {
                (0, 0) => Vec::new(),
                (0, nv) => std::iter::repeat_with(ProcessingTime::none)
                    .take(nv)
                    .collect(),
                (_, 0) => Vec::new(),
                (nb, _nv) => self
                    .processing_times
                    .chunks_exact(nb)
                    .map(|vessel_chunk| {
                        let min_time = vessel_chunk
                            .iter()
                            .copied()
                            .filter_map(|pt| pt.into())
                            .min();
                        ProcessingTime::from_option(min_time)
                    })
                    .collect(),
            };

        let mut chronological_vessels: Vec<_> = (0..self.num_vessels)
            .filter_map(|i| {
                shortest_processing_times[i]
                    .into_option()
                    .map(|min_processing_time| ChronologicalVesselData {
                        index: VesselIndex::new(i),
                        arrival_time: self.arrival_times[i],
                        min_processing_time,
                        weight: self.vessel_weights[i],
                    })
            })
            .collect();
        chronological_vessels.sort_unstable_by_key(|v| (v.arrival_time, v.min_processing_time));

        Model {
            arrival_times: self.arrival_times,
            latest_departure_times: self.latest_departure_times,
            vessel_weights: self.vessel_weights,
            processing_times: self.processing_times,
            opening_times: opening_times_model,
            shortest_processing_times,
            arrival_sorted_vessels: chronological_vessels,
        }
    }
}

impl<T> std::fmt::Debug for ModelBuilder<T>
where
    T: PrimInt + Signed + Copy + MinusOne + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelBuilder")
            .field("num_berths", &self.num_berths)
            .field("num_vessels", &self.num_vessels)
            .field("opening_times", &self.opening_times)
            .field("arrival_times", &self.arrival_times)
            .field("latest_departure_times", &self.latest_departure_times)
            .field("vessel_weights", &self.vessel_weights)
            .field("processing_times", &self.processing_times)
            .finish()
    }
}

impl<T> std::fmt::Display for ModelBuilder<T>
where
    T: PrimInt + Signed + Copy + MinusOne,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ModelBuilder(num_vessels: {}, num_berths: {})",
            self.num_vessels, self.num_berths
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bollard_core::math::interval::ClosedOpenInterval;

    fn v(i: usize) -> VesselIndex {
        VesselIndex::new(i)
    }
    fn b(i: usize) -> BerthIndex {
        BerthIndex::new(i)
    }
    fn opt(pt: ProcessingTime<i64>) -> Option<i64> {
        Option::<i64>::from(pt)
    }

    #[test]
    fn test_flatten_index_basic() {
        assert_eq!(flatten_index(5, v(0), b(0)), 0);
        assert_eq!(flatten_index(5, v(0), b(4)), 4);
    }

    #[test]
    fn test_flatten_index_multiple() {
        assert_eq!(flatten_index(5, v(1), b(0)), 5);
        assert_eq!(flatten_index(5, v(3), b(2)), 17);
    }

    #[test]
    fn test_unconstrained_berth_contains_expected() {
        let r = unconstrained_berth::<i64>();
        assert!(r.contains(&0));
        assert!(r.contains(&(i64::MAX - 1)));
    }

    #[test]
    fn test_unconstrained_berth_excludes_expected() {
        let r = unconstrained_berth::<i64>();
        assert!(!r.contains(&i64::MAX)); // half-open boundary
        assert!(!r.contains(&-1));
    }

    #[test]
    fn test_builder_defaults_dimensions() {
        let bldr = ModelBuilder::<i64>::new(3, 2);
        assert_eq!(bldr.num_berths, 3);
        assert_eq!(bldr.num_vessels, 2);
        assert_eq!(bldr.opening_times.len(), 3);
        assert_eq!(bldr.arrival_times.len(), 2);
        assert_eq!(bldr.latest_departure_times.len(), 2);
        assert_eq!(bldr.vessel_weights.len(), 2);
        assert_eq!(bldr.processing_times.len(), 6);
    }

    #[test]
    fn test_builder_defaults_values() {
        let bldr = ModelBuilder::<i64>::new(2, 2);
        assert!(bldr.arrival_times.iter().all(|&t| t == 0));
        assert!(bldr.latest_departure_times.iter().all(|&t| t == i64::MAX));
        assert!(bldr.vessel_weights.iter().all(|&t| t == 1));
        assert!(bldr.processing_times.iter().all(|pt| pt.is_none()));
    }

    #[test]
    fn test_builder_defaults_opening_times_membership() {
        let bldr = ModelBuilder::<i64>::new(2, 1);
        for rs in &bldr.opening_times {
            assert!(rs.contains(&0));
            assert!(rs.contains(&(i64::MAX - 1)));
            assert!(!rs.contains(&i64::MAX));
            assert!(!rs.contains(&-1));
        }
    }

    #[test]
    fn test_opening_remove_interval_membership() {
        let mut bldr = ModelBuilder::<i64>::new(2, 1);
        bldr.add_berth_closing_time(b(0), ClosedOpenInterval::new(10, 20));
        let m = bldr.build();
        let opening = m.berth_opening_times(b(0));
        let contains = |x| opening.iter().any(|i| i.contains_point(x));
        assert!(contains(9));
        assert!(!contains(10));
        assert!(!contains(15));
        assert!(contains(20));
    }

    #[test]
    fn test_opening_reopen_interval_membership() {
        let mut bldr = ModelBuilder::<i64>::new(2, 1);
        bldr.add_berth_closing_time(b(0), ClosedOpenInterval::new(10, 20));
        bldr.add_berth_opening_time(b(0), ClosedOpenInterval::new(15, 18));
        let m = bldr.build();
        let opening = m.berth_opening_times(b(0));
        let contains = |x| opening.iter().any(|i| i.contains_point(x));
        assert!(contains(15));
        assert!(contains(17));
        assert!(!contains(18)); // half-open boundary
    }

    #[test]
    fn test_opening_multiple_mutations_membership() {
        let mut bldr = ModelBuilder::<i64>::new(1, 1);
        bldr.add_berth_closing_times(
            b(0),
            [
                ClosedOpenInterval::new(100, 200),
                ClosedOpenInterval::new(300, 400),
            ],
        );
        bldr.add_berth_opening_times(
            b(0),
            [
                ClosedOpenInterval::new(150, 160),
                ClosedOpenInterval::new(350, 360),
            ],
        );
        let m = bldr.build();
        let opening = m.berth_opening_times(b(0));
        let contains = |x| opening.iter().any(|i| i.contains_point(x));
        assert!(contains(150));
        assert!(!contains(160));
        assert!(contains(350));
        assert!(!contains(360));
        assert!(contains(401));
    }

    #[test]
    fn test_arrival_set_get() {
        let mut bldr = ModelBuilder::<i64>::new(1, 3);
        bldr.set_vessel_arrival_time(v(0), 10)
            .set_vessel_arrival_time(v(1), 20)
            .set_vessel_arrival_time(v(2), 30);
        let m = bldr.build();
        assert_eq!(m.vessel_arrival_time(v(0)), 10);
        assert_eq!(m.vessel_arrival_time(v(1)), 20);
        assert_eq!(m.vessel_arrival_time(v(2)), 30);
    }

    #[test]
    fn test_departure_set_get() {
        let mut bldr = ModelBuilder::<i64>::new(1, 2);
        bldr.set_vessel_latest_departure_time(v(0), 100)
            .set_vessel_latest_departure_time(v(1), 200);
        let m = bldr.build();
        assert_eq!(m.vessel_latest_departure_time(v(0)), 100);
        assert_eq!(m.vessel_latest_departure_time(v(1)), 200);
    }

    #[test]
    fn test_weight_set_get() {
        let mut bldr = ModelBuilder::<i64>::new(1, 2);
        bldr.set_vessel_weight(v(0), 5).set_vessel_weight(v(1), 7);
        let m = bldr.build();
        assert_eq!(m.vessel_weight(v(0)), 5);
        assert_eq!(m.vessel_weight(v(1)), 7);
    }

    #[test]
    fn test_views_lengths() {
        let m = ModelBuilder::<i64>::new(2, 3).build();
        assert_eq!(m.vessel_arrival_times().len(), 3);
        assert_eq!(m.vessel_latest_departure_times().len(), 3);
        assert_eq!(m.vessel_weights().len(), 3);
        assert_eq!(m.vessel_processing_times().len(), 6);
        assert_eq!(m.vessel_opening_times().len(), 2);
        assert_eq!(m.vessel_shortest_processing_times().len(), 3);
    }

    #[test]
    fn test_processing_set_get_none_and_some() {
        let mut bldr = ModelBuilder::<i64>::new(3, 2);
        bldr.set_vessel_processing_time(v(0), b(0), ProcessingTime::some(5))
            .set_vessel_processing_time(v(0), b(2), ProcessingTime::some(3))
            .set_vessel_processing_time(v(1), b(1), ProcessingTime::some(10));
        let m = bldr.build();

        assert_eq!(opt(m.vessel_processing_time(v(0), b(0))), Some(5));
        assert_eq!(opt(m.vessel_processing_time(v(0), b(1))), None);
        assert_eq!(opt(m.vessel_processing_time(v(0), b(2))), Some(3));
        assert_eq!(opt(m.vessel_processing_time(v(1), b(0))), None);
        assert_eq!(opt(m.vessel_processing_time(v(1), b(1))), Some(10));
        assert_eq!(opt(m.vessel_processing_time(v(1), b(2))), None);
    }

    #[test]
    fn test_processing_unchecked_matches_checked() {
        let mut bldr = ModelBuilder::<i64>::new(2, 1);
        bldr.set_vessel_processing_time(v(0), b(1), ProcessingTime::some(42));
        let m = bldr.build();

        unsafe {
            assert_eq!(
                Option::<i64>::from(m.vessel_processing_time_unchecked(v(0), b(0))),
                opt(m.vessel_processing_time(v(0), b(0)))
            );
            assert_eq!(
                Option::<i64>::from(m.vessel_processing_time_unchecked(v(0), b(1))),
                opt(m.vessel_processing_time(v(0), b(1)))
            );
        }
    }

    #[test]
    fn test_allowed_on_berth_basic() {
        let mut bldr = ModelBuilder::<i64>::new(2, 2);
        bldr.set_vessel_processing_time(v(1), b(0), ProcessingTime::some(7));
        let m = bldr.build();

        assert!(!m.vessel_allowed_on_berth(v(0), b(0)));
        assert!(!m.vessel_allowed_on_berth(v(0), b(1)));
        assert!(m.vessel_allowed_on_berth(v(1), b(0)));
        assert!(!m.vessel_allowed_on_berth(v(1), b(1)));
    }

    #[test]
    fn test_shortest_all_none() {
        let m = ModelBuilder::<i64>::new(3, 2).build();
        assert!(
            m.vessel_shortest_processing_times()
                .iter()
                .all(|pt| Option::<i64>::from(*pt).is_none())
        );
    }

    #[test]
    fn test_shortest_mixed_values() {
        let mut bldr = ModelBuilder::<i64>::new(4, 3);
        bldr.set_vessel_processing_time(v(0), b(0), ProcessingTime::some(5))
            .set_vessel_processing_time(v(0), b(2), ProcessingTime::some(3))
            .set_vessel_processing_time(v(2), b(0), ProcessingTime::some(8))
            .set_vessel_processing_time(v(2), b(1), ProcessingTime::some(8))
            .set_vessel_processing_time(v(2), b(2), ProcessingTime::some(10));
        let m = bldr.build();

        assert_eq!(opt(m.vessel_shortest_processing_time(v(0))), Some(3));
        assert_eq!(opt(m.vessel_shortest_processing_time(v(1))), None);
        assert_eq!(opt(m.vessel_shortest_processing_time(v(2))), Some(8));
    }

    #[test]
    fn test_shortest_single_berth() {
        let mut bldr = ModelBuilder::<i64>::new(1, 3);
        bldr.set_vessel_processing_time(v(0), b(0), ProcessingTime::some(100))
            .set_vessel_processing_time(v(2), b(0), ProcessingTime::some(50));
        let m = bldr.build();

        assert_eq!(opt(m.vessel_shortest_processing_time(v(0))), Some(100));
        assert_eq!(opt(m.vessel_shortest_processing_time(v(1))), None);
        assert_eq!(opt(m.vessel_shortest_processing_time(v(2))), Some(50));
    }

    #[test]
    fn test_shortest_varied_patterns() {
        let mut bldr = ModelBuilder::<i64>::new(5, 3);
        bldr.set_vessel_processing_time(v(0), b(1), ProcessingTime::some(9))
            .set_vessel_processing_time(v(0), b(3), ProcessingTime::some(2))
            .set_vessel_processing_time(v(0), b(4), ProcessingTime::some(2));
        for k in 0..5 {
            bldr.set_vessel_processing_time(v(2), b(k), ProcessingTime::some(7 - k as i64));
        }
        let m = bldr.build();
        assert_eq!(opt(m.vessel_shortest_processing_time(v(0))), Some(2));
        assert_eq!(opt(m.vessel_shortest_processing_time(v(1))), None);
        assert_eq!(opt(m.vessel_shortest_processing_time(v(2))), Some(3));
    }

    #[test]
    fn test_processing_flat_layout_matches_flatten_index() {
        let mut bldr = ModelBuilder::<i64>::new(3, 4);
        for vi in 0..4 {
            for bi in 0..3 {
                bldr.set_vessel_processing_time(
                    v(vi),
                    b(bi),
                    ProcessingTime::some(vi as i64 * 100 + bi as i64),
                );
            }
        }
        let m = bldr.build();
        for vi in 0..4 {
            for bi in 0..3 {
                let flat = flatten_index(3, v(vi), b(bi));
                let pt = m.vessel_processing_times()[flat];
                assert_eq!(opt(pt), Some(vi as i64 * 100 + bi as i64));
            }
        }
    }

    #[test]
    fn test_display_format() {
        let m = ModelBuilder::<i64>::new(4, 7).build();
        let s = format!("{}", m);
        assert!(s.contains("Model("));
        assert!(s.contains("num_vessels: 7"));
        assert!(s.contains("num_berths: 4"));
    }

    #[test]
    fn test_debug_format_contains_fields() {
        let mut bldr = ModelBuilder::<i64>::new(2, 2);
        bldr.set_vessel_arrival_time(v(0), 10)
            .set_vessel_latest_departure_time(v(1), 999)
            .set_vessel_weight(v(1), 5)
            .set_vessel_processing_time(v(0), b(0), ProcessingTime::some(3));
        let m = bldr.build();

        let s = format!("{:?}", m);
        assert!(s.contains("Model"));
        assert!(s.contains("arrival_times"));
        assert!(s.contains("latest_departure_times"));
        assert!(s.contains("vessel_weights"));
        assert!(s.contains("processing_times"));
        assert!(s.contains("opening_times"));
        assert!(s.contains("shortest_processing_times"));
        assert!(s.contains("10"));
        assert!(s.contains("999"));
        assert!(s.contains("5"));
        assert!(s.contains("3"));
    }

    #[test]
    fn test_unchecked_arrival_matches_checked() {
        let m = ModelBuilder::<i64>::new(1, 2).build();
        unsafe {
            assert_eq!(
                m.vessel_arrival_time_unchecked(v(0)),
                m.vessel_arrival_time(v(0))
            );
            assert_eq!(
                m.vessel_arrival_time_unchecked(v(1)),
                m.vessel_arrival_time(v(1))
            );
        }
    }

    #[test]
    fn test_unchecked_departure_matches_checked() {
        let mut bldr = ModelBuilder::<i64>::new(1, 2);
        bldr.set_vessel_latest_departure_time(v(0), 123)
            .set_vessel_latest_departure_time(v(1), 456);
        let m = bldr.build();
        unsafe {
            assert_eq!(
                m.vessel_latest_departure_time_unchecked(v(0)),
                m.vessel_latest_departure_time(v(0))
            );
            assert_eq!(
                m.vessel_latest_departure_time_unchecked(v(1)),
                m.vessel_latest_departure_time(v(1))
            );
        }
    }

    #[test]
    fn test_unchecked_weight_matches_checked() {
        let mut bldr = ModelBuilder::<i64>::new(1, 2);
        bldr.set_vessel_weight(v(0), 9).set_vessel_weight(v(1), 8);
        let m = bldr.build();
        unsafe {
            assert_eq!(m.vessel_weight_unchecked(v(0)), m.vessel_weight(v(0)));
            assert_eq!(m.vessel_weight_unchecked(v(1)), m.vessel_weight(v(1)));
        }
    }

    #[test]
    fn test_unchecked_shortest_matches_checked() {
        let mut bldr = ModelBuilder::<i64>::new(2, 2);
        bldr.set_vessel_processing_time(v(0), b(0), ProcessingTime::some(33));
        let m = bldr.build();
        unsafe {
            assert_eq!(
                m.vessel_shortest_processing_time_unchecked(v(0)),
                m.vessel_shortest_processing_time(v(0))
            );
            assert_eq!(
                m.vessel_shortest_processing_time_unchecked(v(1)),
                m.vessel_shortest_processing_time(v(1))
            );
        }
    }

    #[test]
    fn test_builder_display_and_debug() {
        let mut bldr = ModelBuilder::<i64>::new(3, 5);
        bldr.set_vessel_arrival_time(v(2), 77)
            .set_vessel_latest_departure_time(v(4), 999)
            .set_vessel_weight(v(1), 3);
        let dbg_str = format!("{:?}", bldr);
        let disp_str = format!("{}", bldr);
        assert!(dbg_str.contains("ModelBuilder"));
        assert!(dbg_str.contains("num_berths"));
        assert!(dbg_str.contains("num_vessels"));
        assert!(dbg_str.contains("opening_times"));
        assert!(dbg_str.contains("arrival_times"));
        assert!(dbg_str.contains("latest_departure_times"));
        assert!(dbg_str.contains("vessel_weights"));
        assert!(dbg_str.contains("processing_times"));
        assert!(disp_str.contains("ModelBuilder("));
        assert!(disp_str.contains("num_berths: 3"));
        assert!(disp_str.contains("num_vessels: 5"));
    }

    #[test]
    fn test_opening_times_add_empty_iterators_is_idempotent() {
        let mut bldr = ModelBuilder::<i64>::new(2, 1);
        bldr.add_berth_opening_times(b(0), std::iter::empty::<ClosedOpenInterval<i64>>());
        bldr.add_berth_closing_times(b(1), std::iter::empty::<ClosedOpenInterval<i64>>());
        let m = bldr.build();
        // Both berths should remain unconstrained membership-wise.
        let contains0 = |x| {
            m.berth_opening_times(b(0))
                .iter()
                .any(|i| i.contains_point(x))
        };
        let contains1 = |x| {
            m.berth_opening_times(b(1))
                .iter()
                .any(|i| i.contains_point(x))
        };
        assert!(contains0(0));
        assert!(contains0(i64::MAX - 1));
        assert!(!contains0(i64::MAX));
        assert!(contains1(0));
        assert!(contains1(i64::MAX - 1));
        assert!(!contains1(i64::MAX));
    }

    #[test]
    fn test_opening_times_idempotent_add_close_same_interval() {
        let mut bldr = ModelBuilder::<i64>::new(1, 1);
        let interval = ClosedOpenInterval::new(100, 200);
        bldr.add_berth_closing_time(b(0), interval);
        // closing again should be idempotent
        bldr.add_berth_closing_time(b(0), interval);
        // re-open then re-open again should be idempotent
        let reopen = ClosedOpenInterval::new(120, 130);
        bldr.add_berth_opening_time(b(0), reopen);
        bldr.add_berth_opening_time(b(0), reopen);
        let m = bldr.build();
        let opening = m.berth_opening_times(b(0));
        let contains = |x| opening.iter().any(|i| i.contains_point(x));
        assert!(!contains(100));
        assert!(contains(125)); // reopened
        assert!(!contains(130)); // half-open
        assert!(contains(200)); // after closed section
    }

    #[test]
    fn test_processing_time_boundary_values() {
        let mut bldr = ModelBuilder::<i64>::new(2, 2);
        bldr.set_vessel_processing_time(v(0), b(0), ProcessingTime::some(0))
            .set_vessel_processing_time(v(0), b(1), ProcessingTime::some(i64::MAX - 1))
            .set_vessel_processing_time(v(1), b(0), ProcessingTime::none())
            .set_vessel_processing_time(v(1), b(1), ProcessingTime::some(1));
        let m = bldr.build();

        assert_eq!(
            Option::<i64>::from(m.vessel_processing_time(v(0), b(0))),
            Some(0)
        );
        assert_eq!(
            Option::<i64>::from(m.vessel_processing_time(v(0), b(1))),
            Some(i64::MAX - 1)
        );
        assert_eq!(
            Option::<i64>::from(m.vessel_processing_time(v(1), b(0))),
            None
        );
        assert_eq!(
            Option::<i64>::from(m.vessel_processing_time(v(1), b(1))),
            Some(1)
        );

        assert_eq!(
            Option::<i64>::from(m.vessel_shortest_processing_time(v(0))),
            Some(0)
        );
        assert_eq!(
            Option::<i64>::from(m.vessel_shortest_processing_time(v(1))),
            Some(1)
        );
    }

    #[test]
    fn test_allowed_on_berth_unchecked_more_cases() {
        let mut bldr = ModelBuilder::<i64>::new(3, 2);
        bldr.set_vessel_processing_time(v(0), b(2), ProcessingTime::some(9));
        let m = bldr.build();
        unsafe {
            assert!(!m.vessel_allowed_on_berth_unchecked(v(0), b(0)));
            assert!(!m.vessel_allowed_on_berth_unchecked(v(0), b(1)));
            assert!(m.vessel_allowed_on_berth_unchecked(v(0), b(2)));
            assert!(!m.vessel_allowed_on_berth_unchecked(v(1), b(2)));
        }
    }

    #[test]
    fn test_opening_time_first_berth_checked_vs_unchecked() {
        let mut bldr = ModelBuilder::<i64>::new(2, 1);
        bldr.add_berth_closing_time(b(0), ClosedOpenInterval::new(50, 60));
        let m = bldr.build();
        let checked = m.berth_opening_times(b(0));
        unsafe {
            let unchecked = m.berth_opening_times_unchecked(b(0));
            assert_eq!(checked.len(), unchecked.len());
            for (a, c) in checked.iter().zip(unchecked.iter()) {
                assert_eq!(a, c);
            }
        }
    }

    #[test]
    fn test_processing_times_large_values_and_layout_consistency() {
        let mut bldr = ModelBuilder::<i64>::new(4, 3);
        // Fill with large deterministic values
        for vi in 0..3 {
            for bi in 0..4 {
                let val = (vi as i64) * (i64::MAX / 4) + bi as i64;
                bldr.set_vessel_processing_time(v(vi), b(bi), ProcessingTime::some(val));
            }
        }
        let m = bldr.build();
        // Verify layout via flatten_index and conversion
        for vi in 0..3 {
            for bi in 0..4 {
                let flat = flatten_index(4, v(vi), b(bi));
                let pt = m.vessel_processing_times()[flat];
                let expected = (vi as i64) * (i64::MAX / 4) + bi as i64;
                assert_eq!(Option::<i64>::from(pt), Some(expected));
            }
        }
    }

    #[test]
    fn test_opening_time_membership_boundaries() {
        let mut bldr = ModelBuilder::<i64>::new(1, 1);
        let x = ClosedOpenInterval::new(0, 10);
        bldr.add_berth_closing_time(b(0), x);
        let m = bldr.build();
        let opening = m.berth_opening_times(b(0));
        let contains = |val| opening.iter().any(|i| i.contains_point(val));
        // Before 0 not open (unconstrained starts at 0)
        assert!(contains(0) == false);
        assert!(contains(9) == false);
        assert!(contains(10) == true); // immediately after closed block resumes open
    }

    #[test]
    fn test_empty_model_builder_build() {
        let bldr = ModelBuilder::<i64>::new(0, 0);
        let model = bldr.build();
        assert_eq!(model.num_berths(), 0);
        assert_eq!(model.num_vessels(), 0);
        assert!(model.vessel_arrival_times().is_empty());
        assert!(model.vessel_latest_departure_times().is_empty());
        assert!(model.vessel_weights().is_empty());
        assert!(model.vessel_processing_times().is_empty());
        assert!(model.vessel_opening_times().is_empty());
        assert!(model.vessel_shortest_processing_times().is_empty());
    }

    #[test]
    fn test_arrival_sorted_vessels_basic_order() {
        // 4 vessels, set arrivals out of order; only some have processing times
        let mut bldr = ModelBuilder::<i64>::new(2, 4);
        // arrivals: v0=30, v1=10, v2=20, v3=40
        bldr.set_vessel_arrival_time(VesselIndex::new(0), 30)
            .set_vessel_arrival_time(VesselIndex::new(1), 10)
            .set_vessel_arrival_time(VesselIndex::new(2), 20)
            .set_vessel_arrival_time(VesselIndex::new(3), 40);

        // Provide processing times so shortest exists for v0, v1, v2; leave v3 with None to ensure exclusion
        bldr.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(0),
            ProcessingTime::some(7),
        )
        .set_vessel_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(0),
            ProcessingTime::some(5),
        )
        .set_vessel_processing_time(
            VesselIndex::new(2),
            BerthIndex::new(1),
            ProcessingTime::some(9),
        );
        // v3: no processing times => excluded

        let m = bldr.build();
        let sorted = m.vessel_arrival_sorted_vessels();

        // Expect indices in arrival order: v1 (10), v2 (20), v0 (30). v3 excluded.
        let indices: Vec<usize> = sorted.iter().map(|v| v.index().get()).collect();
        assert_eq!(indices, vec![1, 2, 0]);

        // Sanity-check payloads (arrival and min_processing_time match)
        assert_eq!(sorted[0].arrival_time(), 10);
        assert_eq!(sorted[0].min_processing_time(), 5);
        assert_eq!(sorted[1].arrival_time(), 20);
        assert_eq!(sorted[1].min_processing_time(), 9);
        assert_eq!(sorted[2].arrival_time(), 30);
        assert_eq!(sorted[2].min_processing_time(), 7);
    }

    #[test]
    fn test_arrival_sorted_vessels_tiebreaker_and_exclusion() {
        // Create vessels with same arrival, differing shortest processing times,
        // and ensure vessels with None are excluded.
        let mut bldr = ModelBuilder::<i64>::new(3, 5);
        // arrivals: v0=50, v1=50, v2=10, v3=50, v4=60
        bldr.set_vessel_arrival_time(VesselIndex::new(0), 50)
            .set_vessel_arrival_time(VesselIndex::new(1), 50)
            .set_vessel_arrival_time(VesselIndex::new(2), 10)
            .set_vessel_arrival_time(VesselIndex::new(3), 50)
            .set_vessel_arrival_time(VesselIndex::new(4), 60);

        // Processing times:
        // v2 has shortest = 3 (arrival 10) => should be first
        bldr.set_vessel_processing_time(
            VesselIndex::new(2),
            BerthIndex::new(1),
            ProcessingTime::some(3),
        );

        // v0, v1, v3 all have same arrival (50), tie-break on shortest processing time
        bldr.set_vessel_processing_time(
            VesselIndex::new(0),
            BerthIndex::new(0),
            ProcessingTime::some(10),
        ) // min=10
        .set_vessel_processing_time(
            VesselIndex::new(1),
            BerthIndex::new(2),
            ProcessingTime::some(7),
        ) // min=7
        .set_vessel_processing_time(
            VesselIndex::new(3),
            BerthIndex::new(1),
            ProcessingTime::some(12),
        ); // min=12

        // v4: leave without any processing time => excluded
        let m = bldr.build();
        let sorted = m.vessel_arrival_sorted_vessels();

        // Expected order:
        // - v2 first (arrival 10)
        // - among arrival=50: v1 (min=7), v0 (min=10), v3 (min=12)
        // - v4 excluded
        let indices: Vec<usize> = sorted.iter().map(|v| v.index().get()).collect();
        assert_eq!(indices, vec![2, 1, 0, 3]);

        // Validate tie-break ordering by min_processing_time for the arrival=50 group
        assert_eq!(sorted[1].arrival_time(), 50);
        assert_eq!(sorted[1].min_processing_time(), 7);
        assert_eq!(sorted[2].arrival_time(), 50);
        assert_eq!(sorted[2].min_processing_time(), 10);
        assert_eq!(sorted[3].arrival_time(), 50);
        assert_eq!(sorted[3].min_processing_time(), 12);

        // Ensure no vessel with None shortest processing time is present
        assert!(sorted.iter().all(|v| v.min_processing_time() >= 0));
    }
}
