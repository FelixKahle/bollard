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

use bollard_core::math::interval::ClosedOpenInterval;
use bollard_core::num::constants::{self, MinusOne};
use bollard_core::utils::index::{TypedIndex, TypedIndexTag};
use num_traits::{PrimInt, Signed};

/// A tag type for vessel indices.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct VesselIndexTag;

impl TypedIndexTag for VesselIndexTag {
    const NAME: &'static str = "VesselIndex";
}

/// A typed index for vessels.
pub type VesselIndex = TypedIndex<VesselIndexTag>;

/// A tag type for berth indices.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BerthIndexTag;

impl TypedIndexTag for BerthIndexTag {
    const NAME: &'static str = "BerthIndex";
}

/// A typed index for berths.
pub type BerthIndex = TypedIndex<BerthIndexTag>;

/// A processing time that may be absent.
///
/// Instead of using `Option<T>`, this type uses a sentinel encoding to avoid
/// the additional discriminant that `Option` typically introduces for integer
/// types. In hot loops and dense collections, keeping the value to a single
/// machine word can improve cache locality and reduce memory traffic.
///
/// Encoding:
/// - Non-negative values (>= 0) represent a concrete processing time.
/// - Negative values (<= -1) are reserved to indicate absence.
///
/// This convention assumes valid processing times are non-negative. If negative
/// values are meaningful in your domain, use `Option<T>` instead.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProcessingTime<T>(T)
where
    T: Signed;

impl<T> ProcessingTime<T>
where
    T: Copy + Signed + constants::MinusOne,
{
    const NONE_SENTINEL: T = T::MINUS_ONE;

    /// Creates a `ProcessingTime` from an `Option<T>`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(5i32));
    /// assert!(some_time.is_some());
    /// assert_eq!(some_time.raw(), 5);
    /// ```
    #[inline]
    pub fn from_option(value: Option<T>) -> Self {
        match value {
            Some(v) => ProcessingTime(v),
            None => ProcessingTime(Self::NONE_SENTINEL),
        }
    }

    /// Creates a `ProcessingTime` from a raw value without checking for sentinel.
    /// If you pass a negative value, it will be treated as `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::ProcessingTime;
    ///
    /// let time = ProcessingTime::from_raw(10i32);
    /// assert!(time.is_some());
    /// assert_eq!(time.raw(), 10);
    /// ```
    #[inline]
    pub const fn from_raw(value: T) -> Self {
        ProcessingTime(value)
    }

    /// Creates a `ProcessingTime` representing `Some`.
    ///
    /// # Panics
    ///
    /// This function will panic if the provided value is negative.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::some(5i32);
    /// assert!(some_time.is_some());
    /// assert_eq!(some_time.raw(), 5);
    /// ```
    pub fn some(value: T) -> Self
    where
        T: PartialOrd,
    {
        assert!(
            value > Self::NONE_SENTINEL,
            "Value must be non-negative to represent Some"
        );

        ProcessingTime(value)
    }

    /// Creates a `ProcessingTime` representing `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::ProcessingTime;
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// assert!(none_time.is_none());
    /// ```
    #[inline]
    pub fn none() -> Self {
        ProcessingTime(Self::NONE_SENTINEL)
    }

    /// Checks if the `ProcessingTime` represents `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::ProcessingTime;
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// assert!(none_time.is_none());
    /// ```
    #[inline]
    pub fn is_none(&self) -> bool
    where
        T: PartialOrd,
    {
        self.0 <= Self::NONE_SENTINEL
    }

    /// Checks if the `ProcessingTime` represents `Some`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(3i32));
    /// assert!(some_time.is_some());
    /// ```
    #[inline]
    pub fn is_some(&self) -> bool
    where
        T: PartialOrd,
    {
        !self.is_none()
    }

    /// Returns the raw value, including sentinel if present.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::ProcessingTime;
    ///
    /// let time = ProcessingTime::from_option(Some(7i32));
    /// assert_eq!(time.raw(), 7);
    /// ```
    #[inline]
    pub fn raw(&self) -> T {
        self.0
    }

    /// Converts the `ProcessingTime` back into an `Option<T>`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(4i32));
    /// assert_eq!(some_time.as_option(), Some(4));
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// assert_eq!(none_time.as_option(), None);
    /// ```
    #[inline]
    pub fn as_option(&self) -> Option<T>
    where
        T: PartialOrd,
    {
        if self.is_none() { None } else { Some(self.0) }
    }

    /// Unwraps the `ProcessingTime`, panicking if it is `None`.
    ///
    /// # Panics
    ///
    /// This function will panic if called on a `ProcessingTime` that represents `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(6i32));
    /// assert_eq!(some_time.unwrap(), 6);
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// // The following line would panic:
    /// // none_time.unwrap();
    /// ```
    pub fn unwrap(&self) -> T
    where
        T: PartialOrd,
    {
        if self.is_none() {
            panic!("called `ProcessingTime::unwrap()` on a `None` value")
        }
        self.0
    }

    /// Unwraps the `ProcessingTime`, returning a default value if it is `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(8i32));
    /// assert_eq!(some_time.unwrap_or(0), 8);
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// assert_eq!(none_time.unwrap_or(0), 0);
    /// ```
    #[inline]
    pub fn unwrap_or(&self, default: T) -> T
    where
        T: PartialOrd,
    {
        if self.is_none() { default } else { self.0 }
    }

    /// Unwraps the `ProcessingTime`, computing a default value if it is `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(9i32));
    /// assert_eq!(some_time.unwrap_or_else(|| 1 + 1), 9);
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// assert_eq!(none_time.unwrap_or_else(|| 1 + 1), 2);
    /// ```
    #[inline]
    pub fn unwrap_or_else<F>(&self, f: F) -> T
    where
        T: PartialOrd,
        F: FnOnce() -> T,
    {
        if self.is_none() { f() } else { self.0 }
    }
}

impl<T> std::fmt::Debug for ProcessingTime<T>
where
    T: Copy + Signed + PartialOrd + constants::MinusOne + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_none() {
            write!(f, "ProcessingTime(None)")
        } else {
            write!(f, "ProcessingTime(Some({:?}))", self.0)
        }
    }
}

impl<T> std::fmt::Display for ProcessingTime<T>
where
    T: Copy + Signed + PartialOrd + constants::MinusOne + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_none() {
            write!(f, "ProcessingTime(None)")
        } else {
            write!(f, "ProcessingTime({})", self.0)
        }
    }
}

impl<T> From<Option<T>> for ProcessingTime<T>
where
    T: Copy + Signed + constants::MinusOne,
{
    #[inline]
    fn from(value: Option<T>) -> Self {
        ProcessingTime::from_option(value)
    }
}

impl<T> From<ProcessingTime<T>> for Option<T>
where
    T: Copy + Signed + PartialOrd + constants::MinusOne,
{
    #[inline]
    fn from(val: ProcessingTime<T>) -> Self {
        val.as_option()
    }
}

#[inline(always)]
fn flatten_index(num_berths: usize, vessel_index: VesselIndex, berth_index: BerthIndex) -> usize {
    vessel_index.get() * num_berths + berth_index.get()
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
    arrival_times: Vec<T>,                             // len = num_vessels
    latest_departure_times: Vec<T>,                    // len = num_vessels
    vessel_weights: Vec<T>,                            // len = num_vessels
    processing_times: Vec<ProcessingTime<T>>,          // len = num_vessels * num_berths
    opening_times: Vec<Vec<ClosedOpenInterval<T>>>,    // len = num_berths.
    shortest_processing_times: Vec<ProcessingTime<T>>, // len = num_vessels
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
    /// # use bollard_model::ModelBuilder;
    ///
    /// let model = ModelBuilder::<i32>::new(2, 3).build().unwrap();
    /// assert_eq!(model.num_vessels(), 2);
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
    /// # use bollard_model::ModelBuilder;
    ///
    /// let model = ModelBuilder::<i32>::new(2, 3).build().unwrap();
    /// assert_eq!(model.num_berths(), 3);
    /// ```
    #[inline]
    pub fn num_berths(&self) -> usize {
        self.opening_times.len()
    }

    /// Returns a slice of arrival times for all vessels.
    #[inline]
    pub fn arrival_times(&self) -> &[T] {
        &self.arrival_times
    }

    /// Returns a slice of vessel weights for all vessels.
    #[inline]
    pub fn vessel_weights(&self) -> &[T] {
        &self.vessel_weights
    }

    /// Returns a slice of latest departure times for all vessels.
    #[inline]
    pub fn latest_departure_times(&self) -> &[T] {
        &self.latest_departure_times
    }

    /// Returns a slice of processing times.
    ///
    /// The slice is flat and contains `num_vessels * num_berths` elements.
    #[inline]
    pub fn processing_times(&self) -> &[ProcessingTime<T>] {
        &self.processing_times
    }

    /// Returns a slice containing the vectors of opening intervals for all berths.
    #[inline]
    pub fn opening_times(&self) -> &[Vec<ClosedOpenInterval<T>>] {
        &self.opening_times
    }

    /// Returns a slice of shortest processing times for all vessels across all berths.
    #[inline]
    pub fn shortest_processing_times(&self) -> &[ProcessingTime<T>] {
        &self.shortest_processing_times
    }

    /// Returns the arrival time of the vessel at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, VesselIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.vessel_arrival(VesselIndex::new(0), 10);
    /// let model = builder.build().unwrap();
    ///
    /// assert_eq!(model.arrival_time(VesselIndex::new(0)), 10);
    /// ```
    #[inline]
    pub fn arrival_time(&self, vessel_index: VesselIndex) -> T {
        self.arrival_times[vessel_index.get()]
    }

    /// Returns the arrival time of the vessel at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is within the bounds of `0..num_vessels()`.
    /// Accessing an out-of-bounds index is undefined behavior.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, VesselIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.vessel_arrival(VesselIndex::new(0), 10);
    /// let model = builder.build().unwrap();
    ///
    /// unsafe {
    ///     assert_eq!(model.arrival_time_unchecked(VesselIndex::new(0)), 10);
    /// }
    /// ```
    #[inline]
    pub unsafe fn arrival_time_unchecked(&self, vessel_index: VesselIndex) -> T {
        unsafe { *self.arrival_times.get_unchecked(vessel_index.get()) }
    }

    /// Returns the latest departure time of the vessel at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, VesselIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.vessel_latest_departure(VesselIndex::new(0), 50);
    /// let model = builder.build().unwrap();
    ///
    /// assert_eq!(model.latest_departure_time(VesselIndex::new(0)), 50);
    /// ```
    #[inline]
    pub fn latest_departure_time(&self, vessel_index: VesselIndex) -> T {
        self.latest_departure_times[vessel_index.get()]
    }

    /// Returns the latest departure time of the vessel at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is within the bounds of `0..num_vessels()`.
    /// Accessing an out-of-bounds index is undefined behavior.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, VesselIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.vessel_latest_departure(VesselIndex::new(0), 50);
    /// let model = builder.build().unwrap();
    ///
    /// unsafe {
    ///     assert_eq!(model.latest_departure_time_unchecked(VesselIndex::new(0)), 50);
    /// }
    /// ```
    #[inline]
    pub unsafe fn latest_departure_time_unchecked(&self, vessel_index: VesselIndex) -> T {
        unsafe {
            *self
                .latest_departure_times
                .get_unchecked(vessel_index.get())
        }
    }

    /// Returns the weight of the vessel at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, VesselIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.vessel_weight(VesselIndex::new(0), 15);
    /// let model = builder.build().unwrap();
    ///
    /// assert_eq!(model.vessel_weight(VesselIndex::new(0)), 15);
    /// ```
    #[inline]
    pub fn vessel_weight(&self, vessel_index: VesselIndex) -> T {
        self.vessel_weights[vessel_index.get()]
    }

    /// Returns the weight of the vessel at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is within the bounds of `0..num_vessels()`.
    /// Accessing an out-of-bounds index is undefined behavior.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, VesselIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.vessel_weight(VesselIndex::new(0), 15);
    /// let model = builder.build().unwrap();
    ///
    /// unsafe {
    ///    assert_eq!(model.vessel_weight_unchecked(VesselIndex::new(0)), 15);
    /// }
    /// ```
    #[inline]
    pub fn vessel_weight_unchecked(&self, vessel_index: VesselIndex) -> T {
        unsafe { *self.vessel_weights.get_unchecked(vessel_index.get()) }
    }

    /// Returns the processing time for a specific vessel at a specific berth.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` or `berth_index` are out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, VesselIndex, BerthIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.processing_time(VesselIndex::new(0), BerthIndex::new(0), 5);
    /// let model = builder.build().unwrap();
    ///
    /// let pt = model.processing_time(VesselIndex::new(0), BerthIndex::new(0));
    /// assert_eq!(pt.unwrap(), 5);
    /// ```
    #[inline]
    pub fn processing_time(
        &self,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
    ) -> ProcessingTime<T> {
        let flat_index = flatten_index(self.num_berths(), vessel_index, berth_index);
        self.processing_times[flat_index]
    }

    /// Returns the processing time for a specific vessel at a specific berth without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is within `0..num_vessels()` and
    /// `berth_index` is within `0..num_berths()`.
    /// Accessing an out-of-bounds index is undefined behavior.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, VesselIndex, BerthIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.processing_time(VesselIndex::new(0), BerthIndex::new(0), 5);
    /// let model = builder.build().unwrap();
    ///
    /// unsafe {
    ///     let pt = model.processing_time_unchecked(VesselIndex::new(0), BerthIndex::new(0));
    ///     assert_eq!(pt.unwrap(), 5);
    /// }
    /// ```
    #[inline]
    pub unsafe fn processing_time_unchecked(
        &self,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
    ) -> ProcessingTime<T> {
        let flat_index = flatten_index(self.num_berths(), vessel_index, berth_index);
        unsafe { *self.processing_times.get_unchecked(flat_index) }
    }

    /// Checks if a vessel is allowed to be processed at a specific berth.
    ///
    /// A vessel is allowed if the processing time is `Some` (non-negative).
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, VesselIndex, BerthIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.processing_unavailable(VesselIndex::new(0), BerthIndex::new(0));
    /// let model = builder.build().unwrap();
    ///
    /// assert_eq!(model.allowed_on_berth(VesselIndex::new(0), BerthIndex::new(0)), false);
    /// ```
    #[inline]
    pub fn allowed_on_berth(&self, vessel_index: VesselIndex, berth_index: BerthIndex) -> bool
    where
        T: MinusOne,
    {
        let pt = self.processing_time(vessel_index, berth_index);
        pt.is_some()
    }

    /// Checks if a vessel is allowed to be processed at a specific berth without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` and `berth_index` are valid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, VesselIndex, BerthIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.processing_unavailable(VesselIndex::new(0), BerthIndex::new(0));
    /// let model = builder.build().unwrap();
    ///
    /// unsafe {
    ///     assert_eq!(model.allowed_on_berth_unchecked(VesselIndex::new(0), BerthIndex::new(0)), false);
    /// }
    /// ```
    #[inline]
    pub unsafe fn allowed_on_berth_unchecked(
        &self,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
    ) -> bool
    where
        T: MinusOne,
    {
        let pt = unsafe { self.processing_time_unchecked(vessel_index, berth_index) };
        pt.is_some()
    }

    /// Returns the slice of opening time intervals for a specific berth.
    ///
    /// # Panics
    ///
    /// Panics if `berth_index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, BerthIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.berth_opening_time(BerthIndex::new(0), 10, 20);
    /// let model = builder.build().unwrap();
    ///
    /// let openings = model.opening_time(BerthIndex::new(0));
    /// assert_eq!(openings.len(), 1);
    /// assert_eq!(openings[0].start(), 10);
    /// ```
    #[inline]
    pub fn opening_time(&self, berth_index: BerthIndex) -> &[ClosedOpenInterval<T>] {
        &self.opening_times[berth_index.get()]
    }

    /// Returns the slice of opening time intervals for a specific berth without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `berth_index` is valid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, BerthIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// let model = builder.build().unwrap();
    ///
    /// unsafe {
    ///     let openings = model.opening_time_unchecked(BerthIndex::new(0));
    ///     assert!(!openings.is_empty());
    /// }
    /// ```
    #[inline]
    pub unsafe fn opening_time_unchecked(
        &self,
        berth_index: BerthIndex,
    ) -> &[ClosedOpenInterval<T>] {
        unsafe { self.opening_times.get_unchecked(berth_index.get()) }
    }

    /// Returns the shortest processing time for a vessel across all berths.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, VesselIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 2);
    /// builder.vessel_processing_times(VesselIndex::new(0), vec![Some(10), Some(5)]);
    /// let model = builder.build().unwrap();
    ///
    /// let min_pt = model.shortest_processing_time(VesselIndex::new(0));
    /// assert_eq!(min_pt.unwrap(), 5);
    /// ```
    #[inline]
    pub fn shortest_processing_time(&self, vessel_index: VesselIndex) -> ProcessingTime<T> {
        self.shortest_processing_times[vessel_index.get()]
    }

    /// Returns the shortest processing time for a vessel across all berths without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is valid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, VesselIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 2);
    /// builder.vessel_processing_times(VesselIndex::new(0), vec![Some(10), Some(5)]);
    /// let model = builder.build().unwrap();
    ///
    /// unsafe {
    ///     let min_pt = model.shortest_processing_time_unchecked(VesselIndex::new(0));
    ///     assert_eq!(min_pt.unwrap(), 5);
    /// }
    /// ```
    #[inline]
    pub unsafe fn shortest_processing_time_unchecked(
        &self,
        vessel_index: VesselIndex,
    ) -> ProcessingTime<T> {
        unsafe {
            *self
                .shortest_processing_times
                .get_unchecked(vessel_index.get())
        }
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

/// Error indicating that two opening intervals for a berth overlap or are adjacent.
///
/// For closed-open intervals, adjacency is defined as touching at the boundary:
/// `[a, b)` and `[b, c)` are considered adjacent (no gap). The model requires
/// opening windows per berth to be strictly disjoint with a positive gap.
///
/// Fields:
/// - `berth_index`: index of the berth where the violation occurred.
/// - `first`, `second`: the conflicting intervals.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BerthOpeningsOverlapOrAdjacencyError<T>
where
    T: PrimInt,
{
    pub berth_index: usize,
    pub first: ClosedOpenInterval<T>,
    pub second: ClosedOpenInterval<T>,
}

impl<T> std::fmt::Display for BerthOpeningsOverlapOrAdjacencyError<T>
where
    T: PrimInt + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Berth openings overlap or are adjacent in berth {}: {} and {}",
            self.berth_index, self.first, self.second
        )
    }
}

impl<T> std::error::Error for BerthOpeningsOverlapOrAdjacencyError<T> where
    T: PrimInt + std::fmt::Debug + std::fmt::Display
{
}

/// Error indicating that opening intervals for a berth are not sorted by start time.
///
/// Opening windows must be sorted ascending by `start()` to allow linear validation
/// and efficient algorithms.
///
/// Fields:
/// - `berth_index`: index of the berth with unsorted intervals.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BerthOpeningsUnsortedError {
    pub berth_index: usize,
}

impl std::fmt::Display for BerthOpeningsUnsortedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Berth openings in berth {} are not sorted",
            self.berth_index
        )
    }
}

impl std::error::Error for BerthOpeningsUnsortedError {}

/// Error indicating that an opening interval is empty (`start >= end`).
///
/// Empty intervals are treated as invalid configuration and rejected during
/// model build-time validation.
///
/// Fields:
/// - `index`: the berth index containing the empty interval.
/// - `interval`: the offending interval instance.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EmptyBerthOpeningError<T>
where
    T: PrimInt,
{
    pub index: usize,
    pub interval: ClosedOpenInterval<T>,
}

impl<T> std::fmt::Display for EmptyBerthOpeningError<T>
where
    T: PrimInt + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Berth {} has an empty opening time interval: {}",
            self.index, self.interval
        )
    }
}

impl<T> std::error::Error for EmptyBerthOpeningError<T> where
    T: PrimInt + std::fmt::Debug + std::fmt::Display
{
}

/// Aggregate error type for failures that can occur while building a [`Model`].
///
/// Variants:
/// - `EmptyBerthOpening`: an interval had `start >= end`.
/// - `BerthOpeningsOverlap`: intervals in a berth overlapped or were adjacent.
/// - `BerthOpeningsUnsorted`: intervals in a berth were not sorted by start time.
///
/// Returned by:
/// - `ModelBuilder::build`, after validating berth opening windows.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ModelBuildError<T>
where
    T: PrimInt,
{
    EmptyBerthOpening(EmptyBerthOpeningError<T>),
    BerthOpeningsOverlap(BerthOpeningsOverlapOrAdjacencyError<T>),
    BerthOpeningsUnsorted(BerthOpeningsUnsortedError),
}

impl<T> std::fmt::Display for ModelBuildError<T>
where
    T: PrimInt + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelBuildError::EmptyBerthOpening(err) => write!(f, "{}", err),
            ModelBuildError::BerthOpeningsOverlap(err) => write!(f, "{}", err),
            ModelBuildError::BerthOpeningsUnsorted(err) => write!(f, "{}", err),
        }
    }
}

impl<T> std::error::Error for ModelBuildError<T> where
    T: PrimInt + std::fmt::Debug + std::fmt::Display
{
}

impl<T> From<BerthOpeningsOverlapOrAdjacencyError<T>> for ModelBuildError<T>
where
    T: PrimInt,
{
    fn from(err: BerthOpeningsOverlapOrAdjacencyError<T>) -> Self {
        ModelBuildError::BerthOpeningsOverlap(err)
    }
}

impl<T> From<BerthOpeningsUnsortedError> for ModelBuildError<T>
where
    T: PrimInt,
{
    fn from(err: BerthOpeningsUnsortedError) -> Self {
        ModelBuildError::BerthOpeningsUnsorted(err)
    }
}

impl<T> From<EmptyBerthOpeningError<T>> for ModelBuildError<T>
where
    T: PrimInt,
{
    fn from(err: EmptyBerthOpeningError<T>) -> Self {
        ModelBuildError::EmptyBerthOpening(err)
    }
}

fn validate_berth_openings<T>(
    openings: &[Vec<ClosedOpenInterval<T>>],
) -> Result<(), ModelBuildError<T>>
where
    T: PrimInt + PartialOrd,
{
    for (berth_idx, intervals) in openings.iter().enumerate() {
        for interval in intervals {
            if interval.is_empty() {
                return Err(ModelBuildError::EmptyBerthOpening(EmptyBerthOpeningError {
                    index: berth_idx,
                    interval: *interval,
                }));
            }
        }

        for i in 1..intervals.len() {
            let prev = unsafe { *intervals.get_unchecked(i - 1) };
            let curr = unsafe { *intervals.get_unchecked(i) };

            if prev.start() > curr.start() {
                return Err(ModelBuildError::BerthOpeningsUnsorted(
                    BerthOpeningsUnsortedError {
                        berth_index: berth_idx,
                    },
                ));
            }

            if !prev.disjoint(curr) {
                return Err(ModelBuildError::BerthOpeningsOverlap(
                    BerthOpeningsOverlapOrAdjacencyError {
                        berth_index: berth_idx,
                        first: prev,
                        second: curr,
                    },
                ));
            }
        }
    }

    Ok(())
}

/// Builder for constructing a validated `Model`.
///
/// Responsibilities:
/// - Initializes default values (`arrival=0`, `latest departure=max`, `opening=[0, max)`, `processing=None`).
/// - Offers random-access setters for per-vessel and per-berth data.
/// - Performs validation ensuring berth opening intervals are non-empty, sorted,
///   and strictly disjoint (no overlaps or adjacency).
/// - Computes per-vessel shortest processing times across berths.
///
/// Usage:
/// - Create with `ModelBuilder::new` specifying the number of vessels and berths.
/// - Call setters like `vessel_arrival`, `vessel_latest_departure`,
///   `berth_opening_time`, and `processing_time`.
/// - Finalize with `build`, which returns `Result<Model<T>, ModelBuildError<T>>`.
#[derive(Clone)]
pub struct ModelBuilder<T>
where
    T: PrimInt + Signed,
{
    arrival_times: Vec<T>,
    latest_departure_times: Vec<T>,
    vessel_weights: Vec<T>,
    processing_times: Vec<ProcessingTime<T>>,
    opening_times: Vec<Vec<ClosedOpenInterval<T>>>,
}

impl<T> ModelBuilder<T>
where
    T: PrimInt + Signed + Copy + std::fmt::Debug + std::fmt::Display,
{
    /// Creates a new `ModelBuilder` with pre-allocated and initialized storage.
    ///
    /// - Arrival times default to `0`.
    /// - Latest departure times default to `T::max_value()`.
    /// - Processing times default to `None`.
    /// - Opening times default to `[0, T::max_value())` for each berth.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::ModelBuilder;
    ///
    /// let builder = ModelBuilder::<i32>::new(5, 3);
    /// assert_eq!(builder.num_vessels(), 5);
    /// assert_eq!(builder.num_berths(), 3);
    /// ```
    pub fn new(num_vessels: usize, num_berths: usize) -> Self
    where
        T: MinusOne,
    {
        let arrival_times = vec![T::zero(); num_vessels];
        let latest_departure_times = vec![T::max_value(); num_vessels];
        let vessel_weights = vec![T::one(); num_vessels];
        let processing_times = vec![ProcessingTime::none(); num_vessels * num_berths];
        let opening_times = vec![Vec::new(); num_berths];

        Self {
            arrival_times,
            latest_departure_times,
            vessel_weights,
            processing_times,
            opening_times,
        }
    }

    /// Returns the number of vessels configured in the builder.
    #[inline]
    pub fn num_vessels(&self) -> usize {
        self.arrival_times.len()
    }

    /// Returns the number of berths configured in the builder.
    #[inline]
    pub fn num_berths(&self) -> usize {
        self.opening_times.len()
    }

    /// Sets the arrival time for a specific vessel.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, VesselIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.vessel_arrival(VesselIndex::new(0), 10);
    /// ```
    pub fn vessel_arrival(&mut self, index: VesselIndex, time: T) -> &mut Self {
        self.arrival_times[index.get()] = time;
        self
    }

    /// Sets the latest departure time (deadline) for a specific vessel.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, VesselIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.vessel_latest_departure(VesselIndex::new(0), 100);
    /// ```
    pub fn vessel_latest_departure(&mut self, index: VesselIndex, time: T) -> &mut Self {
        self.latest_departure_times[index.get()] = time;
        self
    }

    /// Sets the weight for a specific vessel.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, VesselIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.vessel_weight(VesselIndex::new(0), 20);
    /// ```
    pub fn vessel_weight(&mut self, index: VesselIndex, weight: T) -> &mut Self {
        self.vessel_weights[index.get()] = weight;
        self
    }

    /// Sets both arrival and latest departure for a vessel.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, VesselIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.vessel_window(VesselIndex::new(0), 10, 100);
    /// ```
    pub fn vessel_window(&mut self, index: VesselIndex, arrival: T, departure: T) -> &mut Self {
        let i = index.get();
        self.arrival_times[i] = arrival;
        self.latest_departure_times[i] = departure;
        self
    }

    /// Sets the opening time for a specific berth to a single interval.
    ///
    /// This overwrites any existing intervals for this berth.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, BerthIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.berth_opening_time(BerthIndex::new(0), 0, 50);
    /// ```
    pub fn berth_opening_time(&mut self, index: BerthIndex, start: T, end: T) -> &mut Self {
        self.opening_times[index.get()] = vec![ClosedOpenInterval::new(start, end)];
        self
    }

    /// Adds an opening time interval for a specific berth.
    ///
    /// This appends the interval to any existing intervals for this berth.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::math::interval::ClosedOpenInterval;
    /// # use bollard_model::{ModelBuilder, BerthIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.push_berth_interval(BerthIndex::new(0), ClosedOpenInterval::new(60, 80));
    /// ```
    pub fn push_berth_interval(
        &mut self,
        index: BerthIndex,
        interval: ClosedOpenInterval<T>,
    ) -> &mut Self {
        self.opening_times[index.get()].push(interval);
        self
    }

    /// Sets the opening time for a specific berth to a single interval using a `ClosedOpenInterval` object.
    ///
    /// This overwrites any existing intervals for this berth.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bollard_core::math::interval::ClosedOpenInterval;
    /// # use bollard_model::{ModelBuilder, BerthIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.berth_interval(BerthIndex::new(0), ClosedOpenInterval::new(10, 20));
    /// ```
    pub fn berth_interval(
        &mut self,
        index: BerthIndex,
        interval: ClosedOpenInterval<T>,
    ) -> &mut Self {
        self.opening_times[index.get()] = vec![interval];
        self
    }

    /// Sets the opening time intervals for a specific berth.
    ///
    /// This allows defining multiple disjoint opening windows for a single berth.
    ///
    /// # Note
    ///
    /// The intervals will overwrite any existing intervals for this berth.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bollard_core::math::interval::ClosedOpenInterval;
    /// # use bollard_model::{ModelBuilder, BerthIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.berth_opening_intervals(BerthIndex::new(0), vec![
    ///     ClosedOpenInterval::new(0, 10),
    ///     ClosedOpenInterval::new(20, 30)
    /// ]);
    /// ```
    pub fn berth_opening_intervals<I>(&mut self, index: BerthIndex, intervals: I) -> &mut Self
    where
        I: IntoIterator<Item = ClosedOpenInterval<T>>,
    {
        self.opening_times[index.get()] = intervals.into_iter().collect();
        self
    }

    /// Sets the processing time for a specific vessel at a specific berth.
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds or if `time` is negative.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, VesselIndex, BerthIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.processing_time(VesselIndex::new(0), BerthIndex::new(0), 15);
    /// ```
    pub fn processing_time(
        &mut self,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
        time: T,
    ) -> &mut Self
    where
        T: PartialOrd + MinusOne,
    {
        let flat_index = flatten_index(self.num_berths(), vessel_index, berth_index);
        self.processing_times[flat_index] = ProcessingTime::some(time);
        self
    }

    /// Sets the processing time to `None` (unavailable) for a vessel at a berth.
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, VesselIndex, BerthIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 1);
    /// builder.processing_unavailable(VesselIndex::new(0), BerthIndex::new(0));
    /// ```
    pub fn processing_unavailable(
        &mut self,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
    ) -> &mut Self
    where
        T: PartialOrd + MinusOne,
    {
        let flat_index = flatten_index(self.num_berths(), vessel_index, berth_index);
        self.processing_times[flat_index] = ProcessingTime::none();
        self
    }

    /// Bulk sets processing times for a single vessel across all berths.
    ///
    /// This expects an iterator yielding values for berth 0, berth 1, etc.
    /// If `times` yields fewer items than berths, remaining berths are left unchanged.
    /// `None` in the iterator means unavailable.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{ModelBuilder, VesselIndex};
    ///
    /// let mut builder = ModelBuilder::<i32>::new(1, 3);
    /// builder.vessel_processing_times(VesselIndex::new(0), vec![Some(5), None, Some(8)]);
    /// ```
    pub fn vessel_processing_times<I>(&mut self, vessel_index: VesselIndex, times: I) -> &mut Self
    where
        I: IntoIterator<Item = Option<T>>,
        T: PartialOrd + MinusOne,
    {
        let num_berths = self.opening_times.len();
        let start_index = vessel_index.get() * num_berths;

        for (offset, time) in times.into_iter().enumerate().take(num_berths) {
            self.processing_times[start_index + offset] = ProcessingTime::from_option(time);
        }
        self
    }

    /// Builds the `Model` after validating constraints.
    ///
    /// # Errors
    /// Returns a `ModelBuildError` if:
    /// - Any opening time interval is empty (start >= end).
    /// - Opening times for any single berth are not sorted by start time.
    /// - Opening times for any single berth overlap or are adjacent (are not strictly disjoint with a gap).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::ModelBuilder;
    ///
    /// let builder = ModelBuilder::<i32>::new(2, 2);
    /// let model = builder.build().expect("Failed to build model");
    /// ```
    pub fn build(self) -> Result<Model<T>, ModelBuildError<T>>
    where
        T: MinusOne,
    {
        let default_interval = ClosedOpenInterval::new(T::zero(), T::max_value());
        let num_vessels = self.arrival_times.len();
        let num_berths = self.opening_times.len();

        assert_eq!(
            self.latest_departure_times.len(),
            num_vessels,
            "Internal Error: Latest departure times length mismatch"
        );
        assert_eq!(
            self.processing_times.len(),
            num_vessels * num_berths,
            "Internal Error: Processing times length mismatch"
        );

        let opening_times: Vec<Vec<ClosedOpenInterval<T>>> = self
            .opening_times
            .into_iter()
            .map(|mut intervals| {
                if intervals.is_empty() {
                    intervals.push(default_interval);
                }
                intervals
            })
            .collect();

        validate_berth_openings(&opening_times)?;

        let shortest_processing_times = Self::compute_shortest_processing_times(
            &self.processing_times,
            self.arrival_times.len(),
            opening_times.len(), // Use length of the local opening_times
        );

        Ok(Model {
            arrival_times: self.arrival_times,
            latest_departure_times: self.latest_departure_times,
            vessel_weights: self.vessel_weights,
            processing_times: self.processing_times,
            opening_times, // Use the modified local variable
            shortest_processing_times,
        })
    }

    fn compute_shortest_processing_times(
        processing_times: &[ProcessingTime<T>],
        num_vessels: usize,
        num_berths: usize,
    ) -> Vec<ProcessingTime<T>>
    where
        T: MinusOne,
    {
        let mut shortest_times = Vec::with_capacity(num_vessels);

        for vessel_idx in 0..num_vessels {
            let mut min_time: Option<T> = None;
            for berth_idx in 0..num_berths {
                let flat_index = vessel_idx * num_berths + berth_idx;
                let pt = processing_times[flat_index];

                if let Some(time) = pt.as_option() {
                    min_time = match min_time {
                        Some(current_min) => Some(current_min.min(time)),
                        None => Some(time),
                    };
                }
            }

            shortest_times.push(ProcessingTime::from_option(min_time));
        }

        shortest_times
    }
}

/// An assignment solution mapping vessels to berths and their planned start times.
///
/// The solution is represented as two parallel arrays:
/// - `vessel_berths[vessel]`: the assigned berth for the vessel at index `vessel`.
/// - `vessel_start_times[vessel]`: the planned start time for that vessel.
///   Both arrays must conceptually have the same length (one entry per vessel).
///
/// This type does not enforce constraints or perform validation; it is a simple
/// container intended to be produced by scheduling/optimization routines and
/// consumed by reporting or downstream systems.
#[derive(Clone)]
pub struct Solution<T>
where
    T: PrimInt + Signed,
{
    vessel_berths: Vec<BerthIndex>,
    vessel_start_times: Vec<T>,
}

impl<T> Solution<T>
where
    T: PrimInt + Signed + Copy + std::fmt::Debug + std::fmt::Display,
{
    /// Creates a new `Solution` from parallel vectors of berths and start times.
    ///
    /// Semantics:
    /// - `vessel_berths[v]` and `vessel_start_times[v]` together describe the assignment
    ///   for vessel `v`.
    /// - If the two vectors have different lengths, the `Display` implementation will
    ///   report inconsistency; indexing beyond the shorter length must be avoided by
    ///   consumers.
    #[inline]
    pub fn new(vessel_berths: Vec<BerthIndex>, vessel_start_times: Vec<T>) -> Self {
        Self {
            vessel_berths,
            vessel_start_times,
        }
    }

    /// Returns the assigned berths per vessel as a slice.
    ///
    /// Each position `i` corresponds to vessel `i`. The associated start time
    /// can be found at `vessel_start_times()[i]`.
    #[inline]
    pub fn vessel_berths(&self) -> &[BerthIndex] {
        &self.vessel_berths
    }

    /// Returns the planned start times per vessel as a slice.
    ///
    /// Each position `i` corresponds to vessel `i`. The associated berth
    /// can be found at `vessel_berths()[i]`.
    #[inline]
    pub fn vessel_start_times(&self) -> &[T] {
        &self.vessel_start_times
    }

    /// Returns the number of vessels represented in this solution.
    ///
    /// The solution stores two parallel arrays:
    /// - `vessel_berths[v]` and
    /// - `vessel_start_times[v]`
    ///
    /// This method returns the minimum of their lengths to ensure that downstream
    /// consumers can safely index the solution without risking out-of-bounds access
    /// in case of length mismatches. If the arrays are equal length, it equals
    /// the total number of vessels.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{BerthIndex, Solution};
    ///
    /// let sol = Solution::<i32>::new(
    ///     vec![BerthIndex::new(0), BerthIndex::new(1)],
    ///     vec![10, 20]
    /// );
    /// assert_eq!(sol.num_vessels(), 2);
    ///
    /// // If mismatched, min length is reported
    /// let sol2 = Solution::<i32>::new(vec![BerthIndex::new(0)], vec![5, 6, 7]);
    /// assert_eq!(sol2.num_vessels(), 1);
    /// ```
    #[inline]
    pub fn num_vessels(&self) -> usize {
        core::cmp::min(self.vessel_berths.len(), self.vessel_start_times.len())
    }

    /// Returns the berth and start time assignment for the given vessel index.
    ///
    /// This is a safe accessor that returns `None` if `vessel_index` is out of bounds
    /// according to [`Solution::num_vessels`]. When arrays are mismatched in length,
    /// the shorter length determines the valid range.
    ///
    /// # Arguments
    ///
    /// - `vessel_index`: the index of the vessel
    ///
    /// # Returns
    ///
    /// - `Some((berth, start_time))` if the index is valid
    /// - `None` if the index is out of range
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_model::{BerthIndex, Solution};
    ///
    /// let sol = Solution::<i32>::new(
    ///     vec![BerthIndex::new(2), BerthIndex::new(0)],
    ///     vec![100, 50]
    /// );
    ///
    /// assert_eq!(sol.vessel_assignment(0).unwrap().0.get(), 2);
    /// assert_eq!(sol.vessel_assignment(0).unwrap().1, 100);
    /// assert_eq!(sol.vessel_assignment(1).unwrap().0.get(), 0);
    /// assert_eq!(sol.vessel_assignment(1).unwrap().1, 50);
    /// assert!(sol.vessel_assignment(2).is_none());
    /// ```
    pub fn vessel_assignment(&self, vessel_index: usize) -> Option<(BerthIndex, T)> {
        if vessel_index < self.num_vessels() {
            let berth = unsafe { *self.vessel_berths.get_unchecked(vessel_index) };
            let start_time = unsafe { *self.vessel_start_times.get_unchecked(vessel_index) };
            Some((berth, start_time))
        } else {
            None
        }
    }
}

impl<T> std::fmt::Debug for Solution<T>
where
    T: PrimInt + Signed + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Solution")
            .field("vessel_berths", &self.vessel_berths)
            .field("vessel_start_times", &self.vessel_start_times)
            .finish()
    }
}

impl<T> std::fmt::Display for Solution<T>
where
    T: PrimInt + Signed + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let num_berths = self.vessel_berths.len();
        let num_starts = self.vessel_start_times.len();

        if num_berths == 0 && num_starts == 0 {
            return write!(f, "Solution: no vessels assigned");
        }

        if num_berths != num_starts {
            writeln!(
                f,
                "Solution: inconsistent lengths (berths: {}, start_times: {})",
                num_berths, num_starts
            )?;
        }

        let n = core::cmp::min(num_berths, num_starts);
        writeln!(f, "Solution: {} vessel(s)", n)?;

        for i in 0..n {
            let berth = unsafe { self.vessel_berths.get_unchecked(i) };
            let start_time = unsafe { self.vessel_start_times.get_unchecked(i) };
            writeln!(
                f,
                "  Vessel {}: Berth {}, Start {}",
                i,
                berth.get(),
                start_time
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bollard_core::math::interval::ClosedOpenInterval;

    fn make_small_model_i32(vessels: usize, berths: usize) -> Model<i32> {
        let builder = ModelBuilder::<i32>::new(vessels, berths);
        builder.build().unwrap()
    }

    #[test]
    fn test_processing_time_from_option_some() {
        let pt = ProcessingTime::<i32>::from_option(Some(5));
        assert!(pt.is_some());
        assert_eq!(pt.raw(), 5);
        assert_eq!(pt.as_option(), Some(5));
        assert_eq!(format!("{:?}", pt), "ProcessingTime(Some(5))");
        assert_eq!(format!("{}", pt), "ProcessingTime(5)");
    }

    #[test]
    fn test_processing_time_from_option_none() {
        let pt = ProcessingTime::<i32>::from_option(None);
        assert!(pt.is_none());
        assert_eq!(pt.as_option(), None);
        assert_eq!(pt.unwrap_or(42), 42);
        assert_eq!(pt.unwrap_or_else(|| 43), 43);
        assert_eq!(format!("{:?}", pt), "ProcessingTime(None)");
        assert_eq!(format!("{}", pt), "ProcessingTime(None)");
    }

    #[test]
    fn test_processing_time_from_raw_non_negative_is_some() {
        let pt = ProcessingTime::<i32>::from_raw(0);
        assert!(pt.is_some());
        assert_eq!(pt.raw(), 0);
    }

    #[test]
    fn test_processing_time_from_raw_negative_is_none() {
        // From_raw does not check sentinel, but is_none interprets negative as none
        let pt = ProcessingTime::<i32>::from_raw(-1);
        assert!(pt.is_none());
        assert_eq!(pt.as_option(), None);
    }

    #[test]
    fn test_processing_time_some_valid() {
        let pt = ProcessingTime::<i32>::some(7);
        assert!(pt.is_some());
        assert_eq!(pt.unwrap(), 7);
    }

    #[test]
    #[should_panic(expected = "Value must be non-negative to represent Some")]
    fn test_processing_time_some_negative_panics() {
        let _ = ProcessingTime::<i32>::some(-1);
    }

    #[test]
    fn test_processing_time_none_is_none() {
        let pt = ProcessingTime::<i32>::none();
        assert!(pt.is_none());
        assert_eq!(pt.as_option(), None);
    }

    #[test]
    fn test_processing_time_is_some_vs_none_boundaries() {
        let none = ProcessingTime::<i32>::from_raw(-1);
        let zero = ProcessingTime::<i32>::from_raw(0);
        let pos = ProcessingTime::<i32>::from_raw(123);

        assert!(none.is_none());
        assert!(!none.is_some());
        assert!(zero.is_some());
        assert!(pos.is_some());
    }

    #[test]
    fn test_processing_time_into_from_option_roundtrip() {
        let opt: Option<i32> = Some(17);
        let pt: ProcessingTime<i32> = ProcessingTime::from(opt);
        let back: Option<i32> = Option::from(pt);
        assert_eq!(back, Some(17));

        let opt_none: Option<i32> = None;
        let pt_none: ProcessingTime<i32> = ProcessingTime::from(opt_none);
        let back_none: Option<i32> = Option::from(pt_none);
        assert_eq!(back_none, None);
    }

    #[test]
    fn test_processing_time_unwrap_panics_on_none() {
        let pt_none: ProcessingTime<i32> = ProcessingTime::none();
        let result = std::panic::catch_unwind(|| {
            let _ = pt_none.unwrap();
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_processing_time_debug_and_display() {
        let some = ProcessingTime::<i32>::some(3);
        let none = ProcessingTime::<i32>::none();

        assert_eq!(format!("{:?}", some), "ProcessingTime(Some(3))");
        assert_eq!(format!("{:?}", none), "ProcessingTime(None)");
        assert_eq!(format!("{}", some), "ProcessingTime(3)");
        assert_eq!(format!("{}", none), "ProcessingTime(None)");
    }

    // Private helper flatten_index tests
    #[test]
    fn test_flatten_index_basic() {
        let v = VesselIndex::new(2);
        let b = BerthIndex::new(3);
        assert_eq!(super::flatten_index(5, v, b), 2 * 5 + 3);
    }

    #[test]
    fn test_flatten_index_zeroes() {
        let v = VesselIndex::new(0);
        let b = BerthIndex::new(0);
        assert_eq!(super::flatten_index(1, v, b), 0);
    }

    // ModelBuilder defaults
    #[test]
    fn test_model_builder_new_defaults() {
        let mb = ModelBuilder::<i32>::new(3, 2);
        assert_eq!(mb.num_vessels(), 3);
        assert_eq!(mb.num_berths(), 2);

        // After build, check defaults
        let m = mb.build().unwrap();
        assert_eq!(m.num_vessels(), 3);
        assert_eq!(m.num_berths(), 2);

        // Arrival times default to 0
        assert_eq!(m.arrival_times(), &[0, 0, 0]);

        // Latest departure times default to max_value
        assert_eq!(m.latest_departure_times(), &[i32::MAX, i32::MAX, i32::MAX]);

        // Vessel weights default to 1
        assert_eq!(m.vessel_weights(), &[1, 1, 1]);

        // Opening times default to [0, max)
        let openings0 = m.opening_time(BerthIndex::new(0));
        assert_eq!(openings0.len(), 1);
        assert_eq!(openings0[0].start(), 0);
        assert_eq!(openings0[0].end(), i32::MAX);

        let openings1 = m.opening_time(BerthIndex::new(1));
        assert_eq!(openings1.len(), 1);
        assert_eq!(openings1[0].start(), 0);
        assert_eq!(openings1[0].end(), i32::MAX);

        // Processing times default to None
        for v in 0..3 {
            for b in 0..2 {
                let pt = m.processing_time(VesselIndex::new(v), BerthIndex::new(b));
                assert!(pt.is_none(), "Expected None at v={}, b={}", v, b);
            }
        }
    }

    // ModelBuilder setters
    #[test]
    fn test_model_builder_setters_arrival_and_departure() {
        let mut mb = ModelBuilder::<i32>::new(2, 1);
        mb.vessel_arrival(VesselIndex::new(0), 10)
            .vessel_latest_departure(VesselIndex::new(0), 100)
            .vessel_window(VesselIndex::new(1), 20, 200)
            .vessel_weight(VesselIndex::new(0), 7)
            .vessel_weight(VesselIndex::new(1), 3);

        let m = mb.build().unwrap();
        assert_eq!(m.arrival_time(VesselIndex::new(0)), 10);
        assert_eq!(m.latest_departure_time(VesselIndex::new(0)), 100);
        assert_eq!(m.arrival_time(VesselIndex::new(1)), 20);
        assert_eq!(m.latest_departure_time(VesselIndex::new(1)), 200);

        // Weights set via builder should be reflected in the model
        assert_eq!(m.vessel_weight(VesselIndex::new(0)), 7);
        assert_eq!(m.vessel_weight(VesselIndex::new(1)), 3);
        assert_eq!(m.vessel_weights(), &[7, 3]);
    }

    #[test]
    fn test_model_builder_set_opening_time_single_interval() {
        let mut mb = ModelBuilder::<i32>::new(1, 2);
        mb.berth_opening_time(BerthIndex::new(0), 5, 50)
            .berth_interval(BerthIndex::new(1), ClosedOpenInterval::new(10, 20));

        let m = mb.build().unwrap();
        let o0 = m.opening_time(BerthIndex::new(0));
        assert_eq!(o0.len(), 1);
        assert_eq!(o0[0], ClosedOpenInterval::new(5, 50));

        let o1 = m.opening_time(BerthIndex::new(1));
        assert_eq!(o1.len(), 1);
        assert_eq!(o1[0], ClosedOpenInterval::new(10, 20));
    }

    #[test]
    fn test_model_builder_set_opening_time_multiple_intervals() {
        let mut mb = ModelBuilder::<i32>::new(1, 1);
        mb.berth_opening_intervals(
            BerthIndex::new(0),
            vec![
                ClosedOpenInterval::new(0, 10),
                ClosedOpenInterval::new(20, 30),
            ],
        );
        let m = mb.build().unwrap();
        let o = m.opening_time(BerthIndex::new(0));
        assert_eq!(o.len(), 2);
        assert_eq!(o[0], ClosedOpenInterval::new(0, 10));
        assert_eq!(o[1], ClosedOpenInterval::new(20, 30));
    }

    #[test]
    fn test_model_builder_processing_time_and_unavailable() {
        let mut mb = ModelBuilder::<i32>::new(1, 2);
        mb.processing_time(VesselIndex::new(0), BerthIndex::new(0), 15)
            .processing_unavailable(VesselIndex::new(0), BerthIndex::new(1));

        let m = mb.build().unwrap();
        assert_eq!(
            m.processing_time(VesselIndex::new(0), BerthIndex::new(0))
                .unwrap(),
            15
        );
        assert!(
            m.processing_time(VesselIndex::new(0), BerthIndex::new(1))
                .is_none()
        );
    }

    #[test]
    fn test_model_builder_vessel_processing_times_bulk() {
        let mut mb = ModelBuilder::<i32>::new(2, 3);
        mb.vessel_processing_times(VesselIndex::new(0), vec![Some(5), None, Some(8)]);
        mb.vessel_processing_times(VesselIndex::new(1), vec![None, None, Some(1)]);
        let m = mb.build().unwrap();

        // Check v0 across berths
        assert_eq!(
            m.processing_time(VesselIndex::new(0), BerthIndex::new(0))
                .unwrap(),
            5
        );
        assert!(
            m.processing_time(VesselIndex::new(0), BerthIndex::new(1))
                .is_none()
        );
        assert_eq!(
            m.processing_time(VesselIndex::new(0), BerthIndex::new(2))
                .unwrap(),
            8
        );

        // Check v1 across berths
        assert!(
            m.processing_time(VesselIndex::new(1), BerthIndex::new(0))
                .is_none()
        );
        assert!(
            m.processing_time(VesselIndex::new(1), BerthIndex::new(1))
                .is_none()
        );
        assert_eq!(
            m.processing_time(VesselIndex::new(1), BerthIndex::new(2))
                .unwrap(),
            1
        );
    }

    // Model getters and allowed checks
    #[test]
    fn test_model_allowed_on_berth_true_false() {
        let mut mb = ModelBuilder::<i32>::new(1, 2);
        mb.processing_time(VesselIndex::new(0), BerthIndex::new(0), 3)
            .processing_unavailable(VesselIndex::new(0), BerthIndex::new(1));
        let m = mb.build().unwrap();

        assert!(m.allowed_on_berth(VesselIndex::new(0), BerthIndex::new(0)));
        assert!(!m.allowed_on_berth(VesselIndex::new(0), BerthIndex::new(1)));
    }

    #[test]
    fn test_model_opening_time_unchecked_matches_checked() {
        let m = make_small_model_i32(1, 1);
        unsafe {
            let u = m.opening_time_unchecked(BerthIndex::new(0));
            let c = m.opening_time(BerthIndex::new(0));
            assert_eq!(u, c);
        }
    }

    #[test]
    fn test_model_arrival_and_departure_unchecked_matches_checked() {
        let mut mb = ModelBuilder::<i32>::new(1, 1);
        mb.vessel_window(VesselIndex::new(0), 10, 200)
            .vessel_weight(VesselIndex::new(0), 9);
        let m = mb.build().unwrap();
        unsafe {
            assert_eq!(
                m.arrival_time_unchecked(VesselIndex::new(0)),
                m.arrival_time(VesselIndex::new(0))
            );
            assert_eq!(
                m.latest_departure_time_unchecked(VesselIndex::new(0)),
                m.latest_departure_time(VesselIndex::new(0))
            );
        }
        // Checked weight equals the slice value
        assert_eq!(m.vessel_weight(VesselIndex::new(0)), m.vessel_weights()[0]);
    }

    #[test]
    fn test_model_processing_time_unchecked_matches_checked() {
        let mut mb = ModelBuilder::<i32>::new(1, 1);
        mb.processing_time(VesselIndex::new(0), BerthIndex::new(0), 11);
        let m = mb.build().unwrap();
        unsafe {
            assert_eq!(
                m.processing_time_unchecked(VesselIndex::new(0), BerthIndex::new(0))
                    .unwrap(),
                m.processing_time(VesselIndex::new(0), BerthIndex::new(0))
                    .unwrap()
            );
        }
    }

    // Shortest processing time computation
    #[test]
    fn test_shortest_processing_time_computation_simple() {
        let mut mb = ModelBuilder::<i32>::new(1, 3);
        mb.vessel_processing_times(VesselIndex::new(0), vec![Some(7), Some(5), None]);
        let m = mb.build().unwrap();
        assert_eq!(m.shortest_processing_time(VesselIndex::new(0)).unwrap(), 5);
        // Weights do not affect shortest processing time computation
        assert_eq!(m.vessel_weights().len(), m.num_vessels());
    }

    #[test]
    fn test_shortest_processing_time_all_none_is_none() {
        let mut mb = ModelBuilder::<i32>::new(2, 2);
        mb.vessel_processing_times(VesselIndex::new(0), vec![None, None]);
        mb.vessel_processing_times(VesselIndex::new(1), vec![None, None]);
        let m = mb.build().unwrap();
        assert!(m.shortest_processing_time(VesselIndex::new(0)).is_none());
        assert!(m.shortest_processing_time(VesselIndex::new(1)).is_none());
        // Vessel weights slice exists for each vessel
        assert_eq!(m.vessel_weights().len(), 2);
    }

    #[test]
    fn test_shortest_processing_time_mixed_values() {
        let mut mb = ModelBuilder::<i32>::new(2, 3);
        mb.vessel_processing_times(VesselIndex::new(0), vec![Some(9), Some(1), Some(5)]);
        mb.vessel_processing_times(VesselIndex::new(1), vec![None, Some(10), Some(3)]);
        let m = mb.build().unwrap();

        assert_eq!(m.shortest_processing_time(VesselIndex::new(0)).unwrap(), 1);
        assert_eq!(m.shortest_processing_time(VesselIndex::new(1)).unwrap(), 3);
        // Ensure weights do not modify result ordering
        assert_eq!(m.vessel_weights().len(), 2);
    }

    // Validation: Empty interval
    #[test]
    fn test_build_fails_on_empty_interval() {
        let mut mb = ModelBuilder::<i32>::new(1, 1);
        // Empty: start == end
        mb.berth_interval(BerthIndex::new(0), ClosedOpenInterval::new(10, 10));

        let err = mb.build().err().expect("Expected error");
        match err {
            ModelBuildError::EmptyBerthOpening(e) => {
                assert_eq!(e.index, 0);
                assert_eq!(e.interval, ClosedOpenInterval::new(10, 10));
                assert_eq!(
                    format!("{}", e),
                    "Berth 0 has an empty opening time interval: [10, 10)"
                );
            }
            _ => panic!("Wrong error variant: {:?}", err),
        }
    }

    #[test]
    fn test_build_fails_on_unsorted_intervals() {
        let mut mb = ModelBuilder::<i32>::new(1, 1);
        // Intervals out of order by start
        mb.berth_opening_intervals(
            BerthIndex::new(0),
            vec![
                ClosedOpenInterval::new(20, 30),
                ClosedOpenInterval::new(0, 10),
            ],
        );

        let err = mb.build().err().expect("Expected error");
        match err {
            ModelBuildError::BerthOpeningsUnsorted(e) => {
                assert_eq!(e.berth_index, 0);
                assert_eq!(format!("{}", e), "Berth openings in berth 0 are not sorted");
            }
            _ => panic!("Wrong error variant: {:?}", err),
        }
    }

    #[test]
    fn test_build_fails_on_overlapping_intervals() {
        let mut mb = ModelBuilder::<i32>::new(1, 1);
        // Overlapping: [0, 10) and [5, 15)
        mb.berth_opening_intervals(
            BerthIndex::new(0),
            vec![
                ClosedOpenInterval::new(0, 10),
                ClosedOpenInterval::new(5, 15),
            ],
        );

        let err = mb.build().err().expect("Expected error");
        match err {
            ModelBuildError::BerthOpeningsOverlap(e) => {
                assert_eq!(e.berth_index, 0);
                assert_eq!(e.first, ClosedOpenInterval::new(0, 10));
                assert_eq!(e.second, ClosedOpenInterval::new(5, 15));
                assert!(format!("{}", e).contains("overlap or are adjacent"));
            }
            _ => panic!("Wrong error variant: {:?}", err),
        }
    }

    #[test]
    fn test_build_fails_on_adjacent_intervals() {
        let mut mb = ModelBuilder::<i32>::new(1, 1);
        // Adjacent: [0, 10) and [10, 20) should be considered invalid by validation
        mb.berth_opening_intervals(
            BerthIndex::new(0),
            vec![
                ClosedOpenInterval::new(0, 10),
                ClosedOpenInterval::new(10, 20),
            ],
        );

        let err = mb.build().err().expect("Expected error");
        match err {
            ModelBuildError::BerthOpeningsOverlap(e) => {
                assert_eq!(e.berth_index, 0);
                assert_eq!(e.first, ClosedOpenInterval::new(0, 10));
                assert_eq!(e.second, ClosedOpenInterval::new(10, 20));
            }
            _ => panic!("Wrong error variant: {:?}", err),
        }
    }

    #[test]
    fn test_push_berth_interval_appends_and_validates_overlap() {
        let mut mb = ModelBuilder::<i32>::new(1, 1);
        // Use push_berth_interval to append multiple intervals to the same berth.
        mb.push_berth_interval(BerthIndex::new(0), ClosedOpenInterval::new(0, 10));
        mb.push_berth_interval(BerthIndex::new(0), ClosedOpenInterval::new(5, 15)); // overlap

        // Build should fail due to overlap
        let err = mb
            .build()
            .err()
            .expect("Expected error due to overlapping intervals via push");
        match err {
            ModelBuildError::BerthOpeningsOverlap(e) => {
                assert_eq!(e.berth_index, 0);
                assert_eq!(e.first, ClosedOpenInterval::new(0, 10));
                assert_eq!(e.second, ClosedOpenInterval::new(5, 15));
            }
            _ => panic!("Wrong error variant: {:?}", err),
        }
    }

    // Edge cases
    #[test]
    fn test_builder_with_zero_vessels() {
        let mb = ModelBuilder::<i32>::new(0, 2);
        let m = mb.build().unwrap();
        assert_eq!(m.num_vessels(), 0);
        assert_eq!(m.num_berths(), 2);
        assert_eq!(m.arrival_times().len(), 0);
        assert_eq!(m.latest_departure_times().len(), 0);
        assert_eq!(m.shortest_processing_times().len(), 0);
        assert_eq!(m.vessel_weights().len(), 0);
        // Opening times still present per berth
        assert_eq!(m.opening_times().len(), 2);
    }

    #[test]
    fn test_builder_with_zero_berths() {
        let mb = ModelBuilder::<i32>::new(2, 0);
        let m = mb.build().unwrap();
        assert_eq!(m.num_vessels(), 2);
        assert_eq!(m.num_berths(), 0);
        assert_eq!(m.processing_times().len(), 0);
        assert_eq!(m.opening_times().len(), 0);
        // Shortest processing times should be computed as None per vessel due to no berths
        assert_eq!(m.shortest_processing_times().len(), 2);
        assert!(m.shortest_processing_time(VesselIndex::new(0)).is_none());
        assert!(m.shortest_processing_time(VesselIndex::new(1)).is_none());
        // Vessel weights present per vessel, default to 1
        assert_eq!(m.vessel_weights().len(), 2);
        assert_eq!(m.vessel_weights(), &[1, 1]);
    }

    #[test]
    fn test_model_display_and_debug() {
        let m = make_small_model_i32(2, 3);
        let s = format!("{}", m);
        assert!(s.contains("Model(num_vessels: 2, num_berths: 3)"));
        let d = format!("{:?}", m);
        assert!(d.contains("Model"));
        assert!(d.contains("arrival_times"));
        assert!(d.contains("latest_departure_times"));
        assert!(d.contains("processing_times"));
        assert!(d.contains("opening_times"));
        assert!(d.contains("shortest_processing_times"));
        // Debug should include vessel_weights
        assert!(d.contains("vessel_weights"));
        // Display may not include an explicit "vessel_weights" label; do not assert it
    }

    #[test]
    fn test_solution_new_and_accessors() {
        let berths = vec![BerthIndex::new(0), BerthIndex::new(2), BerthIndex::new(1)];
        let starts = vec![10i32, 25, 7];
        let sol = Solution::<i32>::new(berths.clone(), starts.clone());

        assert_eq!(sol.vessel_berths().len(), 3);
        assert_eq!(sol.vessel_start_times().len(), 3);

        assert_eq!(sol.vessel_berths()[0].get(), 0);
        assert_eq!(sol.vessel_berths()[1].get(), 2);
        assert_eq!(sol.vessel_berths()[2].get(), 1);

        assert_eq!(sol.vessel_start_times()[0], 10);
        assert_eq!(sol.vessel_start_times()[1], 25);
        assert_eq!(sol.vessel_start_times()[2], 7);
    }

    #[test]
    fn test_solution_display_non_empty() {
        let berths = vec![BerthIndex::new(3), BerthIndex::new(1)];
        let starts = vec![100i32, 50];
        let sol = Solution::<i32>::new(berths, starts);

        let s = format!("{}", sol);
        // Header and count
        assert!(s.contains("Solution: 2 vessel(s)"));
        // Each vessel line
        assert!(s.contains("Vessel 0"));
        assert!(s.contains("Berth 3"));
        assert!(s.contains("Start 100"));
        assert!(s.contains("Vessel 1"));
        assert!(s.contains("Berth 1"));
        assert!(s.contains("Start 50"));
        // It should have newline-separated lines; basic sanity
        assert!(s.lines().count() >= 3);
    }

    #[test]
    fn test_solution_display_empty() {
        let sol = Solution::<i32>::new(vec![], vec![]);
        let s = format!("{}", sol);
        assert_eq!(s, "Solution: no vessels assigned");
    }

    #[test]
    fn test_solution_display_mismatched_lengths_reports_inconsistency() {
        // Mismatch: 2 berths, 1 start
        let sol = Solution::<i32>::new(vec![BerthIndex::new(0), BerthIndex::new(1)], vec![10]);
        let s = format!("{}", sol);
        assert!(
            s.contains("inconsistent lengths"),
            "Display should report inconsistent lengths, got: {}",
            s
        );
        // It should still render up to min length
        assert!(s.contains("Solution: 1 vessel(s)"));
        assert!(s.contains("Vessel 0"));
    }

    #[test]
    fn test_solution_debug_includes_fields() {
        let sol = Solution::<i32>::new(vec![BerthIndex::new(2)], vec![33]);
        let d = format!("{:?}", sol);
        assert!(d.contains("Solution"));
        assert!(d.contains("vessel_berths"));
        assert!(d.contains("vessel_start_times"));
        // Ensure content is displayed
        assert!(d.contains("2"));
        assert!(d.contains("33"));
    }

    #[test]
    fn test_solution_indices_align_with_lengths() {
        // Ensure that the nth position in berths aligns with nth start time semantically
        let sol = Solution::<i32>::new(
            vec![BerthIndex::new(5), BerthIndex::new(0), BerthIndex::new(4)],
            vec![12, 99, 7],
        );

        // Vessel 0
        assert_eq!(sol.vessel_berths()[0].get(), 5);
        assert_eq!(sol.vessel_start_times()[0], 12);
        // Vessel 1
        assert_eq!(sol.vessel_berths()[1].get(), 0);
        assert_eq!(sol.vessel_start_times()[1], 99);
        // Vessel 2
        assert_eq!(sol.vessel_berths()[2].get(), 4);
        assert_eq!(sol.vessel_start_times()[2], 7);
    }

    #[test]
    fn test_solution_large_values_and_signed() {
        // Verify display with large and negative times
        let sol = Solution::<i64>::new(vec![BerthIndex::new(9)], vec![-12345]);
        let s = format!("{}", sol);
        assert!(s.contains("Solution: 1 vessel(s)"));
        assert!(s.contains("Berth 9"));
        assert!(s.contains("Start -12345"));
    }

    #[test]
    fn test_solution_zero_vessels_is_consistent() {
        let sol = Solution::<i32>::new(Vec::new(), Vec::new());
        assert_eq!(sol.vessel_berths().len(), 0);
        assert_eq!(sol.vessel_start_times().len(), 0);
        assert_eq!(format!("{}", sol), "Solution: no vessels assigned");
        assert!(format!("{:?}", sol).contains("Solution"));
    }

    #[test]
    fn test_solution_num_vessels_equal_lengths() {
        let sol = Solution::<i32>::new(
            vec![BerthIndex::new(0), BerthIndex::new(1), BerthIndex::new(2)],
            vec![10, 20, 30],
        );
        assert_eq!(sol.num_vessels(), 3);
    }

    #[test]
    fn test_solution_num_vessels_mismatched_lengths_min() {
        // berths shorter -> min is 2
        let sol_short_berths = Solution::<i32>::new(
            vec![BerthIndex::new(1), BerthIndex::new(3)],
            vec![5, 6, 7, 8],
        );
        assert_eq!(sol_short_berths.num_vessels(), 2);

        // starts shorter -> min is 1
        let sol_short_starts =
            Solution::<i32>::new(vec![BerthIndex::new(0), BerthIndex::new(2)], vec![100]);
        assert_eq!(sol_short_starts.num_vessels(), 1);

        // both empty -> min is 0
        let sol_empty = Solution::<i32>::new(vec![], vec![]);
        assert_eq!(sol_empty.num_vessels(), 0);
    }

    #[test]
    fn test_solution_vessel_assignment_in_range() {
        let sol = Solution::<i32>::new(
            vec![BerthIndex::new(5), BerthIndex::new(0), BerthIndex::new(4)],
            vec![12, 99, 7],
        );
        let a0 = sol.vessel_assignment(0).expect("assignment 0");
        assert_eq!(a0.0.get(), 5);
        assert_eq!(a0.1, 12);

        let a1 = sol.vessel_assignment(1).expect("assignment 1");
        assert_eq!(a1.0.get(), 0);
        assert_eq!(a1.1, 99);

        let a2 = sol.vessel_assignment(2).expect("assignment 2");
        assert_eq!(a2.0.get(), 4);
        assert_eq!(a2.1, 7);
    }

    #[test]
    fn test_solution_vessel_assignment_out_of_range() {
        let sol = Solution::<i32>::new(vec![BerthIndex::new(2)], vec![33]);
        assert!(sol.vessel_assignment(1).is_none());
        assert!(sol.vessel_assignment(100).is_none());
    }

    #[test]
    fn test_solution_vessel_assignment_respects_min_length_on_mismatch() {
        // Mismatch: 3 berths, 2 starts -> num_vessels is 2
        let sol = Solution::<i32>::new(
            vec![BerthIndex::new(1), BerthIndex::new(2), BerthIndex::new(3)],
            vec![10, 20],
        );
        assert_eq!(sol.num_vessels(), 2);
        assert!(sol.vessel_assignment(0).is_some());
        assert!(sol.vessel_assignment(1).is_some());
        assert!(sol.vessel_assignment(2).is_none()); // out of range due to min length
    }
}
