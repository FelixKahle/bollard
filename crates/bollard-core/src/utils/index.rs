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

/// A trait to tag typed indices with a name for debugging and display purposes.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::utils::index::TypedIndexTag;
///
/// #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
/// struct MyTag;
///
/// impl TypedIndexTag for MyTag {
///     const NAME: &'static str = "MyIndex";
/// }
/// ```
pub trait TypedIndexTag: Clone {
    const NAME: &'static str;
}

/// A strongly typed index that is associated with a specific tag type `T`.
///
/// This struct wraps a `usize` index and uses a phantom type parameter `T`
/// to provide type safety and prevent mixing indices of different types.
///
/// # Examples
///
/// ```rust
/// # use bollard_core::utils::index::{TypedIndex, TypedIndexTag};
///
/// #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
/// struct MyTag;
///
/// impl TypedIndexTag for MyTag {
///    const NAME: &'static str = "MyIndex";
/// }
///
/// type MyIndex = TypedIndex<MyTag>;
///
/// let index = MyIndex::new(5);
/// assert_eq!(index.get(), 5);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypedIndex<T> {
    index: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T> TypedIndex<T> {
    /// Creates a new `TypedIndex` with the given `usize` index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::utils::index::{TypedIndex, TypedIndexTag};
    ///
    /// #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
    /// struct MyTag;
    ///
    /// impl TypedIndexTag for MyTag {
    ///    const NAME: &'static str = "MyIndex";
    /// }
    ///
    /// type MyIndex = TypedIndex<MyTag>;
    ///
    /// let index = MyIndex::new(5);
    /// assert_eq!(index.get(), 5);
    /// ```
    pub const fn new(index: usize) -> Self {
        Self {
            index,
            _marker: std::marker::PhantomData,
        }
    }

    /// Returns the underlying `usize` index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bollard_core::utils::index::{TypedIndex, TypedIndexTag};
    ///
    /// #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
    /// struct MyTag;
    ///
    /// impl TypedIndexTag for MyTag {
    ///    const NAME: &'static str = "MyIndex";
    /// }
    ///
    /// type MyIndex = TypedIndex<MyTag>;
    ///
    /// let index = MyIndex::new(5);
    /// assert_eq!(index.get(), 5);
    /// ```
    pub const fn get(&self) -> usize {
        self.index
    }
}

impl<T> std::fmt::Debug for TypedIndex<T>
where
    T: TypedIndexTag,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", T::NAME, self.index)
    }
}

impl<T> std::fmt::Display for TypedIndex<T>
where
    T: TypedIndexTag,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", T::NAME, self.index)
    }
}

impl<T> From<usize> for TypedIndex<T> {
    fn from(index: usize) -> Self {
        Self::new(index)
    }
}

impl<T> From<TypedIndex<T>> for usize {
    fn from(typed_index: TypedIndex<T>) -> Self {
        typed_index.index
    }
}

macro_rules! impl_index_op {
    ($trait_name:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $op:tt) => {
        impl<T> std::ops::$trait_name<usize> for TypedIndex<T> {
            type Output = Self;

            fn $method(self, rhs: usize) -> Self::Output {
                Self::new(self.index $op rhs)
            }
        }
        impl<T> std::ops::$assign_trait<usize> for TypedIndex<T> {
            fn $assign_method(&mut self, rhs: usize) {
                self.index = self.index $op rhs;
            }
        }
    };
}

impl_index_op!(Add, add, AddAssign, add_assign, +);
impl_index_op!(Sub, sub, SubAssign, sub_assign, -);
impl_index_op!(Mul, mul, MulAssign, mul_assign, *);
impl_index_op!(Div, div, DivAssign, div_assign, /);
impl_index_op!(Rem, rem, RemAssign, rem_assign, %);

#[cfg(test)]
mod tests {
    use super::*;

    // Define a dummy tag for testing purposes
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
    struct TestTag;

    impl TypedIndexTag for TestTag {
        const NAME: &'static str = "TestIdx";
    }

    // Type alias for convenience inside tests
    type TestIndex = TypedIndex<TestTag>;

    #[test]
    fn test_new_and_get() {
        let idx = TestIndex::new(10);
        assert_eq!(idx.get(), 10);
    }

    #[test]
    fn test_conversions() {
        // From usize
        let idx: TestIndex = 42.into();
        assert_eq!(idx.get(), 42);

        // Into usize
        let val: usize = idx.into();
        assert_eq!(val, 42);
    }

    #[test]
    fn test_debug_and_display() {
        let idx = TestIndex::new(7);
        // Uses the NAME const from the trait
        assert_eq!(format!("{}", idx), "TestIdx(7)");
        assert_eq!(format!("{:?}", idx), "TestIdx(7)");
    }

    #[test]
    fn test_arithmetic_ops() {
        let idx = TestIndex::new(10);

        // Test operators (consuming self/copy)
        assert_eq!((idx + 5).get(), 15);
        assert_eq!((idx - 5).get(), 5);
        assert_eq!((idx * 2).get(), 20);
        assert_eq!((idx / 2).get(), 5);
        assert_eq!((idx % 3).get(), 1);
    }

    #[test]
    fn test_assignment_ops() {
        let mut idx = TestIndex::new(10);

        idx += 5;
        assert_eq!(idx.get(), 15);

        idx -= 5;
        assert_eq!(idx.get(), 10);

        idx *= 2;
        assert_eq!(idx.get(), 20);

        idx /= 4;
        assert_eq!(idx.get(), 5);

        idx %= 2;
        assert_eq!(idx.get(), 1);
    }
}
