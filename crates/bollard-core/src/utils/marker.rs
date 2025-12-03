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

/// A brand type to create nominal types.
///
/// This type is used to create nominal types by associating a lifetime `'x` with the type.
/// This prevents accidental mixing of types that are meant to be distinct, even if they have
/// the same underlying representation.
#[repr(transparent)]
#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Brand<'x>(std::marker::PhantomData<fn(&'x ()) -> &'x ()>);

impl<'x> Brand<'x> {
    /// Creates a new `Brand` instance.
    #[inline(always)]
    pub const fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<'x> From<()> for Brand<'x> {
    #[inline(always)]
    fn from(_: ()) -> Self {
        Self::new()
    }
}

impl<'x> std::fmt::Debug for Brand<'x> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Brand")
    }
}

impl<'x> std::fmt::Display for Brand<'x> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Brand")
    }
}

#[cfg(test)]
mod tests {
    use super::Brand;

    #[test]
    fn test_brand_correct_display() {
        let _brand: Brand<'static> = Brand::new();
        let _brand_from: Brand<'static> = Brand::from(());
        assert_eq!(format!("{}", _brand), "Brand");
    }
}
