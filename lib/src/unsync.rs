//! Unsynchronized plain old data vector and image for fast access from multiple threads

use std::cell::UnsafeCell;

pub struct UnsyncVec<T: Copy>(UnsafeCell<Vec<T>>);
pub struct UnsyncRgbaImage(UnsafeCell<image::RgbaImage>);

impl<T: Copy> UnsyncVec<T> {
    pub fn new(v: Vec<T>) -> Self {
        Self(UnsafeCell::new(v))
    }

    pub unsafe fn assign_at(&self, idx: usize, value: T) {
        self.0.get().as_mut().unwrap()[idx] = value;
    }

    pub fn as_ref(&self) -> &[T] {
        unsafe { self.0.get().as_ref() }.unwrap()
    }
}

impl UnsyncRgbaImage {
    pub fn new(img: image::RgbaImage) -> Self {
        Self(UnsafeCell::new(img))
    }

    pub fn as_ref(&self) -> &image::RgbaImage {
        unsafe { self.0.get().as_ref() }.unwrap()
    }

    pub fn into_inner(self) -> image::RgbaImage {
        self.0.into_inner()
    }

    pub fn put_pixel(&self, x: u32, y: u32, pixel: image::Rgba<u8>) {
        unsafe { self.0.get().as_mut() }
            .unwrap()
            .put_pixel(x, y, pixel)
    }
}

unsafe impl<T: Copy> Sync for UnsyncVec<T> {}
unsafe impl Sync for UnsyncRgbaImage {}
