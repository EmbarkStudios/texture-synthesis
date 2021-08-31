// BEGIN - Embark standard lints v0.4
// do not change or add/remove here, but one can add exceptions after this section
// for more info see: <https://github.com/EmbarkStudios/rust-ecosystem/issues/59>
#![deny(unsafe_code)]
#![warn(
    clippy::all,
    clippy::await_holding_lock,
    clippy::char_lit_as_u8,
    clippy::checked_conversions,
    clippy::dbg_macro,
    clippy::debug_assert_with_mut_call,
    clippy::doc_markdown,
    clippy::empty_enum,
    clippy::enum_glob_use,
    clippy::exit,
    clippy::expl_impl_clone_on_copy,
    clippy::explicit_deref_methods,
    clippy::explicit_into_iter_loop,
    clippy::fallible_impl_from,
    clippy::filter_map_next,
    clippy::float_cmp_const,
    clippy::fn_params_excessive_bools,
    clippy::if_let_mutex,
    clippy::implicit_clone,
    clippy::imprecise_flops,
    clippy::inefficient_to_string,
    clippy::invalid_upcast_comparisons,
    clippy::large_types_passed_by_value,
    clippy::let_unit_value,
    clippy::linkedlist,
    clippy::lossy_float_literal,
    clippy::macro_use_imports,
    clippy::manual_ok_or,
    clippy::map_err_ignore,
    clippy::map_flatten,
    clippy::map_unwrap_or,
    clippy::match_on_vec_items,
    clippy::match_same_arms,
    clippy::match_wildcard_for_single_variants,
    clippy::mem_forget,
    clippy::mismatched_target_os,
    clippy::mut_mut,
    clippy::mutex_integer,
    clippy::needless_borrow,
    clippy::needless_continue,
    clippy::option_option,
    clippy::path_buf_push_overwrite,
    clippy::ptr_as_ptr,
    clippy::ref_option_ref,
    clippy::rest_pat_in_fully_bound_structs,
    clippy::same_functions_in_if_condition,
    clippy::semicolon_if_nothing_returned,
    clippy::string_add_assign,
    clippy::string_add,
    clippy::string_lit_as_bytes,
    clippy::string_to_string,
    clippy::todo,
    clippy::trait_duplication_in_bounds,
    clippy::unimplemented,
    clippy::unnested_or_patterns,
    clippy::unused_self,
    clippy::useless_transmute,
    clippy::verbose_file_reads,
    clippy::zero_sized_map_values,
    future_incompatible,
    nonstandard_style,
    rust_2018_idioms
)]
// END - Embark standard lints v0.4
#![allow(unsafe_code)]

//! `texture-synthesis` is a light API for Multiresolution Stochastic Texture Synthesis,
//! a non-parametric example-based algorithm for image generation.
//!
//! First, you build a `Session` via a `SessionBuilder`, which follows the builder pattern. Calling
//! `build` on the `SessionBuilder` loads all of the input images and checks for various errors.
//!
//! `Session` has a `run()` method that takes all of the parameters and inputs added in the session
//! builder to generated an image, which is returned as a `GeneratedImage`.
//!
//! You can save, stream, or inspect the image from `GeneratedImage`.
//!
//! ## Features
//!
//! 1. Single example generation
//! 2. Multi example generation
//! 3. Guided synthesis
//! 4. Style transfer
//! 5. Inpainting
//! 6. Tiling textures
//!
//! Please, refer to the examples folder in the [repository](https://github.com/EmbarkStudios/texture-synthesis) for the features usage examples.
//!
//! ## Usage
//! Session follows a "builder pattern" for defining parameters, meaning you chain functions together.
//!
//! ```no_run
//! // Create a new session with default parameters
//! let session = texture_synthesis::Session::builder()
//!     // Set some parameters
//!     .seed(10)
//!     .nearest_neighbors(20)
//!     // Specify example images
//!     .add_example(&"imgs/1.jpg")
//!     // Build the session
//!     .build().expect("failed to build session");
//!
//! // Generate a new image
//! let generated_img = session.run(None);
//!
//! // Save the generated image to disk
//! generated_img.save("my_generated_img.jpg").expect("failed to save generated image");
//! ```
mod errors;
mod img_pyramid;
use img_pyramid::*;
mod utils;
use utils::*;
mod ms;
use ms::*;
pub mod session;
mod unsync;

pub use image;
use std::path::Path;

pub use errors::Error;
pub use session::{Session, SessionBuilder};
pub use utils::{load_dynamic_image, ChannelMask, ImageSource};

/// Simple dimensions struct
#[derive(Copy, Clone)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Dims {
    pub width: u32,
    pub height: u32,
}

impl Dims {
    pub fn square(size: u32) -> Self {
        Self {
            width: size,
            height: size,
        }
    }
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

/// A buffer of transforms that were used to generate an image from a set of
/// examples, which can be applied to a different set of input images to get
/// a different output image.
pub struct CoordinateTransform {
    buffer: Vec<u32>,
    pub output_size: Dims,
    original_maps: Vec<Dims>,
}

const TRANSFORM_MAGIC: u32 = 0x1234_0001;

impl<'a> CoordinateTransform {
    /// Applies the coordinate transformation from new source images. This
    /// method will fail if the the provided source images aren't the same
    /// number of example images that generated the transform.
    ///
    /// The input images are automatically resized to the dimensions of the
    /// original example images used in the generation of this coordinate
    /// transform
    pub fn apply<E, I>(&self, source: I) -> Result<image::RgbaImage, Error>
    where
        I: IntoIterator<Item = E>,
        E: Into<ImageSource<'a>>,
    {
        let ref_maps: Vec<image::RgbaImage> = source
            .into_iter()
            .zip(self.original_maps.iter())
            .map(|(is, dims)| load_image(is.into(), Some(*dims)))
            .collect::<Result<Vec<_>, Error>>()?;

        // Ensure the number of inputs match the number in that generated this
        // transform, otherwise we would get weird results
        if ref_maps.len() != self.original_maps.len() {
            return Err(Error::MapsCountMismatch(
                ref_maps.len() as u32,
                self.original_maps.len() as u32,
            ));
        }

        let mut img = image::RgbaImage::new(self.output_size.width, self.output_size.height);

        // Populate with pixels from ref maps
        for (i, pix) in img.pixels_mut().enumerate() {
            let x = self.buffer[i * 3];
            let y = self.buffer[i * 3 + 1];
            let map = self.buffer[i * 3 + 2];

            *pix = *ref_maps[map as usize].get_pixel(x, y);
        }

        Ok(img)
    }

    pub fn write<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<usize> {
        use std::mem;
        let mut written = 0;

        // Sanity check that that buffer length corresponds correctly with the
        // supposed dimensions
        if self.buffer.len()
            != self.output_size.width as usize * self.output_size.height as usize * 3
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "buffer length doesn't match dimensions",
            ));
        }

        let header = [
            TRANSFORM_MAGIC,
            self.output_size.width,
            self.output_size.height,
            self.original_maps.len() as u32,
        ];

        fn cast(ina: &[u32]) -> &[u8] {
            unsafe {
                let p = ina.as_ptr();
                let len = ina.len();

                std::slice::from_raw_parts(p.cast::<u8>(), len * mem::size_of::<u32>())
            }
        }

        w.write_all(cast(&header))?;
        written += mem::size_of_val(&header);

        for om in &self.original_maps {
            let dims = [om.width, om.height];
            w.write_all(cast(&dims))?;
            written += mem::size_of_val(&dims);
        }

        w.write_all(cast(&self.buffer))?;
        written += 4 * self.buffer.len();

        Ok(written)
    }

    pub fn read<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        use std::{
            io::{Error, ErrorKind, Read},
            mem,
        };

        fn do_read<R: Read>(r: &mut R, buf: &mut [u32]) -> std::io::Result<()> {
            unsafe {
                let p = buf.as_mut_ptr();
                let len = buf.len();

                let mut slice =
                    std::slice::from_raw_parts_mut(p.cast::<u8>(), len * mem::size_of::<u32>());

                r.read(&mut slice).map(|_| ())
            }
        }

        let mut magic = [0u32];
        do_read(r, &mut magic)?;

        if magic[0] >> 16 != 0x1234 {
            return Err(Error::new(ErrorKind::InvalidData, "invalid magic"));
        }

        let (output_size, original_maps) = match magic[0] & 0x0000_ffff {
            0x1 => {
                let mut header = [0u32; 3];
                do_read(r, &mut header)?;

                let mut omaps = Vec::with_capacity(header[2] as usize);
                for _ in 0..header[2] {
                    let mut dims = [0u32; 2];
                    do_read(r, &mut dims)?;
                    omaps.push(Dims {
                        width: dims[0],
                        height: dims[1],
                    });
                }

                (
                    Dims {
                        width: header[0],
                        height: header[1],
                    },
                    omaps,
                )
            }
            _ => return Err(Error::new(ErrorKind::InvalidData, "invalid version")),
        };

        let buffer = unsafe {
            let len = output_size.width as usize * output_size.height as usize * 3;
            let mut buffer = Vec::with_capacity(len);
            buffer.set_len(len);

            do_read(r, &mut buffer)?;
            buffer
        };

        Ok(Self {
            buffer,
            output_size,
            original_maps,
        })
    }
}

struct Parameters {
    tiling_mode: bool,
    nearest_neighbors: u32,
    random_sample_locations: u64,
    cauchy_dispersion: f32,
    backtrack_percent: f32,
    backtrack_stages: u32,
    resize_input: Option<Dims>,
    output_size: Dims,
    guide_alpha: f32,
    random_resolve: Option<u64>,
    max_thread_count: Option<usize>,
    seed: u64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            tiling_mode: false,
            nearest_neighbors: 50,
            random_sample_locations: 50,
            cauchy_dispersion: 1.0,
            backtrack_percent: 0.5,
            backtrack_stages: 5,
            resize_input: None,
            output_size: Dims::square(500),
            guide_alpha: 0.8,
            random_resolve: None,
            max_thread_count: None,
            seed: 0,
        }
    }
}

impl Parameters {
    fn to_generator_params(&self) -> GeneratorParams {
        GeneratorParams {
            nearest_neighbors: self.nearest_neighbors,
            random_sample_locations: self.random_sample_locations,
            cauchy_dispersion: self.cauchy_dispersion,
            p: self.backtrack_percent,
            p_stages: self.backtrack_stages as i32,
            seed: self.seed,
            alpha: self.guide_alpha,
            max_thread_count: self.max_thread_count.unwrap_or_else(num_cpus::get),
            tiling_mode: self.tiling_mode,
        }
    }
}

/// An image generated by a `Session::run()`
pub struct GeneratedImage {
    inner: ms::Generator,
}

impl GeneratedImage {
    /// Saves the generated image to the specified path
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Error> {
        let path = path.as_ref();
        if let Some(parent_path) = path.parent() {
            std::fs::create_dir_all(&parent_path)?;
        }

        self.inner.color_map.as_ref().save(&path)?;
        Ok(())
    }

    /// Writes the generated image to the specified stream
    pub fn write<W: std::io::Write>(
        self,
        writer: &mut W,
        fmt: image::ImageOutputFormat,
    ) -> Result<(), Error> {
        let dyn_img = self.into_image();
        Ok(dyn_img.write_to(writer, fmt)?)
    }

    /// Saves debug information such as copied patches ids, map ids (if you have
    /// multi example generation) and a map indicating generated pixels the
    /// generator was "uncertain" of.
    pub fn save_debug<P: AsRef<Path>>(&self, dir: P) -> Result<(), Error> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(&dir)?;

        self.inner
            .get_uncertainty_map()
            .save(&dir.join("uncertainty.png"))?;
        let id_maps = self.inner.get_id_maps();
        id_maps[0].save(&dir.join("patch_id.png"))?;
        id_maps[1].save(&dir.join("map_id.png"))?;

        Ok(())
    }

    /// Get the coordinate transform of this generated image, which can be
    /// applied to new example images to get a different output image.
    ///
    /// ```no_run
    /// use texture_synthesis as ts;
    ///
    /// // create a new session
    /// let texsynth = ts::Session::builder()
    ///     //load a single example image
    ///     .add_example(&"imgs/1.jpg")
    ///     .build().unwrap();
    ///
    /// // generate an image
    /// let generated = texsynth.run(None);
    ///
    /// // now we can repeat the same transformation on a different image
    /// let repeated_transform_image = generated
    ///     .get_coordinate_transform()
    ///     .apply(&["imgs/2.jpg"]);
    /// ```
    pub fn get_coordinate_transform(&self) -> CoordinateTransform {
        self.inner.get_coord_transform()
    }

    /// Returns the generated output image
    pub fn into_image(self) -> image::DynamicImage {
        image::DynamicImage::ImageRgba8(self.inner.color_map.into_inner())
    }
}

impl AsRef<image::RgbaImage> for GeneratedImage {
    fn as_ref(&self) -> &image::RgbaImage {
        self.inner.color_map.as_ref()
    }
}

/// Method used for sampling an example image.
pub enum GenericSampleMethod<Img> {
    /// All pixels in the example image can be sampled.
    All,
    /// No pixels in the example image will be sampled.
    Ignore,
    /// Pixels are selectively sampled based on an image.
    Image(Img),
}

pub type SampleMethod<'a> = GenericSampleMethod<ImageSource<'a>>;
pub type SamplingMethod = GenericSampleMethod<image::RgbaImage>;

impl<Img> GenericSampleMethod<Img> {
    #[inline]
    fn is_ignore(&self) -> bool {
        matches!(self, Self::Ignore)
    }
}

impl<'a, IS> From<IS> for SampleMethod<'a>
where
    IS: Into<ImageSource<'a>>,
{
    fn from(is: IS) -> Self {
        SampleMethod::Image(is.into())
    }
}

/// A builder for an `Example`
pub struct ExampleBuilder<'a> {
    img: ImageSource<'a>,
    guide: Option<ImageSource<'a>>,
    sample_method: SampleMethod<'a>,
}

impl<'a> ExampleBuilder<'a> {
    /// Creates a new example builder from the specified image source
    pub fn new<I: Into<ImageSource<'a>>>(img: I) -> Self {
        Self {
            img: img.into(),
            guide: None,
            sample_method: SampleMethod::All,
        }
    }

    /// Use a guide map that describe a 'FROM' transformation.
    ///
    /// Note: If any one example has a guide, then they **all** must have
    /// a guide, otherwise a session will not be created.
    pub fn with_guide<G: Into<ImageSource<'a>>>(mut self, guide: G) -> Self {
        self.guide = Some(guide.into());
        self
    }

    /// Specify how the example image is sampled during texture generation.
    ///
    /// By default, all pixels in the example can be sampled.
    pub fn set_sample_method<M: Into<SampleMethod<'a>>>(mut self, method: M) -> Self {
        self.sample_method = method.into();
        self
    }
}

/// An example to be used in texture generation
pub struct Example<'a> {
    img: ImageSource<'a>,
    guide: Option<ImageSource<'a>>,
    sample_method: SampleMethod<'a>,
}

impl<'a> Example<'a> {
    /// Creates a new example builder from the specified image source
    pub fn builder<I: Into<ImageSource<'a>>>(img: I) -> ExampleBuilder<'a> {
        ExampleBuilder::new(img)
    }

    pub fn image_source(&self) -> &ImageSource<'a> {
        &self.img
    }

    /// Creates a new example input from the specified image source
    pub fn new<I: Into<ImageSource<'a>>>(img: I) -> Self {
        Self {
            img: img.into(),
            guide: None,
            sample_method: SampleMethod::All,
        }
    }

    /// Use a guide map that describe a 'FROM' transformation.
    ///
    /// Note: If any one example has a guide, then they **all** must have
    /// a guide, otherwise a session will not be created.
    pub fn with_guide<G: Into<ImageSource<'a>>>(&mut self, guide: G) -> &mut Self {
        self.guide = Some(guide.into());
        self
    }

    /// Specify how the example image is sampled during texture generation.
    ///
    /// By default, all pixels in the example can be sampled.
    pub fn set_sample_method<M: Into<SampleMethod<'a>>>(&mut self, method: M) -> &mut Self {
        self.sample_method = method.into();
        self
    }

    fn resolve(
        self,
        backtracks: u32,
        resize: Option<Dims>,
        target_guide: &Option<ImagePyramid>,
    ) -> Result<ResolvedExample, Error> {
        let image = ImagePyramid::new(load_image(self.img, resize)?, Some(backtracks));

        let guide = match target_guide {
            Some(tg) => {
                Some(match self.guide {
                    Some(exguide) => {
                        let exguide = load_image(exguide, resize)?;
                        ImagePyramid::new(exguide, Some(backtracks))
                    }
                    None => {
                        // if we do not have an example guide, create it as a b/w maps of the example
                        let mut gm = transform_to_guide_map(image.bottom().clone(), resize, 2.0);
                        match_histograms(&mut gm, tg.bottom());

                        ImagePyramid::new(gm, Some(backtracks))
                    }
                })
            }
            None => None,
        };

        let method = match self.sample_method {
            SampleMethod::All => SamplingMethod::All,
            SampleMethod::Ignore => SamplingMethod::Ignore,
            SampleMethod::Image(src) => {
                let img = load_image(src, resize)?;
                SamplingMethod::Image(img)
            }
        };

        Ok(ResolvedExample {
            image,
            guide,
            method,
        })
    }
}

impl<'a> From<ExampleBuilder<'a>> for Example<'a> {
    fn from(eb: ExampleBuilder<'a>) -> Self {
        Self {
            img: eb.img,
            guide: eb.guide,
            sample_method: eb.sample_method,
        }
    }
}

impl<'a, IS> From<IS> for Example<'a>
where
    IS: Into<ImageSource<'a>>,
{
    fn from(is: IS) -> Self {
        Example::new(is)
    }
}

enum MaskOrImg<'a> {
    Mask(utils::ChannelMask),
    ImageSource(ImageSource<'a>),
}

struct InpaintMask<'a> {
    src: MaskOrImg<'a>,
    example_index: usize,
    dims: Dims,
}

struct ResolvedExample {
    image: ImagePyramid,
    guide: Option<ImagePyramid>,
    method: SamplingMethod,
}

#[cfg(test)]
mod test {
    #[test]
    fn coord_tx_serde() {
        use super::CoordinateTransform as CT;

        let fake_buffer = vec![1, 2, 3, 4, 5, 6];

        let input = CT {
            buffer: fake_buffer.clone(),
            output_size: super::Dims {
                width: 2,
                height: 1,
            },
            original_maps: vec![
                super::Dims {
                    width: 9001,
                    height: 9002,
                },
                super::Dims {
                    width: 20,
                    height: 5,
                },
            ],
        };

        let mut buffer = Vec::new();
        input.write(&mut buffer).unwrap();

        let mut cursor = std::io::Cursor::new(&buffer);
        let deserialized = CT::read(&mut cursor).unwrap();

        assert_eq!(deserialized.buffer, fake_buffer);
        assert_eq!(deserialized.output_size.width, 2);
        assert_eq!(deserialized.output_size.height, 1);

        assert_eq!(
            super::Dims {
                width: 9001,
                height: 9002,
            },
            deserialized.original_maps[0]
        );
        assert_eq!(
            super::Dims {
                width: 20,
                height: 5,
            },
            deserialized.original_maps[1]
        );
    }
}
