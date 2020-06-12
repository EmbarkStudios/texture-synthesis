#![warn(
    clippy::all,
    clippy::doc_markdown,
    clippy::dbg_macro,
    clippy::todo,
    clippy::empty_enum,
    clippy::enum_glob_use,
    clippy::pub_enum_variant_names,
    clippy::mem_forget,
    clippy::use_self,
    clippy::filter_map_next,
    clippy::needless_continue,
    clippy::needless_borrow,
    rust_2018_idioms,
    future_incompatible,
    nonstandard_style
)]

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
use std::path::Path;
mod unsync;

pub use image;
pub use utils::{load_dynamic_image, ChannelMask, ImageSource};

pub use errors::Error;

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

                std::slice::from_raw_parts(p as *const u8, len * mem::size_of::<u32>())
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
                    std::slice::from_raw_parts_mut(p as *mut u8, len * mem::size_of::<u32>());

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
            output_size,
            original_maps,
            buffer,
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
        match self {
            Self::Ignore => true,
            _ => false,
        }
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

impl<'a> Into<Example<'a>> for ExampleBuilder<'a> {
    fn into(self) -> Example<'a> {
        Example {
            img: self.img,
            guide: self.guide,
            sample_method: self.sample_method,
        }
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

/// Builds a session by setting parameters and adding input images, calling
/// `build` will check all of the provided inputs to verify that texture
/// synthesis will provide valid output
#[derive(Default)]
pub struct SessionBuilder<'a> {
    examples: Vec<Example<'a>>,
    target_guide: Option<ImageSource<'a>>,
    inpaint_mask: Option<InpaintMask<'a>>,
    params: Parameters,
}

impl<'a> SessionBuilder<'a> {
    /// Creates a new `SessionBuilder`, can also be created via
    /// `Session::builder()`
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an `Example` from which a generator will synthesize a new image.
    ///
    /// See [`examples/01_single_example_synthesis`](https://github.com/EmbarkStudios/texture-synthesis/tree/main/lib/examples/01_single_example_synthesis.rs)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// let tex_synth = texture_synthesis::Session::builder()
    ///     .add_example(&"imgs/1.jpg")
    ///     .build().expect("failed to build session");
    /// ```
    pub fn add_example<E: Into<Example<'a>>>(mut self, example: E) -> Self {
        self.examples.push(example.into());
        self
    }

    /// Adds Examples from which a generator will synthesize a new image.
    ///
    /// See [`examples/02_multi_example_synthesis`](https://github.com/EmbarkStudios/texture-synthesis/tree/main/lib/examples/02_multi_example_synthesis.rs)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// let tex_synth = texture_synthesis::Session::builder()
    ///     .add_examples(&[&"imgs/1.jpg", &"imgs/2.jpg"])
    ///     .build().expect("failed to build session");
    /// ```
    pub fn add_examples<E: Into<Example<'a>>, I: IntoIterator<Item = E>>(
        mut self,
        examples: I,
    ) -> Self {
        self.examples.extend(examples.into_iter().map(|e| e.into()));
        self
    }

    /// Inpaints an example. Due to how inpainting works, a size must also be
    /// provided, as all examples, as well as the inpaint mask, must be the same
    /// size as each other, as well as the final output image. Using
    /// `resize_input` or `output_size` is ignored if this method is called.
    ///
    /// To prevent sampling from the example, you can specify
    /// `SamplingMethod::Ignore` with `Example::set_sample_method`.
    ///
    /// See [`examples/05_inpaint`](https://github.com/EmbarkStudios/texture-synthesis/tree/main/lib/examples/05_inpaint.rs)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// let tex_synth = texture_synthesis::Session::builder()
    ///     .add_examples(&[&"imgs/1.jpg", &"imgs/3.jpg"])
    ///     .inpaint_example(
    ///         &"masks/inpaint.jpg",
    ///         // This will prevent sampling from the imgs/2.jpg, note that
    ///         // we *MUST* provide at least one example to source from!
    ///         texture_synthesis::Example::builder(&"imgs/2.jpg")
    ///             .set_sample_method(texture_synthesis::SampleMethod::Ignore),
    ///         texture_synthesis::Dims::square(400)
    ///     )
    ///     .build().expect("failed to build session");
    /// ```
    pub fn inpaint_example<I: Into<ImageSource<'a>>, E: Into<Example<'a>>>(
        mut self,
        inpaint_mask: I,
        example: E,
        size: Dims,
    ) -> Self {
        self.inpaint_mask = Some(InpaintMask {
            src: MaskOrImg::ImageSource(inpaint_mask.into()),
            example_index: self.examples.len(),
            dims: size,
        });
        self.examples.push(example.into());
        self
    }

    /// Inpaints an example, using a specific channel in the example image as
    /// the inpaint mask
    ///
    /// # Examples
    ///
    /// ```no_run
    /// let tex_synth = texture_synthesis::Session::builder()
    ///     .inpaint_example_channel(
    ///         // Let's use inpaint the alpha channel
    ///         texture_synthesis::ChannelMask::A,
    ///         &"imgs/bricks.png",
    ///         texture_synthesis::Dims::square(400)
    ///     )
    ///     .build().expect("failed to build session");
    /// ```
    pub fn inpaint_example_channel<E: Into<Example<'a>>>(
        mut self,
        mask: utils::ChannelMask,
        example: E,
        size: Dims,
    ) -> Self {
        self.inpaint_mask = Some(InpaintMask {
            src: MaskOrImg::Mask(mask),
            example_index: self.examples.len(),
            dims: size,
        });
        self.examples.push(example.into());
        self
    }

    /// Loads a target guide map.
    ///
    /// If no `Example` guide maps are provided, this will produce a style
    /// transfer effect, where the Examples are styles and the target guide is
    /// content.
    ///
    /// See [`examples/03_guided_synthesis`](https://github.com/EmbarkStudios/texture-synthesis/tree/main/lib/examples/03_guided_synthesis.rs),
    /// or [`examples/04_style_transfer`](https://github.com/EmbarkStudios/texture-synthesis/tree/main/lib/examples/04_style_transfer.rs),
    pub fn load_target_guide<I: Into<ImageSource<'a>>>(mut self, guide: I) -> Self {
        self.target_guide = Some(guide.into());
        self
    }

    /// Overwrite incoming images sizes
    pub fn resize_input(mut self, dims: Dims) -> Self {
        self.params.resize_input = Some(dims);
        self
    }

    /// Changes pseudo-deterministic seed.
    ///
    /// Global structures will stay same, if the same seed is provided, but
    /// smaller details may change due to undeterministic nature of
    /// multithreading.
    pub fn seed(mut self, value: u64) -> Self {
        self.params.seed = value;
        self
    }

    /// Makes the generator output tiling image.
    ///
    /// Default: false.
    pub fn tiling_mode(mut self, is_tiling: bool) -> Self {
        self.params.tiling_mode = is_tiling;
        self
    }

    /// How many neighboring pixels each pixel is aware of during generation.
    ///
    /// A larger number means more global structures are captured.
    ///
    /// Default: 50
    pub fn nearest_neighbors(mut self, count: u32) -> Self {
        self.params.nearest_neighbors = count;
        self
    }

    /// The number of random locations that will be considered during a pixel
    /// resolution apart from its immediate neighbors.
    ///
    /// If unsure, keep same as nearest neighbors.
    ///
    /// Default: 50
    pub fn random_sample_locations(mut self, count: u64) -> Self {
        self.params.random_sample_locations = count;
        self
    }

    /// Forces the first `n` pixels to be randomly resolved, and prevents them
    /// from being overwritten.
    ///
    /// Can be an enforcing factor of remixing multiple images together.
    pub fn random_init(mut self, count: u64) -> Self {
        self.params.random_resolve = Some(count);
        self
    }

    /// The distribution dispersion used for picking best candidate (controls
    /// the distribution 'tail flatness').
    ///
    /// Values close to 0.0 will produce 'harsh' borders between generated
    /// 'chunks'. Values closer to 1.0 will produce a smoother gradient on those
    /// borders.
    ///
    /// For futher reading, check out P.Harrison's "Image Texture Tools".
    ///
    /// Default: 1.0
    pub fn cauchy_dispersion(mut self, value: f32) -> Self {
        self.params.cauchy_dispersion = value;
        self
    }

    /// Controls the trade-off between guide and example maps.
    ///
    /// If doing style transfer, set to about 0.8-0.6 to allow for more global
    /// structures of the style.
    ///
    /// If you'd like the guide maps to be considered through all generation
    /// stages, set to 1.0, which will prevent guide maps weight "decay" during
    /// the score calculation.
    ///
    /// Default: 0.8
    pub fn guide_alpha(mut self, value: f32) -> Self {
        self.params.guide_alpha = value;
        self
    }

    /// The percentage of pixels to be backtracked during each `p_stage`.
    /// Range (0,1).
    ///
    /// Default: 0.5
    pub fn backtrack_percent(mut self, value: f32) -> Self {
        self.params.backtrack_percent = value;
        self
    }

    /// Controls the number of backtracking stages.
    ///
    /// Backtracking prevents 'garbage' generation. Right now, the depth of the
    /// image pyramid for multiresolution synthesis depends on this parameter as
    /// well.
    ///
    /// Default: 5
    pub fn backtrack_stages(mut self, stages: u32) -> Self {
        self.params.backtrack_stages = stages;
        self
    }

    /// Specify size of the generated image.
    ///
    /// Default: 500x500
    pub fn output_size(mut self, dims: Dims) -> Self {
        self.params.output_size = dims;
        self
    }

    /// Controls the maximum number of threads that will be spawned at any one
    /// time in parallel.
    ///
    /// This number is allowed to exceed the number of logical cores on the
    /// system, but it should generally be kept at or below that number.
    ///
    /// Setting this number to `1` will result in completely deterministic
    /// image generation, meaning that redoing generation with the same inputs
    /// will always give you the same outputs.
    ///
    /// Default: The number of logical cores on this system.
    pub fn max_thread_count(mut self, count: usize) -> Self {
        self.params.max_thread_count = Some(count);
        self
    }

    /// Creates a `Session`, or returns an error if invalid parameters or input
    /// images were specified.
    pub fn build(mut self) -> Result<Session, Error> {
        self.check_parameters_validity()?;
        self.check_images_validity()?;

        struct InpaintExample {
            inpaint_mask: image::RgbaImage,
            color_map: image::RgbaImage,
            example_index: usize,
        }

        let (inpaint, out_size, in_size) = match self.inpaint_mask {
            Some(inpaint_mask) => {
                let dims = inpaint_mask.dims;
                let inpaint_img = match inpaint_mask.src {
                    MaskOrImg::ImageSource(img) => load_image(img, Some(dims))?,
                    MaskOrImg::Mask(mask) => {
                        let example_img = &mut self.examples[inpaint_mask.example_index].img;

                        let dynamic_img = utils::load_dynamic_image(example_img.clone())?;
                        let inpaint_src = ImageSource::Image(dynamic_img.clone());

                        // Replace the example image source so we don't load it twice
                        *example_img = ImageSource::Image(dynamic_img);

                        let inpaint_mask = load_image(inpaint_src, Some(dims))?;

                        utils::apply_mask(inpaint_mask, mask)
                    }
                };

                let color_map = load_image(
                    self.examples[inpaint_mask.example_index].img.clone(),
                    Some(dims),
                )?;

                (
                    Some(InpaintExample {
                        inpaint_mask: inpaint_img,
                        color_map,
                        example_index: inpaint_mask.example_index,
                    }),
                    dims,
                    Some(dims),
                )
            }
            None => (None, self.params.output_size, self.params.resize_input),
        };

        let target_guide = match self.target_guide {
            Some(tg) => {
                let tg_img = load_image(tg, Some(out_size))?;

                let num_guides = self.examples.iter().filter(|ex| ex.guide.is_some()).count();
                let tg_img = if num_guides == 0 {
                    transform_to_guide_map(tg_img, None, 2.0)
                } else {
                    tg_img
                };

                Some(ImagePyramid::new(
                    tg_img,
                    Some(self.params.backtrack_stages as u32),
                ))
            }
            None => None,
        };

        let example_len = self.examples.len();

        let mut examples = Vec::with_capacity(example_len);
        let mut guides = if target_guide.is_some() {
            Vec::with_capacity(example_len)
        } else {
            Vec::new()
        };
        let mut methods = Vec::with_capacity(example_len);

        for example in self.examples {
            let resolved = example.resolve(self.params.backtrack_stages, in_size, &target_guide)?;

            examples.push(resolved.image);

            if let Some(guide) = resolved.guide {
                guides.push(guide);
            }

            methods.push(resolved.method);
        }

        // Initialize generator based on availability of an inpaint_mask.
        let generator = match inpaint {
            None => Generator::new(out_size),
            Some(inpaint) => Generator::new_from_inpaint(
                out_size,
                inpaint.inpaint_mask,
                inpaint.color_map,
                inpaint.example_index,
            ),
        };

        let session = Session {
            examples,
            guides: target_guide.map(|tg| GuidesPyramidStruct {
                target_guide: tg,
                example_guides: guides,
            }),
            sampling_methods: methods,
            params: self.params,
            generator,
        };

        Ok(session)
    }

    fn check_parameters_validity(&self) -> Result<(), Error> {
        if self.params.cauchy_dispersion < 0.0 || self.params.cauchy_dispersion > 1.0 {
            return Err(Error::InvalidRange(errors::InvalidRange {
                min: 0.0,
                max: 1.0,
                value: self.params.cauchy_dispersion,
                name: "cauchy-dispersion",
            }));
        }

        if self.params.backtrack_percent < 0.0 || self.params.backtrack_percent > 1.0 {
            return Err(Error::InvalidRange(errors::InvalidRange {
                min: 0.0,
                max: 1.0,
                value: self.params.backtrack_percent,
                name: "backtrack-percent",
            }));
        }

        if self.params.guide_alpha < 0.0 || self.params.guide_alpha > 1.0 {
            return Err(Error::InvalidRange(errors::InvalidRange {
                min: 0.0,
                max: 1.0,
                value: self.params.guide_alpha,
                name: "guide-alpha",
            }));
        }

        if let Some(max_count) = self.params.max_thread_count {
            if max_count == 0 {
                return Err(Error::InvalidRange(errors::InvalidRange {
                    min: 1.0,
                    max: 1024.0,
                    value: max_count as f32,
                    name: "max-thread-count",
                }));
            }
        }

        if self.params.random_sample_locations == 0 {
            return Err(Error::InvalidRange(errors::InvalidRange {
                min: 1.0,
                max: 1024.0,
                value: self.params.random_sample_locations as f32,
                name: "m-rand",
            }));
        }

        Ok(())
    }

    fn check_images_validity(&self) -> Result<(), Error> {
        // We must have at least one example image to source pixels from
        let input_count = self
            .examples
            .iter()
            .filter(|ex| !ex.sample_method.is_ignore())
            .count();

        if input_count == 0 {
            return Err(Error::NoExamples);
        }

        // If we have more than one example guide, then *every* example
        // needs a guide
        let num_guides = self.examples.iter().filter(|ex| ex.guide.is_some()).count();
        if num_guides != 0 && self.examples.len() != num_guides {
            return Err(Error::ExampleGuideMismatch(
                self.examples.len() as u32,
                num_guides as u32,
            ));
        }

        Ok(())
    }
}

struct ResolvedExample {
    image: ImagePyramid,
    guide: Option<ImagePyramid>,
    method: SamplingMethod,
}

/// Texture synthesis session.
///
/// Calling `run()` will generate a new image and return it, consuming the
/// session in the process. You can provide a `GeneratorProgress` implementation
/// to periodically get updates with the currently generated image and the
/// number of pixels that have been resolved both in the current stage and
/// globally.
///
/// # Example
/// ```no_run
/// let tex_synth = texture_synthesis::Session::builder()
///     .seed(10)
///     .tiling_mode(true)
///     .add_example(&"imgs/1.jpg")
///     .build().expect("failed to build session");
///
/// let generated_img = tex_synth.run(None);
/// generated_img.save("my_generated_img.jpg").expect("failed to save image");
/// ```
pub struct Session {
    examples: Vec<ImagePyramid>,
    guides: Option<GuidesPyramidStruct>,
    sampling_methods: Vec<SamplingMethod>,
    generator: Generator,
    params: Parameters,
}

impl Session {
    /// Creates a new session with default parameters.
    pub fn builder<'a>() -> SessionBuilder<'a> {
        SessionBuilder::default()
    }

    /// Runs the generator and outputs a generated image.
    pub fn run(mut self, progress: Option<Box<dyn GeneratorProgress>>) -> GeneratedImage {
        // random resolve
        // TODO: Instead of consuming the generator, we could instead make the
        // seed and random_resolve parameters, so that you could rerun the
        // generator with the same inputs
        if let Some(count) = self.params.random_resolve {
            let lvl = self.examples[0].pyramid.len();
            let imgs: Vec<_> = self
                .examples
                .iter()
                .map(|a| ImageBuffer::from(&a.pyramid[lvl - 1])) //take the blurriest image
                .collect();

            self.generator
                .resolve_random_batch(count as usize, &imgs, self.params.seed);
        }

        // run generator
        self.generator.resolve(
            &self.params.to_generator_params(),
            &self.examples,
            progress,
            &self.guides,
            &self.sampling_methods,
        );

        GeneratedImage {
            inner: self.generator,
        }
    }
}

/// Helper struct for passing progress information to external callers
pub struct ProgressStat {
    /// The current amount of work that has been done
    pub current: usize,
    /// The total amount of work to do
    pub total: usize,
}

/// The current state of the image generator
pub struct ProgressUpdate<'a> {
    /// The currenty resolved image
    pub image: &'a image::RgbaImage,
    /// The total progress for the final image
    pub total: ProgressStat,
    /// The progress for the current stage
    pub stage: ProgressStat,
}

/// Allows the generator to update external callers with the current
/// progress of the image synthesis
pub trait GeneratorProgress {
    fn update(&mut self, info: ProgressUpdate<'_>);
}

impl<G> GeneratorProgress for G
where
    G: FnMut(ProgressUpdate<'_>) + Send,
{
    fn update(&mut self, info: ProgressUpdate<'_>) {
        self(info)
    }
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
