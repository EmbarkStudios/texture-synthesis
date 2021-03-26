use crate::*;

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
