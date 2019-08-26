//! `texture-synthesis` is a light API for Multiresolution Stochastic Texture Synthesis,
//! a non-parametric example-based algorithm for image generation.
//!  All the interactions with the algorithm happen through `Session`.
//! `Session` implements API for loading/saving images and changing parameters.
//! During `Session.run()`, Session pre-processes all the data, checks for errors and calls `multires_stochastic_texture_synthesis.rs`
//! to generate a new image.
//! # Features
//! 1. Single example generation
//! 2. Multi example generation
//! 3. Guided synthesis
//! 4. Style transfer
//! 5. Inpainting
//! 6. Tiling textures
//!
//! Please, refer to the examples folder in the [repository](https://github.com/EmbarkStudios/texture-synthesis) for the features usage examples.
//!
//! # Usage
//! Session follows a "builder pattern" for defining parameters, meaning you chain functions together.
//! ```
//! //create a new session with default parameters
//! let tex_synth = Session::new()
//!                 //set parameters
//!                 .seed(10)
//!                 .nearest_neighbours(20)
//!                 //load example image(s)
//!                 .load_examples(&vec![imgs/1.jpg]);
//! //generate a new image
//! let generated_img = tex_synth.run().unwrap();
//! //save
//! tex_synth.save("my_generated_img.jpg").unwrap();
//! ```
mod img_pyramid;
use img_pyramid::*;
mod utils;
use utils::*;
mod progress_window;
use progress_window::*;
mod multires_stochastic_texture_synthesis;
use multires_stochastic_texture_synthesis::*;
use std::error::Error;
use std::path::Path;
use std::time::Duration;

struct ImagePaths {
    examples: Option<Vec<String>>,
    example_guides: Option<Vec<String>>,
    target_guide: Option<String>,
    sampling_masks: Option<Vec<Option<String>>>,
    inpaint_mask: Option<String>,
}

impl Default for ImagePaths {
    fn default() -> Self {
        Self {
            examples: None,
            example_guides: None,
            target_guide: None,
            sampling_masks: None,
            inpaint_mask: None,
        }
    }
}

struct Parameters {
    tiling_mode: bool,
    nearest_neighbours: u32,
    random_sample_locations: u64,
    cauchy_dispersion: f32,
    backtrack_percent: f32,
    backtrack_stages: u32,
    resize_input: Option<(u32, u32)>,
    output_size: (u32, u32),
    show_progress: bool,
    guide_alpha: f32,
    inpaint_example_id: Option<u32>,
    random_resolve: Option<u64>,
    seed: u64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            tiling_mode: false,
            nearest_neighbours: 50,
            random_sample_locations: 50,
            cauchy_dispersion: 1.0,
            backtrack_percent: 0.5,
            backtrack_stages: 5,
            resize_input: None,
            output_size: (500, 500),
            show_progress: true,
            guide_alpha: 0.8,
            inpaint_example_id: None,
            random_resolve: None,
            seed: 0,
        }
    }
}

impl Parameters {
    fn to_generator_params(&self) -> GeneratorParams {
        GeneratorParams {
            nearest_neighbours: self.nearest_neighbours,
            random_sample_locations: self.random_sample_locations,
            caushy_dispersion: self.cauchy_dispersion,
            p: self.backtrack_percent,
            p_stages: self.backtrack_stages as i32,
            seed: self.seed,
            alpha: self.guide_alpha,
        }
    }
}

/// Texture synthesis session.
///
/// Session follows a "builder pattern" for defining parameters, meaning you chain functions together.
/// # Example
/// ```
/// let tex_synth = Session::new()
///                 .seed(10)
///                 .tiling_mode(true)
///                 .load_examples(&vec![imgs/1.jpg]);
/// let generated_img = tex_synth.run().unwrap();
/// tex_synth.save("my_generated_img.jpg").unwrap();
/// ```
pub struct Session {
    img_paths: ImagePaths,
    params: Parameters,
    generator: Option<multires_stochastic_texture_synthesis::Generator>,
}

impl Session {
    /// Creates a new session with default parameters.
    pub fn new() -> Self {
        Session {
            img_paths: ImagePaths::default(),
            params: Parameters::default(),
            generator: None,
        }
    }

    /// Loads example image(s) from which generator will synthesize a new image.
    ///
    /// See [example files](https://github.com/EmbarkStudios/texture-synthesis/tree/master/examples): `examples/01_single_example_synthesis.rs` and `examples/02_single_example_synthesis.rs`.
    ///
    /// # Example
    /// ```
    /// //single example generation
    /// let tex_synth = Session::new().load_examples(&vec!["imgs/1.jpg"]);
    /// //multi example generation
    /// let tex_synth = Session::new().load_examples(&vec!["imgs/1.jpg", "imgs/2.jpg"]);
    /// ```
    pub fn load_examples(mut self, paths: &[&str]) -> Self {
        self.img_paths.examples = Some(paths.to_vec().iter().map(|a| String::from(*a)).collect());
        self
    }

    /// Loads guide maps that describe a 'FROM' transformation for examples.
    /// Make sure to also provide a target guide map to specify 'TO' transformation for the output image.
    ///
    /// See [example file](https://github.com/EmbarkStudios/texture-synthesis/tree/master/examples): `examples/03_guided_synthesis.rs`.
    pub fn load_example_guides(mut self, paths: &[&str]) -> Self {
        self.img_paths.example_guides =
            Some(paths.to_vec().iter().map(|a| String::from(*a)).collect());
        self
    }

    /// Loads a target guide map.
    /// If no example guide maps are provided, this will produce style transfer effect, where example is style and target guide is content.
    ///
    /// See [example files](https://github.com/EmbarkStudios/texture-synthesis/tree/master/examples): `examples/03_guided_synthesis.rs` and `examples/04_style_transfer.rs`.
    pub fn load_target_guide(mut self, path: &str) -> Self {
        self.img_paths.target_guide = Some(String::from(path));
        self
    }

    /// Loads masks that specify which parts of example images are allowed to be sampled from.
    /// This way you can prevent generator from copying undesired elements of examples.
    /// You can also say "None" to NOT include a map for certain example(s).
    ///
    /// See [example files](https://github.com/EmbarkStudios/texture-synthesis/tree/master/examples): `examples/05_inpaint.rs` and `examples/06_tiling_texture.rs`.
    ///
    /// # Example
    /// ```
    /// let tex_synth = Session::new()
    ///                 .load_examples(&vec!["imgs/1.jpg", "imgs/2.jpg", "imgs/3.jpg"])
    ///                 .load_sampling_masks(&vec["None", "masks/2.jpg", "None"]);
    /// ```
    pub fn load_sampling_masks(mut self, paths: &[&str]) -> Self {
        self.img_paths.sampling_masks = Some(
            paths
                .to_vec()
                .iter()
                .map(|a| match *a {
                    "None" => None,
                    _ => Some(String::from(*a)),
                })
                .collect(),
        );
        self
    }

    /// Inpaints an example (example should be provided through load_examples function).
    /// Specify the index of the example to inpaint with example_id.
    /// Note: right now, only possible to inpaint existing example map.
    /// To prevent sampling from the example, you can add a fully black sampling mask to it.
    ///
    /// See [example file](https://github.com/EmbarkStudios/texture-synthesis/tree/master/examples): `examples/05_inpaint.rs`.
    ///
    /// # Example
    /// ```
    /// let tex_synth = Session::new()
    ///                 .load_examples(&vec!["imgs/1.jpg", "imgs/2.jpg", "imgs/3.jpg"])
    ///                 .inpaint_example("masks/inpaint.jpg", 0); //this will inpaint "imgs/1.jpg" example
    ///                 .inpaint_example("masks/inpaint.jpg", 1); //this will inpaint "imgs/2.jpg" example
    ///                 .load_sampling_masks(&vec!["None", "masks/black.jpg", "None"]); //this will prevent any sampling from "imgs/2.jpg" example
    /// ```
    pub fn inpaint_example(mut self, inpaint_mask: &str, example_id: u32) -> Self {
        self.img_paths.inpaint_mask = Some(String::from(inpaint_mask));
        self.params.inpaint_example_id = Some(example_id);
        self
    }

    /// Overwrite incoming images sizes
    pub fn resize_input(mut self, w: u32, h: u32) -> Self {
        self.params.resize_input = Some((w, h));
        self
    }

    /// Changes pseudo-deterministic seed.
    /// Global structures will stay same, if same seed is provided, but smaller details may change due to undeterministic nature of multithreading.
    pub fn seed(mut self, value: u64) -> Self {
        self.params.seed = value;
        self
    }

    /// Makes the generator output tiling image.
    /// Default: false.
    pub fn tiling_mode(mut self, is_tiling: bool) -> Self {
        self.params.tiling_mode = is_tiling;
        self
    }

    /// How many neighbouring pixels each pixel is aware of during the generation (bigger number -> more global structures are captured).
    /// Default: 50
    pub fn nearest_neighbours(mut self, count: u32) -> Self {
        self.params.nearest_neighbours = count;
        self
    }

    /// Creates a pop-up window showing the progress of the generator.
    /// Default: true
    pub fn show_progress(mut self, is_true: bool) -> Self {
        self.params.show_progress = is_true;
        self
    }

    /// How many random locations will be considered during a pixel resolution apart from its immediate neighbours.
    /// If unsure, keep same as nearest neighbours.
    /// Default: 50
    pub fn random_sample_locations(mut self, count: u64) -> Self {
        self.params.random_sample_locations = count;
        self
    }

    /// Make first X pixels to be randomly resolved and prevent them from being overwritten.
    /// Can be an enforcing factor of remixing multiple images together.
    pub fn random_init(mut self, count: u64) -> Self {
        self.params.random_resolve = Some(count);
        self
    }

    /// The distribution dispersion used for picking best candidate (controls the distribution 'tail flatness').
    /// Values close to 0.0 will produce 'harsh' borders between generated 'chunks'. Values closer to 1.0 will produce a smoother gradient on those borders.
    /// For futher reading, check out P.Harrison's "Image Texture Tools".
    /// Default: 1.0
    pub fn cauchy_dispersion(mut self, value: f32) -> Self {
        self.params.cauchy_dispersion = value;
        self
    }

    /// Controls the trade-off between guide and example map.
    /// If doing style transfer, set to about 0.8-0.6 to allow for more global structures of the style.
    /// If you'd like the guide maps to be considered through all generation stages, set to 1.0 (which would prevent guide maps weight "decay" during the score calculation).
    /// Default: 0.8
    pub fn guide_alpha(mut self, value: f32) -> Self {
        self.params.guide_alpha = value;
        self
    }

    /// The percentage of pixels to be backtracked during each p_stage. Range (0,1).
    /// Default: 0.5
    pub fn backtrack_percent(mut self, value: f32) -> Self {
        self.params.backtrack_percent = value;
        self
    }

    /// Controls the number of backtracking stages. Backtracking prevents 'garbage' generation.
    /// Right now, the depth of image pyramid for multiresolution synthesis
    /// depends on this parameter as well.
    /// Default: 5
    pub fn backtrack_stages(mut self, stages: u32) -> Self {
        self.params.backtrack_stages = stages;
        self
    }

    /// Specify size of the generated image.
    /// Default: 500x500
    pub fn output_size(mut self, w: u32, h: u32) -> Self {
        self.params.output_size = (w, h);
        self
    }

    /// Runs the generator and outputs a generated image.
    /// Now, only runs Multiresolution Stochastic Texture Synthesis. Might be interseting to include more algorithms in the future.
    pub fn run(&mut self) -> Result<image::RgbaImage, Box<dyn Error>> {
        self.check_parameters_validity()?;
        self.check_images_validity()?;

        //pre-process input data
        let progress_window = self.init_progress_window();
        let examples = self.init_examples()?;
        let guides = self.init_guides()?;
        let sampling_masks = self.init_sampling_masks()?;

        //initialize generator based on availability of an inpaint_mask.
        let mut generator = match self.img_paths.inpaint_mask {
            None => multires_stochastic_texture_synthesis::Generator::new(self.params.output_size),
            Some(ref inpaint_path) => {
                multires_stochastic_texture_synthesis::Generator::new_from_inpaint(
                    self.params.output_size,
                    &load_image(inpaint_path, &Some(self.params.output_size))?,
                    &load_image(
                        &self.img_paths.examples.as_ref().unwrap()
                            [self.params.inpaint_example_id.unwrap() as usize],
                        &Some(self.params.output_size),
                    )?,
                    self.params.inpaint_example_id.unwrap(),
                )
            }
        };

        //random resolve
        if let Some(count) = self.params.random_resolve {
            let lvl = examples[0].pyramid.len();
            let imgs: Vec<image::RgbaImage> = examples
                .iter()
                .map(|a| a.pyramid[lvl - 1].clone()) //take the blurriest image
                .collect();
            let imgs_ref = imgs.iter().collect::<Vec<_>>();
            generator.resolve_random_batch(count as usize, &imgs_ref, self.params.seed);
        }

        //run generator
        generator.main_resolve_loop(
            &self.params.to_generator_params(),
            &examples,
            progress_window,
            guides,
            &sampling_masks,
            self.params.tiling_mode,
        );

        self.generator = Some(generator);

        Ok(self.generator.as_ref().unwrap().color_map.clone())
    }

    /// Saves the generated image.
    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        if let Some(ref generator) = self.generator {
            Ok(save_image(&String::from(path), &generator.color_map)?)
        } else {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "Nothing to save. Make sure to run your session",
            )));
        }
    }

    /// Saves debug information such as copied patches ids, map ids (if you have multi example generation)
    /// and uncertainty map indicating generated pixels generator was "uncertain" of.
    pub fn save_debug(&self, folder_path: &str) -> Result<(), Box<dyn Error>> {
        if let Some(ref generator) = self.generator {
            let parent_path = Path::new(folder_path);
            generator
                .get_uncertainty_map()
                .save(Path::new(&parent_path.join("uncertainty.png")))?;
            let id_maps = generator.get_id_maps();
            id_maps[0].save(Path::new(&parent_path.join("patch_id.png")))?;
            id_maps[1].save(Path::new(&parent_path.join("map_id.png")))?;
            Ok(())
        } else {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "Nothing to save. Make sure to run your session before saving",
            )));
        }
    }

    fn load_multiple_as_pyramid(
        &self,
        paths: &[String],
        resize: &Option<(u32, u32)>,
    ) -> Result<Vec<ImagePyramid>, Box<dyn Error>> {
        Ok(load_image_multiple(paths, resize)?
            .iter()
            .map(|a| ImagePyramid::new(&a, Some(self.params.backtrack_stages as u32)))
            .collect())
    }

    fn load_single_as_pyramid(
        &self,
        path: &String,
        resize: &Option<(u32, u32)>,
    ) -> Result<ImagePyramid, Box<dyn Error>> {
        Ok(self
            .load_multiple_as_pyramid(&[path.clone()], resize)?
            .pop()
            .unwrap())
    }

    fn load_examples_as_guides(
        &self,
        paths: &[String],
        match_target: &image::RgbaImage,
    ) -> Result<Vec<ImagePyramid>, Box<dyn Error>> {
        Ok(
            load_images_as_guide_maps(paths, &self.params.resize_input, 2.0)?
                .iter()
                .map(|a| {
                    ImagePyramid::new(
                        &match_histograms(&a, match_target),
                        Some(self.params.backtrack_stages as u32),
                    )
                })
                .collect(),
        )
    }

    fn load_and_preprocess_target_guide(
        &self,
        path: &String,
    ) -> Result<ImagePyramid, Box<dyn Error>> {
        Ok(ImagePyramid::new(
            &load_images_as_guide_maps(&[path.clone()], &Some(self.params.output_size), 2.0)?
                .pop()
                .unwrap(),
            Some(self.params.backtrack_stages as u32),
        ))
    }

    fn init_examples(&self) -> Result<Vec<ImagePyramid>, Box<dyn Error>> {
        Ok(self.load_multiple_as_pyramid(
            &self.img_paths.examples.as_ref().unwrap(),
            &self.params.resize_input,
        )?)
    }

    fn init_guides(&self) -> Result<Option<GuidesPyramidStruct>, Box<dyn Error>> {
        Ok(match self.img_paths.target_guide {
            None => None,
            Some(ref target_guide_path) => match self.img_paths.example_guides {
                Some(ref example_guides_path) => Some(GuidesPyramidStruct {
                    //if we have example guides, load images as is
                    example_guides: self
                        .load_multiple_as_pyramid(example_guides_path, &self.params.resize_input)?,
                    target_guide: self.load_single_as_pyramid(
                        target_guide_path,
                        &Some(self.params.output_size),
                    )?,
                }),
                None => {
                    //if we do not have example guides, create them as a b/w maps of examples
                    let target_guide = self.load_and_preprocess_target_guide(target_guide_path)?;
                    let example_guides = self.load_examples_as_guides(
                        &self.img_paths.examples.as_ref().unwrap(),
                        &target_guide.reconstruct(),
                    )?;
                    Some(GuidesPyramidStruct {
                        target_guide,
                        example_guides,
                    })
                }
            },
        })
    }

    fn init_progress_window(&self) -> Option<Box<dyn GeneratorProgress>> {
        if self.params.show_progress {
            Some(Box::new(WindowStruct::new(
                self.params.output_size,
                Duration::from_millis(10),
            )))
        } else {
            None
        }
    }

    fn init_sampling_masks(&self) -> Result<Vec<Option<image::RgbaImage>>, Box<dyn Error>> {
        match self.img_paths.sampling_masks {
            None => Ok(vec![None; self.img_paths.examples.as_ref().unwrap().len()]),
            Some(ref paths) => {
                let mut sampling_masks: Vec<Option<image::RgbaImage>> = Vec::new();
                for p in paths.iter() {
                    match p {
                        None => sampling_masks.push(None),
                        Some(ref valid_path) => {
                            let mask = load_image(valid_path, &self.params.resize_input)?;
                            sampling_masks.push(Some(mask));
                        }
                    }
                }
                Ok(sampling_masks)
            }
        }
    }

    fn check_parameters_validity(&self) -> Result<(), Box<dyn Error>> {
        if self.params.cauchy_dispersion < 0.0 || self.params.cauchy_dispersion > 1.0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid parameter range: cauchy dispersion. Make sure it is within 0.0-1.0 range",
            )));
        }

        if self.params.backtrack_percent < 0.0 || self.params.backtrack_percent > 1.0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid parameter range: backtrack percent. Make sure it is within 0.0-1.0 range",
            )));
        }

        if self.params.guide_alpha < 0.0 || self.params.guide_alpha > 1.0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid parameter range: guide alpha. Make sure it is within 0.0-1.0 range",
            )));
        }

        if self.params.inpaint_example_id.is_some() {
            if let Some(resize_input) = self.params.resize_input {
                if resize_input.0 != self.params.output_size.0
                    || resize_input.1 != self.params.output_size.1
                {
                    return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Input and output sizes dont match. Make sure resize_input = output_size if using inpaint",
            )));
                }
            } else {
                return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Input and output sizes dont match. Make sure resize_input = output_size if using inpaint",
            )));
            }
        }

        Ok(())
    }

    fn check_images_validity(&self) -> Result<(), Box<dyn Error>> {
        if self.img_paths.examples.is_none() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Missing input: example image(s)",
            )));
        }

        let example_maps_count = self.img_paths.examples.as_ref().unwrap().len();
        if let Some(ref example_guides) = self.img_paths.example_guides {
            assert_eq!(
                example_guides.len(),
                example_maps_count,
                "Mismatch of maps: {} example guide(s) vs {} example(s)",
                example_guides.len(),
                example_maps_count
            );
        }

        if let Some(ref sampling_masks) = self.img_paths.sampling_masks {
            assert_eq!(
                sampling_masks.len(),
                example_maps_count,
                "Mismatch of maps: {} sampling mask(s) vs {} example(s)",
                sampling_masks.len(),
                example_maps_count
            );
        }

        Ok(())
    }
}
