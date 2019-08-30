use structopt::StructOpt;

use std::path::PathBuf;
use texture_synthesis::{
    image::ImageOutputFormat as ImgFmt, Example, ImageSource, SampleMethod, Session,
};

fn parse_size(input: &str) -> Result<(u32, u32), std::num::ParseIntError> {
    let mut i = input.splitn(2, 'x');

    let x: u32 = i.next().unwrap_or("").parse()?;
    let y: u32 = match i.next() {
        Some(num) => num.parse()?,
        None => x,
    };
    Ok((x, y))
}

fn parse_img_fmt(input: &str) -> Result<ImgFmt, String> {
    let fmt = match input {
        "png" => ImgFmt::PNG,
        "jpg" => ImgFmt::JPEG(75),
        "bmp" => ImgFmt::BMP,
        other => {
            return Err(format!(
                "image format `{}` not one of: 'png', 'jpg', 'bmp'",
                other
            ))
        }
    };

    Ok(fmt)
}

#[derive(StructOpt)]
struct Generate {
    /// A target guidance map
    #[structopt(long, parse(from_os_str))]
    target_guide: Option<PathBuf>,
    /// Path(s) to guide maps for the example output.
    #[structopt(long = "guides", parse(from_os_str))]
    example_guides: Vec<PathBuf>,
    /// Path(s) to example images used to synthesize a new image
    #[structopt(long, parse(from_os_str))]
    examples: Vec<PathBuf>,
}

#[derive(StructOpt)]
struct TransferStyle {
    /// The image from which the style will be be sourced
    #[structopt(long)]
    style: PathBuf,
    /// The image used as a guide for the generated output
    #[structopt(long)]
    guide: PathBuf,
}

#[derive(StructOpt)]
enum Subcommand {
    /// Transfers the style from an example onto a target guide
    #[structopt(name = "transfer-style")]
    TransferStyle(TransferStyle),
    /// Generates a new image from 1 or more examples
    #[structopt(name = "generate")]
    Generate(Generate),
}

#[derive(StructOpt)]
struct Tweaks {
    /// The number of neighbouring pixels each pixel is aware of during the generation,
    /// larger numbers means more global structures are captured.
    #[structopt(long = "k-neighs", default_value = "20")]
    k_neighbors: u32,
    /// The number of random locations that will be considered during a pixel resolution,
    /// apart from its immediate neighbours. If unsure of this parameter, keep as the same as k-neigh.
    #[structopt(long = "m-rand", default_value = "20")]
    m_rand: u64,
    /// The distribution dispersion used for picking best candidate (controls the distribution 'tail flatness').
    /// Values close to 0.0 will produce 'harsh' borders between generated 'chunks'.
    /// Values closer to 1.0 will produce a smoother gradient on those borders.
    #[structopt(long, default_value = "1.0")]
    cauchy: f32,
    /// The percentage of pixels to be backtracked during each p_stage. Range (0.0, 1.0).
    #[structopt(long = "backtrack-pct", default_value = "0.35")]
    backtrack_percentage: f32,
    /// The number of backtracking stages. Backtracking prevents 'garbage' generation.
    #[structopt(long = "backtrack-stages", default_value = "5")]
    backtrack_stages: u32,
    /// Show a window with the current progress of the generation
    #[structopt(long = "window")]
    show_window: bool,
    /// Show a window with the current progress of the generation
    #[structopt(long = "no-progress")]
    no_progress: bool,
    /// A seed value for the random generator to give pseudo-deterministic result.
    /// Smaller details will be different from generation to generation due to the
    /// non-deterministic nature of multi-threading
    #[structopt(long)]
    seed: Option<u64>,
    /// Alpha parameter controls the 'importance' of the user guide maps. If you want
    /// to preserve more details from the example map, make sure the number < 1.0. Range (0.0 - 1.0)
    #[structopt(long, default_value = "1.0")]
    alpha: f32,
    /// The number of randomly initialized pixels before the main resolve loop starts
    #[structopt(long = "rand-init")]
    rand_init: Option<u64>,
    /// Enables tiling of the output image
    #[structopt(long = "tiling")]
    enable_tiling: bool,
}

#[derive(StructOpt)]
#[structopt(
    name = "texture-synthesis",
    about = "Synthesizes images based on example images"
)]
struct Opt {
    /// Path(s) to sample masks used to determine which pixels in an example can be used as inputs
    /// during generation, any example that doesn't have a mask, or uses `ALL`, will consider
    /// all pixels in the example. If `IGNORE` is specified, then the example image won't be used
    /// at all, which is useful with `--inpaint`.
    ///
    /// The sample masks must be specified in the same order as the examples
    #[structopt(long = "sample-masks")]
    sample_masks: Vec<String>,
    /// Path to an inpaint map image, where black pixels are resolved, and white pixels are kept
    #[structopt(long, parse(from_os_str))]
    inpaint: Option<PathBuf>,
    /// Size of the generated image, in `width x height`, or a single number for both dimensions
    #[structopt(
        long = "out-size",
        default_value = "500",
        parse(try_from_str = "parse_size")
    )]
    output_size: (u32, u32),
    #[structopt(
        long = "out-fmt",
        default_value = "png",
        parse(try_from_str = "parse_img_fmt")
    )]
    output_fmt: texture_synthesis::image::ImageOutputFormat,
    /// Resize input example map(s), in `width x height`, or a single number for both dimensions
    #[structopt(long = "in-size", parse(try_from_str = "parse_size"))]
    input_size: Option<(u32, u32)>,
    /// The path to save the generated image to, use `-` for stdout
    #[structopt(long = "out", short, default_value = "-")]
    output_path: String,
    /// A directory into which debug images are also saved.
    ///
    /// * `patch_id.png` - Map of the `copy islands` from an example
    /// * `map_id.png` - Map of ids of which example was the source of a pixel
    /// * `uncertainty.png` - Map of pixels the generator was uncertain of
    #[structopt(long = "debug-out-dir", parse(from_os_str))]
    debug_output_dir: Option<PathBuf>,
    #[structopt(flatten)]
    tweaks: Tweaks,
    #[structopt(subcommand)]
    cmd: Subcommand,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Opt::from_args();

    let (mut examples, target_guide) = match &args.cmd {
        Subcommand::Generate(gen) => {
            let mut examples: Vec<_> = gen.examples.iter().map(Example::from).collect();
            if !gen.example_guides.is_empty() {
                if examples.len() != gen.example_guides.len() {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!(
                            "Mismatch of maps: {} example guide(s) vs {} example(s)",
                            gen.example_guides.len(),
                            examples.len()
                        ),
                    )));
                }

                for (i, guide) in gen.example_guides.iter().enumerate() {
                    examples[i].with_guide(guide);
                }
            }

            (examples, gen.target_guide.as_ref())
        }
        Subcommand::TransferStyle(ts) => (vec![Example::from(&ts.style)], Some(&ts.guide)),
    };

    if !args.sample_masks.is_empty() {
        for (i, mask) in args.sample_masks.iter().enumerate() {
            // Just ignore sample masks that don't have a corresponding example,
            // though we could also just error out
            if i == examples.len() {
                break;
            }

            let example = &mut examples[i];
            match mask.as_str() {
                "ALL" => example.set_sample_method(SampleMethod::All),
                "IGNORE" => example.set_sample_method(SampleMethod::Ignore),
                path => example.set_sample_method(SampleMethod::Image(ImageSource::Path(
                    &std::path::Path::new(path),
                ))),
            };
        }
    }

    let mut sb = Session::builder();

    // TODO: Make inpaint work with multiple examples
    if let Some(ref inpaint) = args.inpaint {
        let mut inpaint_example = examples.remove(0);

        // If the user hasn't explicitly specified sample masks, assume ignore for the example
        if args.sample_masks.is_empty() {
            inpaint_example.set_sample_method(SampleMethod::Ignore);
        }

        sb = sb.inpaint_example(inpaint, inpaint_example);
    }

    sb = sb
        .add_examples(examples)
        .output_size(args.output_size.0, args.output_size.1)
        .seed(args.tweaks.seed.unwrap_or_default())
        .nearest_neighbours(args.tweaks.k_neighbors)
        .random_sample_locations(args.tweaks.m_rand)
        .cauchy_dispersion(args.tweaks.cauchy)
        .backtrack_percent(args.tweaks.backtrack_percentage)
        .backtrack_stages(args.tweaks.backtrack_stages)
        .guide_alpha(args.tweaks.alpha)
        .tiling_mode(args.tweaks.enable_tiling);

    if let Some(ref tg) = target_guide {
        sb = sb.load_target_guide(tg);
    }
    if let Some(rand_init) = args.tweaks.rand_init {
        sb = sb.random_init(rand_init);
    }

    if let Some(insize) = args.input_size {
        sb = sb.resize_input(insize.0, insize.1);
    }

    let session = sb.build()?;

    let progress: Option<Box<dyn texture_synthesis::GeneratorProgress>> =
        if !args.tweaks.no_progress {
            let progress = ProgressWindow::new();
            let progress = if args.tweaks.show_window {
                progress.with_preview(args.output_size, std::time::Duration::from_millis(100))
            } else {
                progress
            };

            Some(Box::new(progress))
        } else {
            None
        };

    let generated = session.run(progress);

    if let Some(ref dir) = args.debug_output_dir {
        generated.save_debug(dir)?;
    }

    if args.output_path == "-" {
        let out = std::io::stdout();
        let mut out = out.lock();
        generated.write(&mut out, args.output_fmt)?;
    } else {
        generated.save(&args.output_path)?;
    }

    Ok(())
}

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::time::Duration;

pub struct ProgressWindow {
    window: Option<piston_window::PistonWindow>,
    update_freq: Duration,
    last_update: std::time::Instant,

    total_pb: ProgressBar,
    stage_pb: ProgressBar,

    total_len: usize,
    stage_len: usize,
    stage_num: u32,
}

impl ProgressWindow {
    fn new() -> Self {
        let multi_pb = MultiProgress::new();
        let sty = ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {percent}%")
            .progress_chars("##-");

        let total_pb = multi_pb.add(ProgressBar::new(100));
        total_pb.set_style(sty);

        let sty = ProgressStyle::default_bar()
            .template(" stage {msg:>3} {bar:40.cyan/blue} {percent}%")
            .progress_chars("##-");
        let stage_pb = multi_pb.add(ProgressBar::new(100));
        stage_pb.set_style(sty);

        std::thread::spawn(move || {
            let _ = multi_pb.join();
        });

        Self {
            window: None,
            update_freq: Duration::from_millis(10),
            last_update: std::time::Instant::now(),
            total_pb,
            stage_pb,
            total_len: 100,
            stage_len: 100,
            stage_num: 0,
        }
    }

    fn with_preview(mut self, size: (u32, u32), update_every: Duration) -> Self {
        use piston_window::EventLoop;

        let mut window: piston_window::PistonWindow =
            piston_window::WindowSettings::new("Texture Synthesis", [size.0, size.1])
                .exit_on_esc(true)
                .build()
                .unwrap();

        // disallow sleeping
        window.set_bench_mode(true);
        self.window = Some(window);
        self.update_freq = update_every;

        self
    }
}

impl Drop for ProgressWindow {
    fn drop(&mut self) {
        self.total_pb.finish();
        self.stage_pb.finish();
    }
}

impl texture_synthesis::GeneratorProgress for ProgressWindow {
    fn update(&mut self, update: texture_synthesis::ProgressUpdate<'_>) {
        if update.total.total != self.total_len {
            self.total_len = update.total.total;
            self.total_pb.set_length(self.total_len as u64);
        }

        if update.stage.total != self.stage_len {
            self.stage_len = update.stage.total;
            self.stage_pb.set_length(self.stage_len as u64);
            self.stage_num += 1;
            self.stage_pb.set_message(&self.stage_num.to_string());
        }

        self.total_pb.set_position(update.total.current as u64);
        self.stage_pb.set_position(update.stage.current as u64);

        if let Some(ref mut window) = self.window {
            let now = std::time::Instant::now();

            if now - self.last_update < self.update_freq {
                return;
            }

            self.last_update = now;

            //image to texture
            let texture: piston_window::G2dTexture = piston_window::Texture::from_image(
                &mut window.factory,
                &update.image,
                &piston_window::TextureSettings::new(),
            )
            .unwrap();

            if let Some(event) = window.next() {
                window.draw_2d(&event, |context, graphics| {
                    piston_window::clear([1.0; 4], graphics);
                    piston_window::image(
                        &texture,
                        [
                            [
                                context.transform[0][0],
                                context.transform[0][1],
                                context.transform[0][2],
                            ],
                            [
                                context.transform[1][0],
                                context.transform[1][1],
                                context.transform[1][2],
                            ],
                        ],
                        graphics,
                    );
                });
            }
        }
    }
}
