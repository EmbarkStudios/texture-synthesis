use structopt::StructOpt;

use std::path::PathBuf;
use texture_synthesis::{
    image::ImageOutputFormat as ImgFmt, Error, Example, ImageSource, SampleMethod, Session,
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
#[structopt(rename_all = "kebab-case")]
struct Generate {
    /// A target guidance map
    #[structopt(long, parse(from_os_str))]
    target_guide: Option<PathBuf>,
    /// Path(s) to guide maps for the example output.
    #[structopt(long = "guides", parse(from_os_str))]
    example_guides: Vec<PathBuf>,
    /// Path(s) to example images used to synthesize a new image
    #[structopt(parse(from_os_str))]
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
#[structopt(rename_all = "kebab-case")]
struct Tweaks {
    /// The number of neighboring pixels each pixel is aware of during the generation,
    /// larger numbers means more global structures are captured.
    #[structopt(long = "k-neighs", default_value = "50")]
    k_neighbors: u32,
    /// The number of random locations that will be considered during a pixel resolution,
    /// apart from its immediate neighbors. If unsure of this parameter, keep as the same as k-neigh.
    #[structopt(long, default_value = "50")]
    m_rand: u64,
    /// The distribution dispersion used for picking best candidate (controls the distribution 'tail flatness').
    /// Values close to 0.0 will produce 'harsh' borders between generated 'chunks'.
    /// Values closer to 1.0 will produce a smoother gradient on those borders.
    #[structopt(long, default_value = "1.0")]
    cauchy: f32,
    /// The percentage of pixels to be backtracked during each p_stage. Range (0.0, 1.0).
    #[structopt(long = "backtrack-pct", default_value = "0.5")]
    backtrack_percentage: f32,
    /// The number of backtracking stages. Backtracking prevents 'garbage' generation.
    #[structopt(long = "backtrack-stages", default_value = "5")]
    backtrack_stages: u32,
    #[structopt(long = "window")]
    #[cfg(feature = "progress")]
    #[cfg_attr(feature = "progress", structopt(long = "window"))]
    #[cfg_attr(
        feature = "progress",
        doc = "Show a window with the current progress of the generation"
    )]
    show_window: bool,
    /// Show a window with the current progress of the generation
    #[structopt(long)]
    no_progress: bool,
    /// A seed value for the random generator to give pseudo-deterministic result.
    /// Smaller details will be different from generation to generation due to the
    /// non-deterministic nature of multi-threading
    #[structopt(long)]
    seed: Option<u64>,
    /// Alpha parameter controls the 'importance' of the user guide maps. If you want
    /// to preserve more details from the example map, make sure the number < 1.0. Range (0.0 - 1.0)
    #[structopt(long, default_value = "0.8")]
    alpha: f32,
    /// The number of randomly initialized pixels before the main resolve loop starts
    #[structopt(long)]
    rand_init: Option<u64>,
    /// Enables tiling of the output image
    #[structopt(long = "tiling")]
    enable_tiling: bool,
}

#[derive(StructOpt)]
#[structopt(
    name = "texture-synthesis",
    about = "Synthesizes images based on example images",
    rename_all = "kebab-case"
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
        long,
        default_value = "500",
        parse(try_from_str = parse_size)
    )]
    out_size: (u32, u32),
    /// The format to save the generated image as.
    ///
    /// NOTE: this will only apply when stdout is specified via `-o -`, otherwise the image
    /// format is determined by the file extension of the path provided to `-o`
    #[structopt(
        long,
        default_value = "png",
        parse(try_from_str = parse_img_fmt)
    )]
    out_fmt: ImgFmt,
    /// Resize input example map(s), in `width x height`, or a single number for both dimensions
    #[structopt(long, parse(try_from_str = parse_size))]
    in_size: Option<(u32, u32)>,
    /// The path to save the generated image to, the file extensions of the path determines
    /// the image format used. You may use `-` for stdout.
    #[structopt(long = "out", short, parse(from_os_str))]
    output_path: PathBuf,
    /// A directory into which debug images are also saved.
    ///
    /// * `patch_id.png` - Map of the `copy islands` from an example
    /// * `map_id.png` - Map of ids of which example was the source of a pixel
    /// * `uncertainty.png` - Map of pixels the generator was uncertain of
    #[structopt(long, parse(from_os_str))]
    debug_out_dir: Option<PathBuf>,
    /// The maximum number of worker threads that can be active at any one time
    /// while synthesizing images. Defaults to the logical core count.
    #[structopt(short = "t", long = "threads")]
    max_threads: Option<u32>,
    #[structopt(flatten)]
    tweaks: Tweaks,
    #[structopt(subcommand)]
    cmd: Subcommand,
}

fn main() {
    if let Err(e) = real_main() {
        if atty::is(atty::Stream::Stderr) {
            eprintln!("\x1b[31merror\x1b[0m: {}", e);
        } else {
            eprintln!("error: {}", e);
        }

        std::process::exit(1);
    }
}

fn real_main() -> Result<(), Error> {
    let args = Opt::from_args();

    // Check that the extension for the path supplied by the user is one of the ones we support
    {
        match args.output_path.extension().and_then(|ext| ext.to_str()) {
            Some("png") | Some("jpg") | Some("bmp") => {}
            None => {}
            Some(other) => return Err(Error::UnsupportedOutputFormat(other.to_owned())),
        }
    }

    let (mut examples, target_guide) = match &args.cmd {
        Subcommand::Generate(gen) => {
            let mut examples: Vec<_> = gen.examples.iter().map(Example::new).collect();
            if !gen.example_guides.is_empty() {
                for (ex, guide) in examples.iter_mut().zip(gen.example_guides.iter()) {
                    ex.with_guide(guide);
                }
            }

            (examples, gen.target_guide.as_ref())
        }
        Subcommand::TransferStyle(ts) => (vec![Example::new(&ts.style)], Some(&ts.guide)),
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

        // If the user hasn't explicitly specified sample masks, assume they
        // want to use the same mask
        if args.sample_masks.is_empty() {
            inpaint_example.set_sample_method(inpaint);
        }

        sb = sb.inpaint_example(inpaint, inpaint_example);
    }

    sb = sb
        .add_examples(examples)
        .output_size(args.out_size.0, args.out_size.1)
        .seed(args.tweaks.seed.unwrap_or_default())
        .nearest_neighbors(args.tweaks.k_neighbors)
        .random_sample_locations(args.tweaks.m_rand)
        .cauchy_dispersion(args.tweaks.cauchy)
        .backtrack_percent(args.tweaks.backtrack_percentage)
        .backtrack_stages(args.tweaks.backtrack_stages)
        .guide_alpha(args.tweaks.alpha)
        .tiling_mode(args.tweaks.enable_tiling);

    if let Some(mt) = args.max_threads {
        sb = sb.max_thread_count(mt);
    }

    if let Some(ref tg) = target_guide {
        sb = sb.load_target_guide(tg);
    }

    if let Some(rand_init) = args.tweaks.rand_init {
        sb = sb.random_init(rand_init);
    }

    if let Some(insize) = args.in_size {
        sb = sb.resize_input(insize.0, insize.1);
    }

    let session = sb.build()?;

    let progress: Option<Box<dyn texture_synthesis::GeneratorProgress>> =
        if !args.tweaks.no_progress {
            let progress = ProgressWindow::new();

            #[cfg(feature = "progress")]
            let progress = {
                if args.tweaks.show_window {
                    progress.with_preview(args.out_size, std::time::Duration::from_millis(100))?
                } else {
                    progress
                }
            };

            Some(Box::new(progress))
        } else {
            None
        };

    let generated = session.run(progress);

    if let Some(ref dir) = args.debug_out_dir {
        generated.save_debug(dir)?;
    }

    if args.output_path.to_str() == Some("-") {
        let out = std::io::stdout();
        let mut out = out.lock();
        generated.write(&mut out, args.out_fmt)?;
    } else {
        // This won't respect the output format specified by the user,
        // only the extension on the path they specify, but that makes
        // more sense, and is probably better than detecting and emitting
        // an error
        generated.save(&args.output_path)?;
    }

    Ok(())
}

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
#[cfg(feature = "progress")]
use minifb::Window;

pub struct ProgressWindow {
    #[cfg(feature = "progress")]
    window: Option<(Window, std::time::Duration, std::time::Instant)>,

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
            #[cfg(feature = "progress")]
            window: None,
            total_pb,
            stage_pb,
            total_len: 100,
            stage_len: 100,
            stage_num: 0,
        }
    }

    #[cfg(feature = "progress")]
    fn with_preview(
        mut self,
        size: (u32, u32),
        update_every: std::time::Duration,
    ) -> Result<Self, Error> {
        let window = Window::new(
            "Texture Synthesis",
            size.0 as usize,
            size.1 as usize,
            minifb::WindowOptions::default(),
        )
        .unwrap();

        self.window = Some((window, update_every, std::time::Instant::now()));

        Ok(self)
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

        #[cfg(feature = "progress")]
        {
            if let Some((ref mut window, ref dur, ref mut last_update)) = self.window {
                let now = std::time::Instant::now();

                if now - *last_update < *dur {
                    return;
                }

                *last_update = now;

                if !window.is_open() {
                    return;
                }

                let pixels = &update.image;
                if pixels.len() % 4 != 0 {
                    return;
                }

                // The pixel channels are in a different order so the colors are
                // incorrect in the window, but at least the shape and unfilled pixels
                // are still apparent
                let pixels: &[u32] = unsafe {
                    let raw_pixels: &[u8] = pixels;
                    #[allow(clippy::cast_ptr_alignment)]
                    std::mem::transmute(&*(raw_pixels as *const [u8] as *const [u32]))
                };

                // We don't particularly care if this fails
                let _ = window.update_with_buffer(pixels);
            }
        }
    }
}
