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

#[cfg(not(target_arch = "wasm32"))]
mod progress_window;

mod repeat;

use structopt::StructOpt;

use std::path::PathBuf;
use texture_synthesis::{
    image::ImageOutputFormat as ImgFmt, load_dynamic_image, ChannelMask, Dims, Error, Example,
    ImageSource, SampleMethod, Session,
};

fn parse_size(input: &str) -> Result<Dims, std::num::ParseIntError> {
    let mut i = input.splitn(2, 'x');

    let width: u32 = i.next().unwrap_or("").parse()?;
    let height: u32 = match i.next() {
        Some(num) => num.parse()?,
        None => width,
    };

    Ok(Dims { width, height })
}

fn parse_img_fmt(input: &str) -> Result<ImgFmt, String> {
    let fmt = match input {
        "png" => ImgFmt::Png,
        "jpg" => ImgFmt::Jpeg(75),
        "bmp" => ImgFmt::Bmp,
        other => {
            return Err(format!(
                "image format `{}` not one of: 'png', 'jpg', 'bmp'",
                other
            ))
        }
    };

    Ok(fmt)
}

fn parse_mask(input: &str) -> Result<ChannelMask, String> {
    let mask = match &input.to_lowercase()[..] {
        "r" => ChannelMask::R,
        "g" => ChannelMask::G,
        "b" => ChannelMask::B,
        "a" => ChannelMask::A,
        mask => {
            return Err(format!(
                "unknown mask '{}', must be one of 'a', 'r', 'g', 'b'",
                mask
            ))
        }
    };

    Ok(mask)
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
    /// Saves the transforms used to generate the final output image from the
    /// input examples. This can be used by the `repeat` subcommand to reapply
    /// the same transform to different examples to get a new output image.
    #[structopt(long = "save-transform")]
    save_transform: Option<PathBuf>,
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
#[structopt(rename_all = "kebab-case")]
struct FlipAndRotate {
    /// Path(s) to example images used to synthesize a new image. Each example
    /// is rotated 4 times, and flipped once around each axis, resulting in a
    /// total of 7 example inputs per example, so it is recommended you only
    /// use 1 example input, even if you can pass as many as you like.
    #[structopt(parse(from_os_str))]
    examples: Vec<PathBuf>,
}

#[derive(StructOpt)]
enum Subcommand {
    /// Transfers the style from an example onto a target guide
    #[structopt(name = "transfer-style")]
    TransferStyle(TransferStyle),
    /// Generates a new image from 1 or more examples
    #[structopt(name = "generate")]
    Generate(Generate),
    /// Generates a new image from 1 or more examples, extended with their
    /// flipped and rotated versions
    #[structopt(name = "flip-and-rotate")]
    FlipAndRotate(FlipAndRotate),
    /// Repeats transforms from a previous generate command onto the provided
    /// inputs to generate a new output image
    #[structopt(name = "repeat")]
    Repeat(repeat::Args),
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
    #[cfg(not(target_arch = "wasm32"))]
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
    /// Flag to extract inpaint from one of the example's channels
    #[structopt(long, parse(try_from_str = parse_mask), conflicts_with = "inpaint")]
    inpaint_channel: Option<ChannelMask>,
    /// Size of the generated image, in `width x height`, or a single number for both dimensions
    #[structopt(
        long,
        default_value = "500",
        parse(try_from_str = parse_size)
    )]
    out_size: Dims,
    /// Output format detection when writing to a file is based on the extension, but when
    /// writing to stdout by passing `-` you must specify the format if you want something
    /// other than the default.
    #[structopt(
        long,
        default_value = "png",
        parse(try_from_str = parse_img_fmt)
    )]
    stdout_fmt: ImgFmt,
    /// Resize input example map(s), in `width x height`, or a single number for both dimensions
    #[structopt(long, parse(try_from_str = parse_size))]
    in_size: Option<Dims>,
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
    ///
    /// Note that setting this to `1` will allow you to generate 100%
    /// deterministic output images (considering all other inputs are
    /// the same)
    #[structopt(short = "t", long = "threads")]
    max_threads: Option<usize>,
    #[structopt(flatten)]
    tweaks: Tweaks,
    #[structopt(subcommand)]
    cmd: Subcommand,
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}

fn real_main() -> Result<(), Error> {
    let args = Opt::from_args();

    // Check that the output format or extension for the path supplied by the user is one of the ones we support
    {
        if args.output_path.to_str() != Some("-") {
            match args.output_path.extension().and_then(|ext| ext.to_str()) {
                Some("png") | Some("jpg") | Some("bmp") => {}
                Some(other) => return Err(Error::UnsupportedOutputFormat(other.to_owned())),
                None => return Err(Error::UnsupportedOutputFormat(String::new())),
            }
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
        Subcommand::FlipAndRotate(fr) => {
            let example_imgs = fr
                .examples
                .iter()
                .map(|path| load_dynamic_image(ImageSource::Path(path)))
                .collect::<Result<Vec<_>, _>>()?;

            let mut transformed: Vec<Example<'_>> = Vec::with_capacity(example_imgs.len() * 7);
            for img in &example_imgs {
                transformed.push(Example::new(img.fliph()));
                transformed.push(Example::new(img.rotate90()));
                transformed.push(Example::new(img.fliph().rotate90()));
                transformed.push(Example::new(img.rotate180()));
                transformed.push(Example::new(img.fliph().rotate180()));
                transformed.push(Example::new(img.rotate270()));
                transformed.push(Example::new(img.fliph().rotate270()));
            }

            let mut examples: Vec<_> = example_imgs.into_iter().map(Example::new).collect();
            examples.append(&mut transformed);

            (examples, None)
        }
        Subcommand::TransferStyle(ts) => (vec![Example::new(&ts.style)], Some(&ts.guide)),
        Subcommand::Repeat(rep) => {
            return repeat::cmd(rep, &args);
        }
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
                path => example.set_sample_method(SampleMethod::Image(ImageSource::from_path(
                    std::path::Path::new(path),
                ))),
            };
        }
    }

    let mut sb = Session::builder();

    // TODO: Make inpaint work with multiple examples
    match (args.inpaint_channel, &args.inpaint) {
        (Some(channel), None) => {
            let inpaint_example = examples.remove(0);

            sb = sb.inpaint_example_channel(channel, inpaint_example, args.out_size);
        }
        (None, Some(inpaint)) => {
            let mut inpaint_example = examples.remove(0);

            // If the user hasn't explicitly specified sample masks, assume they
            // want to use the same mask
            if args.sample_masks.is_empty() {
                inpaint_example.set_sample_method(inpaint);
            }

            sb = sb.inpaint_example(inpaint, inpaint_example, args.out_size);
        }
        (None, None) => {}
        (Some(_), Some(_)) => unreachable!("we prevent this combination with 'conflicts_with'"),
    }

    sb = sb
        .add_examples(examples)
        .output_size(args.out_size)
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

    if let Some(tg) = target_guide {
        sb = sb.load_target_guide(tg);
    }

    if let Some(rand_init) = args.tweaks.rand_init {
        sb = sb.random_init(rand_init);
    }

    if let Some(insize) = args.in_size {
        sb = sb.resize_input(insize);
    }

    let session = sb.build()?;

    #[cfg(not(target_arch = "wasm32"))]
    let progress: Option<Box<dyn texture_synthesis::GeneratorProgress>> =
        if !args.tweaks.no_progress {
            let progress = progress_window::ProgressWindow::new();

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

    #[cfg(target_arch = "wasm32")]
    let progress = None;

    let generated = session.run(progress);

    if let Some(ref dir) = args.debug_out_dir {
        generated.save_debug(dir)?;
    }

    if let Subcommand::Generate(gen) = args.cmd {
        if let Some(ref st_path) = gen.save_transform {
            if let Err(e) = repeat::save_coordinate_transform(&generated, st_path) {
                // Continue going, presumably the user will be ok with this
                // failing if they can at least get the actual generated image
                eprintln!(
                    "unable to save coordinate transform to '{}': {}",
                    st_path.display(),
                    e
                );
            }
        }
    }

    if args.output_path.to_str() == Some("-") {
        let out = std::io::stdout();
        let mut out = out.lock();
        generated.write(&mut out, args.stdout_fmt)?;
    } else {
        generated.save(&args.output_path)?;
    }

    Ok(())
}
