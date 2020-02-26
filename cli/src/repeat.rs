use std::{
    fs,
    path::{Path, PathBuf},
};
use structopt::StructOpt;
use texture_synthesis::{self as ts, Error, ImageSource};

#[derive(StructOpt)]
#[structopt(rename_all = "kebab-case")]
pub(crate) struct Args {
    /// Path to the transform to apply to the examples. Will default to
    /// `generate.tranform` in the current directory if not specified.
    #[structopt(long = "transform", parse(from_os_str))]
    transform: PathBuf,
    /// Path(s) to example images used to synthesize a new image
    #[structopt(parse(from_os_str))]
    examples: Vec<PathBuf>,
}

pub(crate) fn cmd(args: &Args, global_opts: &crate::Opt) -> Result<(), Error> {
    let transform_path = &args.transform;

    let xform = {
        let mut transform = fs::File::open(&transform_path).map_err(|e| {
            eprintln!(
                "unable to open transform path '{}': {}",
                transform_path.display(),
                e
            );
            e
        })?;

        ts::CoordinateTransform::read(&mut transform).map_err(|e| {
            eprintln!(
                "failed to load transform from '{}': {}",
                transform_path.display(),
                e
            );
            e
        })?
    };

    // Load the example images. We resize each of them to the dimensions of the
    // coordinate transform as each pixel can be sampled from any of them, at
    // any location within the dimensions of the transform
    let inputs = args
        .examples
        .iter()
        .map(|path| ImageSource::from_path(path));

    let new_img = xform.apply(inputs)?;

    if global_opts.output_path.to_str() == Some("-") {
        let out = std::io::stdout();
        let mut out = out.lock();

        let dyn_img = ts::image::DynamicImage::ImageRgba8(new_img);
        dyn_img.write_to(&mut out, global_opts.stdout_fmt.clone())?;
    } else {
        new_img.save(&global_opts.output_path)?;
    }

    Ok(())
}

pub(crate) fn save_coordinate_transform(
    generated: &ts::GeneratedImage,
    path: &Path,
) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    {
        let xform = generated.get_coordinate_transform();
        let mut output = std::io::BufWriter::new(std::fs::File::create(path)?);
        xform.write(&mut output)?;
    }

    Ok(())
}
