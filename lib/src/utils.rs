use crate::{Dims, Error};
use std::path::Path;

/// Helper type used to transform image sources
#[derive(Clone)]
pub enum Transformation {
    /// Flips an image horizontally
    FlipH,
    /// Flips an image vertically
    FlipV,
    /// Rotates the image by 90 degrees (clockwise)
    Rot90,
    /// Rotates the image by 180 degrees (clockwise)
    Rot180,
    /// Rotates the image by 270 degrees (clockwise)
    Rot270,
}

/// Helper type used to pass image data to the Session
#[derive(Clone)]
pub enum ImageSource<'a> {
    /// A raw buffer of image data, see `image::load_from_memory` for details
    /// on what is supported
    Memory(&'a [u8]),
    /// The path to an image to load from disk. The image format is inferred
    /// from the file extension, see `image::open` for details
    Path(&'a Path),
    /// An already loaded image that is passed directly to the generator
    Image(image::DynamicImage),
}

impl<'a> From<image::DynamicImage> for ImageSource<'a> {
    fn from(img: image::DynamicImage) -> Self {
        ImageSource::Image(img)
    }
}

impl<'a, S> From<&'a S> for ImageSource<'a>
where
    S: AsRef<Path> + 'a,
{
    fn from(path: &'a S) -> Self {
        Self::Path(path.as_ref())
    }
}

pub(crate) fn get_dynamic_image(
    src: ImageSource<'_>,
) -> Result<image::DynamicImage, image::ImageError> {
    return match src {
        ImageSource::Memory(data) => image::load_from_memory(data),
        ImageSource::Path(path) => image::open(path),
        ImageSource::Image(img) => Ok(img),
    };
}

pub(crate) fn load_image(
    src: ImageSource<'_>,
    resize: Option<Dims>,
    transformations: Vec<Transformation>,
) -> Result<image::RgbaImage, Error> {
    let mut img = get_dynamic_image(src)?;
    for t in transformations {
        match t {
            Transformation::FlipH => img = img.fliph(),
            Transformation::FlipV => img = img.flipv(),
            Transformation::Rot90 => img = img.rotate90(),
            Transformation::Rot180 => img = img.rotate180(),
            Transformation::Rot270 => img = img.rotate270(),
        }
    }
    Ok(match resize {
        None => img.to_rgba(),
        Some(ref size) => {
            use image::GenericImageView;

            if img.width() != size.width || img.height() != size.height {
                image::imageops::resize(
                    &img.to_rgba(),
                    size.width,
                    size.height,
                    image::imageops::CatmullRom,
                )
            } else {
                img.to_rgba()
            }
        }
    })
}

pub(crate) fn transform_to_guide_map(
    image: image::RgbaImage,
    size: Option<Dims>,
    blur_sigma: f32,
) -> image::RgbaImage {
    use image::GenericImageView;
    let dyn_img = image::DynamicImage::ImageRgba8(image);

    if let Some(s) = size {
        if dyn_img.width() != s.width || dyn_img.height() != s.height {
            dyn_img.resize(s.width, s.height, image::imageops::Triangle);
        }
    }

    dyn_img.blur(blur_sigma).grayscale().to_rgba()
}

pub(crate) fn get_histogram(img: &image::RgbaImage) -> Vec<u32> {
    let mut hist = vec![0; 256]; //0-255 incl

    let pixels = &img;

    //populate the hist
    for pixel_value in pixels
        .iter()
        .step_by(/*since RGBA image, we only care for 1st channel*/ 4)
    {
        hist[*pixel_value as usize] += 1; //increment histogram by 1
    }

    hist
}

//source will be modified to fit the target
pub(crate) fn match_histograms(source: &mut image::RgbaImage, target: &image::RgbaImage) {
    let target_hist = get_histogram(target);
    let source_hist = get_histogram(source);

    //get commutative distrib
    let target_cdf = get_cdf(&target_hist);
    let source_cdf = get_cdf(&source_hist);

    //clone the source image, modify and return
    let (dx, dy) = source.dimensions();

    for x in 0..dx {
        for y in 0..dy {
            let pixel_value = source.get_pixel(x, y)[0]; //we only care about the first channel
            let pixel_source_cdf = source_cdf[pixel_value as usize];

            //now need to find by value similar cdf in the target
            let new_pixel_val = target_cdf
                .iter()
                .position(|cdf| *cdf > pixel_source_cdf)
                .unwrap_or((pixel_value + 1) as usize) as u8
                - 1;

            let new_color: image::Rgba<u8> =
                image::Rgba([new_pixel_val, new_pixel_val, new_pixel_val, 255]);
            source.put_pixel(x, y, new_color);
        }
    }
}

pub(crate) fn get_cdf(a: &[u32]) -> Vec<f32> {
    let mut cumm = vec![0.0; 256];

    for i in 0..a.len() {
        if i != 0 {
            cumm[i] = cumm[i - 1] + (a[i] as f32);
        } else {
            cumm[i] = a[i] as f32;
        }
    }

    //normalize
    let max = cumm[255];
    for i in cumm.iter_mut() {
        *i /= max;
    }

    cumm
}
