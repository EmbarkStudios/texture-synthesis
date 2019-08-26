use std::error::Error;
use std::path::Path;

pub fn load_image(
    path: &String,
    resize: &Option<(u32, u32)>,
) -> Result<image::RgbaImage, Box<dyn Error>> {
    let img = image::open(path)?;

    Ok(match resize {
        None => img.to_rgba(),
        Some(ref size) => {
            image::imageops::resize(&img.to_rgba(), size.0, size.1, image::imageops::CatmullRom)
        }
    })
}

pub fn load_image_multiple(
    paths: &[String],
    resize: &Option<(u32, u32)>,
) -> Result<Vec<image::RgbaImage>, Box<dyn Error>> {
    paths
        .iter()
        .map(|p| load_image(p, resize))
        .collect::<Result<Vec<image::RgbaImage>, Box<dyn Error>>>()
}

pub fn save_image(path: &String, image: &image::RgbaImage) -> Result<(), Box<dyn Error>> {
    let path = Path::new(&path);
    if let Some(parent_path) = path.parent() {
        std::fs::create_dir_all(&parent_path)?;
    }

    image.save(&path)?;

    Ok(())
}

pub fn load_images_as_guide_maps(
    path_vec: &[String],
    size: &Option<(u32, u32)>,
    blur_sigma: f32,
) -> Result<Vec<image::RgbaImage>, Box<dyn Error>> {
    let mut img_vec: Vec<image::RgbaImage> = Vec::new();

    for path in path_vec.iter() {
        let img = match size {
            None => image::open(path)?.blur(blur_sigma).grayscale(),
            Some(s) => image::open(path)?
                .resize(s.0, s.1, image::imageops::Triangle)
                .blur(blur_sigma)
                .grayscale(),
        };

        img_vec.push(img.to_rgba());
    }
    Ok(img_vec)
}

pub fn get_histogram(a: &[u8]) -> Vec<u32> {
    let mut hist = vec![0; 256]; //0-255 incl

    //populate the hist
    for i in 0..a.len() {
        //since RGBA image, we only care for 1st channel
        if i % 4 == 0 {
            let pixel_value = a[i];
            hist[pixel_value as usize] += 1; //increment histogram by 1
        }
    }

    hist
}

//source will be modified to fit the target
pub fn match_histograms(source: &image::RgbaImage, target: &image::RgbaImage) -> image::RgbaImage {
    let target_hist = get_histogram(&target.clone().into_raw());
    let source_hist = get_histogram(&source.clone().into_raw());

    //get commutative distrib
    let target_cdf = get_cdf(&target_hist);
    let source_cdf = get_cdf(&source_hist);

    //clone the source image, modify and return
    let mut source_modified = source.clone();
    let (dx, dy) = source.dimensions();

    for x in 0..dx {
        for y in 0..dy {
            let pixel_value = source.get_pixel(x, y)[0]; //we only care for the first channel
            let pixel_source_cdf = source_cdf[pixel_value as usize];

            //now need to find by value similar cdf in the target
            let new_pixel_val = target_cdf
                .iter()
                .position(|cdf| *cdf > pixel_source_cdf)
                .unwrap_or((pixel_value + 1) as usize) as u8
                - 1;

            let new_color: image::Rgba<u8> =
                image::Rgba([new_pixel_val, new_pixel_val, new_pixel_val, 255]);
            source_modified.put_pixel(x, y, new_color);
        }
    }

    source_modified
}

pub fn get_cdf(a: &[u32]) -> Vec<f32> {
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
