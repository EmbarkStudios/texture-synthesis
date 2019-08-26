pub struct ImagePyramid {
    pub pyramid: Vec<image::RgbaImage>,
    pub levels: u32,
}

impl ImagePyramid {
    pub fn new(in_img: &image::RgbaImage, levels: Option<u32>) -> Self {
        let lvls = levels.unwrap_or_else(|| {
            //auto-calculate max number of downsampling
            let (dimx, dimy) = in_img.dimensions();
            (f64::from(dimx.max(dimy))).log2() as u32 // pow(2, x) ~ img => x ~ log2(img)
        });

        Self {
            pyramid: ImagePyramid::build_gaussian(lvls, in_img),
            levels: lvls,
        }
    }

    //build gaussian pyramid by downsampling the image by 2
    fn build_gaussian(in_lvls: u32, in_img: &image::RgbaImage) -> Vec<image::RgbaImage> {
        let mut imgs = Vec::new();
        let (dimx, dimy) = in_img.dimensions();

        //going from lowest to largest resolution (to match the texture synthesis generation order)
        for i in (0..in_lvls).rev() {
            let p = u32::pow(2, i);
            if i > 0 {
                imgs.push(image::imageops::resize(
                    &image::imageops::resize(in_img, dimx / p, dimy / p, image::imageops::Gaussian),
                    dimx,
                    dimy,
                    image::imageops::Gaussian,
                ));
            } else {
                imgs.push(in_img.clone());
            }
        }

        imgs
    }

    pub fn reconstruct(&self) -> image::RgbaImage {
        self.gaussian_reconstruct()
    }

    fn gaussian_reconstruct(&self) -> image::RgbaImage {
        self.pyramid[self.levels as usize - 1].clone()
    }
}
