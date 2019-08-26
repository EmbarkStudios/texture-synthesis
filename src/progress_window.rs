use crate::multires_stochastic_texture_synthesis::GeneratorProgress;
use std::time::Duration;

pub struct WindowStruct {
    window: piston_window::PistonWindow,
    update_freq: Duration, //in sec
}

impl WindowStruct {
    pub fn new(size: (u32, u32), update_every: Duration) -> Self {
        let mut my_return = Self {
            window: piston_window::WindowSettings::new("Texture Synthesis", [size.0, size.1])
                .exit_on_esc(true)
                .build()
                .unwrap(),
            update_freq: update_every,
        };
        use piston_window::*;
        my_return.window.set_bench_mode(true); //disallow sleeping
        my_return
    }
}

impl GeneratorProgress for WindowStruct {
    fn update(&mut self, in_image: &image::RgbaImage) {
        //image to texture
        let texture: piston_window::G2dTexture = piston_window::Texture::from_image(
            &mut self.window.factory,
            &in_image,
            &piston_window::TextureSettings::new(),
        )
        .unwrap();

        if let Some(event) = self.window.next() {
            self.window.draw_2d(&event, |context, graphics| {
                //clear([1.0; 4], graphics);
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

        std::thread::sleep(self.update_freq);
    }
}