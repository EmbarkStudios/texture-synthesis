#[cfg(feature = "progress")]
use texture_synthesis::{Dims, Error};

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
    pub fn new() -> Self {
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
    pub fn with_preview(
        mut self,
        size: Dims,
        update_every: std::time::Duration,
    ) -> Result<Self, Error> {
        let window = Window::new(
            "Texture Synthesis",
            size.width as usize,
            size.height as usize,
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
                let _ = window.update_with_buffer(
                    pixels,
                    update.image.width() as usize,
                    update.image.height() as usize,
                );
            }
        }
    }
}
