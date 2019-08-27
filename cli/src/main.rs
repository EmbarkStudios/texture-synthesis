use clap::{App, Arg};
use rand::Rng;
use std::path::Path;

fn parse_size(input: &str) -> (u32, u32) {
    let a: Vec<&str> = input.split('x').collect();
    let x: u32 = a[0].parse().expect("couldn't parse size");
    let y: u32 = a[1].parse().expect("couldn't parse size");
    (x, y)
}

fn parse_string_vector(input: &str) -> Vec<&str> {
    input.split(',').collect()
}

fn main() {
    let user_params = App::new("Rget")
        .version("0.1.0")
        .author("anastasia opara and tomasz stachowiak")
        .about("texture synthesis")
        .arg(
            Arg::with_name("Example Map Path")
                .long("examples")
                .required(true)
                .takes_value(true)
                .help("Path to the example image(s). If multiple please seperate with a comma with no space. Example: img/1.jpg,img/5.jpg")
        ).arg(
            Arg::with_name("Inpaint Map Path")
                .long("inpaint")
                .takes_value(true)
                .help("Path to the inpaint map image, where black is pixels to resolve, and white pixels to keep")
        ).arg(
            Arg::with_name("Valid Samples Map Path")
                .long("sample-masks")
                .takes_value(true)
                .help("Path to the inpaint map image, where black is pixels to resolve, and white pixels to keep. If no valid map is availale, you can put it as 'None'")
        ).arg(
            Arg::with_name("Example Guidance Map Path")
                .long("example-guide")
                .takes_value(true)
                .help("Path to the example guidance map(s)")
        ).arg(
            Arg::with_name("Target Guidance Map Path")
                .long("target-guide")
                .takes_value(true)
                .help("Path to the target guidance map")
        ).arg(
            Arg::with_name("Output Size")
                .long("out-size")
                .takes_value(true)
                .help("Size of the generated image. Please separate with 'x' (ex '100x100'). Default: 500x500")
        ).arg(
            Arg::with_name("Input Size")
                .long("in-size")
                .takes_value(true)
                .help("Resizes the example map. Please separate with 'x' (ex '100x100'). Default: input img size")
        ).arg(
            Arg::with_name("Save Path")
                .long("save")
                .takes_value(true)
                .help("Save path for the generated image (include name, ex final.jpg). Default: 'generated.jpg'")
        ).arg(
            Arg::with_name("K Nearest Neighbours")
                .long("k-neighs")
                .takes_value(true)
                .help("How many neighbouring pixels each pixel is aware of during the generation (bigger number -> more global structures are captured). Default: 20")
        ).arg(
            Arg::with_name("Random M Sample Locations")
                .long("m-rand")
                .takes_value(true)
                .help("How many random locations will be considered during a pixel resolution apart from its immediate neighbours (if unsure, keep same as k-neighbours). Default: 20")
        ).arg(
            Arg::with_name("Cauchy Dispersion")
                .long("cauchy")
                .takes_value(true)
                .help("The distribution dispersion used for picking best candidate (controls the distribution 'tail flatness'). Values close to 0.0 will produce 'harsh' borders between generated 'chunks'. Values  closer to 1.0 will produce a smoother gradient on those borders. Default: 1.0")
        ).arg(
            Arg::with_name("Backtracking Percentage")
                .long("backtrack-p")
                .takes_value(true)
                .help("The percentage of pixels to be backtracked during each p_stage. Range (0,1). Default: 0.35")
        ).arg(
            Arg::with_name("Backtracking Stages")
                .long("backtrack-s")
                .takes_value(true)
                .help("Controls the number of backtracking stages. Backtracking prevents 'garbage' generation. Default: 5")
        ).arg(
            Arg::with_name("no-window")
                .long("no-window")
                .help("Disable showing progress with window")
        ).arg(
            Arg::with_name("Seed")
                .long("seed")
                .takes_value(true)
                .help("Random seed. Gives pseudo-deterministic result. Smaller details will be different from generation to generation due to the nondeterministic nature of multi-threading")
        ).arg(
            Arg::with_name("Alpha")
                .long("alpha")
                .takes_value(true)
                .help("Alpha parameter controls the 'importance' of the user guide maps. If you want to preserve more details from the example map, make sure the number < 1.0. Range (0.0 - 1.0)")
        ).arg(
            Arg::with_name("Random Init")
                .long("rand-init")
                .takes_value(true)
                .help("The number of randomly initialized pixels before the main resolve loop starts")
        ).arg(
            Arg::with_name("Debug Maps")
                .long("debug-maps")
                .help("Outputs patch_id, map_id, uncertainty")
        ).arg(
            Arg::with_name("Tiling")
                .long("tiling")
                .help("Enable tiling")
        )
        .get_matches();

    let outsize = parse_size(user_params.value_of("Output Size").unwrap_or("500x500"));
    let save_path = String::from(
        user_params
            .value_of("Save Path")
            .unwrap_or("out/generated.jpg"),
    );

    let mut tex_synth = texture_synthesis::Session::new()
        .load_examples(&parse_string_vector(
            user_params
                .value_of("Example Map Path")
                .expect("couldn't parse examples paths"),
        ))
        .seed(
            user_params
                .value_of("Seed")
                .unwrap_or(&rand::thread_rng().gen::<u64>().to_string())
                .parse::<u64>()
                .expect("couldn't parse seed"),
        )
        .output_size(outsize.0, outsize.1)
        .nearest_neighbours(
            user_params
                .value_of("K Nearest Neighbours")
                .unwrap_or("20")
                .parse::<u32>()
                .expect("couldn't parse k neigh"),
        )
        .random_sample_locations(
            user_params
                .value_of("Random M Sample Locations")
                .unwrap_or("20")
                .parse::<u64>()
                .expect("couldn't parse m-rand"),
        )
        .cauchy_dispersion(
            user_params
                .value_of("Cauchy Dispersion")
                .unwrap_or("1.0")
                .parse::<f32>()
                .expect("couldn't parse caushy"),
        )
        .backtrack_percent(
            user_params
                .value_of("Backtracking Percentage")
                .unwrap_or("0.35")
                .parse::<f32>()
                .expect("couldn't parse backtrack-p"),
        )
        .backtrack_stages(
            user_params
                .value_of("Backtracking Stages")
                .unwrap_or("5")
                .parse::<u32>()
                .expect("couldn't parse backtrack-s"),
        )
        .guide_alpha(
            user_params
                .value_of("Alpha")
                .unwrap_or("1.0")
                .parse::<f32>()
                .expect("couldn't parse alpha"),
        );

    if user_params.is_present("Example Guidance Map Path") {
        tex_synth = tex_synth.load_example_guides(&parse_string_vector(
            user_params
                .value_of("Example Guidance Map Path")
                .expect("Couldn't parse example guide maps"),
        ));
    }

    if user_params.is_present("Target Guidance Map Path") {
        tex_synth = tex_synth.load_target_guide(
            &user_params
                .value_of("Target Guidance Map Path")
                .expect("Couldn't parse target guide map"),
        );
    }

    if user_params.is_present("Valid Samples Map Path") {
        tex_synth = tex_synth.load_sampling_masks(&parse_string_vector(
            &user_params
                .value_of("Valid Samples Map Path")
                .expect("Couldn't parse sampling masks"),
        ));
    }

    if user_params.is_present("Inpaint Map Path") {
        tex_synth = tex_synth.inpaint_example(
            &user_params
                .value_of("Inpaint Map Path")
                .expect("Couldn't parse inpaint mask"),
            0,
        );
    }

    if user_params.is_present("Random Init") {
        tex_synth = tex_synth.random_init(
            user_params
                .value_of("Random Init")
                .unwrap_or("0")
                .parse::<u64>()
                .expect("couldn't parse random init"),
        );
    }

    if user_params.is_present("Tiling") {
        tex_synth = tex_synth.tiling_mode(true);
    }

    if user_params.is_present("Input Size") {
        let resize = parse_size(
            user_params
                .value_of("Input Size")
                .expect("Couldn't parse input size"),
        );
        tex_synth = tex_synth.resize_input(resize.0, resize.1);
    }

    let preview = if !user_params.is_present("no-window") {
        Some(create_progress_window(outsize, std::time::Duration::from_millis(100)))
    } else {
        None
    };

    tex_synth.run(preview).unwrap();
    tex_synth.save(&save_path).unwrap();

    if user_params.is_present("Debug Maps") {
        let parent_path = Path::new(&save_path)
            .parent()
            .expect("couldnt make a path for debug imgs");
        tex_synth.save_debug(parent_path.to_str().unwrap()).unwrap();
    }
}

fn create_progress_window(size: (u32, u32), update_every: std::time::Duration) -> Box<dyn texture_synthesis::GeneratorProgress> {
    use std::time::Duration;

    pub struct ProgressWindow {
        window: piston_window::PistonWindow,
        update_freq: Duration,
        last_update: std::time::Instant,
    }

    impl ProgressWindow {
        fn new(size: (u32, u32), update_every: Duration) -> Self {
            let mut my_return = Self {
                window: piston_window::WindowSettings::new("Texture Synthesis", [size.0, size.1])
                    .exit_on_esc(true)
                    .build()
                    .unwrap(),
                update_freq: update_every,
                last_update: std::time::Instant::now(),
            };
            use piston_window::*;
            my_return.window.set_bench_mode(true); //disallow sleeping
            my_return
        }
    }

    impl texture_synthesis::GeneratorProgress for ProgressWindow {
        fn update(&mut self, in_image: &texture_synthesis::image::RgbaImage) {
            let now = std::time::Instant::now();

            if now - self.last_update < self.update_freq {
                return;
            }

            self.last_update = now;

            //image to texture
            let texture: piston_window::G2dTexture = piston_window::Texture::from_image(
                &mut self.window.factory,
                &in_image,
                &piston_window::TextureSettings::new(),
            )
            .unwrap();

            if let Some(event) = self.window.next() {
                self.window.draw_2d(&event, |context, graphics| {
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

    Box::new(ProgressWindow::new(size, update_every))
}
