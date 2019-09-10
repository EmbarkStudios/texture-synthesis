use img_hash::{HashType, ImageHash};
use texture_synthesis as ts;

macro_rules! diff_runs {
    ($name:ident, $expected:expr, $gen:expr) => {
        #[test]
        fn $name() {
            let expected_hash = ImageHash::from_base64($expected).expect("loaded hash");

            let generated = $gen
                // We always use a single thread to ensure we get consistent results
                // across runs
                .max_thread_count(1)
                .build()
                .unwrap()
                .run(None);
            let gen_img = generated.into_image();

            let gen_hash = ImageHash::hash(&gen_img, 8, HashType::DoubleGradient);

            if gen_hash != expected_hash {
                let distance = expected_hash.dist_ratio(&gen_hash);
                let txt_gen = gen_hash.to_base64();

                assert_eq!($expected, txt_gen, "images hashes differed by {}", distance);
            }
        }
    };
}

diff_runs!(single_example, "JKc2MqWo1iNWeJ856Ty6+a1M", {
    ts::Session::builder()
        .add_example(&"../imgs/1.jpg")
        .seed(120)
        .output_size(100, 100)
});

diff_runs!(multi_example, "JFCWyK1a4vJ1eWNTQkPOmdy2", {
    ts::Session::builder()
        .add_examples(&[
            &"../imgs/multiexample/1.jpg",
            &"../imgs/multiexample/2.jpg",
            &"../imgs/multiexample/3.jpg",
            &"../imgs/multiexample/4.jpg",
        ])
        .resize_input(100, 100)
        .random_init(10)
        .seed(211)
        .output_size(100, 100)
});

diff_runs!(guided, "JBQFEgoXm5KCiWZUfHHBhyYK", {
    ts::Session::builder()
        .add_example(
            ts::Example::builder(&"../imgs/2.jpg").with_guide(&"../imgs/masks/2_example.jpg"),
        )
        .load_target_guide(&"../imgs/masks/2_target.jpg")
        .output_size(100, 100)
});

diff_runs!(style_transfer, "JEMRDSUzJ4uhpHMes1Onenz0", {
    ts::Session::builder()
        .add_example(&"../imgs/multiexample/4.jpg")
        .load_target_guide(&"../imgs/tom.jpg")
        .output_size(100, 100)
});

diff_runs!(inpaint, "JNG1tl5SaIkqauco1NEmtikk", {
    ts::Session::builder()
        .inpaint_example(
            &"../imgs/masks/3_inpaint.jpg",
            ts::Example::builder(&"../imgs/3.jpg")
                .set_sample_method(&"../imgs/masks/3_inpaint.jpg"),
        )
        .resize_input(100, 100)
        .output_size(100, 100)
});

diff_runs!(tiling, "JNSV0UiMaMzh2KotmlwojR2K", {
    ts::Session::builder()
        .inpaint_example(
            &"../imgs/masks/1_tile.jpg",
            ts::Example::new(&"../imgs/1.jpg"),
        )
        .resize_input(100, 100)
        .output_size(100, 100)
        .tiling_mode(true)
});
