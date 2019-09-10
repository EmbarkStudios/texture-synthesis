use img_hash::{HashType, ImageHash};
use texture_synthesis as ts;

macro_rules! diff_runs {
    ($name:ident, $expected:expr, $gen:expr, $ratio:expr) => {
        #[test]
        fn $name() {
            let expected_hash = ImageHash::from_base64($expected).expect("loaded hash");

            let generated = $gen.build().unwrap().run(None);
            let gen_img = generated.into_image();

            let gen_hash = ImageHash::hash(&gen_img, 8, HashType::DoubleGradient);

            let distance = expected_hash.dist_ratio(&gen_hash);

            if distance > $ratio {
                let txt_gen = gen_hash.to_base64();

                assert!(
                    false,
                    "img difference {} exceeded allowed ratio {}: {}",
                    distance, $ratio, txt_gen,
                );
            }
        }
    };
}

diff_runs!(
    single_example,
    "JKc2MqWo1iNWeJ856Ty6+a1M",
    {
        ts::Session::builder()
            .add_example(&"../imgs/1.jpg")
            .seed(120)
            .output_size(100, 100)
    },
    0.01
);

diff_runs!(
    multi_example,
    "JNq6/DSYlJbAorIgaUKKMEqo",
    {
        ts::Session::builder()
            .add_examples(&[
                &"../imgs/multiexample/1.jpg",
                &"../imgs/multiexample/2.jpg",
                &"../imgs/multiexample/3.jpg",
                &"../imgs/multiexample/4.jpg",
            ])
            .resize_input(300, 300)
            //.random_init(10)
            .seed(211)
            .output_size(100, 100)
    },
    0.01
);

diff_runs!(
    guided,
    "JBQFEwoXmpiWmUZUfPFhgwUK",
    {
        ts::Session::builder()
            .add_example(
                ts::Example::builder(&"../imgs/2.jpg").with_guide(&"../imgs/masks/2_example.jpg"),
            )
            .load_target_guide(&"../imgs/masks/2_target.jpg")
            .output_size(100, 100)
    },
    0.175
);

diff_runs!(
    style_transfer,
    "JEFRCyUrFwuhpGOeszKnanz0",
    {
        ts::Session::builder()
            .add_example(&"../imgs/multiexample/4.jpg")
            .load_target_guide(&"../imgs/tom.jpg")
            .output_size(100, 100)
    },
    0.15
);

diff_runs!(
    inpaint,
    "JNG1tl5SaIkqauco1NEmtSkk",
    {
        ts::Session::builder()
            .inpaint_example(
                &"../imgs/masks/3_inpaint.jpg",
                ts::Example::builder(&"../imgs/3.jpg")
                    .set_sample_method(&"../imgs/masks/3_inpaint.jpg"),
            )
            .resize_input(100, 100)
            .output_size(100, 100)
    },
    0.03
);

diff_runs!(
    tiling,
    "JNQV0UiMaMzh2KotmlwojR2K",
    {
        ts::Session::builder()
            .inpaint_example(
                &"../imgs/masks/1_tile.jpg",
                ts::Example::new(&"../imgs/1.jpg"),
            )
            .resize_input(100, 100)
            .output_size(100, 100)
            .tiling_mode(true)
    },
    0.03
);
