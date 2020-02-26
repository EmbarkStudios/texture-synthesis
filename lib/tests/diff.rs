use img_hash::{HashType, ImageHash};
use texture_synthesis as ts;
use ts::Dims;

// The tests below each run the different example code we have and
// compare the image hash against a "known good" hash. The test
// generation is run in a single thread, with the same seed and other
// parameters to produce a consistent hash between runs to detect
// regressions when making changes to the generator. If the hashes
// don't match, the right hand side of the assertion message will
// be the hash of the generated output, which you can copy and paste
// into the second parameter of the test macro if you intended
// to make an algorithm/parameter change that affects output, but
// please don't just update solely to get the test to pass!

// For example, if cargo test output this
//
// thread 'single_example' panicked at 'assertion failed: `(left == right)`
//   left: `"JKc2MqWo1iNWeJ856Ty6+a2M"`,
//  right: `"JKc2MqWo1iNWeJ856Ty6+a1M"`: images hashes differed by 0.014814815', lib/tests/diff.rs:46:1
//
// You would copy `JKc2MqWo1iNWeJ856Ty6+a1M` and paste it over the hash for `single_example` to
// update the hash

use image::{
    imageops, imageops::FilterType, DynamicImage, GenericImageView, GrayImage, Pixel, RgbaImage,
};
use img_hash::HashImage;
const FILTER_TYPE: FilterType = FilterType::Nearest;

struct MyPrecious<T>(T);

impl HashImage for MyPrecious<DynamicImage> {
    type Grayscale = MyPrecious<GrayImage>;

    fn dimensions(&self) -> (u32, u32) {
        <DynamicImage as GenericImageView>::dimensions(&self.0)
    }

    fn resize(&self, width: u32, height: u32) -> Self {
        Self(self.0.resize(width, height, FILTER_TYPE))
    }

    fn grayscale(&self) -> Self::Grayscale {
        MyPrecious(imageops::grayscale(&self.0))
    }

    fn to_bytes(self) -> Vec<u8> {
        self.0.to_bytes()
    }

    fn channel_count() -> u8 {
        <<DynamicImage as GenericImageView>::Pixel as Pixel>::CHANNEL_COUNT
    }

    fn foreach_pixel<F>(&self, mut iter_fn: F)
    where
        F: FnMut(u32, u32, &[u8]),
    {
        for (x, y, px) in self.0.pixels() {
            iter_fn(x, y, px.channels());
        }
    }
}

impl HashImage for MyPrecious<GrayImage> {
    type Grayscale = MyPrecious<GrayImage>;

    fn dimensions(&self) -> (u32, u32) {
        <GrayImage as GenericImageView>::dimensions(&self.0)
    }

    fn resize(&self, width: u32, height: u32) -> Self {
        Self(imageops::resize(&self.0, width, height, FILTER_TYPE))
    }

    fn grayscale(&self) -> Self::Grayscale {
        Self(imageops::grayscale(&self.0))
    }

    fn to_bytes(self) -> Vec<u8> {
        self.0.into_raw()
    }

    fn channel_count() -> u8 {
        <<GrayImage as GenericImageView>::Pixel as Pixel>::CHANNEL_COUNT
    }

    fn foreach_pixel<F>(&self, mut iter_fn: F)
    where
        F: FnMut(u32, u32, &[u8]),
    {
        for (x, y, px) in self.0.enumerate_pixels() {
            iter_fn(x, y, px.channels());
        }
    }
}

impl HashImage for MyPrecious<RgbaImage> {
    type Grayscale = MyPrecious<GrayImage>;

    fn dimensions(&self) -> (u32, u32) {
        <RgbaImage as GenericImageView>::dimensions(&self.0)
    }

    fn resize(&self, width: u32, height: u32) -> Self {
        Self(imageops::resize(&self.0, width, height, FILTER_TYPE))
    }

    fn grayscale(&self) -> Self::Grayscale {
        MyPrecious(imageops::grayscale(&self.0))
    }

    fn to_bytes(self) -> Vec<u8> {
        self.0.into_raw()
    }

    fn channel_count() -> u8 {
        <<RgbaImage as GenericImageView>::Pixel as Pixel>::CHANNEL_COUNT
    }

    fn foreach_pixel<F>(&self, mut iter_fn: F)
    where
        F: FnMut(u32, u32, &[u8]),
    {
        for (x, y, px) in self.0.enumerate_pixels() {
            iter_fn(x, y, px.channels());
        }
    }
}

macro_rules! diff_hash {
    // $name - The name of the test
    // $expected - A base64 encoded string of our expected image hash
    // $gen - A session builder, note that the max_thread_count is always
    // set to 1 regardless
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
            let gen_hash = ImageHash::hash(&MyPrecious(gen_img), 8, HashType::DoubleGradient);

            if gen_hash != expected_hash {
                let distance = expected_hash.dist_ratio(&gen_hash);
                let txt_gen = gen_hash.to_base64();

                assert_eq!($expected, txt_gen, "images hashes differed by {}", distance);
            }
        }
    };
}

diff_hash!(single_example, "JKc2MqWo1iNWeJ856Ty6+a1M", {
    ts::Session::builder()
        .add_example(&"../imgs/1.jpg")
        .seed(120)
        .output_size(Dims::square(100))
});

diff_hash!(multi_example, "JFCWyK1a4vJ1eWNTQkPOmdy2", {
    ts::Session::builder()
        .add_examples(&[
            &"../imgs/multiexample/1.jpg",
            &"../imgs/multiexample/2.jpg",
            &"../imgs/multiexample/3.jpg",
            &"../imgs/multiexample/4.jpg",
        ])
        .resize_input(Dims::square(100))
        .random_init(10)
        .seed(211)
        .output_size(Dims::square(100))
});

diff_hash!(guided, "JBQFEQoXm5CCiWZUfHHBhweK", {
    ts::Session::builder()
        .add_example(
            ts::Example::builder(&"../imgs/2.jpg").with_guide(&"../imgs/masks/2_example.jpg"),
        )
        .load_target_guide(&"../imgs/masks/2_target.jpg")
        .output_size(Dims::square(100))
});

diff_hash!(style_transfer, "JEMRDSUzJ4uhpHMes1Onenz0", {
    ts::Session::builder()
        .add_example(&"../imgs/multiexample/4.jpg")
        .load_target_guide(&"../imgs/tom.jpg")
        .output_size(Dims::square(100))
});

diff_hash!(inpaint, "JNG1tl5SaIkqauco1NEmtikk", {
    ts::Session::builder().inpaint_example(
        &"../imgs/masks/3_inpaint.jpg",
        ts::Example::builder(&"../imgs/3.jpg").set_sample_method(&"../imgs/masks/3_inpaint.jpg"),
        Dims {
            width: 100,
            height: 100,
        },
    )
});

diff_hash!(inpaint_channel, "JOVF4dThzPKa2suWLo1OWrKk", {
    ts::Session::builder().inpaint_example_channel(
        ts::ChannelMask::A,
        &"../imgs/bricks.png",
        ts::Dims::square(400),
    )
});

diff_hash!(tiling, "JFSVUUmMaMzhWSttmlwojR1q", {
    ts::Session::builder()
        .inpaint_example(
            &"../imgs/masks/1_tile.jpg",
            ts::Example::new(&"../imgs/1.jpg"),
            Dims {
                width: 100,
                height: 100,
            },
        )
        .tiling_mode(true)
});

diff_hash!(sample_masks, "JLO1hQBEpakECqIXDiCkqBME", {
    ts::Session::builder()
        .add_example(
            ts::Example::builder(&"../imgs/4.png")
                .set_sample_method(&"../imgs/masks/4_sample_mask.png"),
        )
        .seed(211)
        .output_size(Dims::square(100))
});

diff_hash!(sample_masks_ignore, "JGgWBEwJiqCaKpEiAonGkQRE", {
    ts::Session::builder()
        .add_example(
            ts::Example::builder(&"../imgs/4.png").set_sample_method(ts::SampleMethod::Ignore),
        )
        .add_example(
            ts::Example::builder(&"../imgs/5.png").set_sample_method(ts::SampleMethod::All),
        )
        .seed(211)
        .output_size(Dims::square(200))
});

#[test]
fn repeat_transform() {
    //create a new session
    let texsynth = ts::Session::builder()
        .add_example(&"../imgs/1.jpg")
        .build()
        .unwrap();

    let generated = texsynth.run(None);

    let repeat_transform_img = generated
        .get_coordinate_transform()
        .apply(&["../imgs/1.jpg"])
        .unwrap();

    let hash_synthesized = ImageHash::hash(
        &MyPrecious(generated.into_image()),
        8,
        HashType::DoubleGradient,
    );
    let hash_repeated = ImageHash::hash(
        &MyPrecious(repeat_transform_img),
        8,
        HashType::DoubleGradient,
    );

    assert_eq!(
        hash_synthesized, hash_repeated,
        "repeated transform image hash differs from synthesized image"
    );
}
