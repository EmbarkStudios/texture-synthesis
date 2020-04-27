use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::{Duration, Instant};
use texture_synthesis as ts;

fn single_example(c: &mut Criterion) {
    static DIM: u32 = 25;

    // Load the example image once to reduce variation between runs,
    // though we still do a memcpy each run
    let example_img = ts::image::open("../imgs/1.jpg").unwrap();

    let mut group = c.benchmark_group("single_example");
    group.sample_size(10);

    for dim in [DIM, 2 * DIM, 4 * DIM, 8 * DIM, 16 * DIM].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            b.iter_custom(|iters| {
                let mut total_elapsed = Duration::new(0, 0);
                for _i in 0..iters {
                    let sess = ts::Session::builder()
                        .add_example(example_img.clone())
                        .seed(120)
                        .output_size(ts::Dims::square(dim))
                        .build()
                        .unwrap();

                    let start = Instant::now();
                    black_box(sess.run(None));
                    total_elapsed += start.elapsed();
                }

                total_elapsed
            });
        });
    }
    group.finish();
}

fn multi_example(c: &mut Criterion) {
    static DIM: u32 = 25;

    // Load the example image once to reduce variation between runs,
    // though we still do a memcpy each run
    let example_imgs = [
        ts::image::open("../imgs/multiexample/1.jpg").unwrap(),
        ts::image::open("../imgs/multiexample/2.jpg").unwrap(),
        ts::image::open("../imgs/multiexample/3.jpg").unwrap(),
        ts::image::open("../imgs/multiexample/4.jpg").unwrap(),
    ];

    let mut group = c.benchmark_group("multi_example");
    group.sample_size(10);

    for dim in [DIM, 2 * DIM, 4 * DIM, 8 * DIM, 16 * DIM].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            b.iter_custom(|iters| {
                let mut total_elapsed = Duration::new(0, 0);
                for _i in 0..iters {
                    let sess = ts::Session::builder()
                        .add_examples(example_imgs.iter().cloned())
                        .resize_input(ts::Dims::square(dim))
                        //.random_init(10)
                        .seed(211)
                        .output_size(ts::Dims::square(dim))
                        .build()
                        .unwrap();

                    let start = Instant::now();
                    black_box(sess.run(None));
                    total_elapsed += start.elapsed();
                }

                total_elapsed
            });
        });
    }
    group.finish();
}

fn guided(c: &mut Criterion) {
    static DIM: u32 = 25;

    // Load the example image once to reduce variation between runs,
    // though we still do a memcpy each run
    let example_img = ts::image::open("../imgs/2.jpg").unwrap();
    let guide_img = ts::image::open("../imgs/masks/2_example.jpg").unwrap();
    let target_img = ts::image::open("../imgs/masks/2_target.jpg").unwrap();

    let mut group = c.benchmark_group("guided");
    group.sample_size(10);

    for dim in [DIM, 2 * DIM, 4 * DIM, 8 * DIM, 16 * DIM].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            b.iter_custom(|iters| {
                let mut total_elapsed = Duration::new(0, 0);
                for _i in 0..iters {
                    let sess = ts::Session::builder()
                        .add_example(
                            ts::Example::builder(example_img.clone()).with_guide(guide_img.clone()),
                        )
                        .load_target_guide(target_img.clone())
                        //.random_init(10)
                        .seed(211)
                        .output_size(ts::Dims::square(dim))
                        .build()
                        .unwrap();

                    let start = Instant::now();
                    black_box(sess.run(None));
                    total_elapsed += start.elapsed();
                }

                total_elapsed
            });
        });
    }
    group.finish();
}

fn style_transfer(c: &mut Criterion) {
    static DIM: u32 = 25;

    // Load the example image once to reduce variation between runs,
    // though we still do a memcpy each run
    let example_img = ts::image::open("../imgs/multiexample/4.jpg").unwrap();
    let target_img = ts::image::open("../imgs/tom.jpg").unwrap();

    let mut group = c.benchmark_group("style_transfer");
    group.sample_size(10);

    for dim in [DIM, 2 * DIM, 4 * DIM, 8 * DIM, 16 * DIM].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            b.iter_custom(|iters| {
                let mut total_elapsed = Duration::new(0, 0);
                for _i in 0..iters {
                    let sess = ts::Session::builder()
                        .add_example(example_img.clone())
                        .load_target_guide(target_img.clone())
                        .output_size(ts::Dims::square(dim))
                        .build()
                        .unwrap();

                    let start = Instant::now();
                    black_box(sess.run(None));
                    total_elapsed += start.elapsed();
                }

                total_elapsed
            });
        });
    }
    group.finish();
}

fn inpaint(c: &mut Criterion) {
    static DIM: u32 = 25;

    // Load the example image once to reduce variation between runs,
    // though we still do a memcpy each run
    let example_img = ts::image::open("../imgs/3.jpg").unwrap();
    let inpaint_mask = ts::image::open("../imgs/masks/3_inpaint.jpg").unwrap();

    let mut group = c.benchmark_group("inpaint");
    group.sample_size(10);

    for dim in [DIM, 2 * DIM, 4 * DIM, 8 * DIM, 16 * DIM].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            b.iter_custom(|iters| {
                let mut total_elapsed = Duration::new(0, 0);
                for _i in 0..iters {
                    let sess = ts::Session::builder()
                        .inpaint_example(
                            inpaint_mask.clone(),
                            ts::Example::builder(example_img.clone())
                                .set_sample_method(inpaint_mask.clone()),
                            ts::Dims::square(dim),
                        )
                        .build()
                        .unwrap();

                    let start = Instant::now();
                    black_box(sess.run(None));
                    total_elapsed += start.elapsed();
                }

                total_elapsed
            });
        });
    }
    group.finish();
}

fn inpaint_channel(c: &mut Criterion) {
    static DIM: u32 = 25;

    // Load the example image once to reduce variation between runs,
    // though we still do a memcpy each run
    let example_img = ts::load_dynamic_image(ts::ImageSource::from(&"../imgs/bricks.png")).unwrap();

    let mut group = c.benchmark_group("inpaint_channel");
    group.sample_size(10);

    for dim in [DIM, 2 * DIM, 4 * DIM, 8 * DIM, 16 * DIM].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            b.iter_custom(|iters| {
                let mut total_elapsed = Duration::new(0, 0);
                for _i in 0..iters {
                    let sess = ts::Session::builder()
                        .inpaint_example_channel(
                            ts::ChannelMask::A,
                            ts::Example::builder(example_img.clone()),
                            ts::Dims::square(dim),
                        )
                        .build()
                        .unwrap();

                    let start = Instant::now();
                    black_box(sess.run(None));
                    total_elapsed += start.elapsed();
                }

                total_elapsed
            });
        });
    }
    group.finish();
}

fn tiling(c: &mut Criterion) {
    static DIM: u32 = 25;

    // Load the example image once to reduce variation between runs,
    // though we still do a memcpy each run
    let example_img = ts::image::open("../imgs/1.jpg").unwrap();
    let inpaint_mask = ts::image::open("../imgs/masks/1_tile.jpg").unwrap();

    let mut group = c.benchmark_group("tiling");
    group.sample_size(10);

    for dim in [DIM, 2 * DIM, 4 * DIM, 8 * DIM, 16 * DIM].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            b.iter_custom(|iters| {
                let mut total_elapsed = Duration::new(0, 0);
                for _i in 0..iters {
                    let sess = ts::Session::builder()
                        .inpaint_example(
                            inpaint_mask.clone(),
                            example_img.clone(),
                            ts::Dims::square(dim),
                        )
                        .tiling_mode(true)
                        .build()
                        .unwrap();

                    let start = Instant::now();
                    black_box(sess.run(None));
                    total_elapsed += start.elapsed();
                }

                total_elapsed
            });
        });
    }
    group.finish();
}

fn repeat(c: &mut Criterion) {
    static DIM: u32 = 25;

    // Load the example image once to reduce variation between runs,
    // though we still do a memcpy each run
    let example_img = ts::load_dynamic_image(ts::ImageSource::from(&"../imgs/bricks.png")).unwrap();

    let mut group = c.benchmark_group("repeat");
    group.sample_size(10);

    let mut gen = Vec::with_capacity(5);

    for dim in [DIM, 2 * DIM, 4 * DIM, 8 * DIM, 16 * DIM].iter() {
        let sess = ts::Session::builder()
            .add_example(example_img.clone())
            .output_size(ts::Dims::square(*dim))
            .build()
            .unwrap();

        let genned = sess.run(None);
        gen.push(genned);
    }

    for (genned, dim) in gen
        .into_iter()
        .zip([DIM, 2 * DIM, 4 * DIM, 8 * DIM, 16 * DIM].iter())
    {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &_dim| {
            b.iter_custom(|iters| {
                let mut total_elapsed = Duration::new(0, 0);

                for _i in 0..iters {
                    let img = example_img.clone();
                    let start = Instant::now();
                    black_box(
                        genned
                            .get_coordinate_transform()
                            .apply(std::iter::once(img))
                            .unwrap(),
                    );
                    total_elapsed += start.elapsed();
                }

                total_elapsed
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    single_example,
    multi_example,
    guided,
    style_transfer,
    inpaint,
    inpaint_channel,
    tiling,
    repeat,
);
criterion_main!(benches);
