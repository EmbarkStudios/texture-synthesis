use texture_synthesis as ts;

fn main() -> Result<(), ts::Error> {
    let session = ts::Session::builder()
        .add_example(
            ts::Example::builder(&"imgs/4.png").set_sample_method(ts::SampleMethod::Ignore),
        )
        .add_example(ts::Example::builder(&"imgs/5.png").set_sample_method(ts::SampleMethod::All))
        .seed(211)
        .output_size(ts::Dims::square(200))
        .build()?;

    // generate an image
    let generated = session.run(None);

    // save the image to the disk
    generated.save("out/09.png")
}
