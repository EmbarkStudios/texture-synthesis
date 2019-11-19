use texture_synthesis as ts;

fn main() -> Result<(), ts::Error> {
    let texsynth = ts::Session::builder()
        // Let the generator know that it is using
        .inpaint_example_channel(
            ts::ChannelMask::A,
            &"imgs/bricks.png",
            ts::Dims::square(400),
        )
        .build()?;

    let generated = texsynth.run(None);

    //save the result to the disk
    generated.save("out/06.jpg")
}
