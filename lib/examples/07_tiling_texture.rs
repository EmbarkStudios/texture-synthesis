use texture_synthesis as ts;

fn main() -> Result<(), ts::Error> {
    // Let's start layering some of the "verbs" of texture synthesis
    // if we just run tiling_mode(true) we will generate a completely new image from scratch (try it!)
    // but what if we want to tile an existing image?
    // we can use inpaint!

    let texsynth = ts::Session::builder()
        // load a mask that specifies borders of the image we can modify to make it tiling
        .inpaint_example(
            &"imgs/masks/1_tile.jpg",
            ts::Example::new(&"imgs/1.jpg"),
            ts::Dims::square(400),
        )
        //turn on tiling mode!
        .tiling_mode(true)
        .build()?;

    let generated = texsynth.run(None);

    generated.save("out/07.jpg")
}
