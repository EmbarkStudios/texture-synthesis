use texture_synthesis as ts;

fn main() -> Result<(), ts::Error> {
    let texsynth = ts::Session::builder()
        // load example which will serve as our style, note you can have more than 1!
        .add_example(&"imgs/multiexample/4.jpg")
        // load target which will be the content
        // with style transfer, we do not need to provide example guides
        // they will be auto-generated if none were provided
        .load_target_guide(&"imgs/tom.jpg")
        .build()?;

    // generate an image that applies 'style' to "tom.jpg"
    let generated = texsynth.run(None);

    // save the result to the disk
    generated.save("out/04.jpg")
}
