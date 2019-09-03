use texture_synthesis as ts;

fn main() -> Result<(), ts::Error> {
    let texsynth = ts::Session::builder()
        // NOTE: it is important that example(s) and their corresponding guides have same size(s)
        // you can ensure that by overwriting the input images sizes with .resize_input()
        .add_example(ts::Example::builder(&"imgs/2.jpg").with_guide(&"imgs/masks/2_example.jpg"))
        // load target "heart" shape that we would like the generated image to look like
        // now the generator will take our target guide into account during synthesis
        .load_target_guide(&"imgs/masks/2_target.jpg")
        .build()?;

    let generated = texsynth.run(None);

    // save the image to the disk
    generated.save("out/03.jpg")

    // You can also do a more involved segmentation with guide maps with R G B annotating specific features of your examples
}
