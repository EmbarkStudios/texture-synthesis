use texture_synthesis as ts;

fn main() -> Result<(), ts::Error> {
    // We use the same image for both the example and inpaint, so we just load it once
    let img = ts::load_dynamic_image(ts::DataSource::from(&"imgs/bricks.png"))?;

    // The inpaint is retrieved from the images alpha channel
    let inpaint = ts::ImageSource::from(img.clone()).mask(ts::Mask::A);

    let texsynth = ts::Session::builder()
        // let the generator know which part we would like to fill in
        // if we had more examples, they would be additional information
        // the generator could use to inpaint
        .inpaint_example(inpaint, img, ts::Dims::square(400))
        .build()?;

    let generated = texsynth.run(None);

    //save the result to the disk
    generated.save("out/06.jpg")
}
