use texture_synthesis as ts;

fn main() -> Result<(), ts::Error> {
    let texsynth = ts::Session::builder()
        // let the generator know which part we would like to fill in
        // if we had more examples, they would be additional information
        // the generator could use to inpaint
        .inpaint_example(
            &"imgs/masks/3_inpaint.jpg",
            // load a "corrupted" example with missing red information we would like to fill in
            ts::Example::builder(&"imgs/3.jpg")
                // we would also like to prevent sampling from "corrupted" red areas
                // otherwise, generator will treat that those as valid areas it can copy from in the example,
                // we could also use SampleMethod::Ignore to ignore the example altogether, but we
                // would then need at least 1 other example image to actually source from
                // example.set_sample_method(ts::SampleMethod::Ignore);
                .set_sample_method(&"imgs/masks/3_inpaint.jpg"),
        )
        // during inpaint, it is important to ensure both input and output are the same size
        .resize_input(400, 400)
        .output_size(400, 400)
        .build()?;

    let generated = texsynth.run(None);

    //save the result to the disk
    generated.save("out/05.jpg")
}
