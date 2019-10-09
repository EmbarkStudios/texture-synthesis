use texture_synthesis as ts;

fn main() -> Result<(), ts::Error> {
    //create a new session
    let texsynth = ts::Session::builder()
        //load a single example image
        .mutate_example(&"imgs/multiexample/4.jpg", 0.035, 0.0, ts::Dims::square(400))
        .backtrack_stages(1)
        .build()?;

    //generate an image
    let generated = texsynth.run(None);

    //save the image to the disk
    generated.save("out/08.jpg")
}
