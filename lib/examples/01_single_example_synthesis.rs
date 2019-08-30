fn main() -> Result<(), texture_synthesis::Error> {
    //create a new session
    let texsynth = texture_synthesis::Session::builder()
        //load a single example image
        .add_example(&"imgs/1.jpg")
        .build()?;

    //generate an image
    let generated = texsynth.run(None);

    //save the image to the disk
    generated.save("out/01.jpg")
}
