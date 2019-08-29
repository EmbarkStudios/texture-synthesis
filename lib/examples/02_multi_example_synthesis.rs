fn main() -> Result<(), Box<dyn std::error::Error>> {
    // create a new session
    let texsynth = texture_synthesis::Session::builder()
        // load multiple example image
        .add_examples(&[
            "imgs/multiexample/1.jpg",
            "imgs/multiexample/2.jpg",
            "imgs/multiexample/3.jpg",
            "imgs/multiexample/4.jpg",
        ])
        // we can ensure all of them come with same size
        // that is however optional, the generator doesnt care whether all images are same sizes
        // however, if you have guides or other additional maps, those have to be same size(s) as corresponding example(s)
        .resize_input(300, 300)
        // randomly initialize first 10 pixels
        .random_init(10)
        .seed(211)
        .build()?;

    // generate an image
    let generated = texsynth.run(None);

    // save the image to the disk
    generated.save("out/02.jpg")?;

    //save debug information to see "remixing" borders of different examples in map_id.jpg
    //different colors represent information coming from different maps
    generated.save_debug("out/")
}
