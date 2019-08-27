fn main() {
    //create a new session
    let mut texsynth = texture_synthesis::Session::new()
        //load example
        .load_examples(&vec!["../imgs/2.jpg"])
        //load segmentation of the example
        .load_example_guides(&vec!["../imgs/masks/2_example.jpg"])
        //load target "heart" shape that we would like the generated image to look like
        .load_target_guide("../imgs/masks/2_target.jpg");

    // NOTE: it is important that example(s) and their corresponding guides have same size(s)
    // you can ensure that by overwriting the input images sizes with .resize_input()

    //now the generator will take our target guide into account during synthesis
    texsynth.run(None).unwrap();

    //save the image to the disk
    texsynth.save("out/03.jpg").unwrap();

    //You can also do a more involved segmentation with guide maps with R G B annotating specific features of your examples
}
