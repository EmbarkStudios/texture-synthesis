fn main() {
    //create a new session
    let mut texsynth = texture_synthesis::Session::new()
        //load example(s) which will serve as our style
        .load_examples(&vec!["imgs/multiexample/4.jpg"])
        //load target which will be the content
        //with style transfer, we do not need to provide example guides
        //they will be auto-generated if none were provided
        .load_target_guide("imgs/tom.jpg");

    //generate an image that applies 'style' to "tom.jpg"
    texsynth.run(None).unwrap();

    //save the result to the disk
    texsynth.save("out/04.jpg").unwrap();
}
