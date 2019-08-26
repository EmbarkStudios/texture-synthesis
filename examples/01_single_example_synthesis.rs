extern crate texture_synthesis;

fn main() {
    //create a new session
    let mut texsynth = texture_synthesis::Session::new()
        //load a single example image
        .load_examples(&vec!["imgs/1.jpg"]);
    //generate an image
    texsynth.run().unwrap();

    //save the image to the disk
    texsynth.save("out/01.jpg").unwrap();
}
