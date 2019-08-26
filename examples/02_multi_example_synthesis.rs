extern crate texture_synthesis;

fn main() {
    //create a new session
    let mut texsynth = texture_synthesis::Session::new()
        //load multiple example image
        .load_examples(&vec![
            "imgs/multiexample/1.jpg",
            "imgs/multiexample/2.jpg",
            "imgs/multiexample/3.jpg",
            "imgs/multiexample/4.jpg",
        ])
        //we can ensure all of them come with same size
        //that is however optional, the generator doesnt care whether all images are same sizes
        //however, if you have guides or other additional maps, those have to be same size(s) as corresponding example(s)
        .resize_input(300, 300)
        //randomly initialize first 10 pixels
        .random_init(10)
        .seed(211);

    //generate an image
    texsynth.run().unwrap();

    //save the image to the disk
    texsynth.save("out/02.jpg").unwrap();

    //save debug information to see "remixing" borders of different examples in map_id.jpg
    //different colors represent information coming from different maps
    texsynth.save_debug("out/").unwrap();
}
