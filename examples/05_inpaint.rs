extern crate texture_synthesis;

fn main() {
    //create a new session
    let mut texsynth = texture_synthesis::Session::new()
        //load a "corrupted" example with missing red information we would like to fill in
        .load_examples(&vec!["imgs/3.jpg"])
        //let the generator know which part we would like to fill in
        //since we only have one example, we put 0 in the example_id
        //if we had more example, we could specify the index of which one to inpaint
        //then the rest of example would be additional information the generator could use to inpaint
        .inpaint_example("imgs/masks/3_inpaint.jpg", 0)
        //we would also like to prevent sampling from "corrupted" red areas
        //otherwise, generator will treat that those as valid areas it can copy from in the example
        .load_sampling_masks(&vec!["imgs/masks/3_inpaint.jpg"])
        //during inpaint, it is important to ensure both input and output are the same size
        .resize_input(400, 400)
        .output_size(400, 400);

    //inpaint out image
    texsynth.run().unwrap();

    //save the result to the disk
    texsynth.save("out/05.jpg").unwrap();
}
