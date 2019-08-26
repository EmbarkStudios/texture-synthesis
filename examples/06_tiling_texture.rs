extern crate texture_synthesis;

fn main() {

    //let's start layering some of the "verbs" of texture synthesis
    //if we just run tiling_mode(true) we will generate a completely new image from scratch (try it!)
    //but what if we want to tile an existing image?
    //we can use inpaint!

    //create a new session
    let mut texsynth = texture_synthesis::Session::new()
        //load an image we want to tile
        .load_examples(&vec!["imgs/1.jpg"])
        //load a mask that specifies borders of the image we can modify to make it tiling
        .inpaint_example("imgs/masks/1_tile.jpg", 0)
        //ensure correct sizes
        .resize_input(400, 400)
        .output_size(400, 400)
        //turn on tiling mode!
        .tiling_mode(true);

    //generate image
    texsynth.run().unwrap();

    //save the result to the disk
    texsynth.save("out/06.jpg").unwrap();
}
