use texture_synthesis as ts;

fn main() -> Result<(), ts::Error> {
    //create a new session
    let texsynth = ts::Session::builder()
        //load a single example image
        .add_example(&"imgs/1.jpg")
        .build()?;

    //generate an image
    let generated = texsynth.run(None);

    //now we can apply the same transformation of the generated image
    //onto a new image (which can be used to ensure 1-1 mapping between multiple images)
    //NOTE: it is important to provide same number and image dimensions as the examples used for synthesis
    //otherwise, there will be coordinates mismatch
    let repeat_transform_img = generated.get_coordinate_transform().repeat_transform(&["imgs/1_bw.jpg"])?;

    //save the image to the disk
    //01 and 01_2 images should match perfectly
    repeat_transform_img.save("out/01_2.jpg");
    generated.save("out/01.jpg")
    
}
