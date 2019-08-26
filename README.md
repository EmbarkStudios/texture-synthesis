# 🎨 texture-synthesis

<!--- FIXME: Update crate and repo names here! --->
[![Build Status](https://travis-ci.com/EmbarkStudios/texture-synthesis.svg?branch=master)](https://travis-ci.com/EmbarkStudios/texture-synthesis)
[![Crates.io](https://img.shields.io/crates/v/texture-synthesis.svg)](https://crates.io/crates/texture-synthesis)
[![Docs](https://docs.rs/texture-synthesis/badge.svg)](https://docs.rs/texture-synthesis)
[![Contributor Covenant](https://img.shields.io/badge/contributor%20covenant-v1.4%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)
[![Embark](https://img.shields.io/badge/embark-open%20source-blueviolet.svg)](http://embark.games)

A light API for Multiresolution Stochastic Texture Synthesis [1], a non-parametric example-based algorithm for image generation. 

The repo also includes multiple code examples to get you started (along with test images), and you can find a compiled binary with a command line interface under release tab.

## Features and examples

### 1. Single example generation

![Imgur](https://i.imgur.com/CsZoSPS.jpg)

Generate similar-looking images from a single example.

Below is how to do it with the texture-synthesis API:
```rust
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
```
This code snippet can be found in `examples/01_single_example_synthesis.rs`

To replicate this example with the commandline binary run: 
`texture_synthesis_cmd.exe --examples imgs/1.jpg --save out/01.jpg`

### 2. Multi example generation

![Imgur](https://i.imgur.com/rYaae2w.jpg)

You can also provide multiple example images and the algorithm will "remix" them into a new image.

Below is how to do it with the texture-synthesis API:
```rust
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
```
This code snippet can be found in `examples/02_multi_example_synthesis.rs`

To replicate this example with the commandline binary run: 
`texture_synthesis_cmd.exe --examples imgs/multiexample/1.jpg,imgs/multiexample/2.jpg,imgs/multiexample/3.jpg,imgs/multiexample/4.jpg --rand-init 10 --in-size 300x300 --save out/02.jpg --debug-maps`

### 3. Guided Synthesis

![Imgur](https://i.imgur.com/eAiNZBg.jpg)

We can also guide the generation by providing a transformation "FROM"-"TO" in a form of guide maps

Below is how to do it with the texture-synthesis API:

```rust
extern crate texture_synthesis;

fn main() {
    //create a new session
    let mut texsynth = texture_synthesis::Session::new()
        //load example
        .load_examples(&vec!["imgs/2.jpg"])
        //load example guide map
        .load_example_guides(&vec!["imgs/masks/2_example.jpg"])
        //load target shape that we would like the generated image to look like
        .load_target_guide("imgs/masks/2_target.jpg");

    // NOTE: it is important that example(s) and their corresponding guides have same size(s)
    // you can ensure that by overwriting the input images sizes with .resize_input()

    //now the generator will take our target guide into account during synthesis
    texsynth.run().unwrap();

    //save the image to the disk
    texsynth.save("out/03.jpg").unwrap();

    //You can also do a more involved segmentation with guide maps with R G B annotating specific features of your examples
}
```
This code snippet can be found in `examples/03_guided_synthesis.rs`

To replicate this example with the commandline binary run:
`texture_synthesis_cmd.exe --examples imgs/2.jpg --example-guide imgs/masks/2_example.jpg --target-guide imgs/masks/2_target.jpg --save out/03.jpg`

### 4. Style Transfer

![Imgur](https://i.imgur.com/o9UxFGO.jpg)

Texture synthesis API supports auto-generation of example guide maps, which would produce a style transfer like effect.

Below is how to do it with the texture-synthesis API:

```rust
extern crate texture_synthesis;

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
    texsynth.run().unwrap();

    //save the result to the disk
    texsynth.save("out/04.jpg").unwrap();
}
```
This code snippet can be found in `examples/04_style_transfer.rs`

To replicate this example with the commandline binary run:
`texture_synthesis_cmd.exe --examples imgs/multiexample/4.jpg --target-guide imgs/tom.jpg --alpha 0.8 --save out/04.jpg`

### 5. Inpaint

![Imgur](https://i.imgur.com/FqvV651.jpg)

We can also fill-in missing information with inpaint. By changing the seed, we will get different version of the 'fillment'.

Below is how to do it with the texture-synthesis API:
```rust
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
```
This code snippet can be found in `examples/05_inpaint.rs`

To replicate this example with the commandline binary run:
`texture_synthesis_cmd.exe --examples imgs/3.jpg --inpaint imgs/masks/3_inpaint.jpg --sample-masks imgs/masks/3_inpaint.jpg --in-size 400x400 --out-size 400x400 --save out/05.jpg`

### 6. Tiling texture

![](https://i.imgur.com/nFpCFzy.jpg)

We can make the generated image tile (meaning it will not have seams if you put multiple images together side-by-side). By invoking inpaint mode together with tiling, we can make an existing image tile.

Below is how to do it with the texture-synthesis API:
```rust
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
```
This code snippet can be found in `examples/06_tiling_texture.rs`

To replicate this example with the commandline binary run:
`texture_synthesis_cmd.exe --examples imgs/1.jpg --inpaint imgs/masks/1_tile.jpg --sample-masks imgs/masks/1_tile.jpg --in-size 400x400 --out-size 400x400 --tiling --save out/06.jpg`

### 7. Combining texture synthesis 'verbs'

We can also combine multiple modes together. For example, multi-example guided synthesis:

![](https://i.imgur.com/By64UXG.jpg)

Or chaining multiple stages of generation together:

![](https://i.imgur.com/FzZW3sl.jpg)

## Command line binary

Instruction on how to use:
* download the repo
* open the terminal (on windows: search for `cmd`)
* navigate to the folder containing the `texture_synthesis_cmd.exe` (for ex: cd C:\Downloads\texture-synthesis)
* run `texture_synthesis_cmd.exe --help` (this will give you a list of all commands you can run)
* refer to the examples section in this readme for examples on running the binary

## Limitations
* Struggles with complex semantics beyond pixel color (unless you guide it)
* Not great with regular textures (seams can become obvious)
* Cannot infer new information from existing information (only operates on what’s already there)
* Designed for single exemplars or very small datasets (unlike Deep Learning based approaches)

## Links/references

[1] [Opara & Stachowiak] ["More Like This, Please! Texture Synthesis and Remixing from a Single Example"](https://youtu.be/fMbK7PYQux4)

[2] [Harrison] Image Texture Tools

[3] [Ashikhmin] Synthesizing Natural Textures

[4] [Efros & Leung] Texture Synthesis by Non-parametric Sampling

[5] [Wey & Levoy] Fast Texture Synthesis using Tree-structured Vector Quantization

[6] [De Bonet] Multiresolution Sampling Procedure for Analysis and Synthesis of Texture Images

[7] All the test images in this repo are from [Unsplash](https://unsplash.com/)

## Contributing

We welcome community contributions to this project.

Please read our [Contributor Guide](CONTRIBUTING.md) for more information on how to get started.

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
