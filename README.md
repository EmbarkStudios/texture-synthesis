# 🎨 texture-synthesis

[![Build Status](https://travis-ci.com/EmbarkStudios/texture-synthesis.svg?branch=master)](https://travis-ci.com/EmbarkStudios/texture-synthesis)
[![Crates.io](https://img.shields.io/crates/v/texture-synthesis.svg)](https://crates.io/crates/texture-synthesis)
[![Docs](https://docs.rs/texture-synthesis/badge.svg)](https://docs.rs/texture-synthesis)
[![Contributor Covenant](https://img.shields.io/badge/contributor%20covenant-v1.4%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)
[![Embark](https://img.shields.io/badge/embark-open%20source-blueviolet.svg)](http://embark.games)

A light API for Multiresolution Stochastic Texture Synthesis [1], a non-parametric example-based algorithm for image generation. 

The repo also includes multiple code examples to get you started (along with test images), and you can find a compiled binary with a command line interface under the release tab.

## Features and examples

### 1. Single example generation

![Imgur](https://i.imgur.com/CsZoSPS.jpg)

Generate similar-looking images from a single example.

#### API - [01_single_example_synthesis](lib/examples/01_single_example_synthesis.rs)

```rust
use texture_synthesis as ts;

fn main() -> Result<(), ts::Error> {
    //create a new session
    let texsynth = ts::Session::builder()
        //load a single example image
        .add_example(&"imgs/1.jpg")
        .build()?;

    //generate an image
    let generated = texsynth.run(None);

    //save the image to the disk
    generated.save("out/01.jpg")
}
```

#### CLI

`texture_synthesis --out-fmt jpg generate -- imgs/1.jpg > out/01.jpg`

### 2. Multi example generation

![Imgur](https://i.imgur.com/rYaae2w.jpg)

We can also provide multiple example images and the algorithm will "remix" them into a new image.

#### API - [02_multi_example_synthesis](lib/examples/02_multi_example_synthesis.rs)

```rust
use texture_synthesis as ts;

fn main() -> Result<(), ts::Error> {
    // create a new session
    let texsynth = ts::Session::builder()
        // load multiple example image
        .add_examples(&[
            &"imgs/multiexample/1.jpg",
            &"imgs/multiexample/2.jpg",
            &"imgs/multiexample/3.jpg",
            &"imgs/multiexample/4.jpg",
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
```

#### CLI

`texture_synthesis --rand-init 10 --seed 211 --in-size 300 --debug-out-dir out generate -- imgs/multiexample/1.jpg imgs/multiexample/2.jpg imgs/multiexample/3.jpg imgs/multiexample/4.jpg > out/02.png`

### 3. Guided Synthesis

![Imgur](https://i.imgur.com/eAiNZBg.jpg)

We can also guide the generation by providing a transformation "FROM"-"TO" in a form of guide maps

#### API - [03_guided_synthesis](lib/examples/03_guided_synthesis.rs)

```rust
use texture_synthesis as ts;

fn main() -> Result<(), ts::Error> {
    let texsynth = ts::Session::builder()
        // NOTE: it is important that example(s) and their corresponding guides have same size(s)
        // you can ensure that by overwriting the input images sizes with .resize_input()
        .add_example(ts::Example::builder(&"imgs/2.jpg").with_guide(&"imgs/masks/2_example.jpg"))
        // load target "heart" shape that we would like the generated image to look like
        // now the generator will take our target guide into account during synthesis
        .load_target_guide(&"imgs/masks/2_target.jpg")
        .build()?;

    let generated = texsynth.run(None);

    // save the image to the disk
    generated.save("out/03.jpg")
}
```

#### CLI

`texture_synthesis generate --target-guide imgs/masks/2_target.jpg --guides imgs/masks/2_example.jpg -- imgs/2.jpg > out/03.png`

### 4. Style Transfer

![Imgur](https://i.imgur.com/o9UxFGO.jpg)

Texture synthesis API supports auto-generation of example guide maps, which produces a style transfer-like effect.

#### API - [04_style_transfer](lib/examples/04_style_transfer.rs)

```rust
use texture_synthesis as ts;

fn main() -> Result<(), ts::Error> {
    let texsynth = ts::Session::builder()
        // load example which will serve as our style, note you can have more than 1!
        .add_examples(&[&"imgs/multiexample/4.jpg"])
        // load target which will be the content
        // with style transfer, we do not need to provide example guides
        // they will be auto-generated if none were provided
        .load_target_guide(&"imgs/tom.jpg")
        .alpha(0.8)
        .build()?;

    // generate an image that applies 'style' to "tom.jpg"
    let generated = texsynth.run(None);

    // save the result to the disk
    generated.save("out/04.jpg")
}
```

#### CLI

`texture_synthesis --alpha 0.8 transfer-style --style imgs/multiexample/4.jpg --guide imgs/tom.jpg > out/04.png`

### 5. Inpaint

![Imgur](https://i.imgur.com/FqvV651.jpg)

We can also fill-in missing information with inpaint. By changing the seed, we will get different version of the 'fillment'.

#### API - [05_inpaint](lib/examples/05_inpaint.rs)

```rust
use texture_synthesis as ts;

fn main() -> Result<(), ts::Error> {
    let texsynth = ts::Session::builder()
        // let the generator know which part we would like to fill in
        // if we had more examples, they would be additional information
        // the generator could use to inpaint
        .inpaint_example(
            &"imgs/masks/3_inpaint.jpg",
            // load a "corrupted" example with missing red information we would like to fill in
            ts::Example::builder(&"imgs/3.jpg")
                // we would also like to prevent sampling from "corrupted" red areas
                // otherwise, generator will treat that those as valid areas it can copy from in the example,
                // we could also use SampleMethod::Ignore to ignore the example altogether, but we
                // would then need at least 1 other example image to actually source from
                // example.set_sample_method(ts::SampleMethod::Ignore);
                .set_sample_method(&"imgs/masks/3_inpaint.jpg"),
        )
        // during inpaint, it is important to ensure both input and output are the same size
        .resize_input(400, 400)
        .output_size(400, 400)
        .build()?;

    let generated = texsynth.run(None);

    //save the result to the disk
    generated.save("out/05.jpg")
}
```

#### CLI

`texture_synthesis --in-size 400 --out-size 400 --inpaint imgs/masks/3_inpaint.jpg generate -- imgs/3.jpg > out/05.png`

### 6. Tiling texture

![](https://i.imgur.com/nFpCFzy.jpg)

We can make the generated image tile (meaning it will not have seams if you put multiple images together side-by-side). By invoking inpaint mode together with tiling, we can make an existing image tile.

#### API - [06_tiling_texture](lib/examples/06_tiling_texture.rs)

```rust
use texture_synthesis as ts;

fn main() -> Result<(), ts::Error> {
    // Let's start layering some of the "verbs" of texture synthesis
    // if we just run tiling_mode(true) we will generate a completely new image from scratch (try it!)
    // but what if we want to tile an existing image?
    // we can use inpaint!

    let texsynth = ts::Session::builder()
        // load a mask that specifies borders of the image we can modify to make it tiling
        .inpaint_example(&"imgs/masks/1_tile.jpg", ts::Example::new(&"imgs/1.jpg"))
        //ensure correct sizes
        .resize_input(400, 400)
        .output_size(400, 400)
        //turn on tiling mode!
        .tiling_mode(true)
        .build()?;

    let generated = texsynth.run(None);

    generated.save("out/06.jpg")
}
```

#### CLI

`texture_synthesis --inpaint imgs/masks/1_tile.jpg --in-size 400 --out-size 400 --tiling generate --examples imgs/1.jpg > out/06.png`

### 7. Combining texture synthesis 'verbs'

We can also combine multiple modes together. For example, multi-example guided synthesis:

![](https://i.imgur.com/By64UXG.jpg)

Or chaining multiple stages of generation together:

![](https://i.imgur.com/FzZW3sl.jpg)

## Command line binary

* [Download the binary](https://github.com/EmbarkStudios/texture-synthesis/releases) for your OS, or install it from source, `cargo install .`
* Open a terminal
* Navigate to the directory where you downloaded the binary, if you didn't just `cargo install` it
* Run `texture_synthesis --help` to get a list of all of the options and commands you can run
* Refer to the examples section in this readme for examples of running the binary

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
