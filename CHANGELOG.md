# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2020-02-26
### Added
- Added the [09_sample_masks](lib/examples/09_sample_masks.rs) example
- CLI: Resolved [#38](https://github.com/EmbarkStudios/texture-synthesis/issues/38) by adding the `repeat` subcommand, which can be used to repeat transforms with different input images
- [PR#91](https://github.com/EmbarkStudios/texture-synthesis/pull/91) Added support for compiling and running texture-synthesis as WASM.

### Fixed
- [PR#69](https://github.com/EmbarkStudios/texture-synthesis/pull/69) Improved performance by 1x-2x, especially for smaller inputs. Thanks [@Mr4k](https://github.com/Mr4k)!
- [PR#70](https://github.com/EmbarkStudios/texture-synthesis/pull/70) Improved performance by **another** 1x-2.7x, especially for larger numbers of threads. Thanks (again!) [@Mr4k](https://github.com/Mr4k)!
- CLI: [PR#71](https://github.com/EmbarkStudios/texture-synthesis/pull/71) The prebuilt binary for Windows is now packaged in a zip file isntead of a gzipped tarball to improve end user experience. Thanks [@moppius](https://github.com/moppius)!
- [PR#98](https://github.com/EmbarkStudios/texture-synthesis/pull/98) fixed [#85](https://github.com/EmbarkStudios/texture-synthesis/issues/85), sample masks could hang image generation. Thanks [@Mr4k](https://github.com/Mr4k)!

## [0.7.1] - 2019-11-19
### Fixed
- Update to fix broken CI script

## [0.7.0] - 2019-11-19
### Added
- [PR#57](https://github.com/EmbarkStudios/texture-synthesis/pull/57) CLI: Added the `flip-and-rotate` subcommand which applies flip and rotation transforms to each example input and adds them as additional examples. Thanks [@JD557](https://github.com/JD557)!
- [PR#60](https://github.com/EmbarkStudios/texture-synthesis/pull/60) added the ability to specify a channel to use as the inpaint mask instead of a separate image. Thanks [@khskarl](https://github.com/khskarl)!
- Added `SessionBuilder::inpaint_example_channel`
- CLI: Added `--inpaint-channel <r|g|b|a>`

### Changed
- Replace [`failure`](https://crates.io/crates/failure) crate for error handling with just `std::error::Error`

### Fixed
- Validate that the `--m-rand` / `random_sample_locations` parameter is > 0. [#45](https://github.com/EmbarkStudios/texture-synthesis/issues/45)

## [0.6.0] - 2019-09-23
### Added
- Added support for the alpha channel during generation, which was previously ignored

### Changed
- `SessionBuilder::inpaint_example` now requires a size be provided by which all inputs will be resized, as well setting the output image size. Previously, you had to manually specify matching `output_size` and `resize_input` otherwise you would get a parameter validation error.
- All public methods/functions that took a size either as 2 u32's, or a tuple of them, now use a simple `Dims` struct for clarity.

### Fixed
- [PR#36](https://github.com/EmbarkStudios/texture-synthesis/pull/36) Fixed undefined behavior in `Generator::update`. Thanks for reporting, [@ralfbiedert](https://github.com/ralfbiedert)!

## [0.5.0] - 2019-09-13
### Added
- You can now specify the maximum number of threads that can be used at any one time via `SessionBuilder::max_thread_count`
- CLI: You can now specify the maximum thread count via `-t | --threads`
- Added `From<image::DynamicImage>` for `ImageSource`
- Added integrations tests for different examples, to catch regressions in generation
- Added criterion benchmarks for the different examples, to catch performance regressions

### Changed
- `SampleMethod::From<AsRef<Path>>` is now `SampleMethod::From<Into<ImageSource>>`
- `Example::From<AsRef<Path>>` is now `Example::From<Into<ImageSource>>`
- CLI: Renamed the `--out-fmt` arg to `--stdout-fmt` to indicate it only works when using stdout via `--out -`

### Fixed
- [PR#14](https://github.com/EmbarkStudios/texture-synthesis/pull/14) Vastly improve performance, all benchmarks are sped up from between **1.03** to **1.96**, almost twice as fast! Thanks [@austinjones](https://github.com/austinjones)!
- Disabled unused `rand` default features (OS random number generator)

## [0.4.2] - 2019-09-05
### Added
- Added `Error::UnsupportedOutputFormat`

### Changed
- CLI: `--out` is now required. `-` can still be passed to write to stdout instead of a file.
- CLI: The file extension for the `--out` path is now checked to see if it a supported format.

### Removed
- Removed tga feature in image since it wasn't used

## [0.4.1] - 2019-09-04
### Fixed
- Removed unused `lodepng` dependency

## [0.4.0] - 2019-09-04
### Changed
- Use [`failure`](https://crates.io/crates/failure) for errors
- CLI: Replaced piston_window with [`minifb`](https://crates.io/crates/minifb)
- CLI: Due to how minifb works via X11, the progress window is now an optional feature not enabled when building for musl

### Removed
- Removed several codec features from `image`, only `png`, `jpeg`, `bmp`, and `tga` are supported now

## [0.3.0] - 2019-09-03
### Added
- Added [`Example`](https://github.com/EmbarkStudios/texture-synthesis/blob/7e65b8abb9508841e7acf758cb79dd3f49aac28e/lib/src/lib.rs#L247) and [`ExampleBuilder`](https://github.com/EmbarkStudios/texture-synthesis/blob/7e65b8abb9508841e7acf758cb79dd3f49aac28e/lib/src/lib.rs#L208) which can be used to manipulate an indidivual
example input before being added to a [`SessionBuilder`](https://github.com/EmbarkStudios/texture-synthesis/blob/7e65b8abb9508841e7acf758cb79dd3f49aac28e/lib/src/lib.rs#L342)
- Added [`SampleMethod`](https://github.com/EmbarkStudios/texture-synthesis/blob/7e65b8abb9508841e7acf758cb79dd3f49aac28e/lib/src/lib.rs#L158) used to specify how a particular example is sampled during generation
- Added [`ImageSource`](https://github.com/EmbarkStudios/texture-synthesis/blob/7e65b8abb9508841e7acf758cb79dd3f49aac28e/lib/src/utils.rs#L6) which is a small enum that means image data for examples, guides,
masks, etc, can be specified either as paths, raw byte slices, or already loaded `image::DynamicImage`
- Added [`GeneratedImage`](https://github.com/EmbarkStudios/texture-synthesis/blob/7e65b8abb9508841e7acf758cb79dd3f49aac28e/lib/src/lib.rs#L103) which allows saving, streaming, and inspection of the image
generated by [`Session::run()`](https://github.com/EmbarkStudios/texture-synthesis/blob/7e65b8abb9508841e7acf758cb79dd3f49aac28e/lib/src/lib.rs#L736)

### Changed
- All usage of `&str` paths to load images from disk have been replaced with `ImageSource`
- Moved all of the building functionality in `Session` into `SessionBuilder`
- [`SessionBuilder::inpaint_example`](https://github.com/EmbarkStudios/texture-synthesis/blob/7e65b8abb9508841e7acf758cb79dd3f49aac28e/lib/src/lib.rs#L410) now specifies an ImageSource for the inpaint mask, along with an `Example` to be paired with, rather than the previous use of an index that the user had to keep track of
- [`GeneratorProgress`](https://github.com/EmbarkStudios/texture-synthesis/blob/7e65b8abb9508841e7acf758cb79dd3f49aac28e/lib/src/lib.rs#L789) now gets the total progress and stage progress in addition to the current image
- `Session::run()` can no longer fail, and consumes the `Session`
- `Session::run()` now returns a `GeneratedImage` which can be used to get, save, or stream the generated image (and maybe the debug ones)
- The CLI default values for the various available parameters now match the defaults in the `SessionBuilder`, which means the examples provided in the README.md match

### Removed
- Replaced `load_examples`, `load_example`, `load_example_guides`, and `load_sampling_masks` with
[`add_example`](https://github.com/EmbarkStudios/texture-synthesis/blob/7e65b8abb9508841e7acf758cb79dd3f49aac28e/lib/src/lib.rs#L366) and [`add_examples`](https://github.com/EmbarkStudios/texture-synthesis/blob/7e65b8abb9508841e7acf758cb79dd3f49aac28e/lib/src/lib.rs#L382) which work with `Example`(s)

### Fixed
- The top-level README.md is now deployed with both `texture-synthesis` and `texture-synthesis-cli` on crates.io

## [0.2.0] - 2019-08-27
### Added
- Split lib and cli into separate crates so CLI specific dependencies weren't pulled in
- Add `GeneratorProgress` to allow progress of image generation to be reported to external callers

## [0.1.0] - 2019-08-27
### Added
- Initial add of `texture-synthesis`

[Unreleased]: https://github.com/EmbarkStudios/texture-synthesis/compare/0.8.0...HEAD
[0.8.0]: https://github.com/EmbarkStudios/texture-synthesis/compare/0.7.0...0.7.1
[0.7.1]: https://github.com/EmbarkStudios/texture-synthesis/compare/0.7.0...0.7.1
[0.7.0]: https://github.com/EmbarkStudios/texture-synthesis/compare/0.6.0...0.7.0
[0.6.0]: https://github.com/EmbarkStudios/texture-synthesis/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/EmbarkStudios/texture-synthesis/compare/0.4.2...0.5.0
[0.4.2]: https://github.com/EmbarkStudios/texture-synthesis/compare/0.4.1...0.4.2
[0.4.1]: https://github.com/EmbarkStudios/texture-synthesis/compare/0.4.0...0.4.1
[0.4.0]: https://github.com/EmbarkStudios/texture-synthesis/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/EmbarkStudios/texture-synthesis/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/EmbarkStudios/texture-synthesis/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/EmbarkStudios/texture-synthesis/releases/tag/0.1.0
