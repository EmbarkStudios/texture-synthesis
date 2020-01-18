use std::fmt;

#[derive(Debug)]
pub struct InvalidRange {
    pub(crate) min: f32,
    pub(crate) max: f32,
    pub(crate) value: f32,
    pub(crate) name: &'static str,
}

impl fmt::Display for InvalidRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "parameter '{}' - value '{}' is outside the range of {}-{}",
            self.name, self.value, self.min, self.max
        )
    }
}

#[derive(Debug)]
pub struct SizeMismatch {
    pub(crate) input: (u32, u32),
    pub(crate) output: (u32, u32),
}

impl fmt::Display for SizeMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the input size ({}x{}) must match the output size ({}x{}) when using an inpaint mask",
            self.input.0, self.input.1, self.output.0, self.output.1
        )
    }
}

#[derive(Debug)]
pub enum Error {
    /// An error in the image library occurred, eg failed to load/save
    Image(image::ImageError),
    /// An input parameter had an invalid range specified
    InvalidRange(InvalidRange),
    /// When using inpaint, the input and output sizes must match
    SizeMismatch(SizeMismatch),
    /// If more than 1 example guide is provided, then **all** examples must have
    /// a guide
    ExampleGuideMismatch(u32, u32),
    /// Io is notoriously error free with no problems, but we cover it just in case!
    Io(std::io::Error),
    /// The user specified an image format we don't support as the output
    UnsupportedOutputFormat(String),
    /// There are no examples to source pixels from, either because no examples
    /// were added, or all of them used SampleMethod::Ignore
    NoExamples,
    ///
    MapsCountMismatch(u32, u32),
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Image(err) => Some(err),
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Image(ie) => write!(f, "{}", ie),
            Self::InvalidRange(ir) => write!(f, "{}", ir),
            Self::SizeMismatch(sm) => write!(f, "{}", sm),
            Self::ExampleGuideMismatch(examples, guides) => {
                if examples > guides {
                    write!(
                        f,
                        "{} examples were provided, but only {} guides were",
                        examples, guides
                    )
                } else {
                    write!(
                        f,
                        "{} examples were provided, but {} guides were",
                        examples, guides
                    )
                }
            }
            Self::Io(io) => write!(f, "{}", io),
            Self::UnsupportedOutputFormat(fmt) => {
                write!(f, "the output format '{}' is not supported", fmt)
            }
            Self::NoExamples => write!(
                f,
                "at least 1 example must be available as a sampling source"
            ),
            Self::MapsCountMismatch(input, required) => write!(
                f,
                "{} map(s) were provided, but {} is/are required",
                input, required
            ),
        }
    }
}

impl From<image::ImageError> for Error {
    fn from(ie: image::ImageError) -> Self {
        Self::Image(ie)
    }
}

impl From<std::io::Error> for Error {
    fn from(io: std::io::Error) -> Self {
        Self::Io(io)
    }
}
