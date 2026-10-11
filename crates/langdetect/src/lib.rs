//! A port of [langdetect](https://github.com/Mimino666/langdetect) (itself a port of Nakatani
//! Shuyo's language-detection) that reproduces its results bit for bit, including the random
//! walk: with the same seed, the probabilities equal those of langdetect 1.0.9 on the CPython
//! version selected by [`Compat`].
//!
//! ```
//! use langdetect::{DetectorFactory, Seed};
//!
//! assert_eq!(langdetect::detect("Ein, zwei, drei, vier").unwrap(), "de");
//!
//! let mut factory = DetectorFactory::builtin().clone();
//! factory.seed = Seed::Int(0);
//! let langs = factory.detect_langs("Ceci est une phrase en français.").unwrap();
//! assert_eq!(langs[0].lang, "fr");
//! ```

mod compat;
mod detector;
mod error;
mod factory;
mod random;
mod tables;
pub mod text;

pub use compat::{Compat, SumMode, UnicodeVersion};
pub use detector::{Detector, Language};
pub use error::{Error, ErrorCode};
pub use factory::{DetectOptions, DetectorFactory, LangProfile};
pub use random::Seed;

/// The most probable language of `text` with the built-in profiles and a random seed.
pub fn detect(text: &str) -> Result<String, Error> {
    DetectorFactory::builtin().detect(text)
}

/// Languages of `text` with probability above 0.1, most probable first.
pub fn detect_langs(text: &str) -> Result<Vec<Language>, Error> {
    DetectorFactory::builtin().detect_langs(text)
}

#[cfg(test)]
mod tests {
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn factory_is_send_and_sync() {
        assert_send_sync::<crate::DetectorFactory>();
    }
}
