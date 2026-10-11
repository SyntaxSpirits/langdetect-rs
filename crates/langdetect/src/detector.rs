use std::collections::HashMap;
use std::fmt;

use crate::error::{Error, ErrorCode};
use crate::factory::{DetectOptions, DetectorFactory};
use crate::random::Seed;
use crate::text;

/// A detected language and its probability.
#[derive(Clone, Debug, PartialEq)]
pub struct Language {
    pub lang: String,
    pub prob: f64,
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{:?}", self.lang, self.prob)
    }
}

/// Detects the language of the appended text, the equivalent of `langdetect.detector.Detector`.
#[derive(Clone, Debug)]
pub struct Detector<'f> {
    factory: &'f DetectorFactory,
    text: String,
    langprob: Option<Vec<f64>>,
    pub alpha: f64,
    pub n_trial: i64,
    pub max_text_length: i64,
    prior_map: Option<Vec<f64>>,
    pub seed: Seed,
}

impl<'f> Detector<'f> {
    pub(crate) fn new(factory: &'f DetectorFactory) -> Self {
        Detector {
            factory,
            text: String::new(),
            langprob: None,
            alpha: 0.5,
            n_trial: 7,
            max_text_length: 10000,
            prior_map: None,
            seed: factory.seed.clone(),
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    pub fn set_max_text_length(&mut self, max_text_length: i64) {
        self.max_text_length = max_text_length;
    }

    /// `Detector.set_prior_map`. As in langdetect, a rejected map stays partly applied.
    pub fn set_prior_map(&mut self, prior_map: &HashMap<String, f64>) -> Result<(), Error> {
        let langs = self.factory.langlist();
        let mut prior = vec![0.0; langs.len()];
        let mut sump = 0.0;
        for (i, lang) in langs.iter().enumerate() {
            if let Some(&p) = prior_map.get(lang) {
                if p < 0.0 {
                    self.prior_map = Some(prior);
                    return Err(Error::new(
                        ErrorCode::InitParamError,
                        "Prior probability must be non-negative.",
                    ));
                }
                prior[i] = p;
                sump += p;
            }
        }
        if sump <= 0.0 {
            self.prior_map = Some(prior);
            return Err(Error::new(
                ErrorCode::InitParamError,
                "More one of prior probability must be non-zero.",
            ));
        }
        for p in &mut prior {
            *p /= sump;
        }
        self.prior_map = Some(prior);
        Ok(())
    }

    /// `Detector.append`.
    pub fn append(&mut self, text: &str) {
        self.text
            .push_str(&text::prepare(text, self.max_text_length));
    }

    /// `Detector.detect`: the most probable language, or `"unknown"`.
    pub fn detect(&mut self) -> Result<String, Error> {
        let probabilities = self.get_probabilities()?;
        Ok(probabilities
            .into_iter()
            .next()
            .map_or_else(|| "unknown".to_string(), |l| l.lang))
    }

    /// `Detector.get_probabilities`.
    pub fn get_probabilities(&mut self) -> Result<Vec<Language>, Error> {
        let factory = self.factory;
        Ok(factory.sort_probability(self.langprob()?))
    }

    /// The probability of every language in `factory.langlist()` order.
    pub fn langprob(&mut self) -> Result<&[f64], Error> {
        if self.langprob.is_none() {
            if let Some(cleaned) = text::clean(&self.text) {
                self.text = cleaned;
            }
            let options = DetectOptions {
                alpha: self.alpha,
                n_trial: self.n_trial,
                prior_map: self.prior_map.as_deref(),
            };
            match self.factory.detect_block(&self.text, &options, &self.seed) {
                Ok(langprob) => self.langprob = Some(langprob),
                Err(e) => {
                    // langdetect sets `langprob` before the walk and keeps what it accumulated.
                    if let Some(partial) = e.langprob() {
                        self.langprob = Some(partial.to_vec());
                    }
                    return Err(e);
                }
            }
        }
        Ok(self.langprob.as_deref().unwrap())
    }
}
