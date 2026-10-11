use std::fmt;

/// Error codes of `langdetect.lang_detect_exception.ErrorCode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    NoTextError = 0,
    FormatError = 1,
    FileLoadError = 2,
    DuplicateLangError = 3,
    NeedLoadProfileError = 4,
    CantDetectError = 5,
    CantOpenTrainData = 6,
    TrainDataFormatError = 7,
    InitParamError = 8,
}

/// An error, mirroring the exception langdetect raises in the same situation.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// `LangDetectException(code, message)`.
    LangDetect { code: ErrorCode, message: String },
    /// Python's `ZeroDivisionError`: a profile with a zero n-gram total, or probabilities
    /// that sum to zero (only with degenerate parameters such as an all-zero prior).
    /// `langprob` holds what the detector had accumulated when it was raised during detection.
    ZeroDivision { langprob: Option<Vec<f64>> },
    /// Python's `IndexError`: a `prior_map` or `add_profile` index that does not fit the
    /// probability lists. `langprob` as for `ZeroDivision`.
    Index {
        message: &'static str,
        langprob: Option<Vec<f64>>,
    },
}

impl Error {
    pub(crate) fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Error::LangDetect {
            code,
            message: message.into(),
        }
    }

    pub(crate) fn zero_division() -> Self {
        Error::ZeroDivision { langprob: None }
    }

    pub(crate) fn zero_division_with(langprob: Vec<f64>) -> Self {
        Error::ZeroDivision {
            langprob: Some(langprob),
        }
    }

    pub(crate) fn index(message: &'static str) -> Self {
        Error::Index {
            message,
            langprob: None,
        }
    }

    pub(crate) fn index_with(message: &'static str, langprob: Vec<f64>) -> Self {
        Error::Index {
            message,
            langprob: Some(langprob),
        }
    }

    pub fn code(&self) -> Option<ErrorCode> {
        match self {
            Error::LangDetect { code, .. } => Some(*code),
            _ => None,
        }
    }

    /// The probabilities accumulated before a `ZeroDivision` or `Index` error in detection.
    pub fn langprob(&self) -> Option<&[f64]> {
        match self {
            Error::ZeroDivision { langprob } | Error::Index { langprob, .. } => langprob.as_deref(),
            Error::LangDetect { .. } => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::LangDetect { message, .. } => f.write_str(message),
            Error::ZeroDivision { .. } => f.write_str("float division by zero"),
            Error::Index { message, .. } => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {}
