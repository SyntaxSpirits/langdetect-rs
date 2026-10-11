use std::borrow::Cow;
use std::sync::OnceLock;

use langdetect::{
    text, Compat, DetectOptions, DetectorFactory, Error, LangProfile, Seed, SumMode, UnicodeVersion,
};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyFloat, PyString, PyTuple};
use rayon::prelude::*;

/// Lone surrogates, which UTF-8 cannot hold, travel as the last 2,048 code points of
/// Supplementary Private Use Area-B. Like surrogates, those are outside the built-in profiles,
/// not uppercase, not matched by the URL and e-mail patterns, above U+0300 and unchanged by
/// normalisation, so detection is the same; text returned to Python maps them back.
const SURROGATE_SHIFT: u32 = 0x10F800 - 0xD800;

/// A Python `str` as UTF-8, and whether it contained lone surrogates.
struct Text<'a> {
    text: Cow<'a, str>,
    surrogates: bool,
}

fn text_arg<'a>(obj: &'a Bound<'_, PyAny>) -> PyResult<Text<'a>> {
    let s = obj.cast::<PyString>().map_err(|_| {
        PyTypeError::new_err(format!(
            "expected str, got {}",
            obj.get_type()
                .name()
                .map_or_else(|_| "object".into(), |n| n.to_string())
        ))
    })?;
    if let Ok(text) = s.to_cow() {
        return Ok(Text {
            text,
            surrogates: false,
        });
    }
    let encoded = s.call_method1("encode", ("utf-32-le", "surrogatepass"))?;
    let bytes = encoded.cast::<PyBytes>()?.as_bytes();
    let text = bytes
        .chunks_exact(4)
        .map(|c| {
            let cp = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            let cp = if (0xD800..0xE000).contains(&cp) {
                cp + SURROGATE_SHIFT
            } else {
                cp
            };
            char::from_u32(cp).unwrap()
        })
        .collect();
    Ok(Text {
        text: Cow::Owned(text),
        surrogates: true,
    })
}

/// Text back to Python, restoring the surrogates of an input that had them.
fn to_py_str<'py>(py: Python<'py>, text: &str, surrogates: bool) -> PyResult<Bound<'py, PyAny>> {
    if !surrogates || !text.chars().any(|c| c as u32 >= 0x10F800) {
        return Ok(PyString::new(py, text).into_any());
    }
    let mut bytes = Vec::with_capacity(text.len() * 4);
    for c in text.chars() {
        let cp = c as u32;
        let cp = if cp >= 0x10F800 {
            cp - SURROGATE_SHIFT
        } else {
            cp
        };
        bytes.extend_from_slice(&cp.to_le_bytes());
    }
    PyBytes::new(py, &bytes).call_method1("decode", ("utf-32-le", "surrogatepass"))
}

fn zero_division_message(py: Python<'_>) -> &'static str {
    if py.version_info() >= (3, 14) {
        "division by zero"
    } else {
        "float division by zero"
    }
}

/// Python exception for a core error: ZeroDivisionError, IndexError, or
/// `ValueError(code, message)` that the Python layer turns into LangDetectException.
fn to_py_err(py: Python<'_>, e: Error) -> PyErr {
    match e {
        Error::ZeroDivision { .. } => PyZeroDivisionError::new_err(zero_division_message(py)),
        Error::Index { message, .. } => PyIndexError::new_err(message),
        Error::LangDetect { code, message } => PyValueError::new_err((code as i64, message)),
    }
}

/// `detect_langs` output: (language, probability), most probable first.
type Ranked = Vec<(String, f64)>;

/// `(langprob, error)` of `Factory.detect_block`.
type BlockResult<'py> = (Option<Vec<f64>>, Option<Bound<'py, PyAny>>);

/// A `random.Random().seed(seed)` state, converted once and reused.
#[pyclass(frozen, module = "langdetect_rs._core")]
struct SeedState {
    seed: Seed,
}

#[pymethods]
impl SeedState {
    #[new]
    fn new(state: Vec<u32>) -> PyResult<Self> {
        Seed::from_state(&state)
            .map(|seed| SeedState { seed })
            .ok_or_else(|| PyValueError::new_err("expected 625 integers from random.getstate()"))
    }
}

fn seed_of(seed: Option<&Bound<'_, SeedState>>) -> Seed {
    seed.map_or(Seed::Os, |s| s.get().seed.clone())
}

/// The process that started Rayon's global thread pool. Its threads do not exist in a forked
/// child, so a child runs batches on the calling thread.
static POOL_OWNER: OnceLock<u32> = OnceLock::new();

fn thread_pool_usable() -> bool {
    *POOL_OWNER.get_or_init(std::process::id) == std::process::id()
}

fn map_batch<T: Send>(texts: &[Text<'_>], f: impl Fn(&str) -> T + Sync) -> Vec<T> {
    if thread_pool_usable() {
        texts.par_iter().map(|t| f(&t.text)).collect()
    } else {
        texts.iter().map(|t| f(&t.text)).collect()
    }
}

#[pyclass(module = "langdetect_rs._core")]
struct Factory {
    inner: DetectorFactory,
}

#[pymethods]
impl Factory {
    #[new]
    fn new(neumaier: bool, unidata_version: &str) -> Self {
        let compat = Compat {
            sum: if neumaier {
                SumMode::Neumaier
            } else {
                SumMode::Sequential
            },
            unicode: UnicodeVersion::from_unidata(unidata_version).unwrap_or(UnicodeVersion::V17_0),
        };
        Factory {
            inner: DetectorFactory::new(compat),
        }
    }

    #[staticmethod]
    fn _from_bytes(data: &[u8]) -> PyResult<Self> {
        DetectorFactory::from_bytes(data)
            .map(|inner| Factory { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let bytes = PyBytes::new(py, &slf.borrow().inner.to_bytes());
        let constructor = slf.get_type().getattr("_from_bytes")?;
        PyTuple::new(
            py,
            [
                constructor.into_any(),
                PyTuple::new(py, [bytes])?.into_any(),
            ],
        )
    }

    /// `load_profile` on the built-in profiles. On failure, raises
    /// `ValueError(index, message)`; the profiles before `index` stay added, as in langdetect.
    fn load_builtin(&mut self, files: Vec<String>) -> PyResult<()> {
        self.inner
            .load_builtin(&files)
            .map_err(|(index, e)| PyValueError::new_err((index, e.to_string())))
    }

    /// `DetectorFactory.add_profile(profile, index, langsize)` without the duplicate check,
    /// which the Python layer does. Each probability is computed with Python's own
    /// `1.0 * count / n_words[len - 1]`, so errors are the ones langdetect raises, after the
    /// n-grams before the failing one are applied.
    fn add_profile(
        &mut self,
        py: Python<'_>,
        name: &str,
        freq: &Bound<'_, PyDict>,
        n_words: &Bound<'_, PyAny>,
        index: usize,
        langsize: i64,
    ) -> PyResult<()> {
        // `[0.0] * langsize` is empty for a negative size.
        let langsize = usize::try_from(langsize).unwrap_or(0);
        let one = PyFloat::new(py, 1.0);
        let mut grams: Vec<(String, f64)> = Vec::with_capacity(freq.len());
        let mut failure: Option<(String, PyErr)> = None;
        for (k, v) in freq.iter() {
            let word = text_arg(&k)?.text.into_owned();
            let len = word.chars().count();
            if !(1..=3).contains(&len) {
                continue;
            }
            let prob = one
                .mul(&v)
                .and_then(|x| x.div(n_words.get_item(len - 1)?))
                .and_then(|x| x.extract::<f64>());
            match prob {
                Ok(prob) => grams.push((word, prob)),
                Err(e) => {
                    failure = Some((word, e));
                    break;
                }
            }
        }
        let profile = LangProfile {
            name: name.to_string(),
            freq: grams,
            n_words: vec![1.0; 3],
        };
        self.inner
            .add_profile_at(&profile, index, langsize)
            .map_err(|e| to_py_err(py, e))?;
        match failure {
            None => Ok(()),
            Some((word, e)) => {
                // langdetect creates the n-gram's row before computing its probability.
                self.inner.touch_ngram(&word, langsize);
                Err(e)
            }
        }
    }

    fn langs(&self) -> Vec<String> {
        self.inner.langlist().to_vec()
    }

    #[pyo3(signature = (text, seed=None))]
    fn detect(
        &self,
        py: Python<'_>,
        text: &Bound<'_, PyAny>,
        seed: Option<&Bound<'_, SeedState>>,
    ) -> PyResult<Option<String>> {
        let text = text_arg(text)?;
        let seed = seed_of(seed);
        let ranked = py.detach(|| detect_langs_one(&self.inner, &text.text, &seed));
        Ok(features(py, ranked)?.map(top))
    }

    #[pyo3(signature = (text, seed=None))]
    fn detect_langs(
        &self,
        py: Python<'_>,
        text: &Bound<'_, PyAny>,
        seed: Option<&Bound<'_, SeedState>>,
    ) -> PyResult<Option<Ranked>> {
        let text = text_arg(text)?;
        let seed = seed_of(seed);
        features(
            py,
            py.detach(|| detect_langs_one(&self.inner, &text.text, &seed)),
        )
    }

    #[pyo3(signature = (texts, seed=None))]
    fn detect_batch(
        &self,
        py: Python<'_>,
        texts: Vec<Bound<'_, PyAny>>,
        seed: Option<&Bound<'_, SeedState>>,
    ) -> PyResult<Vec<Option<String>>> {
        Ok(self
            .detect_langs_batch(py, texts, seed)?
            .into_iter()
            .map(|r| r.map(top))
            .collect())
    }

    /// Texts without features give None; any other error is raised, the first in order.
    #[pyo3(signature = (texts, seed=None))]
    fn detect_langs_batch(
        &self,
        py: Python<'_>,
        texts: Vec<Bound<'_, PyAny>>,
        seed: Option<&Bound<'_, SeedState>>,
    ) -> PyResult<Vec<Option<Ranked>>> {
        let texts = texts.iter().map(text_arg).collect::<PyResult<Vec<_>>>()?;
        let seed = seed_of(seed);
        py.detach(|| map_batch(&texts, |t| detect_langs_one(&self.inner, t, &seed)))
            .into_iter()
            .map(|r| features(py, r))
            .collect()
    }

    /// `Detector._detect_block` on an already cleaned text. Returns `(langprob, error)`:
    /// `(None, "features")` when the text has no features; otherwise `langprob` is what
    /// langdetect leaves in `Detector.langprob`, and `error` is None or the exception it raises.
    #[pyo3(signature = (text, alpha, n_trial, prior_map, seed=None))]
    fn detect_block<'py>(
        &self,
        py: Python<'py>,
        text: &Bound<'py, PyAny>,
        alpha: f64,
        n_trial: i64,
        prior_map: Option<Vec<f64>>,
        seed: Option<&Bound<'py, SeedState>>,
    ) -> PyResult<BlockResult<'py>> {
        let text = text_arg(text)?;
        let seed = seed_of(seed);
        let options = DetectOptions {
            alpha,
            n_trial,
            prior_map: prior_map.as_deref(),
        };
        match py.detach(|| self.inner.detect_block(&text.text, &options, &seed)) {
            Ok(langprob) => Ok((Some(langprob), None)),
            Err(Error::LangDetect { .. }) => {
                Ok((None, Some(PyString::new(py, "features").into_any())))
            }
            Err(e) => {
                let langprob = e.langprob().map(<[f64]>::to_vec);
                let err = to_py_err(py, e).into_value(py).into_bound(py).into_any();
                Ok((langprob, Some(err)))
            }
        }
    }
}

fn top(ranked: Ranked) -> String {
    ranked
        .into_iter()
        .next()
        .map_or_else(|| "unknown".to_string(), |(lang, _)| lang)
}

/// Ok(None) for a text without features; other errors become Python exceptions.
fn features<T>(py: Python<'_>, result: Result<T, Error>) -> PyResult<Option<T>> {
    match result {
        Ok(v) => Ok(Some(v)),
        Err(Error::LangDetect { .. }) => Ok(None),
        Err(e) => Err(to_py_err(py, e)),
    }
}

/// `langdetect.detect_langs` with the default detector options.
fn detect_langs_one(factory: &DetectorFactory, text: &str, seed: &Seed) -> Result<Ranked, Error> {
    let mut prepared = text::prepare(text, 10000);
    if let Some(cleaned) = text::clean(&prepared) {
        prepared = cleaned;
    }
    let langprob = factory.detect_block(&prepared, &DetectOptions::default(), seed)?;
    Ok(factory
        .sort_probability(&langprob)
        .into_iter()
        .map(|l| (l.lang, l.prob))
        .collect())
}

/// The text one `Detector.append(text)` call adds. With `substitute=False`, the caller has
/// already replaced URLs and e-mail addresses and normalised Vietnamese.
#[pyfunction]
#[pyo3(signature = (text, max_text_length, substitute=true))]
fn prepare<'py>(
    py: Python<'py>,
    text: &Bound<'py, PyAny>,
    max_text_length: i64,
    substitute: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let t = text_arg(text)?;
    let out = if substitute {
        text::prepare(&t.text, max_text_length)
    } else {
        text::truncate(&t.text, max_text_length)
    };
    to_py_str(py, &out, t.surrogates)
}

/// `Detector.cleaning_text`; None when the text stays unchanged.
#[pyfunction]
fn clean<'py>(py: Python<'py>, text: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let t = text_arg(text)?;
    text::clean(&t.text)
        .map(|c| to_py_str(py, &c, t.surrogates))
        .transpose()
}

/// `NGram.normalize` of one character.
#[pyfunction]
fn normalize_char(ch: char) -> char {
    text::normalize(ch)
}

#[pyfunction]
fn vietnamese_tables() -> (String, String, Vec<String>) {
    text::vietnamese_tables()
}

#[pyfunction]
fn builtin_files() -> Vec<String> {
    DetectorFactory::builtin_files()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Factory>()?;
    m.add_class::<SeedState>()?;
    m.add_function(wrap_pyfunction!(prepare, m)?)?;
    m.add_function(wrap_pyfunction!(clean, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_char, m)?)?;
    m.add_function(wrap_pyfunction!(vietnamese_tables, m)?)?;
    m.add_function(wrap_pyfunction!(builtin_files, m)?)?;
    Ok(())
}
