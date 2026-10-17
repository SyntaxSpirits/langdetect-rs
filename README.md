# langdetect-rs

[![CI](https://github.com/SyntaxSpirits/langdetect-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/SyntaxSpirits/langdetect-rs/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/langdetect-rs.svg)](https://pypi.org/project/langdetect-rs/)
[![Crates.io](https://img.shields.io/crates/v/langdetect.svg)](https://crates.io/crates/langdetect)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A Rust port of [`langdetect`](https://pypi.org/project/langdetect/) that returns the same
results, down to the last bit of every probability. On WiLI-2018 and papluca test samples
it detects languages 40–52 times faster on one core and 245–315 times faster with all
eight cores of an M1 Pro
([benchmarks](https://github.com/SyntaxSpirits/langdetect-rs/blob/main/BENCHMARKS.md)).

`langdetect` is pure Python and has not had a release since 2021, yet it is downloaded about
ten million times a month: document pipelines such as `unstructured` and the IFEval checks of
`lm-eval`, `lighteval` and `inspect-evals` depend on it. Its detection is a random walk over
character n-grams, so other detectors give different answers. `langdetect-rs` reproduces the
walk itself: the same Mersenne Twister, the same Gaussian smoothing, the same floating-point
operations in the same order. With a seed, every probability equals the original's; without
one, results vary from run to run as they do in `langdetect`.

The Python package is `langdetect-rs`; the Rust crate is `langdetect`. This is an independent
project, not affiliated with the authors of `langdetect` or `language-detection`.

## Python

```sh
pip install langdetect-rs
```

Change the import and keep the rest of your code:

```python
from langdetect_rs import DetectorFactory, detect, detect_langs  # was: from langdetect import ...

DetectorFactory.seed = 0
detect("Ein, zwei, drei, vier")
# 'de'
detect_langs("Война и мир — роман Льва Толстого.")
# [ru:0.5714279654116534, bg:0.42857201882885804]
```

`Detector` (with `append`, `set_alpha`, `set_prior_map`, `set_max_text_length`),
`DetectorFactory` (with `load_profile`, `load_json_profile`, `create`, `set_seed`),
`LangDetectException`, `ErrorCode`, `Language` and `utils.lang_profile.LangProfile` work as in
`langdetect` 1.0.9, with the same results, exceptions and messages, also in its corner cases
(profiles loaded into a non-empty factory, priors of the wrong length, failed detections).

To detect many texts, use the batch functions. They run on all CPU cores and release the GIL;
a text without features gives `None` instead of raising:

```python
from langdetect_rs import detect_batch

detect_batch(["Guten Morgen, wie geht es dir heute?", "Bonjour tout le monde", "1234"])
# ['de', 'fr', None]
```

Set `RAYON_NUM_THREADS` to limit the number of threads.

### unstructured

Since 0.27.28, `unstructured` uses `langdetect-rs` when it is installed and
`UNSTRUCTURED_USE_LANGDETECT_RS=true` is set before `unstructured` is imported:

```bash
pip install "unstructured>=0.27.28" langdetect-rs
export UNSTRUCTURED_USE_LANGDETECT_RS=true
```

Detected languages stay the same. If the variable is set but `langdetect-rs` is missing,
`unstructured` logs a warning and uses `langdetect`.

### Code you do not control

When a library imports `langdetect` itself, call `install()` before importing that library:

```python
import langdetect_rs

langdetect_rs.install()  # `import langdetect` now returns langdetect_rs

import lm_eval
```

`install()` covers `langdetect` and its modules `detector`, `detector_factory`,
`lang_detect_exception`, `language`, `utils.lang_profile` and `utils.ngram`. Modules that
imported `langdetect` before the call keep the original.

## Rust

```toml
[dependencies]
langdetect = "0.1"
```

```rust
use langdetect::{DetectorFactory, Seed};

assert_eq!(langdetect::detect("Ein, zwei, drei, vier")?, "de");

let mut factory = DetectorFactory::builtin().clone();
factory.seed = Seed::Int(0);
for lang in factory.detect_langs("Ceci est une phrase en français.")? {
    println!("{lang}");
}
```

`DetectorFactory` is `Send + Sync`. `Compat` selects which Python behaviour to reproduce
(see below); the default is CPython 3.14.

## Compatibility

The tests compare `langdetect-rs` with `langdetect` 1.0.9 and fail on any difference in any
probability: sentences and synthetic texts for all 55 languages, random Unicode including
lone surrogates, property-based tests, every detector option, custom and JSON profiles, prior
maps and error cases, on CPython 3.9 to 3.15. A separate script compares all 127,500 texts of
the WiLI-2018 and papluca test sets.

`langdetect`'s own results depend on the Python it runs on, and `langdetect-rs` follows:

- From Python 3.12, `sum()` uses compensated summation. `langdetect` normalises
  probabilities with `sum()`, so its probabilities on WiLI-2018 and papluca differ between
  Python 3.11 and 3.12 for 38% of texts (in the last bits; the detected language did not
  change).
- `str.isupper()` follows the Unicode version of the Python, and decides when a run of
  capital letters is skipped. Tables for Unicode 13.0 to 17.0 (Python 3.9 to 3.15) are built
  in.
- `langdetect` loads its profiles in `os.listdir` order, which depends on the file system and
  changes the last bits of a probability in about one text in 100,000. If `langdetect` is
  installed, `langdetect-rs` uses its order; otherwise, alphabetical order.

Factories and detectors can be pickled, also across processes, and batches stay usable in
processes forked after a batch call.

Known differences:

- Without a seed, `langdetect` seeds each detection with 2.5 KB from `os.urandom`;
  `langdetect-rs` takes the seed from a per-thread generator that is seeded by the operating
  system and reseeded after `fork`. Results are random either way.
- `DetectorFactory.word_lang_prob_map` and `Detector.random` do not exist, and
  `Detector.set_verbose()` is accepted but prints nothing.
- Lone surrogates are handled as the private-use characters U+10F800 to U+10FFFF. This only
  matters for custom profiles that contain one of these and texts that contain the other.
- Profile names must be strings, and `add_profile` needs a non-negative `index`.
- `LangDetectException` is a separate class from `langdetect`'s (after `install()` they are
  the same).

## Development

```sh
cargo test -p langdetect
cd bindings/python
uv sync --group dev
uv run pytest
cd benchmarks && uv run --group bench python parity.py   # all 127,500 texts
```

## License

Apache-2.0, the same licence as `langdetect` and `language-detection`, whose detection logic,
language profiles and normalisation tables this project ports. See [NOTICE](NOTICE).
