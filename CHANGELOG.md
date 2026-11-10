# Changelog

## Unreleased

- On free-threaded CPython, importing `langdetect_rs` now enables the GIL instead of declaring
  the module safe to run without it.
- On a Python whose `unicodedata` version has no built-in `str.isupper()` tables,
  `langdetect_rs` warns on import (`RuntimeWarning`) that it uses the Unicode 17.0 tables.
- CI tests CPython 3.15 and free-threaded 3.14, and checks a wheel installed without
  `langdetect` against stored `langdetect` results (to a relative tolerance of 1e-9, as
  `langdetect` itself differs in the last bits between platforms). Releases are published only
  after the built wheel and sdist pass that check in clean environments.
- README: processing large corpora and dataframes in chunks with the batch functions.

## 0.1.0 - 2026-10-11

First release.

- Rust port of `langdetect` 1.0.9 with bit-identical results, including the random walk, for
  CPython 3.9 to 3.15 (`Compat` selects the `sum()` and Unicode behaviour).
- The 55 language profiles of `langdetect` are built into the binary.
- Python package `langdetect-rs`: the `langdetect` API (`detect`, `detect_langs`, `Detector`,
  `DetectorFactory`, `LangProfile`, exceptions), `detect_batch` and `detect_langs_batch` on all
  cores without the GIL, and `install()` to replace `import langdetect`.
