# Changelog

## 0.1.0 - 2026-10-11

First release.

- Rust port of `langdetect` 1.0.9 with bit-identical results, including the random walk, for
  CPython 3.9 to 3.15 (`Compat` selects the `sum()` and Unicode behaviour).
- The 55 language profiles of `langdetect` are built into the binary.
- Python package `langdetect-rs`: the `langdetect` API (`detect`, `detect_langs`, `Detector`,
  `DetectorFactory`, `LangProfile`, exceptions), `detect_batch` and `detect_langs_batch` on all
  cores without the GIL, and `install()` to replace `import langdetect`.
