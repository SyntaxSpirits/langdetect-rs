# Benchmarks

`bindings/python/benchmarks/compare.py` times `langdetect` 1.0.9 and `langdetect-rs` on 2,000
texts sampled (seed 1729) from each of two test sets:

- [WiLI-2018](https://huggingface.co/datasets/MartinThoma/wili_2018) test split: Wikipedia
  paragraphs in 235 languages, 364 characters on average in the sample;
- [papluca/language-identification](https://huggingface.co/datasets/papluca/language-identification)
  test split: reviews and short texts in 20 languages, 111 characters on average.

Each figure is the best of three runs (two for `langdetect`), in milliseconds per text.
"Loop" calls `langdetect_rs.detect` once per text; "batch" passes all texts to
`langdetect_rs.detect_batch`, which uses every core. Without a seed is the default;
`DetectorFactory.seed = 0` gives reproducible results and is what `unstructured` sets.

## Apple M1 Pro, 8 cores, macOS 14.5, CPython 3.13.9, langdetect-rs 0.1.0

| Data | Seed | langdetect | Loop | Batch | Loop speed-up | Batch speed-up |
|---|---|---|---|---|---|---|
| papluca | none | 1.452 | 0.0365 | 0.0059 | 39.8× | 245.9× |
| WiLI-2018 | none | 3.455 | 0.0786 | 0.0118 | 44.0× | 292.7× |
| papluca | 0 | 1.465 | 0.0285 | 0.0047 | 51.3× | 314.7× |
| WiLI-2018 | 0 | 3.446 | 0.0660 | 0.0116 | 52.2× | 296.8× |

Starting Python, importing the package and detecting one text takes 244 ms with
`langdetect`, which reads 55 JSON profiles, and 74 ms with `langdetect-rs`.

Raw results: [`bindings/python/benchmarks/results/m1-pro.json`](bindings/python/benchmarks/results/m1-pro.json).

## Parity on the full test sets

`bindings/python/benchmarks/parity.py` runs every text of both test sets (127,500) through
both packages, with seeds rotating over 0, 1, 42, 2**40 + 7, -5 and "seed", and compares the
full probability vector and the `detect_langs` output. On the machine above:

| CPython | Texts | Differences |
|---|---|---|
| 3.9.20 | 127,500 | 0 |
| 3.11.5 | 127,500 | 0 |
| 3.13.9 | 127,500 | 0 |
| 3.14.8 | 127,500 | 0 |

The "Full parity" GitHub workflow runs the same script on Linux, macOS and Windows.

## Reproducing

```sh
cd bindings/python
uv sync --group dev --group bench
cd benchmarks
uv run python compare.py --output results-local
uv run python parity.py
```

The datasets are downloaded from Hugging Face on the first run and cached in
`~/.cache/langdetect-rs-bench` (`LANGDETECT_BENCH_CACHE` overrides it).
