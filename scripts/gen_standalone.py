"""Generate bindings/python/tests/standalone.json, the langdetect 1.0.9 results that
bindings/python/tests/check_standalone.py expects from langdetect_rs without langdetect.

Usage: python gen_standalone.py OUT.json
Run it under a Python up to 3.11 and under one from 3.12, which sum() differently; each run
adds the results for its Python and keeps the texts already in OUT.json. Profiles are loaded
in name order, as langdetect_rs does when langdetect is not installed. Probabilities below
SMALL are left out of langprob.
"""

import json
import os
import random
import sys

from langdetect import DetectorFactory, LangDetectException
from langdetect.detector_factory import PROFILES_DIRECTORY

SEED = 0
SMALL = 1e-12

POOLS = [
    (0x20, 0x7E),
    (0xA0, 0x24F),
    (0x370, 0x52F),
    (0x590, 0x6FF),
    (0x900, 0xDFF),
    (0xE00, 0xEFF),
    (0x3040, 0x30FF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7A3),
]


def make_texts():
    rng = random.Random(2026)
    texts = [
        "",
        "12345 !!! ???",
        "HELLO WORLD THIS IS ALL CAPS",
        "contact someone.name@example.org or https://example.com/path?q=1",
        "mixed English и русский text 混合 テキスト",
        "Đây là một câu tiếng Việt về thời tiết ngày mai.",
        "Le chat dort. " * 900,
    ]
    for name in sorted(os.listdir(PROFILES_DIRECTORY)):
        with open(os.path.join(PROFILES_DIRECTORY, name), encoding="utf-8") as f:
            freq = json.load(f)["freq"]
        grams = sorted((g for g in freq if len(g) == 3 and " " not in g), key=freq.get)[-300:]
        for n_words in (2, 6, 20, 80):
            words = (
                "".join(rng.choice(grams) for _ in range(rng.randint(1, 3))) for _ in range(n_words)
            )
            texts.append(" ".join(words))
    while len(texts) < 300:
        parts = []
        for _ in range(rng.randint(1, 4)):
            lo, hi = rng.choice(POOLS)
            parts.append("".join(chr(rng.randint(lo, hi)) for _ in range(rng.randint(1, 30))))
        text = " ".join(parts)
        texts.append(text.upper() if rng.random() < 0.2 else text)
    return texts


def significant(langlist, langprob):
    return {lang: p for lang, p in zip(langlist, langprob) if p >= SMALL}


def main():
    out = sys.argv[1]
    try:
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"seed": SEED, "texts": make_texts()}

    factory = DetectorFactory()
    listdir = os.listdir
    os.listdir = lambda directory: sorted(listdir(directory))
    try:
        factory.load_profile(PROFILES_DIRECTORY)
    finally:
        os.listdir = listdir
    factory.set_seed(data["seed"])

    results = []
    for text in data["texts"]:
        detector = factory.create()
        detector.append(text)
        try:
            langs = [[lang.lang, lang.prob] for lang in detector.get_probabilities()]
        except LangDetectException:
            results.append({"langs": None, "langprob": None})
            continue
        langprob = significant(factory.langlist, detector.langprob)
        results.append({"langs": langs, "langprob": langprob})
    data["neumaier" if sys.version_info >= (3, 12) else "sequential"] = results

    fields = [f'"seed": {data["seed"]}']
    for key in ("texts", "sequential", "neumaier"):
        if key in data:
            rows = ",\n".join(json.dumps(row, ensure_ascii=False) for row in data[key])
            fields.append(f'"{key}": [\n{rows}\n]')
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("{\n" + ",\n".join(fields) + "\n}\n")


if __name__ == "__main__":
    main()
