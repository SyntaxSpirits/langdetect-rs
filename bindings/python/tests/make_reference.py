"""Write langdetect 1.0.9's results for 300 texts to OUT.json, for check_standalone.py to
compare with langdetect_rs installed without langdetect.

Usage: python make_reference.py OUT.json
Run it on the machine and the Python minor version the check runs on: langdetect's last bits
depend on the platform's log and cos (random.gauss) and on the Python's sum(). Profiles are
loaded in name order, as langdetect_rs does when langdetect is not installed.
"""

import json
import os
import random
import sys

from langdetect import DetectorFactory, LangDetectException
from langdetect.detector_factory import PROFILES_DIRECTORY

SEED = 0

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


def main():
    factory = DetectorFactory()
    listdir = os.listdir
    os.listdir = lambda directory: sorted(listdir(directory))
    try:
        factory.load_profile(PROFILES_DIRECTORY)
    finally:
        os.listdir = listdir
    factory.set_seed(SEED)

    texts = make_texts()
    results = []
    for text in texts:
        detector = factory.create()
        detector.append(text)
        try:
            langs = [[lang.lang, lang.prob] for lang in detector.get_probabilities()]
        except LangDetectException:
            results.append({"langs": None, "langprob": None})
            continue
        results.append({"langs": langs, "langprob": detector.langprob})

    python = "{}.{}".format(*sys.version_info[:2])
    data = {
        "python": python,
        "seed": SEED,
        "langlist": factory.get_lang_list(),
        "texts": texts,
        "results": results,
    }
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
