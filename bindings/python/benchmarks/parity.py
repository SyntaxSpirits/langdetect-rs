"""Compare langdetect_rs with langdetect 1.0.9 on every text of WiLI-2018 and papluca.

For each text the full probability vector of the Detector and the detect_langs() output must be
identical. Seeds rotate over integers, a negative, a large integer and a string.
"""

import argparse
import json
import random
import sys
import time

import langdetect
from data import load
from langdetect.lang_detect_exception import LangDetectException

import langdetect_rs

SEEDS = [0, 1, 42, 2**40 + 7, -5, "seed"]


def outcome(module, text):
    langdetect_rs.detector_factory.init_factory()
    module.detector_factory.init_factory()
    d = module.detector_factory._factory.create()
    d.append(text)
    try:
        langs = [(x.lang, x.prob) for x in d.get_probabilities()]
    except (LangDetectException, langdetect_rs.LangDetectException):
        return None, None
    return d.langprob, langs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    texts = load("papluca") + load("wili")
    if args.limit:
        texts = random.Random(0).sample(texts, args.limit)
    mismatches = []
    start = time.perf_counter()
    for i, text in enumerate(texts):
        seed = SEEDS[i % len(SEEDS)]
        langdetect.DetectorFactory.seed = langdetect_rs.DetectorFactory.seed = seed
        if outcome(langdetect, text) != outcome(langdetect_rs, text):
            mismatches.append({"index": i, "seed": repr(seed), "text": text[:200]})
    report = {
        "python": sys.version.split()[0],
        "texts": len(texts),
        "mismatches": len(mismatches),
        "examples": mismatches[:20],
        "seconds": round(time.perf_counter() - start, 1),
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
