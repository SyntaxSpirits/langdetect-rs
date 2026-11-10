"""Compare langdetect_rs, installed without langdetect, with the langdetect 1.0.9 results in
standalone.json (made by scripts/gen_standalone.py).

Usage: python check_standalone.py [standalone.json]
"""

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import langdetect_rs
from langdetect_rs import PROFILES_DIRECTORY, DetectorFactory, LangDetectException


def digest(langprob):
    return hashlib.sha256(struct.pack(f"<{len(langprob)}d", *langprob)).hexdigest()


def pairs(langs):
    return None if langs is None else [[lang.lang, lang.prob] for lang in langs]


def main():
    if importlib.util.find_spec("langdetect") is not None:
        sys.exit("langdetect is installed; run this check where it is not")
    path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("standalone.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    texts = data["texts"]
    expected = data["neumaier" if sys.version_info >= (3, 12) else "sequential"]

    DetectorFactory.seed = data["seed"]
    batch = langdetect_rs.detect_langs_batch(texts)
    tops = langdetect_rs.detect_batch(texts)
    factory = DetectorFactory()
    factory.load_profile(PROFILES_DIRECTORY)
    factory.set_seed(data["seed"])

    differ = 0
    for i, (text, want) in enumerate(zip(texts, expected)):
        detector = factory.create()
        detector.append(text)
        try:
            got = {
                "langs": pairs(detector.get_probabilities()),
                "langprob": digest(detector.langprob),
            }
        except LangDetectException:
            got = {"langs": None, "langprob": None}
        top = want["langs"] and want["langs"][0][0]
        if got != want or pairs(batch[i]) != want["langs"] or tops[i] != top:
            differ += 1
            print(f"text {i}: expected {want}, got {got}, batch {pairs(batch[i])}, {tops[i]}")
    if differ:
        sys.exit(f"{differ} of {len(texts)} texts differ from langdetect")
    version = sys.version.split()[0]
    print(f"{len(texts)} texts identical to langdetect 1.0.9 on Python {version}")


if __name__ == "__main__":
    main()
