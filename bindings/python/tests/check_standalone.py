"""Compare langdetect_rs, installed without langdetect, with the langdetect 1.0.9 results that
make_reference.py wrote on the same machine. Every probability must be identical.

Usage: python check_standalone.py REFERENCE.json
"""

import importlib.util
import json
import sys
from pathlib import Path

import langdetect_rs
from langdetect_rs import PROFILES_DIRECTORY, DetectorFactory, LangDetectException


def pairs(langs):
    return None if langs is None else [[lang.lang, lang.prob] for lang in langs]


def main():
    if importlib.util.find_spec("langdetect") is not None:
        sys.exit("langdetect is installed; run this check where it is not")
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    python = "{}.{}".format(*sys.version_info[:2])
    if data["python"] != python:
        sys.exit(f"the reference was made with Python {data['python']}, this is {python}")
    texts = data["texts"]

    DetectorFactory.seed = data["seed"]
    batch = langdetect_rs.detect_langs_batch(texts)
    tops = langdetect_rs.detect_batch(texts)
    factory = DetectorFactory()
    factory.load_profile(PROFILES_DIRECTORY)
    factory.set_seed(data["seed"])
    if factory.get_lang_list() != data["langlist"]:
        sys.exit(f"profile order {factory.get_lang_list()}, langdetect's {data['langlist']}")

    differ = 0
    for i, (text, want) in enumerate(zip(texts, data["results"])):
        detector = factory.create()
        detector.append(text)
        try:
            got = {
                "langs": pairs(detector.get_probabilities()),
                "langprob": detector.langprob,
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
