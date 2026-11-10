"""Compare langdetect_rs, installed without langdetect, with the langdetect 1.0.9 results in
standalone.json (made by scripts/gen_standalone.py).

langdetect's random walk calls the platform's log and cos, so its last bits differ between
macOS, glibc, musl and Windows; probabilities are compared with a relative tolerance.

Usage: python check_standalone.py [standalone.json]
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import langdetect_rs
from langdetect_rs import PROFILES_DIRECTORY, DetectorFactory, LangDetectException

REL_TOL = 1e-9
SMALL = 1e-12


def close(a, b):
    return math.isclose(a, b, rel_tol=REL_TOL)


def same_langs(want, got):
    if want is None or got is None:
        return want is got
    return len(want) == len(got) and all(
        wl == gl and close(wp, gp) for (wl, wp), (gl, gp) in zip(want, got)
    )


def same_langprob(want, got):
    if want is None or got is None:
        return want is got
    for lang in want.keys() | got.keys():
        w, g = want.get(lang, 0.0), got.get(lang, 0.0)
        if not (close(w, g) or max(w, g) < SMALL * (1 + REL_TOL)):
            return False
    return True


def same_result(want, got):
    return same_langs(want["langs"], got["langs"]) and same_langprob(
        want["langprob"], got["langprob"]
    )


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
                "langprob": dict(zip(factory.langlist, detector.langprob)),
            }
        except LangDetectException:
            got = {"langs": None, "langprob": None}
        top = want["langs"] and want["langs"][0][0]
        if (
            not same_result(want, got)
            or not same_langs(want["langs"], pairs(batch[i]))
            or tops[i] != top
        ):
            differ += 1
            print(f"text {i}: expected {want}, got {got}, batch {pairs(batch[i])}, {tops[i]}")
    if differ:
        sys.exit(f"{differ} of {len(texts)} texts differ from langdetect")
    version = sys.version.split()[0]
    print(f"{len(texts)} texts match langdetect 1.0.9 on Python {version}")


if __name__ == "__main__":
    main()
