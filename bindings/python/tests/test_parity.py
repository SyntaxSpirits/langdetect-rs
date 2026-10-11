"""Bit-for-bit comparison with langdetect 1.0.9."""

import langdetect
import pytest
from conftest import SEEDS
from hypothesis import given, settings
from hypothesis import strategies as st

import langdetect_rs


def outcome(fn, *args):
    try:
        return "ok", fn(*args)
    except langdetect.LangDetectException as e:
        return "error", e.code, str(e)
    except langdetect_rs.LangDetectException as e:
        return "error", e.code, str(e)


def probabilities(detector):
    return langs(detector.get_probabilities())


def langs(result):
    return [(lang.lang, lang.prob) for lang in result]


def test_detect_langs_matches(corpus, seeded):
    for i, text in enumerate(corpus):
        seeded(SEEDS[i % len(SEEDS)])
        expected = outcome(lambda t: langs(langdetect.detect_langs(t)), text)
        got = outcome(lambda t: langs(langdetect_rs.detect_langs(t)), text)
        assert got == expected, (i, text[:80])


def test_detect_matches(corpus, seeded):
    seeded(0)
    for text in corpus:
        assert outcome(langdetect_rs.detect, text) == outcome(langdetect.detect, text), text[:80]


def test_full_probability_vector_matches(corpus, seeded):
    seeded(7)
    langdetect.detector_factory.init_factory()
    langdetect_rs.detector_factory.init_factory()
    factory = langdetect.detector_factory._factory
    ours = langdetect_rs.detector_factory._factory
    assert ours.langlist == factory.langlist
    for text in corpus:
        a, b = factory.create(), ours.create()
        a.append(text)
        b.append(text)
        ra, rb = outcome(a._detect_block), outcome(b._detect_block)
        assert ra == rb
        assert a.text == b.text
        assert a.langprob == b.langprob, text[:80]


def test_batch_matches_sequential(corpus, seeded):
    seeded(3)
    expected = [outcome(langdetect.detect, t) for t in corpus]
    got = langdetect_rs.detect_batch(corpus)
    for g, e in zip(got, expected):
        assert (e[0] == "error") if g is None else (e == ("ok", g))
    batch = langdetect_rs.detect_langs_batch(corpus)
    for text, result in zip(corpus, batch):
        exp = outcome(lambda t: langs(langdetect.detect_langs(t)), text)
        assert (exp[0] == "error") if result is None else (exp == ("ok", langs(result)))


@pytest.mark.parametrize(
    "setup",
    [
        lambda d: d.set_alpha(0.0),
        lambda d: d.set_alpha(2.5),
        lambda d: d.set_max_text_length(12),
        lambda d: d.set_prior_map({"en": 0.1, "de": 0.5, "fr": 0.4}),
        lambda d: d.set_prior_map({"ja": 3}),
        lambda d: setattr(d, "n_trial", 1),
        lambda d: setattr(d, "n_trial", 13),
    ],
)
def test_detector_options_match(corpus, seeded, setup):
    seeded(11)
    langdetect.detector_factory.init_factory()
    langdetect_rs.detector_factory.init_factory()
    for text in corpus[:200]:
        a = langdetect.detector_factory._factory.create()
        b = langdetect_rs.detector_factory._factory.create()
        for d in (a, b):
            setup(d)
            d.append(text)
            d.append(" second part " + text[:20])
        assert outcome(probabilities, a) == outcome(probabilities, b)
        assert a.langprob == b.langprob


@settings(max_examples=400, deadline=None)
@given(text=st.text(max_size=200), seed=st.one_of(st.none(), st.integers(), st.text(max_size=5)))
def test_random_text(text, seed):
    old = langdetect.DetectorFactory.seed, langdetect_rs.DetectorFactory.seed
    try:
        seed = 0 if seed is None else seed
        langdetect.DetectorFactory.seed = langdetect_rs.DetectorFactory.seed = seed
        expected = outcome(lambda t: langs(langdetect.detect_langs(t)), text)
        assert outcome(lambda t: langs(langdetect_rs.detect_langs(t)), text) == expected
    finally:
        langdetect.DetectorFactory.seed, langdetect_rs.DetectorFactory.seed = old
