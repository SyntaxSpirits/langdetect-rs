"""Corner cases where langdetect's behaviour is unusual; each scenario runs on both packages."""

import os
import pickle
import subprocess
import sys
import textwrap

import langdetect
import pytest
from langdetect.utils.lang_profile import LangProfile as OriginalProfile

import langdetect_rs
from langdetect_rs.utils.lang_profile import LangProfile

MODULES = [
    (langdetect, OriginalProfile),
    (langdetect_rs, LangProfile),
]


def run(scenario):
    """Outcome of scenario(module, LangProfile): its result or the exception type and text."""
    out = []
    for module, profile_class in MODULES:
        try:
            out.append(("ok", scenario(module, profile_class)))
        except Exception as e:
            out.append((type(e).__name__, str(e), getattr(e, "code", None)))
    return out


def same(scenario):
    original, ours = run(scenario)
    assert ours == original
    return original


def probs(detector):
    return [(x.lang, x.prob) for x in detector.get_probabilities()]


def two_langs(m, profile, first=("x", 9), second=("y", 1), char="a"):
    f = m.DetectorFactory()
    f.set_seed(0)
    for name, count in (first, second):
        f.add_profile(profile(name, {char: count}, [10, 0, 0]), len(f.langlist), 2)
    return f


def test_clear_keeps_existing_detectors():
    def scenario(m, profile):
        f = two_langs(m, profile)
        d = f.create()
        d.append("a")
        f.clear()
        return d.detect(), f.get_lang_list()

    assert same(scenario)[1] == ("x", [])


@pytest.mark.parametrize("order", [(1, 0), (0, 0), (1, 1)])
def test_add_profile_honours_index(order):
    def scenario(m, profile):
        f = m.DetectorFactory()
        f.set_seed(0)
        f.add_profile(profile("x", {"a": 9}, [10, 0, 0]), order[0], 2)
        f.add_profile(profile("y", {"a": 1}, [10, 0, 0]), order[1], 2)
        d = f.create()
        d.append("a")
        return probs(d), d.langprob

    same(scenario)


@pytest.mark.parametrize(("index", "langsize"), [(2, 2), (0, 1), (5, 10), (0, 0)])
def test_add_profile_index_out_of_range(index, langsize):
    def scenario(m, profile):
        f = m.DetectorFactory()
        f.set_seed(0)
        try:
            f.add_profile(profile("x", {"a": 1, "b": 2}, [10, 0, 0]), index, langsize)
        except Exception as e:
            error = (type(e).__name__, str(e))
        else:
            error = None
        f.add_profile(profile("y", {"a": 3}, [10, 0, 0]), 1, 2)
        d = f.create()
        d.append("a b")
        try:
            result = probs(d)
        except Exception as e:
            result = (type(e).__name__, str(e))
        return error, f.get_lang_list(), result, d.langprob

    same(scenario)


def test_profiles_loaded_into_a_non_empty_factory():
    one = '{"freq":{"A":3,"B":6,"AB":2},"n_words":[12,3,4],"name":"one"}'
    two = '{"freq":{"A":6,"B":3,"AA":3},"n_words":[12,5,3],"name":"two"}'

    def scenario(m, profile):
        f = m.DetectorFactory()
        f.load_profile(m.PROFILES_DIRECTORY if m is langdetect else langdetect.PROFILES_DIRECTORY)
        f.load_json_profile([one, two])
        f.set_seed(3)
        out = []
        for text in ("AB AB BA", "Guten Morgen", "Ceci est une phrase"):
            d = f.create()
            d.append(text)
            out.append(probs(d))
        return f.get_lang_list(), out

    same(scenario)


# Lone surrogates travel as U+10F800-U+10FFFF, so mixing the two kinds is a documented limit.
@pytest.mark.parametrize("char", ["￿", "\ud800", "\udfff", "\U0010f7ff"])
def test_unusual_characters_in_custom_profiles(char):
    def scenario(m, profile):
        f = two_langs(m, profile, char=char)
        out = []
        for text in ["\ud800", "￿", "a\udfff", "\U0010f7ff", "𐏿 x"]:
            d = f.create()
            d.append(text)
            try:
                out.append((probs(d), d.text))
            except Exception as e:
                out.append((type(e).__name__, str(e), d.text))
        return out

    same(scenario)


def test_state_after_zero_division():
    def scenario(m, profile):
        m.detector_factory.init_factory()
        d = m.detector_factory._factory.create()
        d.seed = 0
        d.append("a")
        with pytest.raises(m.LangDetectException):
            d.set_prior_map({})
        first = run_once(d.detect)
        return first, run_once(d.detect), d.langprob

    same(scenario)


def run_once(fn):
    try:
        return fn()
    except Exception as e:
        return type(e).__name__, str(e)


@pytest.mark.parametrize(
    "setup",
    [
        lambda d: setattr(d, "prior_map", []),
        lambda d: (setattr(d, "prior_map", []), setattr(d, "n_trial", 0)),
        lambda d: setattr(d, "prior_map", [1.0] * 56),
        lambda d: setattr(d, "prior_map", [1.0] * 54),
        lambda d: setattr(d, "prior_map", [0] * 55),
        lambda d: setattr(d, "max_text_length", 2**100),
        lambda d: setattr(d, "max_text_length", 10000.0),
        lambda d: setattr(d, "max_text_length", 3.0),
        lambda d: setattr(d, "max_text_length", -(2**100)),
        lambda d: setattr(d, "n_trial", -(2**100)),
        lambda d: setattr(d, "n_trial", 2.0),
        lambda d: setattr(d, "n_trial", True),
        lambda d: setattr(d, "seed", [1]),
        lambda d: setattr(d, "URL_RE", __import__("re").compile("hello")),
        lambda d: setattr(d, "MAIL_RE", __import__("re").compile("o")),
    ],
)
@pytest.mark.parametrize("text", ["hello world", "1234", ""])
def test_unusual_detector_settings(setup, text):
    def scenario(m, profile):
        m.detector_factory.init_factory()
        d = m.detector_factory._factory.create()
        d.seed = 0
        setup(d)
        try:
            d.append(text)
        except Exception as e:
            return "append", type(e).__name__, str(e)
        first = run_once(lambda: probs(d))
        return first, run_once(lambda: probs(d)), d.langprob, d.text

    same(scenario)


@pytest.mark.parametrize("value", [b"hello", None, 3, bytearray(b"x")])
def test_type_errors_match(value):
    same(lambda m, profile: m.detect(value))
    same(lambda m, profile: m.detect_langs(value))

    def detector(m, profile):
        m.detector_factory.init_factory()
        m.detector_factory._factory.create().append(value)

    same(detector)


@pytest.mark.parametrize(
    "args",
    [
        ({"a": 1}, [0, 0, 0]),
        ({"a": 1}, []),
        ({"a": 1, "bc": 2}, [1]),
        ({"a": 1, "bc": 2}, [1, 0]),
        ({"abcd": 1, "": 2}, []),
        ({"a": "x"}, [1, 1, 1]),
    ],
)
def test_direct_add_profile_errors(args):
    freq, n_words = args

    def scenario(m, profile):
        f = m.DetectorFactory()
        f.set_seed(0)
        error = run_once(lambda: f.add_profile(profile("x", dict(freq), list(n_words)), 0, 2))
        f.add_profile(profile("y", {"a": 2, "bc": 1}, [5, 5, 5]), 1, 2)
        d = f.create()
        d.append("a bc")
        return (
            error[0] if isinstance(error, tuple) else error,
            f.get_lang_list(),
            run_once(lambda: probs(d)),
        )

    same(scenario)


def test_pickle_round_trip(seeded):
    seeded(5)
    for module in (langdetect, langdetect_rs):
        module.detector_factory.init_factory()
    factory = pickle.loads(pickle.dumps(langdetect_rs.detector_factory._factory))
    d = factory.create()
    d.append("Hej, hur mår du idag?")
    d2 = pickle.loads(pickle.dumps(d))
    expected = langdetect.detector_factory._factory.create()
    expected.append("Hej, hur mår du idag?")
    assert probs(d) == probs(expected) == probs(d2)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_batch_after_fork():
    code = textwrap.dedent(
        """
        import os, signal, langdetect_rs
        langdetect_rs.DetectorFactory.seed = 0
        assert langdetect_rs.detect_batch(["hello world"] * 8) == ["en"] * 8
        pid = os.fork()
        if pid == 0:
            signal.alarm(10)
            ok = langdetect_rs.detect_batch(["hello world"] * 8) == ["en"] * 8
            os._exit(0 if ok else 1)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0, status
        print("ok")
        """
    )
    env = dict(os.environ, RAYON_NUM_THREADS="2")
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60
    )
    assert out.stdout.strip() == "ok", out.stderr


def test_partial_builtin_load_keeps_ngram_order():
    def scenario(m, profile):
        f = m.DetectorFactory()
        f.set_seed(0)
        first = run_once(lambda: f.add_profile(profile("x", {"a": 1}, [1, 1, 1]), 0, 0))
        second = run_once(lambda: f.load_profile(m.PROFILES_DIRECTORY))
        out = []
        for text in ["n", "a", "the", "ж"]:
            d = f.create()
            d.append(text)
            out.append(run_once(lambda d=d: probs(d)))
        return first, second[0], f.get_lang_list(), out

    same(scenario)


def test_module_functions_raise_detection_errors(seeded):
    seeded(0)

    def scenario(m, profile):
        f = m.DetectorFactory()
        f.add_profile(profile("x", {"a": 1}, [1, 1, 1]), 0, 1)
        f.add_profile(profile("y", {}, [1, 1, 1]), 1, 2)
        m.detector_factory.init_factory()
        saved = m.detector_factory._factory
        m.detector_factory._factory = f
        try:
            out = [run_once(lambda: m.detect("a")), run_once(lambda: m.detect_langs("a"))]
            if m is langdetect_rs:
                out.append(run_once(lambda: m.detect_batch(["b", "a"])))
            else:
                out.append(out[0])
            return out
        finally:
            m.detector_factory._factory = saved

    same(scenario)


def test_batch_honours_patched_patterns(seeded, monkeypatch):
    import re

    seeded(0)
    texts = ["hello world", "1234", "Bonjour tout le monde"]
    for module in (langdetect, langdetect_rs):
        monkeypatch.setattr(module.detector.Detector, "URL_RE", re.compile("hello"))
    expected = []
    for text in texts:
        try:
            expected.append(langdetect.detect(text))
        except langdetect.LangDetectException:
            expected.append(None)
    assert langdetect_rs.detect_batch(texts) == expected
    assert [
        None if r is None else probs_of(r) for r in langdetect_rs.detect_langs_batch(texts)
    ] == [
        None if e is None else probs_of(langdetect.detect_langs(t)) for t, e in zip(texts, expected)
    ]


def probs_of(result):
    return [(x.lang, x.prob) for x in result]


@pytest.mark.parametrize(
    "setup",
    [
        lambda d: setattr(d, "alpha", None),
        lambda d: setattr(d, "alpha", 1),
        lambda d: setattr(d, "prior_map", ["1"] * 55),
        lambda d: setattr(d, "prior_map", [None] * 55),
        lambda d: (setattr(d, "alpha", None), setattr(d, "n_trial", 0)),
        lambda d: (setattr(d, "prior_map", [None]), setattr(d, "n_trial", 0)),
        lambda d: setattr(d, "prior_map", 5),
    ],
)
@pytest.mark.parametrize("text", ["hello world", "1234"])
def test_unusual_option_types(setup, text):
    def scenario(m, profile):
        m.detector_factory.init_factory()
        d = m.detector_factory._factory.create()
        d.seed = 0
        setup(d)
        d.append(text)
        first = run_once(lambda: probs(d))
        return first, run_once(lambda: probs(d)), d.langprob

    same(scenario)


@pytest.mark.parametrize(
    "freq",
    [{"a": __import__("decimal").Decimal("1")}, {"a": "x"}, {"a": 2.5}, {"a": True}],
)
def test_profile_count_arithmetic(freq):
    def scenario(m, profile):
        f = m.DetectorFactory()
        f.set_seed(0)
        error = run_once(lambda: f.add_profile(profile("x", dict(freq), [10, 0, 0]), 0, 2))
        f.add_profile(profile("y", {"a": 1}, [10, 0, 0]), 1, 2)
        d = f.create()
        d.append("a")
        return error, run_once(lambda: probs(d))

    same(scenario)


def test_negative_langsize():
    def scenario(m, profile):
        f = m.DetectorFactory()
        f.set_seed(0)
        f.add_profile(profile("x", {"a": 9}, [10, 0, 0]), 0, 2)
        error = run_once(lambda: f.add_profile(profile("y", {"a": 1}, [10, 0, 0]), 1, -1))
        third = run_once(lambda: f.add_profile(profile("z", {"b": 1}, [10, 0, 0]), 0, -3))
        d = f.create()
        d.append("a b")
        return error, third, f.get_lang_list(), run_once(lambda: probs(d))

    same(scenario)
