import subprocess
import sys

import langdetect
import pytest

import langdetect_rs
from langdetect_rs import DetectorFactory, ErrorCode, LangDetectException, Language
from langdetect_rs.utils.lang_profile import LangProfile

JSON_LANG1 = (
    '{"freq":{"A":3,"B":6,"C":3,"AB":2,"BC":1,"ABC":2,"BBC":1,"CBA":1},'
    '"n_words":[12,3,4],"name":"lang1"}'
)
JSON_LANG2 = (
    '{"freq":{"A":6,"B":3,"C":3,"AA":3,"AB":2,"ABC":1,"ABA":1,"CAA":1},'
    '"n_words":[12,5,3],"name":"lang2"}'
)


@pytest.fixture
def small_factory():
    factory = DetectorFactory()
    for i, (name, training) in enumerate(
        [("en", "a a a b b c c d e"), ("fr", "a b b c c c d d d"), ("ja", "あ あ あ い う え え")]
    ):
        profile = LangProfile(name)
        for w in training.split():
            profile.add(w)
        factory.add_profile(profile, i, 3)
    return factory


# The tests below are those of langdetect 1.0.9, run against langdetect_rs.


@pytest.mark.parametrize(
    ("text", "lang"), [("a", "en"), ("b d", "fr"), ("d e", "en"), ("ああああa", "ja")]
)
def test_small_profiles(small_factory, text, lang):
    detect = small_factory.create()
    detect.append(text)
    assert detect.detect() == lang


def test_lang_list(small_factory):
    assert small_factory.get_lang_list() == ["en", "fr", "ja"]


def test_factory_from_json_string(small_factory):
    small_factory.clear()
    small_factory.load_json_profile([JSON_LANG1, JSON_LANG2])
    assert small_factory.get_lang_list() == ["lang1", "lang2"]


def test_language():
    lang = Language(None, 0)
    assert lang.lang is None
    assert lang.prob == 0.0
    assert str(lang) == ""
    lang2 = Language("en", 1.0)
    assert str(lang2) == "en:1.0"
    assert Language("a", 0.1) < Language("b", 0.5)
    assert Language("a", 0.1) != Language("b", 0.5)


# Behaviour beyond the original test suite.


def test_json_profiles_match_original():
    seeds = [0, 5]
    for seed in seeds:
        results = []
        for module in (langdetect, langdetect_rs):
            factory = module.DetectorFactory()
            factory.load_json_profile([JSON_LANG1, JSON_LANG2])
            factory.set_seed(seed)
            d = factory.create()
            d.append("ABC ABA CBA")
            results.append([(x.lang, x.prob) for x in d.get_probabilities()])
        assert results[0] == results[1]


def test_load_profile_from_directory_matches_builtin():
    factory = DetectorFactory()
    factory.load_profile(langdetect.PROFILES_DIRECTORY)
    langdetect_rs.detector_factory.init_factory()
    assert factory.langlist == langdetect_rs.detector_factory._factory.langlist
    factory.set_seed(1)
    d = factory.create()
    d.append("Guten Morgen, wie geht es dir heute?")
    assert d.detect() == "de"


@pytest.mark.parametrize(
    ("call", "code", "message"),
    [
        (lambda f: f.create(), ErrorCode.NeedLoadProfileError, "Need to load profiles."),
        (
            lambda f: f.load_json_profile([JSON_LANG1]),
            ErrorCode.NeedLoadProfileError,
            "Need more than 2 profiles.",
        ),
        (
            lambda f: f.load_json_profile([JSON_LANG1, "{"]),
            ErrorCode.FormatError,
            "Profile format error.",
        ),
        (
            lambda f: f.load_json_profile([JSON_LANG1, JSON_LANG1]),
            ErrorCode.FormatError,
            "Profile format error.",
        ),
        (
            lambda f: f.load_json_profile(['{"name":"x","bad":1}', JSON_LANG1]),
            ErrorCode.FormatError,
            "Profile format error.",
        ),
        (
            lambda f: f.load_json_profile(
                ['{"name":"x","freq":{"ab":1},"n_words":[1,0,1]}', JSON_LANG1]
            ),
            ErrorCode.FormatError,
            "Profile format error.",
        ),
    ],
)
def test_factory_errors_match_original(call, code, message):
    for module in (langdetect, langdetect_rs):
        with pytest.raises(module.LangDetectException) as e:
            call(module.DetectorFactory())
        assert e.value.code == code
        assert str(e.value) == message


def test_builtin_twice_is_format_error():
    factory = DetectorFactory()
    factory.load_profile(langdetect_rs.PROFILES_DIRECTORY)
    with pytest.raises(LangDetectException) as e:
        factory.load_profile(langdetect_rs.PROFILES_DIRECTORY)
    assert e.value.code == ErrorCode.FormatError
    assert len(factory.langlist) == 55


@pytest.mark.parametrize("prior", [{"en": -1, "de": 1}, {"xx": 1}, {}])
def test_rejected_prior_map_matches_original(prior, seeded):
    seeded(0)
    langdetect.detector_factory.init_factory()
    langdetect_rs.detector_factory.init_factory()
    states = []
    for module in (langdetect, langdetect_rs):
        d = module.detector_factory._factory.create()
        with pytest.raises(module.LangDetectException) as e:
            d.set_prior_map(prior)
        d.append("hello world")
        try:
            result = [(x.lang, x.prob) for x in d.get_probabilities()]
        except ZeroDivisionError as z:
            result = ("ZeroDivisionError", str(z))
        states.append((e.value.code, str(e.value), d.prior_map, result))
    assert states[0] == states[1]


def test_no_features():
    with pytest.raises(LangDetectException) as e:
        langdetect_rs.detect("1234 !!!")
    assert e.value.code == ErrorCode.CantDetectError
    assert e.value.get_code() == ErrorCode.CantDetectError


def test_type_errors():
    for module in (langdetect, langdetect_rs):
        with pytest.raises(TypeError):
            module.detect(b"bytes")
        with pytest.raises(TypeError):
            module.detect(None)


def test_unseeded_detection_is_sane(seeded):
    seeded(None)
    text = "Ceci est une longue phrase en français pour être sûr du résultat. " * 3
    assert {langdetect_rs.detect(text) for _ in range(20)} == {"fr"}
    assert langdetect_rs.detect_batch([text] * 50) == ["fr"] * 50


def test_seed_types(seeded):
    for seed in [0, -3, 2**100, "abc", b"abc", bytearray(b"abc"), 1.5, True]:
        seeded(seed)
        assert [(x.lang, x.prob) for x in langdetect_rs.detect_langs("Hej, hur mår du idag?")] == [
            (x.lang, x.prob) for x in langdetect.detect_langs("Hej, hur mår du idag?")
        ]


def test_install_aliases_langdetect():
    code = (
        "import langdetect_rs; langdetect_rs.install()\n"
        "import langdetect\n"
        "from langdetect import DetectorFactory, detect_langs, lang_detect_exception\n"
        "from langdetect.detector_factory import detect\n"
        "from langdetect.utils.lang_profile import LangProfile\n"
        "assert langdetect is langdetect_rs\n"
        "assert lang_detect_exception.LangDetectException is langdetect_rs.LangDetectException\n"
        "DetectorFactory.seed = 0\n"
        "print(detect('Dies ist ein deutscher Satz.'))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "de"
