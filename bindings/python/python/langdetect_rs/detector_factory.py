import json
import os
import sys
import unicodedata
from importlib.machinery import PathFinder
from os import path

from . import _core
from ._seed import seed_state
from .detector import MAIL_RE, URL_RE, Detector
from .lang_detect_exception import ErrorCode, LangDetectException
from .language import Language
from .utils.lang_profile import LangProfile

_NEUMAIER_SUM = sys.version_info >= (3, 12)


def _new_core():
    return _core.Factory(_NEUMAIER_SUM, unicodedata.unidata_version)


class DetectorFactory:
    """
    Language Detector Factory Class.

    This class manages an initialization and constructions of Detector.

    Before using language detection library,
    load profiles with DetectorFactory.load_profile(str)
    and set initialization parameters.

    When the language detection,
    construct Detector instance via DetectorFactory.create().
    See also Detector's sample code.
    """

    seed = None

    def __init__(self):
        self._core = _new_core()
        self.langlist = []

    def load_profile(self, profile_directory):
        if profile_directory == PROFILES_DIRECTORY and not path.isdir(profile_directory):
            self._load_builtin()
            return
        list_files = os.listdir(profile_directory)
        if not list_files:
            raise LangDetectException(
                ErrorCode.NeedLoadProfileError, "Not found profile: " + profile_directory
            )

        langsize, index = len(list_files), 0
        for filename in list_files:
            if filename.startswith("."):
                continue
            filename = path.join(profile_directory, filename)
            if not path.isfile(filename):
                continue

            f = None
            try:
                f = open(filename, encoding="utf-8")  # noqa: SIM115 - mirrors langdetect
                json_data = json.load(f)
                profile = LangProfile(**json_data)
                self.add_profile(profile, index, langsize)
                index += 1
            except OSError:
                raise LangDetectException(
                    ErrorCode.FileLoadError, f'Cannot open "{filename}"'
                ) from None
            except Exception:
                raise LangDetectException(
                    ErrorCode.FormatError, f'Profile format error in "{filename}"'
                ) from None
            finally:
                if f:
                    f.close()

    def _load_builtin(self):
        files = _builtin_order()
        try:
            self._core.load_builtin(files)
        except ValueError as e:
            filename = path.join(PROFILES_DIRECTORY, files[e.args[0]])
            raise LangDetectException(
                ErrorCode.FormatError, f'Profile format error in "{filename}"'
            ) from None
        finally:
            self.langlist[:] = self._core.langs()

    def load_json_profile(self, json_profiles):
        langsize, index = len(json_profiles), 0
        if langsize < 2:
            raise LangDetectException(ErrorCode.NeedLoadProfileError, "Need more than 2 profiles.")

        for json_profile in json_profiles:
            try:
                json_data = json.loads(json_profile)
                profile = LangProfile(**json_data)
                self.add_profile(profile, index, langsize)
                index += 1
            except Exception:
                raise LangDetectException(ErrorCode.FormatError, "Profile format error.") from None

    def add_profile(self, profile, index, langsize):
        lang = profile.name
        if lang in self.langlist:
            raise LangDetectException(
                ErrorCode.DuplicateLangError, "Duplicate the same language profile."
            )
        try:
            self._core.add_profile(lang, profile.freq, profile.n_words, index, langsize)
        finally:
            self.langlist[:] = self._core.langs()

    def clear(self):
        self.langlist = []
        self._core = _new_core()

    def create(self, alpha=None):
        """Construct Detector instance with smoothing parameter."""
        detector = self._create_detector()
        if alpha is not None:
            detector.set_alpha(alpha)
        return detector

    def _create_detector(self):
        if not self.langlist:
            raise LangDetectException(ErrorCode.NeedLoadProfileError, "Need to load profiles.")
        return Detector(self)

    def set_seed(self, seed):
        self.seed = seed

    def get_lang_list(self):
        return list(self.langlist)


def _builtin_order():
    """The built-in profiles in the order langdetect loads them on this machine.

    langdetect walks its profile directory in os.listdir order, which only affects the last
    bits of the probabilities. If langdetect is installed, its order is used, else name order.
    """
    files = _core.builtin_files()
    spec = PathFinder.find_spec("langdetect")
    if spec is not None and spec.submodule_search_locations:
        directory = path.join(list(spec.submodule_search_locations)[0], "profiles")
        try:
            listed = os.listdir(directory)
        except OSError:
            listed = []
        if sorted(listed) == files:
            return listed
    return files


PROFILES_DIRECTORY = path.join(path.dirname(__file__), "profiles")
_factory = None


def init_factory():
    global _factory
    if _factory is None:
        factory = DetectorFactory()
        factory.load_profile(PROFILES_DIRECTORY)
        _factory = factory


def _fast_state():
    """The seed state for detection without a Detector object, or False when the Detector path
    is needed (patched patterns, or a seed that raises, whose error langdetect reports later)."""
    if Detector.URL_RE is not URL_RE or Detector.MAIL_RE is not MAIL_RE:
        return False
    try:
        return seed_state(_factory.seed)
    except Exception:
        return False


def _no_features():
    return LangDetectException(ErrorCode.CantDetectError, "No features in text.")


def detect(text):
    init_factory()
    state = _fast_state()
    if state is False:
        detector = _factory.create()
        detector.append(text)
        return detector.detect()
    if not isinstance(text, str):
        URL_RE.sub(" ", text)  # raises langdetect's TypeError
    lang = _factory._core.detect(text, state)
    if lang is None:
        raise _no_features()
    return lang


def detect_langs(text):
    init_factory()
    state = _fast_state()
    if state is False:
        detector = _factory.create()
        detector.append(text)
        return detector.get_probabilities()
    if not isinstance(text, str):
        URL_RE.sub(" ", text)
    result = _factory._core.detect_langs(text, state)
    if result is None:
        raise _no_features()
    return [Language(lang, prob) for lang, prob in result]


def _detect_each(texts, langs):
    out = []
    for text in texts:
        detector = _factory.create()
        detector.append(text)
        try:
            out.append(detector.get_probabilities() if langs else detector.detect())
        except LangDetectException as e:
            if e.code != ErrorCode.CantDetectError:
                raise
            out.append(None)
    return out


def detect_batch(texts):
    """`detect` for many texts in parallel, without the GIL. Texts without features give None
    instead of raising."""
    init_factory()
    texts = list(texts)
    state = _fast_state()
    if state is False:
        return _detect_each(texts, langs=False)
    return _factory._core.detect_batch(texts, state)


def detect_langs_batch(texts):
    """`detect_langs` for many texts in parallel, without the GIL. Texts without features give
    None instead of raising."""
    init_factory()
    texts = list(texts)
    state = _fast_state()
    if state is False:
        return _detect_each(texts, langs=True)
    results = _factory._core.detect_langs_batch(texts, state)
    return [None if r is None else [Language(lang, prob) for lang, prob in r] for r in results]
