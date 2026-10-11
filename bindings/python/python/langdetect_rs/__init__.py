"""Fast drop-in replacement for langdetect with bit-identical results."""

from ._core import __version__
from .detector import Detector
from .detector_factory import (
    PROFILES_DIRECTORY,
    DetectorFactory,
    detect,
    detect_batch,
    detect_langs,
    detect_langs_batch,
)
from .lang_detect_exception import ErrorCode, LangDetectException
from .language import Language

__all__ = [
    "PROFILES_DIRECTORY",
    "Detector",
    "DetectorFactory",
    "ErrorCode",
    "LangDetectException",
    "Language",
    "__version__",
    "detect",
    "detect_batch",
    "detect_langs",
    "detect_langs_batch",
    "install",
]

_SUBMODULES = ("detector", "detector_factory", "lang_detect_exception", "language", "utils")


def install():
    """Make `import langdetect` resolve to langdetect_rs.

    Call it before anything imports langdetect; modules that already imported langdetect keep
    the original. Covers langdetect, its submodules detector, detector_factory,
    lang_detect_exception and language, and utils.lang_profile and utils.ngram.
    """
    import sys

    from .utils import lang_profile, ngram  # noqa: F401

    package = sys.modules[__name__]
    sys.modules["langdetect"] = package
    for name in _SUBMODULES:
        sys.modules["langdetect." + name] = sys.modules[__name__ + "." + name]
    sys.modules["langdetect.utils.lang_profile"] = sys.modules[__name__ + ".utils.lang_profile"]
    sys.modules["langdetect.utils.ngram"] = sys.modules[__name__ + ".utils.ngram"]
