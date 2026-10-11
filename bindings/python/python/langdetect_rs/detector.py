import operator
import re

from . import _core
from ._seed import seed_state
from .lang_detect_exception import ErrorCode, LangDetectException
from .language import Language
from .utils.ngram import NGram

URL_RE = re.compile(r"https?://[-_.?&~;+=/#0-9A-Za-z]{1,2076}")
MAIL_RE = re.compile(r"[-_.0-9A-Za-z]{1,64}@[-_0-9A-Za-z]{1,255}[-_.0-9A-Za-z]{1,255}")
_I64_MAX = 2**63 - 1


class Detector:
    """
    Detector class is to detect language from specified text.
    Its instance is able to be constructed via the factory class DetectorFactory.

    After appending a target text to the Detector instance with .append(string),
    the detector provides the language detection results for target text via .detect() or
    .get_probabilities().

    The detection itself runs in Rust and reproduces langdetect 1.0.9 bit for bit.
    """

    ALPHA_DEFAULT = 0.5
    ALPHA_WIDTH = 0.05

    ITERATION_LIMIT = 1000
    PROB_THRESHOLD = 0.1
    CONV_THRESHOLD = 0.99999
    BASE_FREQ = 10000
    UNKNOWN_LANG = "unknown"

    URL_RE = URL_RE
    MAIL_RE = MAIL_RE

    def __init__(self, factory):
        self._core = factory._core
        self.langlist = factory.langlist
        self.seed = factory.seed
        self.text = ""
        self.langprob = None

        self.alpha = self.ALPHA_DEFAULT
        self.n_trial = 7
        self.max_text_length = 10000
        self.prior_map = None
        self.verbose = False

    def set_verbose(self):
        """Accepted for compatibility; the trace output of langdetect is not produced."""
        self.verbose = True

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_prior_map(self, prior_map):
        """Set prior information about language probabilities."""
        self.prior_map = [0.0] * len(self.langlist)
        sump = 0.0
        for i in range(len(self.prior_map)):
            lang = self.langlist[i]
            if lang in prior_map:
                p = prior_map[lang]
                if p < 0:
                    raise LangDetectException(
                        ErrorCode.InitParamError, "Prior probability must be non-negative."
                    )
                self.prior_map[i] = p
                sump += p
        if sump <= 0.0:
            raise LangDetectException(
                ErrorCode.InitParamError, "More one of prior probability must be non-zero."
            )
        for i in range(len(self.prior_map)):
            self.prior_map[i] /= sump

    def set_max_text_length(self, max_text_length):
        """Specify max size of target text to use for language detection.
        The default value is 10000(10KB).
        """
        self.max_text_length = max_text_length

    def append(self, text):
        """Append the target text for language detection.
        If the total size of target text exceeds the limit size specified by
        Detector.set_max_text_length(int), the rest is cut down.
        """
        limit = self.max_text_length
        if type(limit) is int and self.URL_RE is URL_RE and self.MAIL_RE is MAIL_RE:
            if not isinstance(text, str):
                URL_RE.sub(" ", text)  # raises langdetect's TypeError
            self.text += _core.prepare(text, max(-_I64_MAX - 1, min(limit, _I64_MAX)))
            return
        text = self.URL_RE.sub(" ", text)
        text = self.MAIL_RE.sub(" ", text)
        text = NGram.normalize_vi(text)
        count = len(range(min(len(text), limit)))
        self.text += _core.prepare(text, count, False)

    def cleaning_text(self):
        """Eliminate Latin letters if the text is not written in the Latin alphabet."""
        cleaned = _core.clean(self.text)
        if cleaned is not None:
            self.text = cleaned

    def detect(self):
        """Detect language of the target text and return the language name
        which has the highest probability.
        """
        probabilities = self.get_probabilities()
        if probabilities:
            return probabilities[0].lang
        return self.UNKNOWN_LANG

    def get_probabilities(self):
        if self.langprob is None:
            self._detect_block()
        return self._sort_probability(self.langprob)

    def _detect_block(self):
        # The steps of langdetect's _detect_block, in its order: features, langprob, seed,
        # trial count, prior, alpha; options are only evaluated when a trial runs.
        self.cleaning_text()
        if self._core.detect_block(self.text, 0.0, 0, None)[1] == "features":
            raise _no_features()
        self.langprob = [0.0] * len(self.langlist)
        state = seed_state(self.seed)
        n_trial = min(operator.index(self.n_trial), _I64_MAX)
        if n_trial <= 0:
            return
        prior = None if self.prior_map is None else list(self.prior_map)
        alpha = self.alpha if type(self.alpha) is float else self.alpha + 0.0
        if prior is not None:
            prior = [p if type(p) is float else _times_float(p) for p in prior]
        langprob, error = self._core.detect_block(self.text, alpha, n_trial, prior, state)
        self.langprob = langprob
        if error is not None:
            raise error

    def _sort_probability(self, prob):
        result = [
            Language(lang, p) for (lang, p) in zip(self.langlist, prob) if p > self.PROB_THRESHOLD
        ]
        result.sort(reverse=True)
        return result


def _times_float(p):
    """A prior value as a float, raising what `prob[i] *= float` raises in langdetect."""
    p *= 1.0
    return float(p)


def _no_features():
    return LangDetectException(ErrorCode.CantDetectError, "No features in text.")
