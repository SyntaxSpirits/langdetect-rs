import re

from .. import _core

_VI_FROM, _VI_MARKS, _VI_TO = _core.vietnamese_tables()


class NGram:
    N_GRAM = 3

    def __init__(self):
        self.grams = " "
        self.capitalword = False

    def add_char(self, ch):
        """Append a character into ngram buffer."""
        ch = self.normalize(ch)
        last_char = self.grams[-1]
        if last_char == " ":
            self.grams = " "
            self.capitalword = False
            if ch == " ":
                return
        elif len(self.grams) >= self.N_GRAM:
            self.grams = self.grams[1:]
        self.grams += ch

        if ch.isupper():
            if last_char.isupper():
                self.capitalword = True
        else:
            self.capitalword = False

    def get(self, n):
        """Get n-gram."""
        if self.capitalword:
            return
        if n < 1 or n > self.N_GRAM or len(self.grams) < n:
            return
        if n == 1:
            ch = self.grams[-1]
            if ch == " ":
                return
            return ch
        else:
            return self.grams[-n:]

    @classmethod
    def normalize(cls, ch):
        if 0xD800 <= ord(ch) <= 0xDFFF:
            return ch
        return _core.normalize_char(ch)

    @classmethod
    def normalize_vi(cls, text):
        """Normalizer for Vietnamese.
        Normalize Alphabet + Diacritical Mark(U+03xx) into U+1Exx.
        """

        def repl(m):
            alphabet = cls.TO_NORMALIZE_VI_CHARS.find(m.group(1))
            dmark = cls.DMARK_CLASS.find(m.group(2))  # Diacritical Mark
            return cls.NORMALIZED_VI_CHARS[dmark][alphabet]

        return cls.ALPHABET_WITH_DMARK.sub(repl, text)

    NORMALIZED_VI_CHARS = _VI_TO
    TO_NORMALIZE_VI_CHARS = _VI_FROM
    DMARK_CLASS = _VI_MARKS
    ALPHABET_WITH_DMARK = re.compile("([" + _VI_FROM + "])([" + _VI_MARKS + "])", re.UNICODE)
