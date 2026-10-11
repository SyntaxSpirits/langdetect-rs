import json
import os
import random

import langdetect
import pytest
from langdetect.detector_factory import PROFILES_DIRECTORY

SEEDS = [0, 1, 42, 2**40 + 7, -5, 2**200 + 3, "seed", b"bytes", 3.5]

SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Ceci est une phrase en français, écrite pour tester la détection.",
    "Dies ist ein kurzer deutscher Satz über das Wetter von morgen.",
    "Esta es una frase en español sobre el tiempo de mañana.",
    "Questa è una frase italiana scritta per una prova.",
    "Isto é uma frase em português sobre o tempo de amanhã.",
    "Dit is een Nederlandse zin over het weer van morgen.",
    "Война и мир — роман Льва Толстого.",
    "Це речення українською мовою про погоду на завтра.",
    "Това е изречение на български език за времето утре.",
    "To jest zdanie po polsku o jutrzejszej pogodzie.",
    "Toto je věta v češtině o zítřejším počasí.",
    "Bu, yarınki hava durumu hakkında Türkçe bir cümledir.",
    "Αυτή είναι μια ελληνική πρόταση για τον καιρό.",
    "זהו משפט בעברית על מזג האוויר של מחר.",
    "هذه جملة باللغة العربية عن الطقس غدا.",
    "این یک جمله فارسی درباره هوای فردا است.",
    "यह कल के मौसम के बारे में हिंदी का एक वाक्य है।",
    "আগামীকালের আবহাওয়া সম্পর্কে এটি একটি বাংলা বাক্য।",
    "这是一个关于明天天气的中文句子。",
    "這是一個關於明天天氣的中文句子。",
    "これは明日の天気についての日本語の文です。",
    "이것은 내일 날씨에 관한 한국어 문장입니다.",
    "นี่คือประโยคภาษาไทยเกี่ยวกับสภาพอากาศพรุ่งนี้",
    "Đây là một câu tiếng Việt về thời tiết ngày mai.",
    "Đây là một câu tiếng Việt về thời tiết ngày mai.",
    "Tämä on suomenkielinen lause huomisen säästä.",
    "Detta är en svensk mening om vädret i morgon.",
    "Dette er en dansk sætning om vejret i morgen.",
    "Ez egy magyar mondat a holnapi időjárásról.",
    "Aceasta este o propoziție în limba română despre vreme.",
    "Hii ni sentensi ya Kiswahili kuhusu hali ya hewa kesho.",
    "HELLO WORLD THIS IS ALL CAPS",
    "mixed English и русский text 混合 テキスト",
    "contact me at someone.name@example.org or https://example.com/path?q=1",
    "   leading   and   trailing   spaces   ",
    "",
    "12345 !!! ???",
]


def _profile_texts(rng):
    texts = []
    for name in sorted(os.listdir(PROFILES_DIRECTORY)):
        with open(os.path.join(PROFILES_DIRECTORY, name), encoding="utf-8") as f:
            freq = json.load(f)["freq"]
        grams = sorted((g for g in freq if len(g) == 3 and " " not in g), key=freq.get)[-300:]
        for _ in range(8):
            words = [
                "".join(rng.choice(grams) for _ in range(rng.randint(1, 3)))
                for _ in range(rng.randint(1, 25))
            ]
            texts.append(" ".join(words))
    return texts


def _random_texts(rng):
    pools = [
        (0x20, 0x7E),
        (0xA0, 0x24F),
        (0x300, 0x36F),
        (0x370, 0x52F),
        (0x590, 0x6FF),
        (0x900, 0xDFF),
        (0xE00, 0xEFF),
        (0x1E00, 0x1EFF),
        (0x2000, 0x206F),
        (0x3040, 0x30FF),
        (0x3100, 0x31BF),
        (0x4E00, 0x9FFF),
        (0xAC00, 0xD7A3),
        (0xF900, 0xFFFD),
        (0x10000, 0x1FAFF),
        (0xD800, 0xDFFF),
    ]
    texts = []
    for _ in range(400):
        parts = []
        for _ in range(rng.randint(1, 4)):
            lo, hi = rng.choice(pools)
            parts.append("".join(chr(rng.randint(lo, hi)) for _ in range(rng.randint(1, 30))))
        s = " ".join(parts)
        if rng.random() < 0.2:
            s = s.upper()
        texts.append(s)
    return texts


@pytest.fixture(scope="session")
def corpus():
    rng = random.Random(1234)
    return SENTENCES + _profile_texts(rng) + _random_texts(rng) + ["Le chat dort. " * 900]


@pytest.fixture
def seeded():
    """Yield a function that sets DetectorFactory.seed on both libraries."""
    import langdetect_rs

    old = langdetect.DetectorFactory.seed, langdetect_rs.DetectorFactory.seed

    def set_seed(seed):
        langdetect.DetectorFactory.seed = seed
        langdetect_rs.DetectorFactory.seed = seed

    yield set_seed
    langdetect.DetectorFactory.seed, langdetect_rs.DetectorFactory.seed = old
