"""Evaluation texts: WiLI-2018 test (Wikipedia paragraphs, 235 languages) and the
papluca/language-identification test split (short reviews and sentences, 20 languages)."""

import csv
import io
import os
import urllib.request

import pyarrow.parquet as pq

HF = "https://huggingface.co/datasets"
SOURCES = {
    "wili": f"{HF}/MartinThoma/wili_2018/resolve/main/"
    "WiLI-2018%20dataset/test-00000-of-00001.parquet",
    "papluca": f"{HF}/papluca/language-identification/resolve/main/test.csv",
}
CACHE = os.environ.get("LANGDETECT_BENCH_CACHE", os.path.expanduser("~/.cache/langdetect-rs-bench"))


def _download(name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, os.path.basename(SOURCES[name].split("?")[0]).replace("%20", "_"))
    if not os.path.exists(path):
        with urllib.request.urlopen(SOURCES[name]) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.replace(path + ".part", path)
    return path


def load(name):
    path = _download(name)
    if name == "wili":
        return pq.read_table(path, columns=["sentence"]).column("sentence").to_pylist()
    with open(path, encoding="utf-8") as f:
        return [row["text"] for row in csv.DictReader(io.StringIO(f.read()))]
