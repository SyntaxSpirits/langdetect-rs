"""Time langdetect 1.0.9 and langdetect_rs on samples of WiLI-2018 and papluca.

Reports milliseconds per text for langdetect.detect, langdetect_rs.detect in a loop and
langdetect_rs.detect_batch on all cores, without a seed (the default) and with seed 0, plus
the time to import each package and detect the first text in a fresh process.
"""

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time

import langdetect
from data import load

import langdetect_rs


def per_text(fn, texts, repeats):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(texts)
        best = min(best, time.perf_counter() - start)
    return best / len(texts) * 1000


def loop(detect):
    def run(texts):
        for text in texts:
            try:  # noqa: SIM105 - contextlib.suppress would add its own overhead to the timing
                detect(text)
            except Exception:
                pass

    return run


def cold_start(module):
    code = f"import {module}; {module}.detect('Guten Morgen, wie geht es dir?')"
    best = float("inf")
    for _ in range(5):
        start = time.perf_counter()
        subprocess.run([sys.executable, "-c", code], check=True)
        best = min(best, time.perf_counter() - start)
    return best * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="benchmark-results")
    args = parser.parse_args()

    rng = random.Random(1729)
    datasets = {name: rng.sample(load(name), args.texts) for name in ("papluca", "wili")}
    results = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "langdetect_rs": langdetect_rs.__version__,
        "texts": args.texts,
        "cold_start_ms": {
            "langdetect": round(cold_start("langdetect"), 1),
            "langdetect_rs": round(cold_start("langdetect_rs"), 1),
        },
        "rows": [],
    }
    for seed in (None, 0):
        langdetect.DetectorFactory.seed = langdetect_rs.DetectorFactory.seed = seed
        for name, texts in datasets.items():
            py = per_text(loop(langdetect.detect), texts, max(1, args.repeats - 1))
            rs = per_text(loop(langdetect_rs.detect), texts, args.repeats)
            batch = per_text(langdetect_rs.detect_batch, texts, args.repeats)
            results["rows"].append(
                {
                    "dataset": name,
                    "seed": seed,
                    "avg_chars": round(sum(map(len, texts)) / len(texts)),
                    "langdetect_ms": round(py, 4),
                    "loop_ms": round(rs, 4),
                    "batch_ms": round(batch, 4),
                    "loop_speedup": round(py / rs, 1),
                    "batch_speedup": round(py / batch, 1),
                }
            )
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
