"""Behaviour that depends on the environment: langdetect missing, an unknown Unicode version,
free-threaded Python."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import textwrap
import unicodedata
from pathlib import Path

import pytest

from langdetect_rs import _core

TESTS = Path(__file__).parent
GENERATOR = TESTS.parents[2] / "scripts" / "gen_standalone.py"


def load_check():
    spec = importlib.util.spec_from_file_location("check_standalone", TESTS / "check_standalone.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_python(code, *args):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_without_langdetect():
    out = run_python(
        """
        import importlib.machinery, runpy, sys

        find_spec = importlib.machinery.PathFinder.find_spec

        def hide_langdetect(name, path=None, target=None):
            return None if name == "langdetect" else find_spec(name, path, target)

        importlib.machinery.PathFinder.find_spec = hide_langdetect
        sys.argv = sys.argv[1:]
        runpy.run_path(sys.argv[0], run_name="__main__")
        """,
        str(TESTS / "check_standalone.py"),
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "300 texts match" in out.stdout


@pytest.mark.skipif(not GENERATOR.exists(), reason="scripts not available")
def test_stored_results_are_langdetects(tmp_path):
    copy = tmp_path / "standalone.json"
    shutil.copy(TESTS / "standalone.json", copy)
    out = subprocess.run(
        [sys.executable, str(GENERATOR), str(copy)], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    stored = json.loads((TESTS / "standalone.json").read_text(encoding="utf-8"))
    fresh = json.loads(copy.read_text(encoding="utf-8"))
    assert fresh.keys() == stored.keys()
    assert (fresh["seed"], fresh["texts"]) == (stored["seed"], stored["texts"])
    same_result = load_check().same_result
    for key in ("sequential", "neumaier"):
        pairs = zip(stored[key], fresh[key])
        differ = [i for i, (want, got) in enumerate(pairs) if not same_result(want, got)]
        assert len(fresh[key]) == len(stored[key]) and not differ, (key, differ)


def test_unicode_tables():
    assert _core.unicode_tables(unicodedata.unidata_version) == unicodedata.unidata_version
    assert _core.unicode_tables("99.0.0") == "17.0.0"


def test_unknown_unicode_version_warns_once():
    out = run_python(
        """
        import unicodedata, warnings

        unicodedata.unidata_version = "99.0.0"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import langdetect_rs

            langdetect_rs.DetectorFactory.seed = 0
            print(langdetect_rs.detect("Ein, zwei, drei, vier"))
            print(langdetect_rs.detect_batch(["Bonjour tout le monde"]))
            langdetect_rs.DetectorFactory().load_profile(langdetect_rs.PROFILES_DIRECTORY)
        for w in caught:
            if "Unicode" in str(w.message):
                print(w.category.__name__, w.message)
        """
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [
        "de",
        "['fr']",
        "RuntimeWarning langdetect_rs has no str.isupper() tables for Unicode 99.0.0 (unicodedata"
        " of this Python) and uses those of Unicode 17.0.0; results can differ from langdetect's.",
    ]


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="needs a free-threaded build"
)
def test_free_threaded_import_enables_the_gil():
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}
    out = subprocess.run(
        [sys.executable, "-c", "import sys, langdetect_rs; print(sys._is_gil_enabled())"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout == "True\n"
