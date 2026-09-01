"""Read-only checks for the TechJam workspace and KuaiRand starter kit."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import py_compile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "project.json"
REQUIRED_STARTER_FILES = (
    "baseline.py",
    "baseline_scores.json",
    "data.py",
    "evaluate.py",
    "submit.py",
)
REQUIRED_DATA_FILES = (
    "video_features_basic_pure.csv",
    "user_features_pure.csv",
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)


def result(ok: bool, message: str) -> None:
    marker = "OK" if ok else "MISSING"
    print(f"[{marker}] {message}")


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    starter_dir = ROOT / config["starter_dir"]
    configured_data = os.getenv("TECHJAM_DATA_DIR", config["data_dir"])
    data_dir = Path(configured_data)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    print(f"Python: {sys.version.split()[0]}")
    print(f"Interpreter: {Path(sys.executable)}")
    python_ok = sys.version_info >= (3, 9)
    result(python_ok, "Python 3.9+")

    dependencies_ok = True
    for package in ("numpy", "lightgbm", "sklearn", "torch"):
        installed = importlib.util.find_spec(package) is not None
        result(installed, f"{package} installed")
        dependencies_ok &= installed

    starter_ok = True
    for name in REQUIRED_STARTER_FILES:
        path = starter_dir / name
        exists = path.is_file()
        result(exists, f"starter file: {path.relative_to(ROOT)}")
        starter_ok &= exists
        if exists and path.suffix == ".py":
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                result(False, f"compiles: {name} ({exc.msg})")
                starter_ok = False

    evaluator_path = starter_dir / "evaluate.py"
    if evaluator_path.is_file():
        actual_hash = hashlib.sha256(evaluator_path.read_bytes()).hexdigest()
        expected_hash = config["organizer_integrity"]["evaluator_sha256"]
        evaluator_ok = actual_hash == expected_hash
        result(evaluator_ok, "official evaluator SHA-256 unchanged")
        starter_ok &= evaluator_ok

    data_ok = True
    for name in REQUIRED_DATA_FILES:
        path = data_dir / name
        exists = path.is_file()
        result(exists, f"dataset file: {path}")
        data_ok &= exists

    if python_ok and dependencies_ok and starter_ok and data_ok:
        print("\nReady to reproduce the baseline and run the research agent.")
        return 0

    if not data_ok:
        print("\nEnvironment and starter-kit checks can pass before the dataset is downloaded.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
