from __future__ import annotations

import copy
import math
import statistics
import time
from pathlib import Path
from typing import Any

from .config import validate_config
from .evidence_escalator import ConfirmationAction
from .rolling import build_rolling_splits
from .runner import ExperimentRunner


def _configured(config: dict[str, Any], seed: int) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["hyperparameters"]["seed"] = int(seed)
    validate_config(value)
    return value


def _paired_interval(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [float("-inf"), float("inf")]
    mean = statistics.mean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    # Student-t critical values for the predeclared small paired-seed sets.
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(
        len(values), 1.96
    )
    return [float(mean - critical * standard_error), float(mean + critical * standard_error)]


def run_rolling_confirmation(
    root: Path,
    data_dir: Path,
    starter_dir: Path,
    evaluator_sha256: str | None,
    action: ConfirmationAction,
    output_dir: Path,
) -> dict[str, Any]:
    loader = ExperimentRunner(root, data_dir, starter_dir, evaluator_sha256)
    development = loader.data.load(str(data_dir))
    rows = list(development["train"]) + list(development["valid"])
    folds = build_rolling_splits(rows)
    results: dict[str, Any] = {}
    started = time.monotonic()
    for fold_name, splits in folds.items():
        runner = ExperimentRunner(root, data_dir, starter_dir, evaluator_sha256)
        runner._splits = splits
        runner._encoded = runner.data.encode(splits)
        fold_dir = output_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        reference = runner.run(
            action.reference_config, fold_dir / "reference.npz"
        )
        candidate = runner.run(
            action.candidate_config, fold_dir / "candidate.npz"
        )
        results[fold_name] = {
            "reference": reference,
            "candidate": candidate,
            "delta": float(candidate["primary"]) - float(reference["primary"]),
            "rows": {name: len(values) for name, values in splits.items()},
        }
    deltas = [float(row["delta"]) for row in results.values()]
    return {
        "kind": "rolling",
        "test_labels_used": False,
        "fold_results": results,
        "mean_delta": float(statistics.mean(deltas)),
        "wins": sum(delta > 0 for delta in deltas),
        "folds": len(deltas),
        "training_runs": 2 * len(deltas),
        "runtime_seconds": time.monotonic() - started,
    }


def run_paired_seed_confirmation(
    root: Path,
    data_dir: Path,
    starter_dir: Path,
    evaluator_sha256: str | None,
    action: ConfirmationAction,
    output_dir: Path,
) -> dict[str, Any]:
    if not action.seeds:
        raise ValueError("paired-seed confirmation requires at least one seed")
    runner = ExperimentRunner(root, data_dir, starter_dir, evaluator_sha256)
    results: dict[str, Any] = {}
    deltas: list[float] = []
    started = time.monotonic()
    for seed in action.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        reference = runner.run(
            _configured(action.reference_config, seed), seed_dir / "reference.npz"
        )
        candidate = runner.run(
            _configured(action.candidate_config, seed), seed_dir / "candidate.npz"
        )
        delta = float(candidate["primary"]) - float(reference["primary"])
        deltas.append(delta)
        results[str(seed)] = {
            "reference": reference,
            "candidate": candidate,
            "delta": delta,
        }
    return {
        "kind": "paired_seeds",
        "test_labels_used": False,
        "seed_results": results,
        "paired_mean_delta": float(statistics.mean(deltas)),
        "paired_std": float(statistics.stdev(deltas)) if len(deltas) > 1 else 0.0,
        "approx_95_interval": _paired_interval(deltas),
        "wins": sum(delta > 0 for delta in deltas),
        "seeds": len(deltas),
        "training_runs": 2 * len(deltas),
        "runtime_seconds": time.monotonic() - started,
    }


def run_confirmation(
    root: Path,
    data_dir: Path,
    starter_dir: Path,
    evaluator_sha256: str | None,
    action: ConfirmationAction,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if action.kind == "rolling":
        return run_rolling_confirmation(
            root, data_dir, starter_dir, evaluator_sha256, action, output_dir
        )
    if action.kind == "paired_seeds":
        return run_paired_seed_confirmation(
            root, data_dir, starter_dir, evaluator_sha256, action, output_dir
        )
    raise ValueError(f"unknown confirmation action kind: {action.kind}")
