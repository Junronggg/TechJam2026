"""Validation-only adapter for the untouched organizer NumPy FM implementation."""

from __future__ import annotations

import csv
import gc
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np

from experiment.evaluator import OfficialEvaluator
from experiment.schemas import ExperimentResult, ExperimentStatus, MetricBundle, ModelConfig


TRAIN_DATES = (20220408, 20220421)
VALID_DATES = (20220422, 20220428)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_starter_modules(starter_dir: Path) -> tuple[ModuleType, ModuleType]:
    """Load official source files without modifying or copying them."""
    data_module = _load_module("data", starter_dir / "data.py")
    _load_module("evaluate", starter_dir / "evaluate.py")
    baseline_module = _load_module("baseline", starter_dir / "baseline.py")
    return data_module, baseline_module


def load_train_valid(data_dir: Path) -> dict[str, list[tuple[object, ...]]]:
    """Load only train/validation labels; rows after 2022-04-28 are discarded first."""
    video_to_author: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            video_to_author[row["video_id"]] = row["author_id"]

    splits: dict[str, list[tuple[object, ...]]] = {"train": [], "valid": []}
    files = (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    )
    for filename in files:
        with (data_dir / filename).open(newline="") as handle:
            for row in csv.DictReader(handle):
                date = int(row["date"])
                if TRAIN_DATES[0] <= date <= TRAIN_DATES[1]:
                    split = "train"
                elif VALID_DATES[0] <= date <= VALID_DATES[1]:
                    split = "valid"
                else:
                    # Crucially, do not read the relevance label for test-period rows.
                    continue
                splits[split].append(
                    (
                        date,
                        row["user_id"],
                        row["video_id"],
                        video_to_author.get(row["video_id"], "UNK"),
                        row["tab"],
                        float(row["duration_ms"]),
                        1 if row["long_view"] != "0" else 0,
                    )
                )
    return splits


def run_validation_fm(
    experiment_id: str,
    config: ModelConfig,
    starter_dir: Path,
    data_dir: Path,
    run_dir: Path,
    evaluator_sha256: str,
    verbose: bool = True,
) -> ExperimentResult:
    if config.model != "fm":
        raise ValueError(f"Official FM backend cannot run model={config.model!r}")

    started = time.monotonic()
    official_data, official_baseline = load_starter_modules(starter_dir)
    evaluator = OfficialEvaluator(starter_dir / "evaluate.py", evaluator_sha256)
    evaluator.verify_integrity()
    splits = load_train_valid(data_dir)
    if verbose:
        print(f"loaded validation-only splits: { {name: len(rows) for name, rows in splits.items()} }")
    encoded, dimension = official_data.encode(splits)
    del splits
    gc.collect()

    official_fields = tuple(official_data.FIELDS)
    unknown = set(config.features).difference(official_fields)
    if unknown:
        raise ValueError(f"Official FM adapter does not implement features: {sorted(unknown)}")
    if not config.features:
        raise ValueError("At least one feature is required")
    column_indices = np.asarray([official_fields.index(name) for name in config.features])

    x_train_full, y_train, _ = encoded["train"]
    x_valid_full, y_valid, valid_users = encoded["valid"]
    x_valid = x_valid_full[:, column_indices]
    params = dict(config.hyperparameters)
    k = int(params.get("k", 16))
    learning_rate = float(params.get("lr", 0.001))
    epochs = int(params.get("epochs", 40))
    l2 = float(params.get("l2", 1e-6))
    batch_size = int(params.get("batch_size", 8192))
    patience = int(params.get("patience", 4))

    model = official_baseline.FM(
        dimension, k=k, lr=learning_rate, l2=l2, seed=config.seed
    )
    random = np.random.default_rng(config.seed)
    best_primary = float("-inf")
    best_state: tuple[np.ndarray, np.ndarray, np.float32] | None = None
    best_metrics: MetricBundle | None = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        epoch_started = time.monotonic()
        order = random.permutation(len(y_train))
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            batch_rows = order[start : start + batch_size]
            x_batch = x_train_full[batch_rows][:, column_indices]
            losses.append(model.step(x_batch, y_train[batch_rows]))
        scores = model.predict(x_valid)
        metrics = evaluator.evaluate(valid_users, y_valid, scores)
        if verbose:
            print(
                f"[{experiment_id}] epoch {epoch:2d} | loss {np.mean(losses):.4f} | "
                f"valid GAUC {metrics.gauc:.4f} nDCG@5 {metrics.ndcg_at_5:.4f} "
                f"primary {metrics.primary:.4f} | {time.monotonic() - epoch_started:.1f}s",
                flush=True,
            )
        if metrics.primary > best_primary + 1e-5:
            best_primary = metrics.primary
            best_metrics = metrics
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"[{experiment_id}] early stop at epoch {epoch}", flush=True)
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("FM training completed without a valid checkpoint")
    model.V, model.W, model.b = best_state
    final_scores = model.predict(x_valid)
    if not np.isfinite(final_scores).all():
        raise ValueError("Validation predictions contain NaN or Inf")

    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = run_dir / "validation_predictions.npy"
    checkpoint_path = run_dir / "model.npz"
    np.save(prediction_path, final_scores)
    np.savez_compressed(
        checkpoint_path,
        V=model.V,
        W=model.W,
        b=model.b,
        features=np.asarray(config.features),
    )
    return ExperimentResult(
        experiment_id=experiment_id,
        status=ExperimentStatus.SUCCESS,
        metrics=best_metrics,
        runtime_seconds=time.monotonic() - started,
        checkpoint=str(checkpoint_path),
        prediction_path=str(prediction_path),
    )

