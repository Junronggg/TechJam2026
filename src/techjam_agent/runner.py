from __future__ import annotations

import csv
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import validate_config
from .bpr import bpr_step, build_pair_indices
from .history_features import aggregate, aggregate_pair, smoothed_rate_bucket


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ExperimentRunner:
    def __init__(self, root: Path, data_dir: Path, starter_dir: Path) -> None:
        self.root = root
        self.data_dir = data_dir
        self.starter_dir = starter_dir
        sys.path.insert(0, str(starter_dir))
        self.data = _load_module("techjam_starter_data", starter_dir / "data.py")
        self.evaluate_mod = _load_module("techjam_starter_evaluate", starter_dir / "evaluate.py")
        self.baseline = _load_module("techjam_starter_baseline", starter_dir / "baseline.py")
        self.submit = _load_module("techjam_starter_submit", starter_dir / "submit.py")
        self._splits = None
        self._encoded = None

    def _encoded_for(self, config: dict[str, Any]):
        base, base_dim = self._encoded
        enabled = [key for key in ("user_long_view_rate", "item_long_view_rate")
                   if config["features"][key]]
        if not enabled:
            return base, base_dim
        key_indices = {"user_long_view_rate": 1, "item_long_view_rate": 2}
        columns = {}
        next_offset = base_dim
        for feature in enabled:
            key_index = key_indices[feature]
            stats, global_rate = aggregate(self._splits["train"], key_index)
            for split, rows in self._splits.items():
                values = np.empty(len(rows), dtype=np.int32)
                for index, row in enumerate(rows):
                    leave_out = row[6] if split == "train" else None
                    bucket = smoothed_rate_bucket(row[key_index], stats, global_rate,
                                                  label_to_leave_out=leave_out)
                    values[index] = next_offset + bucket
                columns.setdefault(split, []).append(values)
            next_offset += 20
        encoded = {}
        for split, (X, y, users) in base.items():
            encoded[split] = (np.column_stack([X, *columns[split]]).astype(np.int32), y, users)
        return encoded, next_offset

    def _lightgbm_matrices(self, config: dict[str, Any]):
        base, _ = self._encoded
        # Starter-kit FM IDs use global field offsets. LightGBM categorical columns
        # need independent zero-based IDs or they are treated as extremely sparse.
        field_mins = base["train"][0].min(axis=0)
        categorical = {split: (values[0] - field_mins).astype(np.int32)
                       for split, values in base.items()}
        use_global = config["features"]["continuous_history_stats"]
        use_user_tab = config["features"]["user_tab_long_view_rate"]
        if not use_global and not use_user_tab:
            return categorical
        extra: dict[str, list[np.ndarray]] = {split: [] for split in self._splits}
        if use_global:
            for key_index in (1, 2):  # user_id, video_id
                stats, global_rate = aggregate(self._splits["train"], key_index)
                for split, rows in self._splits.items():
                    rates = np.empty(len(rows), dtype=np.float32)
                    counts = np.empty(len(rows), dtype=np.float32)
                    for index, row in enumerate(rows):
                        positives, impressions = stats.get(row[key_index], [0, 0])
                        if split == "train":
                            positives -= int(row[6]); impressions -= 1
                        rates[index] = (positives + 20.0 * global_rate) / (impressions + 20.0)
                        counts[index] = np.log1p(max(0, impressions))
                    extra[split].extend((rates, counts))
        if use_user_tab:
            stats, global_rate = aggregate_pair(self._splits["train"], 1, 4)
            for split, rows in self._splits.items():
                rates = np.empty(len(rows), dtype=np.float32)
                counts = np.empty(len(rows), dtype=np.float32)
                for index, row in enumerate(rows):
                    positives, impressions = stats.get((row[1], row[4]), [0, 0])
                    if split == "train":
                        positives -= int(row[6]); impressions -= 1
                    rates[index] = (positives + 20.0 * global_rate) / (impressions + 20.0)
                    counts[index] = np.log1p(max(0, impressions))
                extra[split].extend((rates, counts))
        return {split: np.column_stack([categorical[split], *extra[split]]).astype(np.float32)
                for split in self._splits}

    @staticmethod
    def _metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        return {key: int(value) if key in ("users", "rows") else float(value)
                for key, value in metrics.items()}

    def _run_lightgbm(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        try:
            import lightgbm as lgb
        except ModuleNotFoundError as exc:
            raise RuntimeError("LightGBM is required: python -m pip install -r requirements.txt") from exc
        matrices = self._lightgbm_matrices(config)
        enc, _ = self._encoded
        ytr, yva, yte = enc["train"][1], enc["valid"][1], enc["test"][1]
        uva, ute = enc["valid"][2], enc["test"][2]
        hp = config["lightgbm_hyperparameters"]
        model = lgb.LGBMClassifier(
            objective="binary", learning_rate=hp["learning_rate"], num_leaves=hp["num_leaves"],
            n_estimators=hp["n_estimators"], min_child_samples=hp["min_child_samples"],
            subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"],
            reg_lambda=hp["reg_lambda"], random_state=config["hyperparameters"]["seed"],
            n_jobs=-1, verbosity=-1,
        )
        callbacks = [lgb.early_stopping(hp["early_stopping_rounds"], verbose=False),
                     lgb.log_evaluation(period=25)]
        started = time.monotonic()
        model.fit(matrices["train"], ytr, eval_set=[(matrices["valid"], yva)],
                  eval_metric="binary_logloss", categorical_feature=list(range(5)), callbacks=callbacks)
        valid_scores = model.predict_proba(matrices["valid"])[:, 1]
        test_scores = model.predict_proba(matrices["test"])[:, 1]
        valid = self.evaluate_mod.evaluate(uva, yva, valid_scores)
        test = self.evaluate_mod.evaluate(ute, yte, test_scores)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(checkpoint.with_suffix(".txt")))
        np.savez_compressed(checkpoint, test_scores=test_scores,
                           best_iteration=np.asarray(model.best_iteration_))
        return {**self._metrics(valid), "test": self._metrics(test),
                "best_iteration": int(model.best_iteration_),
                "runtime_seconds": float(time.monotonic() - started)}

    def prepare(self) -> None:
        required = ("video_features_basic_pure.csv", "log_standard_4_08_to_4_21_pure.csv",
                    "log_standard_4_22_to_5_08_pure.csv")
        missing = [name for name in required if not (self.data_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"dataset is missing from {self.data_dir}: {', '.join(missing)}")
        print(f"Loading dataset from {self.data_dir} ...", flush=True)
        self._splits = self.data.load(str(self.data_dir))
        self._encoded = self.data.encode(self._splits)
        print("Dataset loaded and encoded: " + ", ".join(
            f"{name}={len(rows):,}" for name, rows in self._splits.items()), flush=True)

    def run(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        validate_config(config)
        if self._encoded is None:
            self.prepare()
        if config["model"] == "lightgbm":
            return self._run_lightgbm(config, checkpoint)
        enc, dim = self._encoded_for(config)
        hp = config["hyperparameters"]
        Xtr, ytr, utr = enc["train"]
        Xva, yva, uva = enc["valid"]
        Xte, yte, ute = enc["test"]
        model = self.baseline.FM(dim, k=hp["embedding_dim"], lr=hp["learning_rate"],
                                 l2=hp["l2"], seed=hp["seed"])
        rng = np.random.default_rng(hp["seed"])
        best_score, best_state, bad, best_epoch = -1.0, None, 0, 0
        started = time.monotonic()
        for epoch in range(1, hp["epochs"] + 1):
            if config["training_objective"] == "bpr":
                positive, negative = build_pair_indices(utr, ytr, rng)
                for start in range(0, len(positive), hp["batch_size"]):
                    selection = slice(start, start + hp["batch_size"])
                    bpr_step(model, Xtr[positive[selection]], Xtr[negative[selection]])
            else:
                idx = rng.permutation(len(ytr))
                for start in range(0, len(idx), hp["batch_size"]):
                    batch = idx[start:start + hp["batch_size"]]
                    model.step(Xtr[batch], ytr[batch])
            metrics = self.evaluate_mod.evaluate(uva, yva, model.predict(Xva))
            print(f"    epoch {epoch:02d} | primary={float(metrics['primary']):.6f}"
                  f" | best={max(best_score, float(metrics['primary'])):.6f}", flush=True)
            if metrics["primary"] > best_score + 1e-5:
                best_score, bad, best_epoch = metrics["primary"], 0, epoch
                best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
            else:
                bad += 1
                if bad >= hp["patience"]:
                    break
        if best_state is None:
            raise RuntimeError("training produced no checkpoint")
        model.V, model.W, model.b = best_state
        valid = self.evaluate_mod.evaluate(uva, yva, model.predict(Xva))
        test_scores = model.predict(Xte)
        test = self.evaluate_mod.evaluate(ute, yte, test_scores)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(checkpoint, V=model.V, W=model.W, b=model.b,
                           test_scores=test_scores, best_epoch=np.asarray(best_epoch))
        return {"GAUC": float(valid["GAUC"]), "nDCG@5": float(valid["nDCG@5"]),
                "primary": float(valid["primary"]),
                "test": self._metrics(test),
                "best_epoch": int(best_epoch),
                "runtime_seconds": float(time.monotonic() - started)}

    def write_submission(self, checkpoint: Path, output: Path) -> None:
        if self._splits is None:
            self.prepare()
        with np.load(checkpoint) as state:
            scores = state["test_scores"]
        output.parent.mkdir(parents=True, exist_ok=True)
        self.submit.write_submission(output, self._splits["test"], scores)
        self.submit.read_submission(output, self._splits["test"])
