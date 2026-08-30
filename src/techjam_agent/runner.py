from __future__ import annotations

import csv
import copy
import hashlib
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import validate_config
from .bpr import bpr_step, build_pair_indices, hybrid_step
from .causal_sequence import strict_past_sequences
from .dcnv2 import DCNv2
from .deepfm import DeepFM, MultiTaskDeepFM
from .ensemble import blend_scores
from .feedback import (
    align_auxiliary_feedback,
    auxiliary_task_count,
    select_auxiliary_feedback,
)
from .history_features import aggregate, aggregate_pair, smoothed_rate_bucket
from .sequence_features import (
    SEQUENCE_FEATURE_DIMS,
    align_event_times,
    strict_sequence_categories,
)
from .sequence_model import LightweightSequenceDeepFM
from .temporal_features import bucket_log_counts, strict_past_window_counts


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ExperimentRunner:
    def __init__(self, root: Path, data_dir: Path, starter_dir: Path,
                 evaluator_sha256: str | None = None) -> None:
        self.root = root
        self.data_dir = data_dir
        self.starter_dir = starter_dir
        self.evaluator_sha256 = evaluator_sha256
        sys.path.insert(0, str(starter_dir))
        self.data = _load_module("techjam_starter_data", starter_dir / "data.py")
        self.evaluate_mod = _load_module("techjam_starter_evaluate", starter_dir / "evaluate.py")
        self.baseline = _load_module("techjam_starter_baseline", starter_dir / "baseline.py")
        self.submit = _load_module("techjam_starter_submit", starter_dir / "submit.py")
        self._splits = None
        self._encoded = None
        self._auxiliary_feedback = None
        self._sequence_categories = None
        self._causal_sequence_cache = {}

    def _auxiliary_for(self, selection: str) -> tuple[np.ndarray, np.ndarray]:
        if self._auxiliary_feedback is None:
            self._auxiliary_feedback = align_auxiliary_feedback(
                self.data_dir, {"train": self._splits["train"]}
            )
        labels, masks = self._auxiliary_feedback
        return select_auxiliary_feedback(
            labels["train"], masks["train"], selection
        )

    def _causal_for(
        self,
        sequence_length: int,
        include_test: bool = False,
    ) -> dict[str, dict[str, np.ndarray]]:
        key = (int(sequence_length), bool(include_test))
        if key not in self._causal_sequence_cache:
            names = tuple(self._splits) if include_test else ("train", "valid")
            splits = {name: self._splits[name] for name in names}
            base = {name: self._encoded[0][name] for name in names}
            event_times = align_event_times(self.data_dir, splits)
            self._causal_sequence_cache[key] = strict_past_sequences(
                splits, event_times, base, max_length=sequence_length
            )
        return self._causal_sequence_cache[key]

    @staticmethod
    def _history_batch(history: dict[str, np.ndarray], selection) -> dict[str, np.ndarray]:
        return {
            name: values[selection]
            for name, values in history.items()
            if name != "length"
        }

    def verify_evaluator(self) -> None:
        if not self.evaluator_sha256:
            return
        path = self.starter_dir / "evaluate.py"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != self.evaluator_sha256:
            raise RuntimeError(f"official evaluator integrity check failed: {actual}")

    def _encoded_for(self, config: dict[str, Any]):
        base, base_dim = self._encoded
        rate_features = [key for key in ("user_long_view_rate", "item_long_view_rate")
                         if config["features"][key]]
        cross_features = [key for key in ("user_tab_cross", "user_author_cross")
                          if config["features"][key]]
        temporal_features = [key for key in
                             ("user_recent_3d_activity", "item_recent_3d_exposure")
                             if config["features"][key]]
        sequence_features = [key for key in SEQUENCE_FEATURE_DIMS
                             if config["features"][key]]
        use_global_context = config["features"]["global_context"]
        if (not rate_features and not cross_features and not temporal_features
                and not sequence_features and not use_global_context):
            return base, base_dim
        key_indices = {"user_long_view_rate": 1, "item_long_view_rate": 2}
        columns = {}
        next_offset = base_dim
        for feature in rate_features:
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
        cross_indices = {"user_tab_cross": (1, 4), "user_author_cross": (1, 3)}
        for feature in cross_features:
            first, second = cross_indices[feature]
            vocabulary = {
                key: index
                for index, key in enumerate(dict.fromkeys(
                    (row[first], row[second]) for row in self._splits["train"]
                ))
            }
            unknown = len(vocabulary)
            for split, rows in self._splits.items():
                values = np.fromiter(
                    (next_offset + vocabulary.get((row[first], row[second]), unknown)
                     for row in rows),
                    dtype=np.int32,
                    count=len(rows),
                )
                columns.setdefault(split, []).append(values)
            next_offset += unknown + 1
        temporal_indices = {"user_recent_3d_activity": 1, "item_recent_3d_exposure": 2}
        for feature in temporal_features:
            counts = strict_past_window_counts(
                self._splits, temporal_indices[feature], window_days=3
            )
            buckets, dimension = bucket_log_counts(counts)
            for split, values in buckets.items():
                columns.setdefault(split, []).append(next_offset + values)
            next_offset += dimension
        if sequence_features:
            if self._sequence_categories is None:
                event_times = align_event_times(self.data_dir, self._splits)
                self._sequence_categories = strict_sequence_categories(
                    self._splits, event_times
                )
            for feature in sequence_features:
                for split in self._splits:
                    values = self._sequence_categories[split][feature]
                    columns.setdefault(split, []).append(next_offset + values)
                next_offset += SEQUENCE_FEATURE_DIMS[feature]
        if use_global_context:
            for split, rows in self._splits.items():
                columns.setdefault(split, []).append(
                    np.full(len(rows), next_offset, dtype=np.int32)
                )
            next_offset += 1
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
        ytr, yva = enc["train"][1], enc["valid"][1]
        uva = enc["valid"][2]
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
        valid = self.evaluate_mod.evaluate(uva, yva, valid_scores)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(checkpoint.with_suffix(".txt")))
        np.savez_compressed(checkpoint, best_iteration=np.asarray(model.best_iteration_))
        return {**self._metrics(valid),
                "best_iteration": int(model.best_iteration_),
                "runtime_seconds": float(time.monotonic() - started)}

    def prepare(self) -> None:
        self.verify_evaluator()
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

    def _ensemble_configs(self, config: dict[str, Any]):
        fm_config = copy.deepcopy(config)
        fm_config["model"] = "fm"
        fm_config["training_objective"] = "bpr"
        fm_config["hyperparameters"]["learning_rate"] = 0.0003
        fm_config["hyperparameters"]["negative_sampling"] = "random"
        deepfm_config = copy.deepcopy(config)
        deepfm_config["model"] = "deepfm"
        deepfm_config["training_objective"] = "bce"
        deepfm_config["hyperparameters"]["learning_rate"] = 0.001
        return fm_config, deepfm_config

    def _run_ensemble(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        started = time.monotonic()
        fm_config, deepfm_config = self._ensemble_configs(config)
        fm_checkpoint = checkpoint.with_name(checkpoint.stem + "_fm.npz")
        deepfm_checkpoint = checkpoint.with_name(checkpoint.stem + "_deepfm.npz")
        self.run(fm_config, fm_checkpoint)
        self.run(deepfm_config, deepfm_checkpoint)
        enc, dim = self._encoded_for(config)
        Xvalid, labels, users = enc["valid"]
        hp = config["hyperparameters"]
        fm = self.baseline.FM(dim, k=hp["embedding_dim"], lr=0.0003,
                              l2=hp["l2"], seed=hp["seed"])
        with np.load(fm_checkpoint) as state:
            fm.V, fm.W, fm.b = state["V"], state["W"], state["b"]
            fm_state = {"fm_V": state["V"].copy(), "fm_W": state["W"].copy(),
                        "fm_b": state["b"].copy()}
        deepfm = DeepFM(dim, Xvalid.shape[1], embedding_dim=hp["embedding_dim"],
                        hidden_dim=hp["deepfm_hidden_dim"], learning_rate=0.001,
                        l2=hp["l2"], seed=hp["seed"])
        with np.load(deepfm_checkpoint) as state:
            deepfm.load_state_dict({name: state[name] for name in deepfm.state_dict()})
            deepfm_state = {f"deepfm_{name}": state[name].copy()
                            for name in deepfm.state_dict()}
        scores = blend_scores(users, fm.predict(Xvalid), deepfm.predict(Xvalid),
                              hp["ensemble_deepfm_weight"])
        metrics = self.evaluate_mod.evaluate(users, labels, scores)
        np.savez_compressed(checkpoint, **fm_state, **deepfm_state)
        return {**self._metrics(metrics), "best_epoch": 0,
                "runtime_seconds": float(time.monotonic() - started)}

    def run(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        validate_config(config)
        if self._encoded is None:
            self.prepare()
        if config["model"] == "lightgbm":
            return self._run_lightgbm(config, checkpoint)
        if config["model"] == "ensemble":
            return self._run_ensemble(config, checkpoint)
        enc, dim = self._encoded_for(config)
        hp = config["hyperparameters"]
        Xtr, ytr, utr = enc["train"]
        Xva, yva, uva = enc["valid"]
        causal_sequence = (
            self._causal_for(hp["sequence_length"])
            if config["model"] == "sequence_deepfm"
            else None
        )
        if config["model"] == "multitask_deepfm":
            auxiliary_train, auxiliary_mask = self._auxiliary_for(
                hp["auxiliary_signals"]
            )
            model = MultiTaskDeepFM(
                dim,
                Xtr.shape[1],
                embedding_dim=hp["embedding_dim"],
                hidden_dim=hp["deepfm_hidden_dim"],
                learning_rate=hp["learning_rate"],
                l2=hp["l2"],
                seed=hp["seed"],
                auxiliary_tasks=auxiliary_task_count(hp["auxiliary_signals"]),
            )
        elif config["model"] == "sequence_deepfm":
            model = LightweightSequenceDeepFM(
                dim,
                Xtr.shape[1],
                embedding_dim=hp["embedding_dim"],
                hidden_dim=hp["deepfm_hidden_dim"],
                learning_rate=hp["learning_rate"],
                l2=hp["l2"],
                seed=hp["seed"],
                sequence_length=hp["sequence_length"],
            )
        elif config["model"] == "deepfm":
            model = DeepFM(
                dim,
                Xtr.shape[1],
                embedding_dim=hp["embedding_dim"],
                hidden_dim=hp["deepfm_hidden_dim"],
                learning_rate=hp["learning_rate"],
                l2=hp["l2"],
                seed=hp["seed"],
            )
        elif config["model"] == "dcnv2":
            model = DCNv2(
                dim,
                Xtr.shape[1],
                embedding_dim=hp["embedding_dim"],
                hidden_dim=hp["deepfm_hidden_dim"],
                cross_layers=hp["dcn_cross_layers"],
                cross_rank=hp["dcn_low_rank"],
                learning_rate=hp["learning_rate"],
                l2=hp["l2"],
                seed=hp["seed"],
            )
        else:
            model = self.baseline.FM(dim, k=hp["embedding_dim"], lr=hp["learning_rate"],
                                     l2=hp["l2"], seed=hp["seed"])
        rng = np.random.default_rng(hp["seed"])
        best_score, best_state, bad, best_epoch = -1.0, None, 0, 0
        started = time.monotonic()
        for epoch in range(1, hp["epochs"] + 1):
            if config["training_objective"] in ("bpr", "hybrid"):
                negative_scores = (
                    model.predict(Xtr) if hp["negative_sampling"] == "hard" else None
                )
                positive, negative = build_pair_indices(
                    utr,
                    ytr,
                    rng,
                    hp["pairs_per_positive"],
                    negative_scores=negative_scores,
                    hard_negative_candidates=hp["hard_negative_candidates"],
                )
                for start in range(0, len(positive), hp["batch_size"]):
                    selection = slice(start, start + hp["batch_size"])
                    positive_x = Xtr[positive[selection]]
                    negative_x = Xtr[negative[selection]]
                    if config["training_objective"] == "hybrid":
                        if config["model"] == "deepfm":
                            model.hybrid_step(
                                positive_x, negative_x, hp["hybrid_bpr_weight"]
                            )
                        else:
                            hybrid_step(
                                model, positive_x, negative_x, hp["hybrid_bpr_weight"]
                            )
                    elif config["model"] == "deepfm":
                        model.bpr_step(positive_x, negative_x)
                    else:
                        bpr_step(model, positive_x, negative_x)
            else:
                idx = rng.permutation(len(ytr))
                for start in range(0, len(idx), hp["batch_size"]):
                    batch = idx[start:start + hp["batch_size"]]
                    if config["model"] == "multitask_deepfm":
                        model.multitask_step(
                            Xtr[batch],
                            ytr[batch],
                            auxiliary_train[batch],
                            hp["auxiliary_loss_weight"],
                            auxiliary_mask[batch],
                            "mse" if hp["auxiliary_signals"] == "log_watch" else "bce",
                        )
                    elif config["model"] == "sequence_deepfm":
                        model.step(
                            Xtr[batch],
                            self._history_batch(causal_sequence["train"], batch),
                            ytr[batch],
                        )
                    else:
                        model.step(Xtr[batch], ytr[batch])
            valid_scores = (
                model.predict(Xva, causal_sequence["valid"])
                if config["model"] == "sequence_deepfm"
                else model.predict(Xva)
            )
            metrics = self.evaluate_mod.evaluate(uva, yva, valid_scores)
            print(f"    epoch {epoch:02d} | primary={float(metrics['primary']):.6f}"
                  f" | best={max(best_score, float(metrics['primary'])):.6f}", flush=True)
            if metrics["primary"] > best_score + 1e-5:
                best_score, bad, best_epoch = metrics["primary"], 0, epoch
                best_state = (model.state_dict() if config["model"] in
                              ("deepfm", "multitask_deepfm", "sequence_deepfm", "dcnv2") else
                              (model.V.copy(), model.W.copy(), np.float32(model.b)))
            else:
                bad += 1
                if bad >= hp["patience"]:
                    break
        if best_state is None:
            raise RuntimeError("training produced no checkpoint")
        if config["model"] in ("deepfm", "multitask_deepfm", "sequence_deepfm", "dcnv2"):
            model.load_state_dict(best_state)
        else:
            model.V, model.W, model.b = best_state
        valid_scores = (
            model.predict(Xva, causal_sequence["valid"])
            if config["model"] == "sequence_deepfm"
            else model.predict(Xva)
        )
        valid = self.evaluate_mod.evaluate(uva, yva, valid_scores)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        state = (model.state_dict() if config["model"] in
                 ("deepfm", "multitask_deepfm", "sequence_deepfm", "dcnv2") else
                 {"V": model.V, "W": model.W, "b": model.b})
        np.savez_compressed(checkpoint, **state, best_epoch=np.asarray(best_epoch))
        return {"GAUC": float(valid["GAUC"]), "nDCG@5": float(valid["nDCG@5"]),
                "primary": float(valid["primary"]),
                "best_epoch": int(best_epoch),
                "runtime_seconds": float(time.monotonic() - started)}

    def finalize(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        if self._splits is None:
            self.prepare()
        enc, dim = self._encoded_for(config)
        Xtest, ytest, users = enc["test"]
        if config["model"] == "lightgbm":
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(checkpoint.with_suffix(".txt")))
            scores = model.predict(self._lightgbm_matrices(config)["test"])
        elif config["model"] == "ensemble":
            hp = config["hyperparameters"]
            fm = self.baseline.FM(dim, k=hp["embedding_dim"], lr=0.0003,
                                  l2=hp["l2"], seed=hp["seed"])
            deepfm = DeepFM(dim, Xtest.shape[1], embedding_dim=hp["embedding_dim"],
                            hidden_dim=hp["deepfm_hidden_dim"], learning_rate=0.001,
                            l2=hp["l2"], seed=hp["seed"])
            with np.load(checkpoint) as state:
                fm.V, fm.W, fm.b = state["fm_V"], state["fm_W"], state["fm_b"]
                deepfm.load_state_dict({name: state[f"deepfm_{name}"]
                                        for name in deepfm.state_dict()})
            scores = blend_scores(users, fm.predict(Xtest), deepfm.predict(Xtest),
                                  hp["ensemble_deepfm_weight"])
        elif config["model"] in ("deepfm", "multitask_deepfm", "sequence_deepfm", "dcnv2"):
            hp = config["hyperparameters"]
            if config["model"] == "dcnv2":
                model = DCNv2(
                    dim,
                    Xtest.shape[1],
                    embedding_dim=hp["embedding_dim"],
                    hidden_dim=hp["deepfm_hidden_dim"],
                    cross_layers=hp["dcn_cross_layers"],
                    cross_rank=hp["dcn_low_rank"],
                    learning_rate=hp["learning_rate"],
                    l2=hp["l2"],
                    seed=hp["seed"],
                )
            elif config["model"] == "sequence_deepfm":
                model = LightweightSequenceDeepFM(
                    dim,
                    Xtest.shape[1],
                    embedding_dim=hp["embedding_dim"],
                    hidden_dim=hp["deepfm_hidden_dim"],
                    learning_rate=hp["learning_rate"],
                    l2=hp["l2"],
                    seed=hp["seed"],
                    sequence_length=hp["sequence_length"],
                )
            else:
                model_class = (
                    MultiTaskDeepFM
                    if config["model"] == "multitask_deepfm"
                    else DeepFM
                )
                model = model_class(
                    dim,
                    Xtest.shape[1],
                    embedding_dim=hp["embedding_dim"],
                    hidden_dim=hp["deepfm_hidden_dim"],
                    learning_rate=hp["learning_rate"],
                    l2=hp["l2"],
                    seed=hp["seed"],
                    **({"auxiliary_tasks": auxiliary_task_count(hp["auxiliary_signals"])}
                       if config["model"] == "multitask_deepfm" else {}),
                )
            with np.load(checkpoint) as state:
                model.load_state_dict({name: state[name] for name in model.state_dict()})
            scores = (
                model.predict(
                    Xtest,
                    self._causal_for(hp["sequence_length"], include_test=True)["test"],
                )
                if config["model"] == "sequence_deepfm"
                else model.predict(Xtest)
            )
        else:
            hp = config["hyperparameters"]
            model = self.baseline.FM(dim, k=hp["embedding_dim"], lr=hp["learning_rate"],
                                     l2=hp["l2"], seed=hp["seed"])
            with np.load(checkpoint) as state:
                model.V, model.W, model.b = state["V"], state["W"], state["b"]
            scores = model.predict(Xtest)
        metrics = self.evaluate_mod.evaluate(users, ytest, scores)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.submit.write_submission(output, self._splits["test"], scores)
        self.submit.read_submission(output, self._splits["test"])
        return self._metrics(metrics)
