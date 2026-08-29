from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ALLOWED_VALUES = {
    "embedding_dim": (8, 16, 24, 32, 48, 64),
    "learning_rate": (0.0003, 0.0005, 0.001, 0.002, 0.005),
    "epochs": (10, 20, 30, 40),
    "l2": (0.0, 1e-6, 1e-5, 1e-4),
    "batch_size": (4096, 8192, 16384),
    "patience": (3, 4, 5),
    "pairs_per_positive": (1, 2, 4),
    "seed": (0,),
}
FEATURE_KEYS = ("user_long_view_rate", "item_long_view_rate", "continuous_history_stats",
                "user_tab_long_view_rate")
MODELS = ("fm", "lightgbm")
OBJECTIVES = ("bce", "bpr")
LIGHTGBM_KEYS = {
    "learning_rate", "num_leaves", "n_estimators", "min_child_samples", "subsample",
    "colsample_bytree", "reg_lambda", "early_stopping_rounds",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("model") not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    if config.get("training_objective") not in OBJECTIVES:
        raise ValueError(f"training_objective must be one of {OBJECTIVES}")
    if config["model"] == "lightgbm" and config["training_objective"] != "bce":
        raise ValueError("LightGBM currently supports only the BCE objective")
    hp = config.get("hyperparameters")
    if not isinstance(hp, dict):
        raise ValueError("hyperparameters must be an object")
    unknown = set(hp) - set(ALLOWED_VALUES)
    if unknown:
        raise ValueError(f"unsupported hyperparameters: {sorted(unknown)}")
    missing = set(ALLOWED_VALUES) - set(hp)
    if missing:
        raise ValueError(f"missing hyperparameters: {sorted(missing)}")
    for key, allowed in ALLOWED_VALUES.items():
        if hp[key] not in allowed:
            raise ValueError(f"{key}={hp[key]!r} is outside the allowed experiment space")
    features = config.get("features")
    if not isinstance(features, dict) or set(features) != set(FEATURE_KEYS):
        raise ValueError(f"features must contain exactly: {list(FEATURE_KEYS)}")
    if any(type(features[key]) is not bool for key in FEATURE_KEYS):
        raise ValueError("feature flags must be booleans")
    lgb = config.get("lightgbm_hyperparameters")
    if not isinstance(lgb, dict) or set(lgb) != LIGHTGBM_KEYS:
        raise ValueError(f"lightgbm_hyperparameters must contain exactly: {sorted(LIGHTGBM_KEYS)}")
    if config["model"] == "fm" and any(features[key] for key in
                                        ("continuous_history_stats", "user_tab_long_view_rate")):
        raise ValueError("continuous statistical features require model='lightgbm'")


def apply_changes(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    if not changes or not isinstance(changes, dict):
        raise ValueError("proposal changes must be a non-empty object")
    allowed = set(ALLOWED_VALUES) | set(FEATURE_KEYS) | {"model", "training_objective"}
    if set(changes) - allowed:
        raise ValueError(f"unsupported proposal keys: {sorted(set(changes) - allowed)}")
    candidate = copy.deepcopy(base)
    for key, value in changes.items():
        if key in ("model", "training_objective"):
            candidate[key] = value
        else:
            target = candidate["features"] if key in FEATURE_KEYS else candidate["hyperparameters"]
            target[key] = value
    validate_config(candidate)
    return candidate


def experiment_key(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))
