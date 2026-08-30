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
    "negative_sampling": ("random", "hard"),
    "hard_negative_candidates": (2, 4),
    "deepfm_hidden_dim": (16, 32, 64),
    "hybrid_bpr_weight": (0.25, 0.5, 0.75),
    "ensemble_deepfm_weight": (0.3, 0.4, 0.5),
    "auxiliary_loss_weight": (0.05, 0.1, 0.2, 0.5),
    "auxiliary_signals": (
        "click", "like", "completion", "click_like", "click_like_completion",
        "log_watch"
    ),
    "dcn_cross_layers": (1, 2, 3),
    "dcn_low_rank": (8, 16, 32),
    "sequence_length": (16, 32),
    "feature_control": ("real", "constant", "shuffled", "random_same_cardinality"),
    "seed": (0, 1, 2, 3, 4),
}
FEATURE_KEYS = (
    "user_long_view_rate",
    "item_long_view_rate",
    "continuous_history_stats",
    "user_tab_long_view_rate",
    "user_tab_cross",
    "user_author_cross",
    "user_recent_3d_activity",
    "item_recent_3d_exposure",
    "prior_video_positive",
    "author_positive_recency",
    "prior_video_count",
    "previous_author_same",
    "global_context",
)
MODELS = (
    "fm", "deepfm", "multitask_deepfm", "sequence_deepfm", "dcnv2",
    "ensemble", "lightgbm",
)
OBJECTIVES = ("bce", "bpr", "hybrid")
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
    if config["model"] == "ensemble" and config["training_objective"] != "hybrid":
        raise ValueError("The FM/DeepFM ensemble requires training_objective='hybrid'")
    if config["model"] == "multitask_deepfm" and config["training_objective"] != "bce":
        raise ValueError("Multi-task DeepFM currently supports only the BCE objective")
    if config["model"] == "dcnv2" and config["training_objective"] != "bce":
        raise ValueError("DCNv2 currently supports only the BCE objective")
    if config["model"] == "sequence_deepfm" and config["training_objective"] != "bce":
        raise ValueError("Sequence DeepFM currently supports only the BCE objective")
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
    if hp["feature_control"] != "real":
        controlled = [key for key in (
            "prior_video_positive", "author_positive_recency",
            "prior_video_count", "previous_author_same",
        ) if features[key]]
        if len(controlled) != 1:
            raise ValueError(
                "a placebo feature_control requires exactly one supported categorical history feature"
            )
    lgb = config.get("lightgbm_hyperparameters")
    if not isinstance(lgb, dict) or set(lgb) != LIGHTGBM_KEYS:
        raise ValueError(f"lightgbm_hyperparameters must contain exactly: {sorted(LIGHTGBM_KEYS)}")
    if config["model"] in ("fm", "deepfm", "multitask_deepfm", "sequence_deepfm", "dcnv2") and any(features[key] for key in
                                                       ("continuous_history_stats", "user_tab_long_view_rate")):
        raise ValueError("continuous statistical features require model='lightgbm'")
    if config["model"] == "lightgbm" and any(features[key] for key in
                                              ("user_tab_cross", "user_author_cross",
                                               "user_recent_3d_activity",
                                               "item_recent_3d_exposure",
                                               "prior_video_positive",
                                               "author_positive_recency",
                                               "prior_video_count",
                                               "previous_author_same",
                                               "global_context")):
        raise ValueError("categorical crosses and temporal buckets require an FM-family model")


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
