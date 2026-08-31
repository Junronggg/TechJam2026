from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .operator_registry import (
    ALLOWED_VALUES,
    FEATURE_FIELDS,
    HYPERPARAMETER_FIELDS,
    MODEL_SPECS,
    MODELS,
    OBJECTIVES,
    OPERATORS,
    ROOT_FIELDS,
)

FEATURE_KEYS = FEATURE_FIELDS
# Compatibility marker used by the validation-only evidence producers imported
# from the main branch.  Feature semantics remain defined by FEATURE_FIELDS and
# the operator registry; this version is metadata, not a model setting.
FEATURE_SCHEMA_VERSION = "v3"
LIGHTGBM_KEYS = {
    "learning_rate", "num_leaves", "n_estimators", "min_child_samples", "subsample",
    "colsample_bytree", "reg_lambda", "early_stopping_rounds",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Backfill optional fields added after an experiment was archived."""
    normalized = copy.deepcopy(config)
    features = normalized.get("features")
    if isinstance(features, dict):
        for key in FEATURE_KEYS:
            features.setdefault(key, False)
    hyperparameters = normalized.get("hyperparameters")
    if isinstance(hyperparameters, dict):
        for key in HYPERPARAMETER_FIELDS:
            default = OPERATORS[key].default
            if default is not None:
                hyperparameters.setdefault(key, default)
    return normalized


def validate_config(config: dict[str, Any]) -> None:
    config = normalize_config(config)
    if config.get("model") not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    if config.get("training_objective") not in OBJECTIVES:
        raise ValueError(f"training_objective must be one of {OBJECTIVES}")
    model_spec = MODEL_SPECS[config["model"]]
    if config["model"] == "custom":
        branch = config.get("code_branch")
        if not isinstance(branch, str) or not branch.strip():
            raise ValueError("custom model requires a non-empty code_branch path")
    if config["training_objective"] not in model_spec.objectives:
        raise ValueError(
            f"{config['model']} does not support objective="
            f"{config['training_objective']!r}"
        )
    hp = config.get("hyperparameters")
    if not isinstance(hp, dict):
        raise ValueError("hyperparameters must be an object")
    unknown = set(hp) - set(HYPERPARAMETER_FIELDS)
    if unknown:
        raise ValueError(f"unsupported hyperparameters: {sorted(unknown)}")
    missing = set(HYPERPARAMETER_FIELDS) - set(hp)
    if missing:
        raise ValueError(f"missing hyperparameters: {sorted(missing)}")
    for key, allowed in ALLOWED_VALUES.items():
        if hp[key] not in allowed:
            raise ValueError(f"{key}={hp[key]!r} is outside the allowed experiment space")
        spec = OPERATORS[key]
        if (spec.default is not None and hp[key] != spec.default and
                config["model"] not in spec.models):
            raise ValueError(f"{key} is not compatible with model={config['model']!r}")
        if (spec.default is not None and hp[key] != spec.default and
                config["training_objective"] not in spec.objectives):
            raise ValueError(
                f"{key} is not compatible with training_objective="
                f"{config['training_objective']!r}"
            )
    ensemble_size = hp["ensemble_size"]
    if config["model"] == "fm_ensemble" and ensemble_size <= 1:
        raise ValueError("fm_ensemble requires ensemble_size greater than 1")
    if config["model"] != "fm_ensemble" and ensemble_size != 1:
        raise ValueError("ensemble_size greater than 1 requires model='fm_ensemble'")
    ensemble_seed_set = hp["ensemble_seed_set"]
    if config["model"] != "fm_ensemble" and ensemble_seed_set != "sequential":
        raise ValueError("a custom ensemble_seed_set requires model='fm_ensemble'")
    if ensemble_seed_set != "sequential":
        seeds = [int(value) for value in ensemble_seed_set.split(",")]
        if len(seeds) != ensemble_size or len(set(seeds)) != len(seeds):
            raise ValueError("ensemble_seed_set must contain ensemble_size unique seeds")
        if any(seed not in ALLOWED_VALUES["seed"] for seed in seeds):
            raise ValueError("ensemble_seed_set contains an unsupported seed")
    features = config.get("features")
    if not isinstance(features, dict) or set(features) != set(FEATURE_KEYS):
        raise ValueError(f"features must contain exactly: {list(FEATURE_KEYS)}")
    if any(type(features[key]) is not bool for key in FEATURE_KEYS):
        raise ValueError("feature flags must be booleans")
    if not model_spec.supports_engineered_features and any(features.values()):
        raise ValueError(
            f"{config['model']} currently supports base fields only; disable engineered features"
        )
    for key in FEATURE_KEYS:
        if not features[key]:
            continue
        spec = OPERATORS[key]
        if config["model"] not in spec.models:
            raise ValueError(f"{key} is not compatible with model={config['model']!r}")
        if config["training_objective"] not in spec.objectives:
            raise ValueError(
                f"{key} is not compatible with training_objective="
                f"{config['training_objective']!r}"
            )
        for required_field, required_value in spec.requires:
            actual = config.get(required_field)
            if actual != required_value:
                raise ValueError(
                    f"{key} requires {required_field}={required_value!r}, got {actual!r}"
                )
    lgb = config.get("lightgbm_hyperparameters")
    if not isinstance(lgb, dict) or set(lgb) != LIGHTGBM_KEYS:
        raise ValueError(f"lightgbm_hyperparameters must contain exactly: {sorted(LIGHTGBM_KEYS)}")


def apply_changes(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    if not changes or not isinstance(changes, dict):
        raise ValueError("proposal changes must be a non-empty object")
    allowed = set(OPERATORS)
    if set(changes) - allowed:
        raise ValueError(f"unsupported proposal keys: {sorted(set(changes) - allowed)}")
    candidate = normalize_config(base)
    for key, value in changes.items():
        target_name = OPERATORS[key].target
        if target_name == "root":
            candidate[key] = value
        else:
            candidate[target_name][key] = value
    # Branch metadata belongs to the generated source, not to a built-in
    # configuration transition. Clear stale hashes when replacing a branch so
    # candidate IDs and duplicate detection remain content-based.
    if candidate.get("model") != "custom":
        candidate.pop("code_branch", None)
        candidate.pop("code_branch_sha256", None)
        candidate.pop("code_branch_name", None)
    elif "code_branch" in changes:
        candidate.pop("code_branch_sha256", None)
        candidate.pop("code_branch_name", None)
    validate_config(candidate)
    for key, value in changes.items():
        spec = OPERATORS[key]
        if (spec.target == "root" or
                (spec.target == "features" and not value) or
                (spec.default is not None and value == spec.default)):
            continue
        if candidate["model"] not in spec.models:
            raise ValueError(f"{key} is not compatible with model={candidate['model']!r}")
        if candidate["training_objective"] not in spec.objectives:
            raise ValueError(
                f"{key} is not compatible with training_objective="
                f"{candidate['training_objective']!r}"
            )
    return candidate


def experiment_key(config: dict[str, Any]) -> str:
    """Return a scientific key that ignores inactive model-specific knobs.

    For example, FM learning rate is carried through a switch to LightGBM for
    schema stability but does not affect that experiment. Treating it as active
    would let the agent unknowingly repeat the same LightGBM run.
    """
    canonical = normalize_config(config)
    # Display metadata must not make the same generated source look like a
    # different scientific experiment.
    canonical.pop("code_branch_name", None)
    model = canonical.get("model")
    objective = canonical.get("training_objective")
    hyperparameters = canonical.get("hyperparameters")
    if isinstance(hyperparameters, dict):
        for key in tuple(hyperparameters):
            spec = OPERATORS.get(key)
            if spec is None:
                continue
            if model not in spec.models or objective not in spec.objectives:
                hyperparameters.pop(key, None)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))
