"""Validation-only feasibility evidence producers.

These builders never train models, never read Markdown, and never access the
test split. Coverage and correlation writers emit schema-shaped JSON only from
caller-supplied validation arrays or already-computed scalars.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .config import FEATURE_KEYS
from .experiment_planner import FEATURE_FAMILIES


COVERAGE_SCHEMA_VERSION = 1
CORRELATION_SCHEMA_VERSION = 1
LEAKAGE_REGISTRY_VERSION = 1
COVERAGE_SPLIT = "validation"
COVERAGE_SUMMARY_PATH = "runs/candidate_history_audit/summary.json"
LEAKAGE_REGISTRY_PATH = "configs/feature_leakage_registry.json"

# FEATURE_KEYS that analysis/candidate_history_audit.py actually computes.
COVERAGE_FEATURE_KEYS = (
    "prior_video_positive",
    "author_positive_recency",
    "prior_video_count",
    "previous_author_same",
)

CORRELATION_OFFICIAL_CHECKPOINTS = (
    "runs/sequence_ablation/fm_bpr.npz",
    "runs/dcnv2_ablation/deepfm.npz",
    "runs/dcnv2_ablation/dcnv2.npz",
)
CORRELATION_ROLLING_CHECKPOINTS = (
    "runs/rolling_sequence/fold_1/fm_bpr.npz",
    "runs/rolling_sequence/fold_2/fm_bpr.npz",
    "runs/rolling_sequence/fold_3/fm_bpr.npz",
    "runs/rolling_dcnv2/fold_1/deepfm.npz",
    "runs/rolling_dcnv2/fold_2/deepfm.npz",
    "runs/rolling_dcnv2/fold_3/deepfm.npz",
    "runs/rolling_dcnv2/fold_1/dcnv2.npz",
    "runs/rolling_dcnv2/fold_2/dcnv2.npz",
    "runs/rolling_dcnv2/fold_3/dcnv2.npz",
)
CORRELATION_EXISTING_SUMMARIES = (
    "runs/dcnv2_ensemble/summary.json",
    "runs/enhanced_ensemble/summary.json",
    "runs/conditional_complementarity/summary.json",
)
MARKDOWN_FORBIDDEN = frozenset({"TRY.md", "AGENT-TRY.md"})

# Audited FEATURE_KEYS implementations. Safe only when code proves validation
# and test labels are unused and every temporal aggregation is strict-past.
LEAKAGE_AUDIT: dict[str, dict[str, Any]] = {
    "user_long_view_rate": {
        "status": "uncertain",
        "leakage_safe": True,
        "strict_past": False,
        "implementation_source": (
            "src/techjam_agent/runner.py:_encoded_for + "
            "src/techjam_agent/history_features.py:aggregate"
        ),
        "rationale": (
            "Stats are fit on the train split with leave-one-out on the current "
            "train label. Validation and test labels are not read, but the "
            "feature is a global train-label rate rather than a strict-past "
            "temporal window. Person 1 must confirm whether train-label target "
            "encoding is acceptable."
        ),
    },
    "item_long_view_rate": {
        "status": "uncertain",
        "leakage_safe": True,
        "strict_past": False,
        "implementation_source": (
            "src/techjam_agent/runner.py:_encoded_for + "
            "src/techjam_agent/history_features.py:aggregate"
        ),
        "rationale": (
            "Same train-only leave-one-out target-rate encoding as "
            "user_long_view_rate. Validation and test labels are unused, but "
            "aggregation is not a strict-past temporal window. Person 1 must "
            "confirm."
        ),
    },
    "continuous_history_stats": {
        "status": "uncertain",
        "leakage_safe": True,
        "strict_past": False,
        "implementation_source": (
            "src/techjam_agent/runner.py:_lightgbm_matrices + "
            "src/techjam_agent/history_features.py:aggregate"
        ),
        "rationale": (
            "Continuous user and item rates/counts use train-only aggregates "
            "with train leave-one-out. Validation and test labels are unused, "
            "but the statistic is not strict-past. Person 1 must confirm."
        ),
    },
    "user_tab_long_view_rate": {
        "status": "uncertain",
        "leakage_safe": True,
        "strict_past": False,
        "implementation_source": (
            "src/techjam_agent/runner.py:_lightgbm_matrices + "
            "src/techjam_agent/history_features.py:aggregate_pair"
        ),
        "rationale": (
            "User-tab rates use train-only pair aggregates with train "
            "leave-one-out. Validation and test labels are unused, but "
            "aggregation is not a strict-past temporal window. Person 1 must "
            "confirm."
        ),
    },
    "user_tab_cross": {
        "status": "safe",
        "leakage_safe": True,
        "strict_past": True,
        "implementation_source": "src/techjam_agent/runner.py:_encoded_for",
        "rationale": (
            "Vocabulary is the set of train (user, tab) pairs. No labels are "
            "read on any split. There is no temporal aggregation, so no "
            "future or validation/test label can enter the encoding."
        ),
    },
    "user_author_cross": {
        "status": "safe",
        "leakage_safe": True,
        "strict_past": True,
        "implementation_source": "src/techjam_agent/runner.py:_encoded_for",
        "rationale": (
            "Vocabulary is the set of train (user, author) pairs. No labels "
            "are read on any split. There is no temporal aggregation."
        ),
    },
    "user_recent_3d_activity": {
        "status": "safe",
        "leakage_safe": True,
        "strict_past": True,
        "implementation_source": (
            "src/techjam_agent/temporal_features.py:strict_past_window_counts"
        ),
        "rationale": (
            "Counts prior-day exposures in a rolling window. Labels are never "
            "read. Same-day rows cannot increment the count used for that day."
        ),
    },
    "item_recent_3d_exposure": {
        "status": "safe",
        "leakage_safe": True,
        "strict_past": True,
        "implementation_source": (
            "src/techjam_agent/temporal_features.py:strict_past_window_counts"
        ),
        "rationale": (
            "Same strict-past, label-free window counts as "
            "user_recent_3d_activity, keyed by video_id."
        ),
    },
    "prior_video_positive": {
        "status": "safe",
        "leakage_safe": True,
        "strict_past": True,
        "implementation_source": (
            "src/techjam_agent/sequence_features.py:strict_sequence_categories"
        ),
        "rationale": (
            "Train positives enter history only at strictly earlier timestamps. "
            "Validation and test start from the final train state and never "
            "update positive history with their own labels."
        ),
    },
    "author_positive_recency": {
        "status": "safe",
        "leakage_safe": True,
        "strict_past": True,
        "implementation_source": (
            "src/techjam_agent/sequence_features.py:strict_sequence_categories"
        ),
        "rationale": (
            "Recency uses the last train positive timestamp for that "
            "(user, author). Validation and test labels never update "
            "last_positive_author_time."
        ),
    },
    "prior_video_count": {
        "status": "safe",
        "leakage_safe": True,
        "strict_past": True,
        "implementation_source": (
            "src/techjam_agent/sequence_features.py:strict_sequence_categories"
        ),
        "rationale": (
            "Exposure counts update after the current timestamp group is "
            "emitted. Validation and test may update counts from inputs only; "
            "labels on those splits are not read."
        ),
    },
    "previous_author_same": {
        "status": "safe",
        "leakage_safe": True,
        "strict_past": True,
        "implementation_source": (
            "src/techjam_agent/sequence_features.py:strict_sequence_categories"
        ),
        "rationale": (
            "Previous-author state updates after the current timestamp group. "
            "No labels are read for this flag."
        ),
    },
    "global_context": {
        "status": "safe",
        "leakage_safe": True,
        "strict_past": True,
        "implementation_source": "src/techjam_agent/runner.py:_encoded_for",
        "rationale": (
            "Constant field identifier appended to every row. No labels and no "
            "temporal aggregation are used."
        ),
    },
}


def write_versioned_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_validation_only(test_labels_used: bool, what: str) -> None:
    if test_labels_used:
        raise ValueError(f"{what} refuses test_labels_used=true")


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def coverage_stats(feature: str, values: Any) -> dict[str, Any]:
    if feature not in COVERAGE_FEATURE_KEYS:
        raise ValueError(f"unsupported coverage feature: {feature}")
    array = _as_float_array(values)
    total_rows = int(array.size)
    if feature == "author_positive_recency":
        eligible_rows = int(np.count_nonzero(array != -1.0))
    elif feature == "prior_video_count":
        eligible_rows = int(np.count_nonzero(array > 0))
    else:
        eligible_rows = int(np.count_nonzero(array == 1))
    coverage = (eligible_rows / total_rows) if total_rows else 0.0
    return {
        "feature": feature,
        "split": COVERAGE_SPLIT,
        "coverage": float(coverage),
        "eligible_rows": eligible_rows,
        "total_rows": total_rows,
    }


def coverage_summary_from_signals(
    signals: Mapping[str, Any],
    *,
    test_labels_used: bool = False,
) -> dict[str, Any]:
    """Build coverage JSON for supported FEATURE_KEYS that were actually computed."""
    _require_validation_only(test_labels_used, "coverage producer")
    features = {
        name: coverage_stats(name, signals[name])
        for name in COVERAGE_FEATURE_KEYS
        if name in signals
    }
    return {
        "version": COVERAGE_SCHEMA_VERSION,
        "test_labels_used": False,
        "split": COVERAGE_SPLIT,
        "features": features,
    }


def coverage_manifest_sources() -> list[dict[str, Any]]:
    return [
        {
            "id": f"coverage_{feature}_v1",
            "family": FEATURE_FAMILIES[feature],
            "kind": "feature_coverage",
            "path": COVERAGE_SUMMARY_PATH,
            "pointer": ["features", feature],
            "validation_only": True,
            "optional": True,
            "applies_to": {"features": {feature: True}},
        }
        for feature in COVERAGE_FEATURE_KEYS
    ]


def correlation_pair(
    model_a: str,
    model_b: str,
    correlation: float,
    *,
    split: str = "validation",
    test_labels_used: bool = False,
) -> dict[str, Any]:
    _require_validation_only(test_labels_used, "correlation producer")
    if split in {"test", "final_test"}:
        raise ValueError("correlation producer refuses test-split predictions")
    value = float(correlation)
    if not -1.0 <= value <= 1.0:
        raise ValueError("correlation must be in [-1, 1]")
    return {
        "correlation": value,
        "models": [model_a, model_b],
        "split": split,
        "test_labels_used": False,
    }


def correlation_from_scores(
    model_a: str,
    model_b: str,
    scores_a: Any,
    scores_b: Any,
    *,
    split: str = "validation",
    test_labels_used: bool = False,
) -> dict[str, Any]:
    _require_validation_only(test_labels_used, "correlation producer")
    left = _as_float_array(scores_a)
    right = _as_float_array(scores_b)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("correlation requires matching validation score vectors")
    return correlation_pair(
        model_a,
        model_b,
        float(np.corrcoef(left, right)[0, 1]),
        split=split,
    )


def correlation_summary_from_pairs(
    pairs: Iterable[Mapping[str, Any]],
    *,
    test_labels_used: bool = False,
) -> dict[str, Any]:
    _require_validation_only(test_labels_used, "correlation producer")
    keyed: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        models = pair["models"]
        key = "_".join(models)
        keyed[key] = dict(pair)
    return {
        "version": CORRELATION_SCHEMA_VERSION,
        "test_labels_used": False,
        "split": "validation",
        "pairs": keyed,
    }


def missing_correlation_inputs(root: Path) -> dict[str, list[str]]:
    """Report prediction files Person 1 must supply. Does not invent values."""
    def absent(relative_paths: Iterable[str]) -> list[str]:
        return [
            relative
            for relative in relative_paths
            if not (root / relative).is_file()
        ]

    return {
        "official_validation_checkpoints": absent(CORRELATION_OFFICIAL_CHECKPOINTS),
        "rolling_validation_checkpoints": absent(CORRELATION_ROLLING_CHECKPOINTS),
        "existing_correlation_summaries": absent(CORRELATION_EXISTING_SUMMARIES),
    }


def format_missing_correlation_inputs(missing: Mapping[str, list[str]]) -> str:
    lines = [
        "Required validation prediction files/checkpoints are missing. "
        "Refusing to fabricate correlation output.",
        "Person 1 must supply:",
    ]
    for group, paths in missing.items():
        lines.append(f"{group}:")
        if paths:
            lines.extend(f"  - {path}" for path in paths)
        else:
            lines.append("  - (none missing)")
    return "\n".join(lines)


def build_leakage_registry(*, test_labels_used: bool = False) -> dict[str, Any]:
    _require_validation_only(test_labels_used, "leakage registry")
    if set(LEAKAGE_AUDIT) != set(FEATURE_KEYS):
        raise ValueError("leakage registry must cover every FEATURE_KEYS entry")
    features = {}
    for name in FEATURE_KEYS:
        row = dict(LEAKAGE_AUDIT[name])
        row["feature"] = name
        features[name] = row
    return {
        "version": LEAKAGE_REGISTRY_VERSION,
        "test_labels_used": False,
        "features": features,
    }


def leakage_rows_requiring_confirmation() -> tuple[str, ...]:
    return tuple(
        name for name in FEATURE_KEYS
        if LEAKAGE_AUDIT[name]["status"] != "safe"
    )


def leakage_manifest_sources() -> list[dict[str, Any]]:
    return [
        {
            "id": f"leakage_{feature}_v1",
            "family": FEATURE_FAMILIES[feature],
            "kind": "leakage_status",
            "path": LEAKAGE_REGISTRY_PATH,
            "pointer": ["features", feature],
            "validation_only": True,
            "applies_to": {"features": {feature: True}},
        }
        for feature in FEATURE_KEYS
    ]
