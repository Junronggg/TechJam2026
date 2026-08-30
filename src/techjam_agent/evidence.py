from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import FEATURE_SCHEMA_VERSION


POLICY_SCHEMA_VERSION = 1
SUPPORTED_KINDS = {
    "rolling_aggregate",
    "rolling_fold_field",
    "paired_seed",
    "placebo_margin",
    "single_delta",
}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite evidence value: {value!r}")
    return number


def _pointer(value: Any, path: list[str]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError("/".join(path))
        current = current[key]
    return current


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rolling_result(aggregate: Any, threshold: float) -> dict[str, Any]:
    if not isinstance(aggregate, dict):
        raise ValueError("rolling aggregate must be an object")
    mean_delta = _finite(aggregate["mean_delta"])
    wins, folds = int(aggregate["wins"]), int(aggregate["folds"])
    if folds < 1 or not 0 <= wins <= folds:
        raise ValueError("invalid rolling wins/folds")
    if folds >= 3 and wins == folds and mean_delta > threshold:
        signal = "positive"
    elif folds >= 3 and (wins <= folds // 2 or mean_delta < -threshold):
        signal = "negative"
    else:
        signal = "uncertain"
    return {
        "signal": signal,
        "mean_delta": mean_delta,
        "wins": wins,
        "folds": folds,
        "robust": folds >= 3,
    }


def _extract_result(payload: dict[str, Any], source: dict[str, Any],
                    threshold: float) -> dict[str, Any]:
    kind = source.get("kind")
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"unsupported evidence kind: {kind!r}")
    pointer = source.get("pointer", [])
    if not isinstance(pointer, list) or any(not isinstance(key, str) for key in pointer):
        raise ValueError("source pointer must be a list of strings")
    selected = _pointer(payload, pointer)

    if kind == "rolling_aggregate":
        return _rolling_result(selected, threshold)
    if kind == "rolling_fold_field":
        if not isinstance(selected, dict):
            raise ValueError("rolling fold source must resolve to an object")
        field = source.get("delta_field")
        if not isinstance(field, str) or not field:
            raise ValueError("rolling_fold_field requires delta_field")
        deltas = [_finite(fold[field]) for fold in selected.values()
                  if isinstance(fold, dict) and field in fold]
        if not deltas:
            raise ValueError(f"no rolling fold contains {field!r}")
        return _rolling_result({
            "mean_delta": sum(deltas) / len(deltas),
            "wins": sum(delta > 0 for delta in deltas),
            "folds": len(deltas),
        }, threshold)
    if kind == "paired_seed":
        if not isinstance(selected, dict):
            raise ValueError("paired-seed aggregate must be an object")
        interval = selected.get("approx_95pct_interval")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("paired-seed evidence requires a two-value interval")
        lower, upper = _finite(interval[0]), _finite(interval[1])
        mean_delta = _finite(selected["paired_mean_delta"])
        wins, seeds = int(selected["wins"]), int(selected["seeds"])
        signal = "positive" if lower > 0 else "negative" if upper < 0 else "uncertain"
        return {
            "signal": signal,
            "mean_delta": mean_delta,
            "wins": wins,
            "seeds": seeds,
            "interval": [lower, upper],
            "robust": seeds >= 4,
        }
    if kind in {"placebo_margin", "single_delta"}:
        delta = _finite(selected)
        if kind == "placebo_margin":
            signal = "positive" if delta > threshold else "negative"
        elif delta > threshold:
            signal = "positive"
        elif delta < -threshold:
            signal = "negative"
        else:
            signal = "noise"
        return {
            "signal": signal,
            "delta": delta,
            "robust": kind == "placebo_margin",
        }
    raise AssertionError("unreachable evidence kind")


def collect_artifact_evidence(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Load explicitly routed validation artifacts without parsing prose or test metrics."""
    if manifest.get("version") != POLICY_SCHEMA_VERSION:
        raise ValueError(f"evidence manifest version must be {POLICY_SCHEMA_VERSION}")
    threshold = _finite(manifest.get("noise_threshold", 0.0002))
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ValueError("evidence manifest sources must be a list")
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("each evidence source must be an object")
        source_id, family, relative = (
            source.get("id"), source.get("family"), source.get("path")
        )
        if not all(isinstance(value, str) and value for value in
                   (source_id, family, relative)):
            raise ValueError("evidence source requires non-empty id, family and path")
        if source_id in seen_ids:
            raise ValueError(f"duplicate evidence source id: {source_id}")
        seen_ids.add(source_id)
        if source.get("validation_only") is not True:
            raise ValueError(f"source {source_id} is not declared validation-only")
        path = root / relative
        payload = _load_object(path)
        if payload.get("test_labels_used") is True:
            raise ValueError(f"source {source_id} used test labels")
        applies_to = source.get("applies_to", {})
        if not isinstance(applies_to, dict):
            raise ValueError("applies_to must be an object")
        feature_scope = applies_to.get("features", {})
        hyperparameter_scope = applies_to.get("hyperparameters", {})
        if not isinstance(feature_scope, dict) or not isinstance(
            hyperparameter_scope, dict
        ):
            raise ValueError("applies_to features/hyperparameters must be objects")
        result = _extract_result(payload, source, threshold)
        records.append({
            "source_id": source_id,
            "family": family,
            "kind": source["kind"],
            "path": relative,
            "sha256": _sha256(path),
            "applies_to": {
                "task": applies_to.get("task", manifest.get("task", "long_view")),
                "feature_schema": applies_to.get(
                    "feature_schema", manifest.get(
                        "feature_schema", FEATURE_SCHEMA_VERSION
                    )
                ),
                "models": sorted(set(applies_to.get("models", []))),
                "training_objectives": sorted(set(
                    applies_to.get("training_objectives", [])
                )),
                "features": dict(sorted(feature_scope.items())),
                "hyperparameters": dict(sorted(hyperparameter_scope.items())),
            },
            "result": result,
        })
    return records


def _policy_for_records(records: list[dict[str, Any]]) -> tuple[str, str, str, float]:
    signals = [record["result"]["signal"] for record in records]
    robust_negative = any(
        record["result"].get("robust") and record["result"]["signal"] == "negative"
        for record in records
    )
    robust_positive = any(
        record["result"].get("robust") and record["result"]["signal"] == "positive"
        for record in records
    )
    negative = signals.count("negative")
    positive = signals.count("positive")
    noise_or_uncertain = signals.count("noise") + signals.count("uncertain")

    if robust_negative or negative >= 2:
        return "stop_direction", "REJECTED", "NOT_ELIGIBLE", 0.90
    if robust_positive and noise_or_uncertain:
        return "gather_evidence", "UNCERTAIN", "ELIGIBLE", 0.70
    if robust_positive and positive > 0:
        return "exploit_with_confirmation", "VALIDATED", "ELIGIBLE", 0.82
    if positive > 0:
        return "gather_evidence", "PROMISING", "ELIGIBLE", 0.55
    if negative > 0 or noise_or_uncertain >= 2:
        return "stop_direction", "REJECTED", "NOT_ELIGIBLE", 0.65
    return "gather_evidence", "INSUFFICIENT", "RESEARCH_ONLY", 0.40


def build_generated_family_policies(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    records = collect_artifact_evidence(root, manifest)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    contexts: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        context_key = json.dumps(record["applies_to"], sort_keys=True)
        key = (record["family"], context_key)
        grouped[key].append(record)
        contexts[key] = record["applies_to"]

    policies = []
    for key in sorted(grouped):
        family, _ = key
        family_records = grouped[key]
        policy, scientific, competition, confidence = _policy_for_records(family_records)
        provenance = [{
            "source_id": record["source_id"],
            "kind": record["kind"],
            "path": record["path"],
            "sha256": record["sha256"],
            "result": record["result"],
        } for record in family_records]
        identity = json.dumps({
            "family": family,
            "applies_to": contexts[key],
            "sources": [(row["source_id"], row["sha256"]) for row in provenance],
        }, sort_keys=True, separators=(",", ":"))
        policies.append({
            "policy_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
            "family": family,
            "policy": policy,
            "scientific_verdict": scientific,
            "competition_status": competition,
            "confidence": confidence,
            "applies_to": contexts[key],
            "expires_if": [
                "task_changed",
                "model_changed_outside_scope",
                "feature_schema_changed",
                "source_artifact_changed",
            ],
            "created_from": provenance,
        })
    return {
        "version": POLICY_SCHEMA_VERSION,
        "source": "generated_from_validation_artifacts",
        "test_metrics_included": False,
        "task": manifest.get("task", "long_view"),
        "feature_schema": manifest.get("feature_schema", FEATURE_SCHEMA_VERSION),
        "family_policies": policies,
    }


def merge_generated_policies(base: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    """Generated artifact evidence overrides legacy manual policy for the same scope."""
    merged = dict(base)
    generated_rows = generated.get("family_policies", [])
    manual_rows = base.get("family_policies", [])
    if not isinstance(generated_rows, list) or not isinstance(manual_rows, list):
        raise ValueError("family_policies must be lists")
    generated_families = {
        row.get("family") for row in generated_rows if isinstance(row, dict)
    }
    merged["family_policies"] = [
        row for row in manual_rows
        if isinstance(row, dict) and row.get("family") not in generated_families
    ] + list(generated_rows)
    merged["generated_policy_metadata"] = {
        key: generated.get(key) for key in (
            "version", "source", "test_metrics_included", "task", "feature_schema"
        )
    }
    return merged
