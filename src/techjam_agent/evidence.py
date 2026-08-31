from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import FEATURE_SCHEMA_VERSION


POLICY_SCHEMA_VERSION = 1
FEASIBILITY_SCHEMA_VERSION = 1
POLICY_KINDS = {
    "rolling_aggregate",
    "rolling_fold_field",
    "paired_seed",
    "placebo_margin",
    "single_delta",
}
FEASIBILITY_KINDS = {
    "feature_coverage",
    "prediction_correlation",
    "family_runtime",
    "leakage_status",
}
SUPPORTED_KINDS = POLICY_KINDS | FEASIBILITY_KINDS
MARKDOWN_SUFFIXES = {".md"}
DEFAULT_FEASIBILITY_THRESHOLDS = {
    "low_coverage": 0.01,
    "high_correlation": 0.99,
    "high_correlation_effect": "soft_stop",
    "runtime_reference_seconds": 60.0,
    "runtime_cost_min": 0.05,
    "runtime_cost_max": 2.0,
}
HIGH_CORRELATION_EFFECTS = frozenset({"hard_block", "soft_stop"})


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
    if kind == "feature_coverage":
        return _feature_coverage_result(selected)
    if kind == "prediction_correlation":
        return _prediction_correlation_result(selected, source)
    if kind == "family_runtime":
        return _family_runtime_result(selected, source)
    if kind == "leakage_status":
        return _leakage_status_result(selected)
    raise AssertionError("unreachable evidence kind")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    number = int(value)
    if number < 0:
        raise ValueError("counts cannot be negative")
    return number


def _feature_coverage_result(selected: Any) -> dict[str, Any]:
    if isinstance(selected, dict):
        coverage = _finite(selected["coverage"])
        eligible = _optional_int(selected.get("eligible_rows"))
        total = _optional_int(selected.get("total_rows"))
    else:
        coverage = _finite(selected)
        eligible = None
        total = None
    if coverage < 0:
        raise ValueError("coverage cannot be negative")
    return {
        "kind": "feature_coverage",
        "coverage": coverage,
        "eligible_rows": eligible,
        "total_rows": total,
    }


def _prediction_correlation_result(selected: Any, source: dict[str, Any]) -> dict[str, Any]:
    if isinstance(selected, dict):
        correlation = _finite(selected["correlation"])
        models = selected.get("models", source.get("models"))
        split = selected.get("split", "validation")
    else:
        correlation = _finite(selected)
        models = source.get("models")
        split = "validation"
    if isinstance(split, str) and split.strip().lower() in {"test", "final_test"}:
        raise ValueError("prediction correlation must use validation predictions")
    if not isinstance(models, list) or len(models) != 2:
        raise ValueError("prediction_correlation requires exactly two models")
    if any(not isinstance(name, str) or not name for name in models):
        raise ValueError("prediction_correlation models must be non-empty strings")
    if not -1.0 <= correlation <= 1.0:
        raise ValueError("correlation must be in [-1, 1]")
    return {
        "kind": "prediction_correlation",
        "correlation": correlation,
        "models": sorted(models),
        "split": split if isinstance(split, str) else "validation",
    }


def _collect_runtimes(node: Any, runtime_key: str, model: str | None) -> list[float]:
    found: list[float] = []
    if isinstance(node, dict):
        if model is None:
            value = node.get(runtime_key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                found.append(_finite(value))
        elif model in node and isinstance(node[model], dict):
            value = node[model].get(runtime_key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                found.append(_finite(value))
        for child in node.values():
            found.extend(_collect_runtimes(child, runtime_key, model))
    elif isinstance(node, list):
        for child in node:
            found.extend(_collect_runtimes(child, runtime_key, model))
    return found


def _family_runtime_result(selected: Any, source: dict[str, Any]) -> dict[str, Any]:
    runtime_key = source.get("runtime_key", "runtime_seconds")
    if not isinstance(runtime_key, str) or not runtime_key:
        raise ValueError("family_runtime requires runtime_key")
    model = source.get("runtime_model")
    if model is not None and (not isinstance(model, str) or not model):
        raise ValueError("runtime_model must be a non-empty string")
    if isinstance(selected, (int, float)) and not isinstance(selected, bool):
        runtimes = [_finite(selected)]
    else:
        runtimes = _collect_runtimes(selected, runtime_key, model)
    if not runtimes:
        raise ValueError("family_runtime found no runtime_seconds")
    runtimes = sorted(runtimes)
    return {
        "kind": "family_runtime",
        "runtime_seconds": runtimes,
        "median_runtime_seconds": float(statistics.median(runtimes)),
        "observations": len(runtimes),
    }


def _leakage_status_result(selected: Any) -> dict[str, Any]:
    if not isinstance(selected, dict):
        raise ValueError("leakage_status must resolve to an object")
    status = selected.get("status")
    leakage_safe = selected.get("leakage_safe")
    strict_past = selected.get("strict_past")
    if status not in (None, "safe", "unsafe", "uncertain"):
        raise ValueError("leakage status must be safe, unsafe, or uncertain")
    if status is None:
        if leakage_safe is False or selected.get("causal") is False:
            status = "unsafe"
        elif leakage_safe is True and strict_past is True:
            status = "safe"
        else:
            status = "uncertain"
    return {
        "kind": "leakage_status",
        "status": status,
        "leakage_safe": leakage_safe if isinstance(leakage_safe, bool) else None,
        "strict_past": strict_past if isinstance(strict_past, bool) else None,
    }


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
        if Path(relative).suffix.lower() in MARKDOWN_SUFFIXES:
            raise ValueError(
                f"source {source_id} uses markdown; structured artifacts only"
            )
        path = root / relative
        if not path.is_file():
            if source.get("optional") is True:
                continue
            raise FileNotFoundError(f"evidence source {source_id} is missing: {relative}")
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


def _normalized_thresholds(raw: Any) -> dict[str, Any]:
    thresholds = dict(DEFAULT_FEASIBILITY_THRESHOLDS)
    if raw is None:
        return thresholds
    if not isinstance(raw, dict):
        raise ValueError("feasibility_thresholds must be an object")
    for key in DEFAULT_FEASIBILITY_THRESHOLDS:
        if key not in raw:
            continue
        value = raw[key]
        if key == "high_correlation_effect":
            if value not in HIGH_CORRELATION_EFFECTS:
                raise ValueError(
                    "high_correlation_effect must be hard_block or soft_stop"
                )
            thresholds[key] = value
            continue
        number = _finite(value)
        if key != "high_correlation" and number <= 0:
            raise ValueError(f"{key} must be positive")
        thresholds[key] = number
    if thresholds["runtime_cost_min"] > thresholds["runtime_cost_max"]:
        raise ValueError("runtime_cost_min cannot exceed runtime_cost_max")
    return thresholds


def build_feasibility_evidence(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Build scoped cheap-feasibility records from the same validation-only manifest."""
    schema = manifest.get("feasibility_schema_version", FEASIBILITY_SCHEMA_VERSION)
    if schema != FEASIBILITY_SCHEMA_VERSION:
        raise ValueError(
            f"feasibility_schema_version must be {FEASIBILITY_SCHEMA_VERSION}"
        )
    records = [
        record for record in collect_artifact_evidence(root, manifest)
        if record["kind"] in FEASIBILITY_KINDS
    ]
    return {
        "version": FEASIBILITY_SCHEMA_VERSION,
        "source": "generated_from_validation_artifacts",
        "test_metrics_included": False,
        "thresholds": _normalized_thresholds(manifest.get("feasibility_thresholds")),
        "records": records,
    }


def attach_feasibility_evidence(
    prior: dict[str, Any],
    feasibility: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(prior)
    merged["feasibility"] = feasibility
    return merged


def feasibility_from_prior(prior_evidence: dict[str, Any] | None) -> dict[str, Any]:
    """Read attached feasibility evidence; missing records are not treated as safe."""
    empty = {
        "version": FEASIBILITY_SCHEMA_VERSION,
        "thresholds": dict(DEFAULT_FEASIBILITY_THRESHOLDS),
        "records": [],
    }
    if not isinstance(prior_evidence, dict):
        return empty
    bundle = prior_evidence.get("feasibility")
    if not isinstance(bundle, dict):
        return empty
    records = bundle.get("records")
    return {
        "version": bundle.get("version", FEASIBILITY_SCHEMA_VERSION),
        "thresholds": _normalized_thresholds(bundle.get("thresholds")),
        "records": records if isinstance(records, list) else [],
    }


def build_generated_family_policies(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    records = [
        record for record in collect_artifact_evidence(root, manifest)
        if record["kind"] in POLICY_KINDS
    ]
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
