from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import FEATURE_KEYS, MODELS, validate_config
from techjam_agent.model_interface import registered_model_ids
from techjam_agent.operator_registry import MODEL_SPECS
from techjam_agent.proposals import legal_candidate_catalog

NEW_MODELS = (
    "deepfm", "dcnv2", "dcnv2_dense", "two_tower",
    "hybrid_blend", "sasrec", "multitask",
    "din", "sasrec_meta", "lightgcn", "lightgcn_hybrid", "custom",
)


def _candidate_configs(payload: Any):
    """Yield config-shaped dicts from common state/config JSON layouts."""
    if not isinstance(payload, dict):
        return

    for key in ("config", "best_config", "incumbent_config", "current_config"):
        value = payload.get(key)
        if isinstance(value, dict):
            yield value

    if (
        isinstance(payload.get("model"), str)
        and isinstance(payload.get("hyperparameters"), dict)
    ):
        yield payload

    # research_state.json can contain nested state/records.
    for value in payload.values():
        if isinstance(value, dict):
            yield from _candidate_configs(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield from _candidate_configs(item)


def _migrate_feature_keys(config: dict[str, Any]) -> dict[str, Any]:
    """
    Make an old config compatible with the CURRENT FEATURE_KEYS schema
    for this diagnostic script only.

    Missing newly-added features default to False.
    Unknown/stale feature keys are dropped.
    """
    migrated = copy.deepcopy(config)
    old = migrated.get("features")
    old = old if isinstance(old, dict) else {}

    migrated["features"] = {
        key: bool(old.get(key, False))
        for key in FEATURE_KEYS
    }
    return migrated


def load_parent() -> tuple[dict[str, Any], str]:
    """
    Prefer the canonical experiment config, then best artifact/state.

    We only need ONE valid parent to ask:
        'can legal_candidate_catalog create switches to the advanced models?'
    """
    paths = (
        ROOT / "configs/experiment.json",
        ROOT / "artifacts/best_config.json",
        ROOT / "configs/research_state.json",
    )

    errors: list[str] = []

    for path in paths:
        if not path.is_file():
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{path}: JSON read failed: {exc}")
            continue

        for index, raw in enumerate(_candidate_configs(payload), start=1):
            parent = _migrate_feature_keys(raw)
            try:
                validate_config(parent)
            except Exception as exc:
                errors.append(f"{path} candidate #{index}: {exc}")
                continue
            return parent, str(path)

    details = "\n  - ".join(errors) if errors else "no candidate files found"
    raise SystemExit(
        "Could not find a valid parent config.\n"
        "Tried configs/experiment.json, artifacts/best_config.json, "
        "and configs/research_state.json.\n"
        f"Details:\n  - {details}"
    )


def main() -> None:
    parent, source = load_parent()

    print(f"Parent source: {source}")
    print("Parent:", parent["model"], parent["training_objective"])
    print("Feature schema:", len(parent["features"]), "keys")
    print()

    print("Layer 1 - config.MODELS")
    for model in NEW_MODELS:
        print(f"  {model:10s}", "OK" if model in MODELS else "MISSING")

    print()
    print("Layer 2 - MODEL_SPECS")
    for model in NEW_MODELS:
        status = "OK" if model in MODEL_SPECS else "MISSING"
        objectives = MODEL_SPECS[model].objectives if model in MODEL_SPECS else ()
        print(f"  {model:10s} {status:7s} objectives={objectives}")

    print()
    registered = set(registered_model_ids())
    print("Layer 3 - model_interface")
    for model in NEW_MODELS:
        print(f"  {model:10s}", "OK" if model in registered else "MISSING")

    print()
    catalog = legal_candidate_catalog(parent, history=[])
    print(f"Layer 4 - proposal catalog ({len(catalog)} total legal candidates)")

    found: dict[str, list[dict[str, Any]]] = {model: [] for model in NEW_MODELS}
    for candidate in catalog:
        model = candidate["changes"].get("model")
        if model in found:
            found[model].append(candidate)

    for model in NEW_MODELS:
        rows = found[model]
        if not rows:
            print(f"  {model:10s} MISSING")
            continue

        print(f"  {model:10s} OK")
        for candidate in rows:
            print(
                "      ",
                candidate["candidate_id"],
                candidate["changes"],
                f"branch={candidate['branch']}",
                f"cost={candidate['cost']}",
            )

    missing = [
        model
        for model in NEW_MODELS
        if (
            model not in MODELS
            or model not in MODEL_SPECS
            or model not in registered
            or not found[model]
        )
    ]

    print()
    if missing:
        raise SystemExit("NOT READY: " + ", ".join(missing))

    print(f"ALL {len(NEW_MODELS)} ADVANCED MODELS ARE AVAILABLE THROUGH THE STANDARD AGENT INTERFACE.")
    print("Selection order is evidence-ranked at runtime; there is no fixed model priority.")


if __name__ == "__main__":
    main()
