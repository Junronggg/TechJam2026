"""Emit validation-only prediction correlations without training.

If the required checkpoints or an already-written correlation summary are
missing, this producer reports exactly what Person 1 must supply and writes
nothing. It never regenerates model predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.feasibility_producers import (
    correlation_pair,
    correlation_summary_from_pairs,
    format_missing_correlation_inputs,
    missing_correlation_inputs,
    write_versioned_json,
)


def emit_from_existing_summary(path: Path) -> dict | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("test_labels_used") is True:
        raise ValueError("correlation producer refuses test_labels_used=true")
    correlations = None
    if isinstance(payload.get("feasibility_correlations"), dict):
        return payload["feasibility_correlations"]
    official = payload.get("official_validation", payload)
    if isinstance(official, dict):
        correlations = official.get("prediction_correlations")
    if not isinstance(correlations, dict):
        return None
    mapping = {
        "fm_deepfm": ("fm", "deepfm"),
        "fm_vs_deepfm": ("fm", "deepfm"),
        "fm_dcnv2": ("fm", "dcnv2"),
        "deepfm_dcnv2": ("deepfm", "dcnv2"),
    }
    pairs = []
    for key, models in mapping.items():
        if key in correlations:
            pairs.append(correlation_pair(models[0], models[1], correlations[key]))
    return correlation_summary_from_pairs(pairs) if pairs else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write validation-only correlation JSON from existing files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/prediction_correlation/summary.json",
    )
    args = parser.parse_args(argv)
    missing = missing_correlation_inputs(ROOT)
    existing = [
        ROOT / relative
        for relative in (
            "runs/dcnv2_ensemble/summary.json",
            "runs/enhanced_ensemble/summary.json",
            "runs/conditional_complementarity/summary.json",
        )
        if (ROOT / relative).is_file()
    ]
    if not existing:
        print(format_missing_correlation_inputs(missing), file=sys.stderr)
        return 2
    summary = emit_from_existing_summary(existing[0])
    if summary is None:
        print(format_missing_correlation_inputs(missing), file=sys.stderr)
        return 2
    write_versioned_json(summary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
