"""Run extra train-only chronological holdouts as a shadow test.

These folds are carved out of the official training period.  They are not the
organizer validation/test split and are used only to check whether a blend
weight survives an additional future-like holdout without touching test labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_rolling_validation import component_configs, metrics, train_scores
from techjam_agent.ensemble import blend_scores
from techjam_agent.runner import ExperimentRunner


# All dates come from the official train period (20220408-20220421).  The
# windows are disjoint and strictly chronological; official validation/test is
# never included in these shadow folds.
SHADOW_FOLDS = (
    ("shadow_1", 20220408, 20220411, 20220412, 20220413),
    ("shadow_2", 20220408, 20220413, 20220414, 20220415),
    ("shadow_3", 20220408, 20220415, 20220416, 20220417),
    ("shadow_4", 20220408, 20220417, 20220418, 20220420),
)


def build_shadow_splits(rows: list[tuple]) -> dict[str, dict[str, list[tuple]]]:
    folds: dict[str, dict[str, list[tuple]]] = {}
    for name, train_start, train_end, valid_start, valid_end in SHADOW_FOLDS:
        train = [row for row in rows if train_start <= int(row[0]) <= train_end]
        valid = [row for row in rows if valid_start <= int(row[0]) <= valid_end]
        if not train or not valid:
            raise ValueError(f"{name} is empty: train={len(train)}, valid={len(valid)}")
        if max(int(row[0]) for row in train) >= min(int(row[0]) for row in valid):
            raise ValueError(f"{name} does not preserve temporal order")
        folds[name] = {"train": train, "valid": valid}
    return folds


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a candidate on train-only shadow folds")
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/shadow_validation")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    initial = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    loader = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
    development = loader.data.load(str(data_dir))
    # Deliberately use only the official train split, not official valid/test.
    folds = build_shadow_splits(list(development["train"]))
    results: dict[str, dict] = {}
    for fold_name, splits in folds.items():
        print(
            f"\n=== {fold_name}: train={len(splits['train']):,} "
            f"valid={len(splits['valid']):,} ===",
            flush=True,
        )
        runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
        runner._splits = splits
        runner._encoded = runner.data.encode(splits)
        fold_dir = output_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        fm_config, deepfm_config = component_configs(initial, temporal=False)
        users, labels, fm_scores = train_scores(
            runner, fm_config, fold_dir / "fm_bpr.npz"
        )
        _, _, deepfm_scores = train_scores(
            runner, deepfm_config, fold_dir / "deepfm.npz"
        )
        reference = metrics(
            runner, users, labels,
            blend_scores(users, fm_scores, deepfm_scores, 0.4, "zscore"),
        )
        candidate = metrics(
            runner, users, labels,
            blend_scores(
                users, fm_scores, deepfm_scores, 0.63,
                "fm_zscore_deepfm_rank",
            ),
        )
        results[fold_name] = {
            "rows": {name: len(values) for name, values in splits.items()},
            "reference_weight_0.4": reference,
            "candidate_weight_0.63": candidate,
            "delta": float(candidate["primary"] - reference["primary"]),
        }
        print(json.dumps(results[fold_name], indent=2), flush=True)

    deltas = [row["delta"] for row in results.values()]
    aggregate = {
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        "wins": int(sum(delta > 0 for delta in deltas)),
        "folds": len(deltas),
        "deltas": [float(delta) for delta in deltas],
    }
    payload = {
        "test_labels_used": False,
        "official_validation_or_test_used": False,
        "description": "Additional chronological holdouts carved only from official train split.",
        "folds": results,
        "aggregate": aggregate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== shadow aggregate ===")
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
