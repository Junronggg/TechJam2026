from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.rolling import build_rolling_splits
from techjam_agent.runner import ExperimentRunner


EXPERIMENTS = {
    "fm_bpr": {},
    "fm_bpr_prior_video": {"prior_video_positive": True},
    "fm_bpr_author_recency": {"author_positive_recency": True},
}


def experiment_config(initial: dict, features: dict[str, bool]) -> dict:
    config = copy.deepcopy(initial)
    config["model"] = "fm"
    config["training_objective"] = "bpr"
    config["hyperparameters"]["learning_rate"] = 0.0003
    config["features"].update(features)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rolling validation for strict candidate-history features"
    )
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/rolling_sequence")
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
    rows = list(development["train"]) + list(development["valid"])
    folds = build_rolling_splits(rows)

    fold_results = {}
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
        fold_results[fold_name] = {}
        for name, features in EXPERIMENTS.items():
            print(f"\n--- {name} ---", flush=True)
            config = experiment_config(initial, features)
            result = runner.run(config, fold_dir / f"{name}.npz")
            fold_results[fold_name][name] = result
        baseline = fold_results[fold_name]["fm_bpr"]["primary"]
        fold_results[fold_name]["deltas"] = {
            name: result["primary"] - baseline
            for name, result in fold_results[fold_name].items()
            if name != "fm_bpr"
        }
        print(json.dumps(fold_results[fold_name], indent=2), flush=True)

    aggregate = {}
    for name in EXPERIMENTS:
        values = [fold[name]["primary"] for fold in fold_results.values()]
        aggregate[name] = {
            "mean_primary": float(np.mean(values)),
            "std_primary": float(np.std(values, ddof=1)),
        }
        if name != "fm_bpr":
            deltas = [
                fold[name]["primary"] - fold["fm_bpr"]["primary"]
                for fold in fold_results.values()
            ]
            aggregate[name].update(
                mean_delta=float(np.mean(deltas)),
                wins=sum(delta > 0 for delta in deltas),
                folds=len(deltas),
            )

    payload = {
        "test_labels_used": False,
        "folds": fold_results,
        "aggregate": aggregate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== aggregate ===")
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
