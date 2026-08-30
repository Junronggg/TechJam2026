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


def model_config(initial: dict, model: str) -> dict:
    config = copy.deepcopy(initial)
    config["model"] = model
    config["training_objective"] = "bce"
    config["hyperparameters"]["learning_rate"] = 0.001
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Rolling validation for DCNv2")
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/rolling_dcnv2")
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
    results = {}
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
        fold_result = {}
        for model in ("deepfm", "dcnv2"):
            print(f"\n--- {model} ---", flush=True)
            fold_result[model] = runner.run(
                model_config(initial, model), fold_dir / f"{model}.npz"
            )
        fold_result["delta"] = (
            fold_result["dcnv2"]["primary"] - fold_result["deepfm"]["primary"]
        )
        results[fold_name] = fold_result
        print(json.dumps(fold_result, indent=2), flush=True)

    deltas = [result["delta"] for result in results.values()]
    aggregate = {
        "mean_delta": float(np.mean(deltas)),
        "wins": sum(delta > 0 for delta in deltas),
        "folds": len(deltas),
    }
    payload = {"test_labels_used": False, "folds": results, "aggregate": aggregate}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== aggregate ===")
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
