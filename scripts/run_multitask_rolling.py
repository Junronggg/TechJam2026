from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_rolling_validation import metrics, train_scores
from techjam_agent.rolling import build_rolling_splits
from techjam_agent.runner import ExperimentRunner


def experiment_configs(initial: dict) -> tuple[dict, dict]:
    deepfm = copy.deepcopy(initial)
    deepfm["model"] = "deepfm"
    deepfm["training_objective"] = "bce"
    deepfm["hyperparameters"]["learning_rate"] = 0.001
    multitask = copy.deepcopy(deepfm)
    multitask["model"] = "multitask_deepfm"
    multitask["hyperparameters"]["auxiliary_loss_weight"] = 0.1
    return deepfm, multitask


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare DeepFM and multi-feedback DeepFM on rolling folds"
    )
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/rolling_multitask")
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
        print(f"\n=== {fold_name}: train={len(splits['train']):,} valid={len(splits['valid']):,} ===")
        runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
        runner._splits = splits
        runner._encoded = runner.data.encode(splits)
        fold_dir = output_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        deepfm_config, multitask_config = experiment_configs(initial)
        users, labels, deepfm_scores = train_scores(
            runner, deepfm_config, fold_dir / "deepfm.npz"
        )
        _, _, multitask_scores = train_scores(
            runner, multitask_config, fold_dir / "multitask_deepfm.npz"
        )
        deepfm_result = metrics(runner, users, labels, deepfm_scores)
        multitask_result = metrics(runner, users, labels, multitask_scores)
        results[fold_name] = {
            "rows": {name: len(values) for name, values in splits.items()},
            "deepfm": deepfm_result,
            "multitask_deepfm": multitask_result,
            "delta": multitask_result["primary"] - deepfm_result["primary"],
        }
        print(json.dumps(results[fold_name], indent=2))

    deltas = [result["delta"] for result in results.values()]
    aggregate = {
        "mean_delta": float(np.mean(deltas)),
        "wins": sum(delta > 0 for delta in deltas),
        "folds": len(deltas),
    }
    payload = {"folds": results, "aggregate": aggregate}
    print(json.dumps({"aggregate": aggregate}, indent=2))
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
