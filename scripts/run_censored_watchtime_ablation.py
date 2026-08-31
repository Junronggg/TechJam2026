from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.runner import ExperimentRunner


def config_for(initial: dict, objective: str | None) -> dict:
    config = copy.deepcopy(initial)
    if objective is None:
        config["model"] = "deepfm"
        config["training_objective"] = "bce"
    else:
        config["model"] = "multitask_deepfm"
        config["training_objective"] = objective
        config["hyperparameters"]["auxiliary_signals"] = "censored_watch"
        config["hyperparameters"]["auxiliary_loss_weight"] = 0.1
    config["hyperparameters"]["learning_rate"] = 0.001
    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare capped-MSE alternatives with one-sided censored watch-time"
    )
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/censored_watchtime_ablation")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    initial = json.loads((ROOT / "configs" / "experiment.json").read_text())
    runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
    configurations = {
        "deepfm_bce": config_for(initial, None),
        "censored_watch_bce": config_for(initial, "bce"),
        "censored_watch_bpr": config_for(initial, "bpr"),
    }
    results = {}
    for name, config in configurations.items():
        print(f"\n=== {name} ===", flush=True)
        results[name] = runner.run(config, output_dir / f"{name}.npz")
    baseline = float(results["deepfm_bce"]["primary"])
    results["censored_watch_bce"]["delta_vs_deepfm_bce"] = (
        float(results["censored_watch_bce"]["primary"]) - baseline
    )
    results["censored_watch_bpr"]["delta_vs_deepfm_bce"] = (
        float(results["censored_watch_bpr"]["primary"]) - baseline
    )
    results["censored_watch_bpr"]["delta_vs_censored_watch_bce"] = (
        float(results["censored_watch_bpr"]["primary"])
        - float(results["censored_watch_bce"]["primary"])
    )
    payload = {
        "selection_split": "validation only (2022-04-22 through 2022-04-28)",
        "test_labels_used": False,
        "target": "log1p(play_time) for incomplete plays; log1p(duration) lower bound "
                  "for completed plays; train-P99 scaled",
        "loss": "squared error for incomplete plays; one-sided squared hinge below the "
                "duration bound for completed plays",
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== summary ===")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
