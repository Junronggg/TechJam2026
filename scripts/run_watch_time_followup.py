from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.runner import ExperimentRunner


def config_for(initial: dict, multitask: bool) -> dict:
    config = copy.deepcopy(initial)
    config["model"] = "multitask_deepfm" if multitask else "deepfm"
    config["training_objective"] = "bce"
    config["hyperparameters"]["learning_rate"] = 0.001
    if multitask:
        config["hyperparameters"]["auxiliary_signals"] = "log_watch"
    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare DeepFM with train-P99-scaled capped log-watch regression"
    )
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/watch_time_followup")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    initial = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
    results = {}
    for name, multitask in (("deepfm", False), ("multitask_log_watch", True)):
        print(f"\n=== {name} ===", flush=True)
        results[name] = runner.run(
            config_for(initial, multitask), output_dir / f"{name}.npz"
        )
        print(json.dumps(results[name], indent=2), flush=True)
    results["multitask_log_watch"]["delta_vs_deepfm"] = (
        results["multitask_log_watch"]["primary"]
        - results["deepfm"]["primary"]
    )
    payload = {
        "selection_split": "validation only (2022-04-22 through 2022-04-28)",
        "test_labels_used": False,
        "target": "log1p(min(play_time_ms, duration_ms)), scaled by train P99; "
        "duration<=0 masked",
        "loss": "sigmoid MSE auxiliary head",
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
