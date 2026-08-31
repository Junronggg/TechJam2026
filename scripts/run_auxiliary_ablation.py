from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.runner import ExperimentRunner


SIGNALS = (
    "click", "like", "completion", "log_watch", "click_like",
    "click_like_completion",
)


def experiment_config(initial: dict, signals: str | None) -> dict:
    config = copy.deepcopy(initial)
    config["model"] = "deepfm" if signals is None else "multitask_deepfm"
    config["training_objective"] = "bce"
    config["hyperparameters"]["learning_rate"] = 0.001
    if signals is not None:
        config["hyperparameters"]["auxiliary_signals"] = signals
    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ablate training-only click, like, and completion targets"
    )
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/auxiliary_ablation")
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
    experiments = [("deepfm", None), *[(f"multitask_{name}", name) for name in SIGNALS]]
    results = {}
    for name, signals in experiments:
        print(f"\n=== {name} ===", flush=True)
        metrics = runner.run(
            experiment_config(initial, signals), output_dir / f"{name}.npz"
        )
        results[name] = {"signals": signals, "metrics": metrics}
        print(json.dumps(results[name], indent=2), flush=True)

    baseline = results["deepfm"]["metrics"]["primary"]
    for result in results.values():
        result["delta_vs_deepfm"] = result["metrics"]["primary"] - baseline
    payload = {
        "selection_split": "validation only (2022-04-22 through 2022-04-28)",
        "test_labels_used": False,
        "completion_definition": "min(play_time_ms, duration_ms) / duration_ms; "
        "duration<=0 is masked",
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
