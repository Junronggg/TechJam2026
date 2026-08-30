from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.runner import ExperimentRunner


def config_for(initial: dict, model: str, epochs: int, sequence_length: int) -> dict:
    config = copy.deepcopy(initial)
    config["model"] = model
    config["training_objective"] = "bce"
    config["hyperparameters"]["learning_rate"] = 0.001
    config["hyperparameters"]["epochs"] = epochs
    config["hyperparameters"]["sequence_length"] = sequence_length
    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cheap controlled test of leakage-safe last-K sequence attention"
    )
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/lightweight_sequence_ablation")
    parser.add_argument("--epochs", type=int, choices=(10, 20, 30, 40), default=10)
    parser.add_argument("--sequence-length", type=int, choices=(16, 32), default=16)
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
    for name, model in (("deepfm", "deepfm"), ("sequence_deepfm", "sequence_deepfm")):
        print(f"\n=== {name} ===", flush=True)
        results[name] = runner.run(
            config_for(initial, model, args.epochs, args.sequence_length),
            output_dir / f"{name}.npz",
        )
        print(json.dumps(results[name], indent=2), flush=True)
    results["sequence_deepfm"]["delta_vs_deepfm"] = (
        results["sequence_deepfm"]["primary"] - results["deepfm"]["primary"]
    )
    payload = {
        "test_labels_used": False,
        "controlled_variables": (
            "Same five base fields, BCE, seed, embedding/hidden dimensions, learning "
            "rate, epoch budget, patience, split, and official evaluator."
        ),
        "sequence": {
            "max_length": args.sequence_length,
            "fields": ["video_id", "author_id", "behavior", "time_gap", "position"],
            "same_timestamp_interaction": False,
            "validation_or_test_labels_in_history": False,
            "attention": "single-head candidate-conditioned pooling",
        },
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
