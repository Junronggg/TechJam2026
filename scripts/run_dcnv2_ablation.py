from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.runner import ExperimentRunner


def model_config(initial: dict, model: str) -> dict:
    config = copy.deepcopy(initial)
    config["model"] = model
    config["training_objective"] = "bce"
    config["hyperparameters"]["learning_rate"] = 0.001
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare DeepFM and low-rank DCNv2")
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/dcnv2_ablation")
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
    for model in ("deepfm", "dcnv2"):
        print(f"\n=== {model} ===", flush=True)
        result = runner.run(model_config(initial, model), output_dir / f"{model}.npz")
        results[model] = result
        print(json.dumps(result, indent=2), flush=True)
    results["dcnv2"]["delta_vs_deepfm"] = (
        results["dcnv2"]["primary"] - results["deepfm"]["primary"]
    )
    payload = {
        "selection_split": "validation only (2022-04-22 through 2022-04-28)",
        "test_labels_used": False,
        "dcnv2": {
            "cross_layers": initial["hyperparameters"]["dcn_cross_layers"],
            "low_rank": initial["hyperparameters"]["dcn_low_rank"],
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
