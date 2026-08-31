from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.runner import ExperimentRunner


def config_for(initial: dict, enabled: bool) -> dict:
    config = copy.deepcopy(initial)
    config["model"] = "fm"
    config["training_objective"] = "bpr"
    config["hyperparameters"]["learning_rate"] = 0.0003
    config["features"]["global_context"] = enabled
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Ablate an explicit FM global context field")
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/global_context_ablation")
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
    for name, enabled in (("fm_bpr", False), ("fm_bpr_global_context", True)):
        print(f"\n=== {name} ===", flush=True)
        results[name] = runner.run(
            config_for(initial, enabled), output_dir / f"{name}.npz"
        )
        print(json.dumps(results[name], indent=2), flush=True)
    results["fm_bpr_global_context"]["delta_vs_fm_bpr"] = (
        results["fm_bpr_global_context"]["primary"] - results["fm_bpr"]["primary"]
    )
    payload = {"test_labels_used": False, "results": results}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== summary ===")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
