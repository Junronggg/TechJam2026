from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.runner import ExperimentRunner


def config_for(initial: dict, seed: int, enabled: bool) -> dict:
    config = copy.deepcopy(initial)
    config["model"] = "fm"
    config["training_objective"] = "bpr"
    config["hyperparameters"]["learning_rate"] = 0.0003
    config["hyperparameters"]["seed"] = seed
    config["features"]["global_context"] = enabled
    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired-seed robustness check for the FM global-context field"
    )
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/global_context_multiseed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
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
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        base = runner.run(
            config_for(initial, seed, False), seed_dir / "fm_bpr.npz"
        )
        context = runner.run(
            config_for(initial, seed, True), seed_dir / "fm_bpr_global_context.npz"
        )
        results[str(seed)] = {
            "fm_bpr": base,
            "fm_bpr_global_context": context,
            "delta": context["primary"] - base["primary"],
        }
        print(json.dumps(results[str(seed)], indent=2), flush=True)

    deltas = np.asarray([item["delta"] for item in results.values()], dtype=np.float64)
    mean = float(np.mean(deltas))
    std = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
    # Two-sided 95% Student-t critical value for four predeclared paired seeds.
    t_critical = 3.182 if len(deltas) == 4 else 1.96
    half_width = t_critical * std / np.sqrt(len(deltas))
    aggregate = {
        "paired_mean_delta": mean,
        "paired_std": std,
        "approx_95pct_interval": [mean - half_width, mean + half_width],
        "wins": int(np.sum(deltas > 0)),
        "seeds": len(deltas),
    }
    payload = {
        "test_labels_used": False,
        "selection_rule": "Report all predeclared paired seeds; do not select the best seed.",
        "results": results,
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
