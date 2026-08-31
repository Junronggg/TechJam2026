from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.runner import ExperimentRunner


EXPERIMENTS = {
    "fm_bpr": {},
    "fm_bpr_prior_video_exposure": {"prior_video_exposure": True},
    "fm_bpr_author_recency": {"author_recency": True},
    "fm_bpr_prior_video_count": {"prior_video_count": True},
    "fm_bpr_previous_author_same": {"previous_author_same": True},
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
        description="Test target-free candidate exposure and author recency signals"
    )
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/candidate_history_followup")
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
    for name, features in EXPERIMENTS.items():
        print(f"\n=== {name} ===", flush=True)
        result = runner.run(
            experiment_config(initial, features), output_dir / f"{name}.npz"
        )
        results[name] = {"features": features, "metrics": result}
        print(json.dumps(results[name], indent=2), flush=True)

    baseline = results["fm_bpr"]["metrics"]["primary"]
    for result in results.values():
        result["delta_vs_fm_bpr"] = result["metrics"]["primary"] - baseline
    payload = {
        "selection_split": "validation only (2022-04-22 through 2022-04-28)",
        "test_labels_used": False,
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
