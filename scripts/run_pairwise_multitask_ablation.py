from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.runner import ExperimentRunner  # noqa: E402


def experiment_config(
    initial: dict,
    objective: str,
    *,
    embedding_dim: int = 16,
    auxiliary_weight: float = 0.1,
) -> dict:
    config = copy.deepcopy(initial)
    config["model"] = "multitask_deepfm"
    config["training_objective"] = objective
    config["hyperparameters"]["learning_rate"] = 0.001
    config["hyperparameters"]["auxiliary_signals"] = "like"
    config["hyperparameters"]["auxiliary_loss_weight"] = auxiliary_weight
    config["hyperparameters"]["embedding_dim"] = embedding_dim
    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare pointwise and pairwise long-view objectives in the same "
            "like-only Multi-task DeepFM."
        )
    )
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/pairwise_multitask_ablation")
    parser.add_argument(
        "--reported-config-only",
        action="store_true",
        help="Only run the externally reported k=32, auxiliary_weight=0.3 variant.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    initial = json.loads(
        (ROOT / "configs" / "experiment.json").read_text(encoding="utf-8")
    )
    runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
    summary_path = output_dir / "summary.json"
    results = {}
    if args.reported_config_only and summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        results.update(previous.get("results", {}))
    experiments = (
        [("multitask_like_bpr_reported_k32_aux03", "bpr", 32, 0.3)]
        if args.reported_config_only
        else [
            ("multitask_like_bce", "bce", 16, 0.1),
            ("multitask_like_bpr", "bpr", 16, 0.1),
        ]
    )
    for name, objective, embedding_dim, auxiliary_weight in experiments:
        print(f"\n=== {name} ===", flush=True)
        config = experiment_config(
            initial,
            objective,
            embedding_dim=embedding_dim,
            auxiliary_weight=auxiliary_weight,
        )
        metrics = runner.run(config, output_dir / f"{name}.npz")
        results[name] = {"config": config, "metrics": metrics}
        print(json.dumps(results[name], indent=2), flush=True)

    pointwise = results["multitask_like_bce"]["metrics"]["primary"]
    pairwise = results["multitask_like_bpr"]["metrics"]["primary"]
    payload = {
        "selection_split": "validation only (2022-04-22 through 2022-04-28)",
        "test_labels_used": False,
        "controlled_change": "long_view objective: pointwise BCE -> within-user BPR",
        "fixed": {
            "model": "MultiTaskDeepFM",
            "auxiliary_signal": "like",
            "auxiliary_loss": "BCE",
            "auxiliary_weight": 0.1,
            "embedding_dim": initial["hyperparameters"]["embedding_dim"],
            "learning_rate": 0.001,
            "seed": initial["hyperparameters"]["seed"],
        },
        "results": results,
        "pairwise_delta_vs_pointwise": pairwise - pointwise,
    }
    reported = results.get("multitask_like_bpr_reported_k32_aux03")
    if reported is not None:
        payload["reported_config_delta_vs_pointwise"] = (
            reported["metrics"]["primary"] - pointwise
        )
    summary_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== summary ===")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
