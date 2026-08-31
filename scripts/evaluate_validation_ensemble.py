from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import apply_changes
from techjam_agent.runner import ExperimentRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate cumulative FM checkpoint ensembles on validation only"
    )
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--learning-rate", type=float, choices=(0.0005, 0.001, 0.002),
                        default=0.0005)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--data-dir")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--all-subsets", action="store_true",
        help="Also evaluate every checkpoint subset with at least two members.",
    )
    args = parser.parse_args()

    project = json.loads((ROOT / "configs" / "project.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))
    config = apply_changes(config, {
        "training_objective": "bpr",
        "learning_rate": args.learning_rate,
        **{feature: True for feature in args.feature},
    })
    data_dir = Path(args.data_dir or project["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    runner = ExperimentRunner(
        ROOT, data_dir, ROOT / project["starter_dir"],
        project["official_evaluator_sha256"],
    )
    runner.prepare()
    encoded, dimension = runner._encoded_for(config)
    X, labels, users = encoded["valid"]
    cumulative = np.zeros(len(labels), dtype=np.float64)
    rows = []
    member_scores = []
    for member_count, checkpoint in enumerate(args.checkpoint, start=1):
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        hp = config["hyperparameters"]
        model = runner.baseline.FM(
            dimension, k=hp["embedding_dim"], lr=hp["learning_rate"],
            l2=hp["l2"], seed=0,
        )
        with np.load(checkpoint) as state:
            model.V, model.W, model.b = state["V"], state["W"], state["b"]
        member_score = model.predict(X)
        member_scores.append(member_score)
        cumulative += member_score
        metrics = runner._metrics(
            runner.evaluate_mod.evaluate(users, labels, cumulative / member_count)
        )
        rows.append({
            "members": member_count,
            "checkpoint": str(checkpoint),
            "metrics": metrics,
        })
        print(
            f"members={member_count} | primary={metrics['primary']:.6f} "
            f"| GAUC={metrics['GAUC']:.6f} | nDCG@5={metrics['nDCG@5']:.6f}",
            flush=True,
        )
    subsets = []
    if args.all_subsets:
        for size in range(2, len(member_scores) + 1):
            for indices in itertools.combinations(range(len(member_scores)), size):
                scores = np.mean([member_scores[index] for index in indices], axis=0)
                metrics = runner._metrics(runner.evaluate_mod.evaluate(users, labels, scores))
                subsets.append({"member_indices": list(indices), "metrics": metrics})
        subsets.sort(key=lambda item: item["metrics"]["primary"], reverse=True)
        for candidate in subsets[:5]:
            print(
                f"subset={candidate['member_indices']} | "
                f"primary={candidate['metrics']['primary']:.6f}",
                flush=True,
            )
    payload = {
        "split": "validation",
        "config": config,
        "cumulative_ensembles": rows,
        **({"subset_search": subsets} if args.all_subsets else {}),
    }
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote validation-only ensemble report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
