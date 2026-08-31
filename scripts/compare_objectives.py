from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import apply_changes
from techjam_agent.isolated import IsolatedExperimentRunner
from techjam_agent.replication import critique_replications, summarize_objective_comparison
from techjam_agent.runner import ExperimentRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare FM+BCE and FM+BPR across seeds using validation metrics only"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--negatives-per-positive", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument(
        "--bpr-learning-rate", type=float, choices=(0.0005, 0.001, 0.002), default=0.001
    )
    parser.add_argument(
        "--bpr-embedding-dim", type=int, choices=(8, 16, 32, 64), default=16
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume", type=Path,
        help="Reuse completed seed rows from a compatible earlier comparison report.",
    )
    args = parser.parse_args()

    project = json.loads((ROOT / "configs" / "project.json").read_text(encoding="utf-8"))
    initial = json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir or project["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    run_id = datetime.now(timezone.utc).strftime("replication_%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "artifacts" / "replications" / f"{run_id}.json"
    checkpoint_dir = output.parent / run_id
    local = ExperimentRunner(
        ROOT, data_dir, ROOT / project["starter_dir"], project["official_evaluator_sha256"]
    )
    runner = IsolatedExperimentRunner(local, project["experiment_timeout_seconds"])
    rows = []
    if args.resume is not None:
        previous = json.loads(args.resume.read_text(encoding="utf-8"))
        expected = {
            "negatives_per_positive": args.negatives_per_positive,
            "bpr_learning_rate": args.bpr_learning_rate,
            "bpr_embedding_dim": args.bpr_embedding_dim,
        }
        mismatched = {
            key: (previous.get(key), value)
            for key, value in expected.items() if previous.get(key) != value
        }
        if mismatched:
            parser.error(f"resume report has incompatible BPR settings: {mismatched}")
        rows = [row for row in previous.get("raw", []) if int(row["seed"]) in args.seeds]
    completed_seeds = {int(row["seed"]) for row in rows}
    for seed in args.seeds:
        if seed in completed_seeds:
            print(f"Reusing completed seed {seed} from {args.resume}", flush=True)
            continue
        bce = apply_changes(initial, {"seed": seed}) if seed != 0 else initial
        bpr = apply_changes(bce, {
            "training_objective": "bpr",
            "negatives_per_positive": args.negatives_per_positive,
            "learning_rate": args.bpr_learning_rate,
            "embedding_dim": args.bpr_embedding_dim,
        })
        record: dict[str, object] = {"seed": seed}
        for objective, config in (("bce", bce), ("bpr", bpr)):
            print(f"\nSeed {seed} | FM+{objective.upper()}", flush=True)
            try:
                record[objective] = runner.run(
                    config, checkpoint_dir / f"seed_{seed}_{objective}.npz"
                )
            except Exception as exc:
                record[objective] = None
                record.setdefault("errors", {})[objective] = {
                    "type": type(exc).__name__, "message": str(exc),
                }
        rows.append(record)
        rows.sort(key=lambda item: int(item["seed"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"raw": rows}, indent=2) + "\n", encoding="utf-8")

    summary = summarize_objective_comparison(rows)
    payload = {
        "benchmark": project["benchmark"],
        "split": "validation",
        "negatives_per_positive": args.negatives_per_positive,
        "bpr_learning_rate": args.bpr_learning_rate,
        "bpr_embedding_dim": args.bpr_embedding_dim,
        "raw": rows,
        "summary": summary,
        "critique": critique_replications(
            summary, float(project["run_limits"]["convergence_epsilon"])
        ),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote validation-only comparison: {output}")
    print(json.dumps(payload["summary"], indent=2))
    return 0 if summary["seeds_total"] == len(args.seeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
