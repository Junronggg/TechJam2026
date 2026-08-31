from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import FEATURE_KEYS, MODELS, OBJECTIVES, apply_changes, validate_config
from techjam_agent.isolated import IsolatedExperimentRunner
from techjam_agent.runner import ExperimentRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one allow-listed train/validation experiment without test evaluation"
    )
    parser.add_argument("--model", choices=MODELS, default="fm")
    parser.add_argument("--objective", choices=OBJECTIVES, default="bpr")
    parser.add_argument(
        "--validation-metric", choices=("primary", "nDCG@5", "GAUC"), default="primary",
        help="Metric used for epoch/blend selection; promotion remains official Primary.",
    )
    parser.add_argument(
        "--blend-mode", choices=("rank", "zscore"), default="rank",
        help="Component normalization for hybrid blends.",
    )
    parser.add_argument("--seed", type=int, choices=range(5), default=0)
    parser.add_argument("--ensemble-size", type=int, choices=(1, 2, 3, 4, 5), default=1)
    parser.add_argument(
        "--ensemble-seed-set", choices=("sequential", "3,4"), default="sequential"
    )
    parser.add_argument("--negatives-per-positive", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument(
        "--negative-sampling-strategy",
        choices=("random", "same_tab", "same_author"),
        default="random",
    )
    parser.add_argument("--feature", action="append", choices=FEATURE_KEYS, default=[])
    parser.add_argument("--learning-rate", type=float, choices=(0.0002, 0.0005, 0.001, 0.002, 0.005))
    parser.add_argument("--embedding-dim", type=int, choices=(8, 16, 32, 64, 128))
    parser.add_argument("--epochs", type=int, choices=(5, 10, 20, 30, 40, 50))
    parser.add_argument("--batch-size", type=int, choices=(1024, 2048, 4096, 8192, 16384, 32768))
    parser.add_argument("--patience", type=int, choices=(2, 3, 4, 5, 7))
    parser.add_argument("--dropout", type=float, choices=(0.0, 0.1, 0.2, 0.3))
    parser.add_argument("--sequence-length", type=int, choices=(10, 20, 50))
    parser.add_argument("--hard-negative-pool-size", type=int, choices=(0, 4, 8, 16))
    parser.add_argument("--auxiliary-weight", type=float, choices=(0.05, 0.1, 0.2, 0.3))
    parser.add_argument("--graph-layers", type=int, choices=(1, 2, 3))
    parser.add_argument("--data-dir")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    project = json.loads((ROOT / "configs" / "project.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))
    changes = {
        "model": args.model,
        "training_objective": args.objective,
        "validation_metric": args.validation_metric,
        "blend_mode": args.blend_mode,
        "seed": args.seed,
        **({"ensemble_size": args.ensemble_size} if args.model == "fm_ensemble" else {}),
        **({"ensemble_seed_set": args.ensemble_seed_set}
           if args.model == "fm_ensemble" else {}),
        **({"negatives_per_positive": args.negatives_per_positive}
           if args.objective in ("bpr", "group_softmax") else {}),
        **({"negative_sampling_strategy": args.negative_sampling_strategy}
           if args.objective in ("bpr", "group_softmax") else {}),
        **({"learning_rate": args.learning_rate} if args.learning_rate is not None else {}),
        **({"embedding_dim": args.embedding_dim} if args.embedding_dim is not None else {}),
        **({"epochs": args.epochs} if args.epochs is not None else {}),
        **({"batch_size": args.batch_size} if args.batch_size is not None else {}),
        **({"patience": args.patience} if args.patience is not None else {}),
        **({"dropout": args.dropout} if args.dropout is not None else {}),
        **({"sequence_length": args.sequence_length} if args.sequence_length is not None else {}),
        **({"hard_negative_pool_size": args.hard_negative_pool_size}
           if args.hard_negative_pool_size is not None else {}),
        **({"auxiliary_weight": args.auxiliary_weight}
           if args.auxiliary_weight is not None else {}),
        **({"graph_layers": args.graph_layers} if args.graph_layers is not None else {}),
        **{feature: True for feature in args.feature},
    }
    changes = {
        key: value for key, value in changes.items()
        if value != config.get(key) and value != config.get("hyperparameters", {}).get(key)
    }
    if changes:
        config = apply_changes(config, changes)
    else:
        validate_config(config)
    data_dir = Path(args.data_dir or project["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    run_id = datetime.now(timezone.utc).strftime("candidate_%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "artifacts" / "validation-candidates" / f"{run_id}.json"
    checkpoint = output.with_suffix(".npz")
    local = ExperimentRunner(
        ROOT, data_dir, ROOT / project["starter_dir"], project["official_evaluator_sha256"]
    )
    metrics = IsolatedExperimentRunner(
        local, project["experiment_timeout_seconds"], project.get("model_timeout_seconds")
    ).run(config, checkpoint)
    payload = {"split": "validation", "config": config, "metrics": metrics}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Validation-only result: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
