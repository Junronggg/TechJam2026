from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.rolling import build_rolling_splits
from techjam_agent.runner import ExperimentRunner


def evaluate_checkpoint(runner: ExperimentRunner, config: dict, checkpoint: Path) -> dict:
    encoded, dimension = runner._encoded_for(config)
    Xvalid, labels, users = encoded["valid"]
    hp = config["hyperparameters"]
    model = runner.baseline.FM(
        dimension,
        k=hp["embedding_dim"],
        lr=hp["learning_rate"],
        l2=hp["l2"],
        seed=hp["seed"],
    )
    with np.load(checkpoint) as state:
        model.V, model.W, model.b = state["V"], state["W"], state["b"]
    return runner._metrics(runner.evaluate_mod.evaluate(users, labels, model.predict(Xvalid)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rolling placebo test for an all-zero FM context field"
    )
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/rolling_constant_context")
    parser.add_argument("--reuse-checkpoints", action="store_true")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    initial = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    config = copy.deepcopy(initial)
    config["model"] = "fm"
    config["training_objective"] = "bpr"
    config["hyperparameters"]["learning_rate"] = 0.0003
    config["features"]["global_context"] = True

    loader = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
    development = loader.data.load(str(data_dir))
    rows = list(development["train"]) + list(development["valid"])
    baseline = json.loads(
        (ROOT / "runs/rolling_sequence/summary.json").read_text(encoding="utf-8")
    )["folds"]
    results = {}
    for fold_name, splits in build_rolling_splits(rows).items():
        print(f"\n=== {fold_name} ===", flush=True)
        runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
        runner._splits = splits
        runner._encoded = runner.data.encode(splits)
        fold_dir = output_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = fold_dir / "constant_context.npz"
        if args.reuse_checkpoints and checkpoint.is_file():
            metrics = evaluate_checkpoint(runner, config, checkpoint)
        else:
            metrics = runner.run(config, checkpoint)
        base_primary = baseline[fold_name]["fm_bpr"]["primary"]
        results[fold_name] = {
            "fm_bpr_primary": base_primary,
            "constant_context": metrics,
            "delta": metrics["primary"] - base_primary,
        }
        print(json.dumps(results[fold_name], indent=2), flush=True)

    deltas = [result["delta"] for result in results.values()]
    aggregate = {
        "mean_delta": float(np.mean(deltas)),
        "wins": sum(delta > 0 for delta in deltas),
        "folds": len(deltas),
    }
    payload = {"test_labels_used": False, "folds": results, "aggregate": aggregate}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== aggregate ===")
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
