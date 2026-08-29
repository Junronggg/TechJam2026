from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.deepfm import DeepFM, MultiTaskDeepFM
from techjam_agent.ensemble import blend_scores
from techjam_agent.rolling import build_rolling_splits
from techjam_agent.runner import ExperimentRunner


def metrics(runner: ExperimentRunner, users, labels, scores):
    return runner._metrics(runner.evaluate_mod.evaluate(users, labels, scores))


def component_configs(initial: dict, temporal: bool):
    fm = copy.deepcopy(initial)
    fm["model"] = "fm"
    fm["training_objective"] = "bpr"
    fm["hyperparameters"]["learning_rate"] = 0.0003
    deepfm = copy.deepcopy(initial)
    deepfm["model"] = "deepfm"
    deepfm["training_objective"] = "bce"
    deepfm["hyperparameters"]["learning_rate"] = 0.001
    for config in (fm, deepfm):
        config["features"]["user_recent_3d_activity"] = temporal
        config["features"]["item_recent_3d_exposure"] = temporal
    return fm, deepfm


def train_scores(runner, config, checkpoint):
    runner.run(config, checkpoint)
    encoded, dimension = runner._encoded_for(config)
    Xvalid, labels, users = encoded["valid"]
    hp = config["hyperparameters"]
    if config["model"] == "fm":
        model = runner.baseline.FM(
            dimension, k=hp["embedding_dim"], lr=hp["learning_rate"],
            l2=hp["l2"], seed=hp["seed"],
        )
        with np.load(checkpoint) as state:
            model.V, model.W, model.b = state["V"], state["W"], state["b"]
    else:
        model_class = (
            MultiTaskDeepFM if config["model"] == "multitask_deepfm" else DeepFM
        )
        model = model_class(
            dimension, Xvalid.shape[1], embedding_dim=hp["embedding_dim"],
            hidden_dim=hp["deepfm_hidden_dim"], learning_rate=hp["learning_rate"],
            l2=hp["l2"], seed=hp["seed"],
        )
        with np.load(checkpoint) as state:
            model.load_state_dict({name: state[name] for name in model.state_dict()})
    return users, labels, model.predict(Xvalid)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run expanding-window rolling validation")
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/rolling_validation")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    initial = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    loader = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
    all_splits = loader.data.load(str(data_dir))
    # Rolling validation must never use the official test period.  Fold 3 crosses
    # the starter kit's train/valid boundary, so combine only those two known
    # development splits and then apply the explicit date windows below.
    all_rows = list(all_splits["train"]) + list(all_splits["valid"])
    folds = build_rolling_splits(all_rows)
    results = {}
    for fold_name, splits in folds.items():
        print(f"\n=== {fold_name}: train={len(splits['train']):,} valid={len(splits['valid']):,} ===")
        runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
        runner._splits = splits
        runner._encoded = runner.data.encode(splits)
        fold_dir = output_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        fm_config, deepfm_config = component_configs(initial, temporal=False)
        users, labels, fm_scores = train_scores(runner, fm_config, fold_dir / "fm.npz")
        _, _, deepfm_scores = train_scores(runner, deepfm_config, fold_dir / "deepfm.npz")
        fm_result = metrics(runner, users, labels, fm_scores)
        ensemble_scores = blend_scores(users, fm_scores, deepfm_scores, 0.4)
        ensemble_result = metrics(runner, users, labels, ensemble_scores)

        temporal_fm, temporal_deepfm = component_configs(initial, temporal=True)
        temporal_users, temporal_labels, temporal_fm_scores = train_scores(
            runner, temporal_fm, fold_dir / "temporal_fm.npz"
        )
        _, _, temporal_deepfm_scores = train_scores(
            runner, temporal_deepfm, fold_dir / "temporal_deepfm.npz"
        )
        temporal_scores = blend_scores(
            temporal_users, temporal_fm_scores, temporal_deepfm_scores, 0.4
        )
        temporal_result = metrics(
            runner, temporal_users, temporal_labels, temporal_scores
        )
        results[fold_name] = {
            "rows": {name: len(rows) for name, rows in splits.items()},
            "fm_bpr": fm_result,
            "ensemble": ensemble_result,
            "ensemble_temporal": temporal_result,
            "ensemble_delta": ensemble_result["primary"] - fm_result["primary"],
            "temporal_delta": temporal_result["primary"] - ensemble_result["primary"],
        }
        print(json.dumps(results[fold_name], indent=2))

    aggregate = {"models": {}, "comparisons": {}}
    for model in ("fm_bpr", "ensemble", "ensemble_temporal"):
        values = [fold[model]["primary"] for fold in results.values()]
        aggregate["models"][model] = {
            "mean_primary": float(np.mean(values)),
            "std_primary": float(np.std(values, ddof=1)),
        }
        print(f"{model}: mean={np.mean(values):.6f} std={np.std(values, ddof=1):.6f}")
    for name, delta_key in (
        ("ensemble_vs_fm_bpr", "ensemble_delta"),
        ("temporal_vs_ensemble", "temporal_delta"),
    ):
        deltas = [fold[delta_key] for fold in results.values()]
        aggregate["comparisons"][name] = {
            "mean_delta": float(np.mean(deltas)),
            "wins": sum(delta > 0 for delta in deltas),
            "folds": len(deltas),
        }
    print(json.dumps({"aggregate": aggregate}, indent=2))
    (output_dir / "summary.json").write_text(
        json.dumps({"folds": results, "aggregate": aggregate}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
