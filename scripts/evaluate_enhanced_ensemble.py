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


def configs(initial: dict) -> dict[str, dict]:
    fm = copy.deepcopy(initial)
    fm["model"] = "fm"
    fm["training_objective"] = "bpr"
    fm["hyperparameters"]["learning_rate"] = 0.0003
    fm_prior = copy.deepcopy(fm)
    fm_prior["features"]["prior_video_positive"] = True

    deepfm = copy.deepcopy(initial)
    deepfm["model"] = "deepfm"
    deepfm["training_objective"] = "bce"
    deepfm["hyperparameters"]["learning_rate"] = 0.001
    multitask_like = copy.deepcopy(deepfm)
    multitask_like["model"] = "multitask_deepfm"
    multitask_like["hyperparameters"]["auxiliary_signals"] = "like"
    return {
        "fm": fm,
        "fm_prior": fm_prior,
        "deepfm": deepfm,
        "multitask_like": multitask_like,
    }


def fm_scores(runner: ExperimentRunner, config: dict, checkpoint: Path):
    encoded, dimension = runner._encoded_for(config)
    Xvalid, labels, users = encoded["valid"]
    hp = config["hyperparameters"]
    model = runner.baseline.FM(
        dimension, k=hp["embedding_dim"], lr=hp["learning_rate"],
        l2=hp["l2"], seed=hp["seed"],
    )
    with np.load(checkpoint) as state:
        model.V, model.W, model.b = state["V"], state["W"], state["b"]
    return users, labels, model.predict(Xvalid)


def deep_scores(
    runner: ExperimentRunner,
    config: dict,
    checkpoint: Path,
    multitask: bool,
):
    encoded, dimension = runner._encoded_for(config)
    Xvalid, labels, users = encoded["valid"]
    hp = config["hyperparameters"]
    model_class = MultiTaskDeepFM if multitask else DeepFM
    model = model_class(
        dimension, Xvalid.shape[1], embedding_dim=hp["embedding_dim"],
        hidden_dim=hp["deepfm_hidden_dim"], learning_rate=hp["learning_rate"],
        l2=hp["l2"], seed=hp["seed"],
        **({"auxiliary_tasks": 1} if multitask else {}),
    )
    with np.load(checkpoint) as state:
        model.load_state_dict({name: state[name] for name in model.state_dict()})
    return users, labels, model.predict(Xvalid)


def evaluate_scope(
    runner: ExperimentRunner,
    initial: dict,
    checkpoints: dict[str, Path],
) -> dict:
    cfg = configs(initial)
    users, labels, fm = fm_scores(runner, cfg["fm"], checkpoints["fm"])
    _, _, fm_prior = fm_scores(
        runner, cfg["fm_prior"], checkpoints["fm_prior"]
    )
    _, _, deepfm = deep_scores(
        runner, cfg["deepfm"], checkpoints["deepfm"], multitask=False
    )
    _, _, multitask_like = deep_scores(
        runner,
        cfg["multitask_like"],
        checkpoints["multitask_like"],
        multitask=True,
    )
    components = {
        "fm_bpr": fm,
        "fm_bpr_prior_video": fm_prior,
        "deepfm": deepfm,
        "multitask_like": multitask_like,
    }
    blends = {
        "base_ensemble": blend_scores(users, fm, deepfm, 0.4),
        "prior_video_ensemble": blend_scores(users, fm_prior, deepfm, 0.4),
        "like_ensemble": blend_scores(users, fm, multitask_like, 0.4),
        "enhanced_ensemble": blend_scores(users, fm_prior, multitask_like, 0.4),
    }
    metrics = {
        name: runner._metrics(runner.evaluate_mod.evaluate(users, labels, scores))
        for name, scores in {**components, **blends}.items()
    }
    baseline = metrics["base_ensemble"]["primary"]
    deltas = {
        name: result["primary"] - baseline
        for name, result in metrics.items()
        if name.endswith("ensemble") and name != "base_ensemble"
    }
    return {
        "metrics": metrics,
        "deltas_vs_base_ensemble": deltas,
        "prediction_correlations": {
            "fm_vs_deepfm": float(np.corrcoef(fm, deepfm)[0, 1]),
            "fm_prior_vs_multitask_like": float(
                np.corrcoef(fm_prior, multitask_like)[0, 1]
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate already-trained sequence and auxiliary checkpoints"
    )
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/enhanced_ensemble")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    initial = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    main_runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
    main_runner.prepare()
    main_result = evaluate_scope(
        main_runner,
        initial,
        {
            "fm": ROOT / "runs/sequence_ablation/fm_bpr.npz",
            "fm_prior": ROOT / "runs/sequence_ablation/fm_bpr_prior_video.npz",
            "deepfm": ROOT / "runs/auxiliary_ablation/deepfm.npz",
            "multitask_like": ROOT / "runs/auxiliary_ablation/multitask_like.npz",
        },
    )
    print("\n=== official validation ===")
    print(json.dumps(main_result, indent=2))

    development = main_runner._splits
    rows = list(development["train"]) + list(development["valid"])
    rolling_results = {}
    for fold_name, splits in build_rolling_splits(rows).items():
        runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
        runner._splits = splits
        runner._encoded = runner.data.encode(splits)
        result = evaluate_scope(
            runner,
            initial,
            {
                "fm": ROOT / f"runs/rolling_sequence/{fold_name}/fm_bpr.npz",
                "fm_prior": ROOT
                / f"runs/rolling_sequence/{fold_name}/fm_bpr_prior_video.npz",
                "deepfm": ROOT / f"runs/rolling_multitask_like/{fold_name}/deepfm.npz",
                "multitask_like": ROOT
                / f"runs/rolling_multitask_like/{fold_name}/multitask_deepfm.npz",
            },
        )
        rolling_results[fold_name] = result
        print(f"\n=== {fold_name} ===")
        print(json.dumps(result, indent=2))

    aggregate = {}
    for name in ("prior_video_ensemble", "like_ensemble", "enhanced_ensemble"):
        deltas = [
            result["deltas_vs_base_ensemble"][name]
            for result in rolling_results.values()
        ]
        aggregate[name] = {
            "mean_delta": float(np.mean(deltas)),
            "wins": sum(delta > 0 for delta in deltas),
            "folds": len(deltas),
        }
    payload = {
        "test_labels_used": False,
        "official_validation": main_result,
        "rolling_folds": rolling_results,
        "rolling_aggregate": aggregate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== rolling aggregate ===")
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
