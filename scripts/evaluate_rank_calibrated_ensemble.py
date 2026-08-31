"""Validate rank-calibrated FM/DeepFM fusion on official and rolling folds.

The trained models and feature set are held fixed.  The only intervention is
the score calibration used before blending, so this is a controlled ensemble
ablation rather than another model search.
"""

from __future__ import annotations

import copy
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.deepfm import DeepFM
from techjam_agent.ensemble import blend_scores
from techjam_agent.rolling import build_rolling_splits
from techjam_agent.runner import ExperimentRunner


def component_configs(initial: dict) -> tuple[dict, dict]:
    fm = copy.deepcopy(initial)
    fm["model"] = "fm"
    fm["training_objective"] = "bpr"
    fm["hyperparameters"]["learning_rate"] = 0.0003
    deepfm = copy.deepcopy(initial)
    deepfm["model"] = "deepfm"
    deepfm["training_objective"] = "bce"
    deepfm["hyperparameters"]["learning_rate"] = 0.001
    return fm, deepfm


def load_component_scores(
    runner: ExperimentRunner,
    fm_config: dict,
    deepfm_config: dict,
    fm_checkpoint: Path,
    deepfm_checkpoint: Path,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    encoded, dimension = runner._encoded_for(fm_config)
    X, labels, users = encoded["valid"]
    hp = fm_config["hyperparameters"]
    fm = runner.baseline.FM(
        dimension, k=hp["embedding_dim"], lr=hp["learning_rate"],
        l2=hp["l2"], seed=hp["seed"],
    )
    with np.load(fm_checkpoint) as state:
        fm.V, fm.W, fm.b = state["V"], state["W"], state["b"]

    encoded, dimension = runner._encoded_for(deepfm_config)
    X_deep, _, _ = encoded["valid"]
    hp = deepfm_config["hyperparameters"]
    deepfm = DeepFM(
        dimension, X_deep.shape[1], embedding_dim=hp["embedding_dim"],
        hidden_dim=hp["deepfm_hidden_dim"], learning_rate=hp["learning_rate"],
        l2=hp["l2"], seed=hp["seed"],
    )
    with np.load(deepfm_checkpoint) as state:
        deepfm.load_state_dict({name: state[name] for name in deepfm.state_dict()})
    return users, labels, fm.predict(X), deepfm.predict(X_deep)


def evaluate_blends(
    runner: ExperimentRunner,
    users,
    labels,
    fm_scores: np.ndarray,
    deepfm_scores: np.ndarray,
) -> dict[str, dict]:
    variants = {
        f"{normalization}_w{weight:g}": (normalization, weight)
        for normalization in (
            "zscore", "fm_zscore_deepfm_rank", "fm_rank_deepfm_zscore"
        )
        for weight in (0.3, 0.4, 0.5, 0.6, 0.63, 0.64, 0.65, 0.7)
    }
    result = {}
    for name, (normalization, weight) in variants.items():
        scores = blend_scores(
            users, fm_scores, deepfm_scores, weight, normalization
        )
        result[name] = runner._metrics(
            runner.evaluate_mod.evaluate(users, labels, scores)
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/rank_calibrated_ensemble")
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
    fm_config, deepfm_config = component_configs(initial)
    loader = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
    loader.prepare()

    official_users, official_labels, official_fm, official_deepfm = load_component_scores(
        loader, fm_config, deepfm_config,
        ROOT / "runs" / "ensemble_rank_calibrated2_fm.npz",
        ROOT / "runs" / "ensemble_rank_calibrated2_deepfm.npz",
    )
    official = evaluate_blends(
        loader, official_users, official_labels, official_fm, official_deepfm
    )
    print("=== official validation ===")
    print(json.dumps(official, indent=2))

    development = loader.data.load(str(data_dir))
    rows = list(development["train"]) + list(development["valid"])
    rolling = {}
    for fold_name, splits in build_rolling_splits(rows).items():
        runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
        runner._splits = splits
        runner._encoded = runner.data.encode(splits)
        users, labels, fm_scores, deepfm_scores = load_component_scores(
            runner,
            fm_config,
            deepfm_config,
            ROOT / "runs" / "rolling_sequence" / fold_name / "fm_bpr.npz",
            ROOT / "runs" / "rolling_dcnv2" / fold_name / "deepfm.npz",
        )
        rolling[fold_name] = evaluate_blends(
            runner, users, labels, fm_scores, deepfm_scores
        )
        print(f"=== {fold_name} ===")
        print(json.dumps(rolling[fold_name], indent=2))

    aggregate = {}
    variant_names = [
        name for name in next(iter(rolling.values()))
        if name != "zscore_w0.4"
    ]
    # ``:g`` formatting produces ``zscore_w0.4``; retain the historical key
    # used by existing summaries for the reference blend.
    reference_name = "zscore_w0.4"
    for name in variant_names:
        deltas = [
            fold[name]["primary"] - fold[reference_name]["primary"]
            for fold in rolling.values()
        ]
        aggregate[name] = {
            "mean_delta_vs_zscore_w04": float(np.mean(deltas)),
            "wins": int(sum(delta > 0 for delta in deltas)),
            "folds": len(deltas),
            "deltas": [float(delta) for delta in deltas],
        }
    # Keep a backwards-compatible alias for consumers of the earlier artifact.
    aggregate["zscore_w065"] = aggregate["zscore_w0.65"]
    aggregate["fm_zscore_deepfm_rank_w04"] = aggregate[
        "fm_zscore_deepfm_rank_w0.4"
    ]
    aggregate["fm_zscore_deepfm_rank_w065"] = aggregate[
        "fm_zscore_deepfm_rank_w0.65"
    ]
    payload = {
        "test_labels_used": False,
        "controlled_change": "FM z-score + DeepFM within-user rank before blend",
        "official_validation": official,
        "rolling_folds": rolling,
        "rolling_aggregate": aggregate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("=== rolling aggregate ===")
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
