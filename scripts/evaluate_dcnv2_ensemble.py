from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.dcnv2 import DCNv2
from techjam_agent.deepfm import DeepFM
from techjam_agent.ensemble import within_user_zscore
from techjam_agent.rolling import build_rolling_splits
from techjam_agent.runner import ExperimentRunner


def configs(initial: dict) -> dict[str, dict]:
    fm = copy.deepcopy(initial)
    fm["model"] = "fm"
    fm["training_objective"] = "bpr"
    fm["hyperparameters"]["learning_rate"] = 0.0003
    deepfm = copy.deepcopy(initial)
    deepfm["model"] = "deepfm"
    deepfm["training_objective"] = "bce"
    deepfm["hyperparameters"]["learning_rate"] = 0.001
    dcnv2 = copy.deepcopy(deepfm)
    dcnv2["model"] = "dcnv2"
    return {"fm": fm, "deepfm": deepfm, "dcnv2": dcnv2}


def load_scores(runner, config, checkpoint):
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
    elif config["model"] == "deepfm":
        model = DeepFM(
            dimension, Xvalid.shape[1], embedding_dim=hp["embedding_dim"],
            hidden_dim=hp["deepfm_hidden_dim"], learning_rate=hp["learning_rate"],
            l2=hp["l2"], seed=hp["seed"],
        )
        with np.load(checkpoint) as state:
            model.load_state_dict({name: state[name] for name in model.state_dict()})
    else:
        model = DCNv2(
            dimension, Xvalid.shape[1], embedding_dim=hp["embedding_dim"],
            hidden_dim=hp["deepfm_hidden_dim"],
            cross_layers=hp["dcn_cross_layers"], cross_rank=hp["dcn_low_rank"],
            learning_rate=hp["learning_rate"], l2=hp["l2"], seed=hp["seed"],
        )
        with np.load(checkpoint) as state:
            model.load_state_dict({name: state[name] for name in model.state_dict()})
    return users, labels, model.predict(Xvalid)


def evaluate_scope(runner, initial, checkpoints):
    cfg = configs(initial)
    users, labels, fm = load_scores(runner, cfg["fm"], checkpoints["fm"])
    _, _, deepfm = load_scores(runner, cfg["deepfm"], checkpoints["deepfm"])
    _, _, dcnv2 = load_scores(runner, cfg["dcnv2"], checkpoints["dcnv2"])
    normalized = {
        "fm": within_user_zscore(users, fm),
        "deepfm": within_user_zscore(users, deepfm),
        "dcnv2": within_user_zscore(users, dcnv2),
    }
    scores = {
        "base_ensemble": 0.6 * normalized["fm"] + 0.4 * normalized["deepfm"],
        "dcnv2_ensemble": 0.6 * normalized["fm"] + 0.4 * normalized["dcnv2"],
        "three_model_ensemble": (
            0.6 * normalized["fm"]
            + 0.2 * normalized["deepfm"]
            + 0.2 * normalized["dcnv2"]
        ),
    }
    metrics = {
        name: runner._metrics(runner.evaluate_mod.evaluate(users, labels, values))
        for name, values in scores.items()
    }
    baseline = metrics["base_ensemble"]["primary"]
    return {
        "metrics": metrics,
        "deltas_vs_base_ensemble": {
            name: result["primary"] - baseline
            for name, result in metrics.items()
            if name != "base_ensemble"
        },
        "prediction_correlations": {
            "fm_deepfm": float(np.corrcoef(fm, deepfm)[0, 1]),
            "fm_dcnv2": float(np.corrcoef(fm, dcnv2)[0, 1]),
            "deepfm_dcnv2": float(np.corrcoef(deepfm, dcnv2)[0, 1]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check DCNv2 ensemble complementarity")
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/dcnv2_ensemble")
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
            "deepfm": ROOT / "runs/dcnv2_ablation/deepfm.npz",
            "dcnv2": ROOT / "runs/dcnv2_ablation/dcnv2.npz",
        },
    )
    print("\n=== official validation ===")
    print(json.dumps(main_result, indent=2))

    rows = list(main_runner._splits["train"]) + list(main_runner._splits["valid"])
    rolling = {}
    for fold_name, splits in build_rolling_splits(rows).items():
        runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
        runner._splits = splits
        runner._encoded = runner.data.encode(splits)
        rolling[fold_name] = evaluate_scope(
            runner,
            initial,
            {
                "fm": ROOT / f"runs/rolling_sequence/{fold_name}/fm_bpr.npz",
                "deepfm": ROOT / f"runs/rolling_dcnv2/{fold_name}/deepfm.npz",
                "dcnv2": ROOT / f"runs/rolling_dcnv2/{fold_name}/dcnv2.npz",
            },
        )
        print(f"\n=== {fold_name} ===")
        print(json.dumps(rolling[fold_name], indent=2))

    aggregate = {}
    for name in ("dcnv2_ensemble", "three_model_ensemble"):
        deltas = [result["deltas_vs_base_ensemble"][name] for result in rolling.values()]
        aggregate[name] = {
            "mean_delta": float(np.mean(deltas)),
            "wins": sum(delta > 0 for delta in deltas),
            "folds": len(deltas),
        }
    payload = {
        "test_labels_used": False,
        "official_validation": main_result,
        "rolling_folds": rolling,
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
