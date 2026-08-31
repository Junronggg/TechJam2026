from __future__ import annotations

import copy
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


def configs(initial: dict) -> tuple[dict, dict, dict]:
    fm = copy.deepcopy(initial)
    fm["model"] = "fm"
    fm["training_objective"] = "bpr"
    fm["hyperparameters"]["learning_rate"] = 0.0003
    context = copy.deepcopy(fm)
    context["features"]["global_context"] = True
    deepfm = copy.deepcopy(initial)
    deepfm["model"] = "deepfm"
    deepfm["training_objective"] = "bce"
    deepfm["hyperparameters"]["learning_rate"] = 0.001
    return fm, context, deepfm


def fm_scores(runner, config, checkpoint):
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


def deepfm_scores(runner, config, checkpoint):
    encoded, dimension = runner._encoded_for(config)
    Xvalid, labels, users = encoded["valid"]
    hp = config["hyperparameters"]
    model = DeepFM(
        dimension, Xvalid.shape[1], embedding_dim=hp["embedding_dim"],
        hidden_dim=hp["deepfm_hidden_dim"], learning_rate=hp["learning_rate"],
        l2=hp["l2"], seed=hp["seed"],
    )
    with np.load(checkpoint) as state:
        model.load_state_dict({name: state[name] for name in model.state_dict()})
    return users, labels, model.predict(Xvalid)


def metrics(runner, users, labels, scores):
    return runner._metrics(runner.evaluate_mod.evaluate(users, labels, scores))


def evaluate_scope(runner, fm_config, context_config, deepfm_config, checkpoints):
    users, labels, fm = fm_scores(runner, fm_config, checkpoints["fm"])
    _, _, context = fm_scores(runner, context_config, checkpoints["context"])
    _, _, deepfm = deepfm_scores(runner, deepfm_config, checkpoints["deepfm"])
    base = blend_scores(users, fm, deepfm, 0.4)
    enhanced = blend_scores(users, context, deepfm, 0.4)
    base_metrics = metrics(runner, users, labels, base)
    enhanced_metrics = metrics(runner, users, labels, enhanced)
    return {
        "base_ensemble": base_metrics,
        "global_context_ensemble": enhanced_metrics,
        "delta": enhanced_metrics["primary"] - base_metrics["primary"],
        "fm_deepfm_correlation": float(np.corrcoef(fm, deepfm)[0, 1]),
        "context_deepfm_correlation": float(np.corrcoef(context, deepfm)[0, 1]),
    }


def main() -> int:
    data_dir = ROOT / "data/KuaiRand-Pure/data"
    output_dir = ROOT / "runs/global_context_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    initial = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    fm_config, context_config, deepfm_config = configs(initial)

    runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
    runner.prepare()
    official = evaluate_scope(
        runner,
        fm_config,
        context_config,
        deepfm_config,
        {
            "fm": ROOT / "runs/global_context_ablation/fm_bpr.npz",
            "context": ROOT / "runs/global_context_ablation/fm_bpr_global_context.npz",
            "deepfm": ROOT / "runs/dcnv2_ablation/deepfm.npz",
        },
    )
    print("\n=== official validation ===")
    print(json.dumps(official, indent=2))

    rows = list(runner._splits["train"]) + list(runner._splits["valid"])
    rolling = {}
    for fold_name, splits in build_rolling_splits(rows).items():
        fold_runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
        fold_runner._splits = splits
        fold_runner._encoded = fold_runner.data.encode(splits)
        rolling[fold_name] = evaluate_scope(
            fold_runner,
            fm_config,
            context_config,
            deepfm_config,
            {
                "fm": ROOT / f"runs/rolling_sequence/{fold_name}/fm_bpr.npz",
                "context": ROOT
                / f"runs/rolling_constant_context/{fold_name}/constant_context.npz",
                "deepfm": ROOT / f"runs/rolling_dcnv2/{fold_name}/deepfm.npz",
            },
        )
        print(f"\n=== {fold_name} ===")
        print(json.dumps(rolling[fold_name], indent=2))

    deltas = [result["delta"] for result in rolling.values()]
    aggregate = {
        "mean_delta": float(np.mean(deltas)),
        "wins": sum(delta > 0 for delta in deltas),
        "folds": len(deltas),
    }
    payload = {
        "test_labels_used": False,
        "official_validation": official,
        "rolling_folds": rolling,
        "rolling_aggregate": aggregate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== aggregate ===")
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
