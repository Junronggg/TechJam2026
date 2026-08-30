from __future__ import annotations

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
from techjam_agent.research_diagnostics import strict_history_lengths
from techjam_agent.rolling import build_rolling_splits
from techjam_agent.runner import ExperimentRunner
from techjam_agent.sequence_features import align_event_times


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


def load_scores(runner: ExperimentRunner, config: dict, checkpoint: Path):
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
    return np.asarray(users, dtype=object), labels, model.predict(Xvalid)


def evaluate_scope(
    runner: ExperimentRunner,
    initial: dict,
    checkpoints: dict[str, Path],
) -> dict:
    config = configs(initial)
    users, labels, fm = load_scores(runner, config["fm"], checkpoints["fm"])
    _, _, deepfm = load_scores(runner, config["deepfm"], checkpoints["deepfm"])
    _, _, dcnv2 = load_scores(runner, config["dcnv2"], checkpoints["dcnv2"])
    champion = 0.6 * within_user_zscore(users, fm) + 0.4 * within_user_zscore(
        users, deepfm
    )
    dcnv2 = within_user_zscore(users, dcnv2)
    event_times = align_event_times(runner.data_dir, runner._splits)
    history_lengths = strict_history_lengths(runner._splits, event_times)["valid"]
    cold = history_lengths <= 2
    # One predeclared, interpretable gate: add DCNv2 only for cold-history rows.
    gated = champion.copy()
    gated[cold] = 0.5 * champion[cold] + 0.5 * dcnv2[cold]
    champion_metrics = runner._metrics(
        runner.evaluate_mod.evaluate(users, labels, champion)
    )
    gated_metrics = runner._metrics(
        runner.evaluate_mod.evaluate(users, labels, gated)
    )
    return {
        "cold_rows": int(np.sum(cold)),
        "cold_fraction": float(np.mean(cold)),
        "champion": champion_metrics,
        "cold_half_dcnv2_gate": gated_metrics,
        "delta": gated_metrics["primary"] - champion_metrics["primary"],
    }


def main() -> int:
    data_dir = ROOT / "data/KuaiRand-Pure/data"
    output_dir = ROOT / "runs/history_gated_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    initial = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    main_runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
    main_runner.prepare()
    official = evaluate_scope(
        main_runner,
        initial,
        {
            "fm": ROOT / "runs/global_context_ablation/fm_bpr.npz",
            "deepfm": ROOT / "runs/dcnv2_ablation/deepfm.npz",
            "dcnv2": ROOT / "runs/dcnv2_ablation/dcnv2.npz",
        },
    )
    print("\n=== official validation ===")
    print(json.dumps(official, indent=2))

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

    deltas = [item["delta"] for item in rolling.values()]
    aggregate = {
        "mean_delta": float(np.mean(deltas)),
        "wins": int(np.sum(np.asarray(deltas) > 0)),
        "folds": len(deltas),
    }
    payload = {
        "test_labels_used": False,
        "gate": "history_len <= 2: 0.5 champion + 0.5 DCNv2; otherwise champion",
        "official_validation": official,
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
