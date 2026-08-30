from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.dcnv2 import DCNv2
from techjam_agent.deepfm import DeepFM, MultiTaskDeepFM
from techjam_agent.ensemble import within_user_zscore
from techjam_agent.research_diagnostics import (
    build_slice_values,
    conditional_complementarity,
    strict_history_lengths,
)
from techjam_agent.runner import ExperimentRunner
from techjam_agent.sequence_model import LightweightSequenceDeepFM
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
    multitask = copy.deepcopy(deepfm)
    multitask["model"] = "multitask_deepfm"
    multitask["hyperparameters"]["auxiliary_signals"] = "like"
    dcnv2 = copy.deepcopy(deepfm)
    dcnv2["model"] = "dcnv2"
    sequence = copy.deepcopy(deepfm)
    sequence["model"] = "sequence_deepfm"
    sequence["hyperparameters"]["epochs"] = 10
    sequence["hyperparameters"]["sequence_length"] = 16
    return {
        "fm": fm,
        "deepfm": deepfm,
        "multitask": multitask,
        "dcnv2": dcnv2,
        "sequence": sequence,
    }


def load_scores(runner: ExperimentRunner, config: dict, checkpoint: Path):
    encoded, dimension = runner._encoded_for(config)
    Xvalid, labels, users = encoded["valid"]
    hp = config["hyperparameters"]
    if config["model"] == "fm":
        model = runner.baseline.FM(
            dimension,
            k=hp["embedding_dim"],
            lr=hp["learning_rate"],
            l2=hp["l2"],
            seed=hp["seed"],
        )
        with np.load(checkpoint) as state:
            model.V, model.W, model.b = state["V"], state["W"], state["b"]
    elif config["model"] == "multitask_deepfm":
        model = MultiTaskDeepFM(
            dimension,
            Xvalid.shape[1],
            embedding_dim=hp["embedding_dim"],
            hidden_dim=hp["deepfm_hidden_dim"],
            learning_rate=hp["learning_rate"],
            l2=hp["l2"],
            seed=hp["seed"],
            auxiliary_tasks=1,
        )
        with np.load(checkpoint) as state:
            model.load_state_dict({name: state[name] for name in model.state_dict()})
    elif config["model"] == "sequence_deepfm":
        model = LightweightSequenceDeepFM(
            dimension,
            Xvalid.shape[1],
            embedding_dim=hp["embedding_dim"],
            hidden_dim=hp["deepfm_hidden_dim"],
            learning_rate=hp["learning_rate"],
            l2=hp["l2"],
            seed=hp["seed"],
            sequence_length=hp["sequence_length"],
        )
        with np.load(checkpoint) as state:
            model.load_state_dict({name: state[name] for name in model.state_dict()})
        return (
            np.asarray(users, dtype=object),
            labels,
            model.predict(Xvalid, runner._causal_for(hp["sequence_length"])["valid"]),
        )
    elif config["model"] == "dcnv2":
        model = DCNv2(
            dimension,
            Xvalid.shape[1],
            embedding_dim=hp["embedding_dim"],
            hidden_dim=hp["deepfm_hidden_dim"],
            cross_layers=hp["dcn_cross_layers"],
            cross_rank=hp["dcn_low_rank"],
            learning_rate=hp["learning_rate"],
            l2=hp["l2"],
            seed=hp["seed"],
        )
        with np.load(checkpoint) as state:
            model.load_state_dict({name: state[name] for name in model.state_dict()})
    else:
        model = DeepFM(
            dimension,
            Xvalid.shape[1],
            embedding_dim=hp["embedding_dim"],
            hidden_dim=hp["deepfm_hidden_dim"],
            learning_rate=hp["learning_rate"],
            l2=hp["l2"],
            seed=hp["seed"],
        )
        with np.load(checkpoint) as state:
            model.load_state_dict({name: state[name] for name in model.state_dict()})
    return np.asarray(users, dtype=object), labels, model.predict(Xvalid)


def main() -> int:
    data_dir = ROOT / "data/KuaiRand-Pure/data"
    output_dir = ROOT / "runs/conditional_complementarity"
    output_dir.mkdir(parents=True, exist_ok=True)
    initial = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    config = configs(initial)
    runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
    runner.prepare()

    users, labels, fm = load_scores(
        runner, config["fm"], ROOT / "runs/global_context_ablation/fm_bpr.npz"
    )
    _, _, deepfm = load_scores(
        runner, config["deepfm"], ROOT / "runs/dcnv2_ablation/deepfm.npz"
    )
    _, _, multitask = load_scores(
        runner,
        config["multitask"],
        ROOT / "runs/auxiliary_ablation/multitask_like.npz",
    )
    _, _, dcnv2 = load_scores(
        runner, config["dcnv2"], ROOT / "runs/dcnv2_ablation/dcnv2.npz"
    )
    _, _, sequence = load_scores(
        runner,
        config["sequence"],
        ROOT / "runs/lightweight_sequence_ablation/sequence_deepfm.npz",
    )
    champion = 0.6 * within_user_zscore(users, fm) + 0.4 * within_user_zscore(
        users, deepfm
    )

    development_splits = {
        "train": runner._splits["train"],
        "valid": runner._splits["valid"],
    }
    event_times = align_event_times(data_dir, development_splits)
    history_lengths = strict_history_lengths(development_splits, event_times)
    slices = build_slice_values(development_splits, history_lengths, "valid")
    candidates = {
        "like_multitask": multitask,
        "dcnv2": dcnv2,
        "lightweight_sequence": sequence,
    }
    comparisons = {
        name: conditional_complementarity(
            runner.evaluate_mod.evaluate,
            users,
            labels,
            champion,
            scores,
            slices,
            min_rows=100,
        )
        for name, scores in candidates.items()
    }
    sequence_blend = 0.9 * champion + 0.1 * within_user_zscore(users, sequence)
    champion_metrics = runner._metrics(
        runner.evaluate_mod.evaluate(users, labels, champion)
    )
    sequence_blend_metrics = runner._metrics(
        runner.evaluate_mod.evaluate(users, labels, sequence_blend)
    )
    fixed_blend_check = {
        "rule": "0.9 champion + 0.1 lightweight_sequence; no weight sweep",
        "champion": champion_metrics,
        "blend": sequence_blend_metrics,
        "delta": sequence_blend_metrics["primary"] - champion_metrics["primary"],
    }
    payload = {
        "test_labels_used": False,
        "history_policy": (
            "Strict earlier milliseconds; same timestamp cannot interact; "
            "validation labels never enter history."
        ),
        "model_a": "0.6 FM+BPR + 0.4 DeepFM+BCE champion",
        "model_b_candidates": list(candidates),
        "comparisons": comparisons,
        "fixed_blend_check": fixed_blend_check,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    for name, comparison in comparisons.items():
        overall = comparison["overall"]
        print(f"\n=== {name} versus champion ===")
        print(json.dumps({
            "primary_delta": overall["primary_delta_b_minus_a"],
            "correlation": overall["within_user_score_correlation"],
            "user_win_rate": overall["model_b_user_win_rate"],
            "pair_error_recovery_rate": overall["pair_error_recovery_rate"],
            "pair_error_introduction_rate": overall["pair_error_introduction_rate"],
        }, indent=2))
        best = max(
            (item for item in comparison.items() if item[0] != "overall"),
            key=lambda item: item[1]["primary_delta_b_minus_a"],
        )
        worst = min(
            (item for item in comparison.items() if item[0] != "overall"),
            key=lambda item: item[1]["primary_delta_b_minus_a"],
        )
        print(json.dumps({
            "best_slice": best[0],
            "best_slice_delta": best[1]["primary_delta_b_minus_a"],
            "worst_slice": worst[0],
            "worst_slice_delta": worst[1]["primary_delta_b_minus_a"],
        }, indent=2))
    print("\n=== fixed sequence blend check ===")
    print(json.dumps(fixed_blend_check, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
