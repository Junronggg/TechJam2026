from __future__ import annotations

import csv
import copy
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.deepfm import DeepFM
from techjam_agent.ensemble import within_user_zscore
from techjam_agent.runner import ExperimentRunner


DEVELOPMENT_CUTOFF = 20220428


def load_random_development_rows(data_dir: Path) -> list[tuple]:
    authors = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            authors[row["video_id"]] = row["author_id"]
    rows = []
    with (data_dir / "log_random_4_22_to_5_08_pure.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for raw in csv.DictReader(handle):
            date = int(raw["date"])
            # Discard post-development rows before parsing their behavior label.
            if date > DEVELOPMENT_CUTOFF:
                continue
            video = raw["video_id"]
            rows.append((
                date,
                raw["user_id"],
                video,
                authors.get(video, "UNK"),
                raw["tab"],
                float(raw["duration_ms"]),
                1 if raw["long_view"] != "0" else 0,
            ))
    if not rows:
        raise ValueError("random-exposure development split is empty")
    return rows


def configs(initial: dict) -> tuple[dict, dict]:
    fm = copy.deepcopy(initial)
    fm["model"] = "fm"
    fm["training_objective"] = "bpr"
    fm["hyperparameters"]["learning_rate"] = 0.0003
    deepfm = copy.deepcopy(initial)
    deepfm["model"] = "deepfm"
    deepfm["training_objective"] = "bce"
    deepfm["hyperparameters"]["learning_rate"] = 0.001
    return fm, deepfm


def evaluate_scope(
    runner: ExperimentRunner,
    initial: dict,
    fm_checkpoint: Path,
    deepfm_checkpoint: Path,
) -> dict:
    fm_config, deepfm_config = configs(initial)
    encoded, dimension = runner._encoded_for(fm_config)
    Xvalid, labels, users = encoded["valid"]
    fm_hp = fm_config["hyperparameters"]
    fm = runner.baseline.FM(
        dimension, k=fm_hp["embedding_dim"], lr=fm_hp["learning_rate"],
        l2=fm_hp["l2"], seed=fm_hp["seed"],
    )
    with np.load(fm_checkpoint) as state:
        fm.V, fm.W, fm.b = state["V"], state["W"], state["b"]
    deep_hp = deepfm_config["hyperparameters"]
    deepfm = DeepFM(
        dimension, Xvalid.shape[1], embedding_dim=deep_hp["embedding_dim"],
        hidden_dim=deep_hp["deepfm_hidden_dim"], learning_rate=deep_hp["learning_rate"],
        l2=deep_hp["l2"], seed=deep_hp["seed"],
    )
    with np.load(deepfm_checkpoint) as state:
        deepfm.load_state_dict({name: state[name] for name in deepfm.state_dict()})
    fm_scores = fm.predict(Xvalid)
    deepfm_scores = deepfm.predict(Xvalid)
    ensemble_scores = (
        0.6 * within_user_zscore(users, fm_scores)
        + 0.4 * within_user_zscore(users, deepfm_scores)
    )
    metrics = {
        "fm_bpr": runner._metrics(
            runner.evaluate_mod.evaluate(users, labels, fm_scores)
        ),
        "deepfm_bce": runner._metrics(
            runner.evaluate_mod.evaluate(users, labels, deepfm_scores)
        ),
        "champion_ensemble": runner._metrics(
            runner.evaluate_mod.evaluate(users, labels, ensemble_scores)
        ),
    }
    metrics["champion_ensemble"]["delta_vs_fm_bpr"] = (
        metrics["champion_ensemble"]["primary"] - metrics["fm_bpr"]["primary"]
    )
    metrics["label_rate"] = float(np.mean(labels))
    return metrics


def main() -> int:
    data_dir = ROOT / "data/KuaiRand-Pure/data"
    output_dir = ROOT / "runs/random_exposure_robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    initial = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
    runner.prepare()
    checkpoints = {
        "fm": ROOT / "runs/global_context_ablation/fm_bpr.npz",
        "deepfm": ROOT / "runs/dcnv2_ablation/deepfm.npz",
    }
    standard = evaluate_scope(
        runner, initial, checkpoints["fm"], checkpoints["deepfm"]
    )

    random_rows = load_random_development_rows(data_dir)
    random_runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
    random_runner._splits = {
        "train": runner._splits["train"],
        "valid": random_rows,
    }
    random_runner._encoded = random_runner.data.encode(random_runner._splits)
    random_exposure = evaluate_scope(
        random_runner, initial, checkpoints["fm"], checkpoints["deepfm"]
    )
    payload = {
        "test_labels_used": False,
        "random_policy": "Only random-exposure rows dated 2022-04-22 through 2022-04-28.",
        "standard_exposure": standard,
        "random_exposure": random_exposure,
        "ensemble_delta_change": (
            random_exposure["champion_ensemble"]["delta_vs_fm_bpr"]
            - standard["champion_ensemble"]["delta_vs_fm_bpr"]
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
