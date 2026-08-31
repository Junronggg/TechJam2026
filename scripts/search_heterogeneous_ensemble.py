"""Search a cached, heterogeneous validation model pool.

This is deliberately an offline capability: models are trained once, their
validation predictions are cached, and the search only evaluates predeclared
normalizations, subsets, and coarse weights.  It therefore captures the useful
part of a large ensemble search without silently retraining or reading test
labels.

The script is also useful when a weak standalone model is not a good model by
itself.  Its prediction is still measured for diversity and can be retained as
an ensemble-only candidate if it recovers errors made by the champion.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.ensemble import within_user_rank, within_user_zscore
from techjam_agent.runner import ExperimentRunner


# The first entries are strong/cache-backed components.  The last entries are
# intentionally heterogeneous and may be weak standalone models; retaining
# them lets the diversity audit decide whether they are useful.
CACHE_POOL: dict[str, str] = {
    "fm_bpr": "runs/ensemble_rank_calibrated2_fm_validation.npz",
    "deepfm_bce": "runs/ensemble_rank_calibrated2_deepfm_validation.npz",
    "multitask_like": "runs/pairwise_multitask_ablation/multitask_like_bce_validation.npz",
    "censored_watch_bce": "runs/censored_watchtime_ablation/censored_watch_bce_validation.npz",
    "fm_watchtime": "runs/fm_watchtime_ablation/fm_watchtime_validation.npz",
    "lambdarank": "runs/action_ablation/lambdarank_validation.npz",
    "adt": "runs/action_ablation/adt_validation.npz",
    "lightgcn": "runs/action_ablation/lightgcn_validation.npz",
}

# State-only checkpoints are materialized on demand.  This does not train a
# model; it only runs inference using an already saved checkpoint.
CHECKPOINT_POOL: dict[str, tuple[str, str]] = {
    "dcnv2": ("dcnv2", "runs/dcnv2_ablation/dcnv2.npz"),
    "lightweight_sequence": (
        "sequence_deepfm",
        "runs/lightweight_sequence_ablation/sequence_deepfm.npz",
    ),
}


def _load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        required = {"users", "labels", "scores"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path} is missing cache fields: {sorted(missing)}")
        users = np.asarray(data["users"], dtype=str)
        labels = np.asarray(data["labels"], dtype=np.float32)
        scores = np.asarray(data["scores"], dtype=np.float32)
    if not (len(users) == len(labels) == len(scores)):
        raise ValueError(f"cache arrays have different lengths: {path}")
    if not np.isfinite(scores).all():
        raise ValueError(f"cache contains non-finite scores: {path}")
    return users, labels, scores


def _config(initial: dict, model: str) -> dict:
    config = json.loads(json.dumps(initial))
    config["model"] = model
    config["training_objective"] = "bpr" if model == "fm" else "bce"
    if model == "fm":
        config["hyperparameters"]["learning_rate"] = 0.0003
    elif model in {"deepfm", "multitask_deepfm", "dcnv2", "sequence_deepfm"}:
        config["hyperparameters"]["learning_rate"] = 0.001
    if model == "multitask_deepfm":
        config["hyperparameters"]["auxiliary_signals"] = "like"
    if model == "sequence_deepfm":
        config["hyperparameters"]["sequence_length"] = 16
    return config


def _checkpoint_scores(
    runner: ExperimentRunner,
    initial: dict,
    model_name: str,
    model_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infer from a saved state so the pool can include existing models."""
    from techjam_agent.dcnv2 import DCNv2
    from techjam_agent.deepfm import DeepFM, MultiTaskDeepFM
    from techjam_agent.sequence_model import LightweightSequenceDeepFM

    config = _config(initial, model_name)
    encoded, dimension = runner._encoded_for(config)
    Xvalid, labels, users = encoded["valid"]
    hp = config["hyperparameters"]
    if model_name == "dcnv2":
        model = DCNv2(
            dimension, Xvalid.shape[1], embedding_dim=hp["embedding_dim"],
            hidden_dim=hp["deepfm_hidden_dim"],
            cross_layers=hp["dcn_cross_layers"], cross_rank=hp["dcn_low_rank"],
            learning_rate=hp["learning_rate"], l2=hp["l2"], seed=hp["seed"],
        )
    elif model_name == "sequence_deepfm":
        model = LightweightSequenceDeepFM(
            dimension, Xvalid.shape[1], embedding_dim=hp["embedding_dim"],
            hidden_dim=hp["deepfm_hidden_dim"], learning_rate=hp["learning_rate"],
            l2=hp["l2"], seed=hp["seed"],
            sequence_length=hp["sequence_length"],
        )
    elif model_name == "multitask_deepfm":
        model = MultiTaskDeepFM(
            dimension, Xvalid.shape[1], embedding_dim=hp["embedding_dim"],
            hidden_dim=hp["deepfm_hidden_dim"], learning_rate=hp["learning_rate"],
            l2=hp["l2"], seed=hp["seed"], auxiliary_tasks=1,
        )
    else:
        model = DeepFM(
            dimension, Xvalid.shape[1], embedding_dim=hp["embedding_dim"],
            hidden_dim=hp["deepfm_hidden_dim"], learning_rate=hp["learning_rate"],
            l2=hp["l2"], seed=hp["seed"],
        )
    with np.load(model_path) as state:
        if model_name == "fm":
            model.V, model.W, model.b = state["V"], state["W"], state["b"]
        else:
            model.load_state_dict({name: state[name] for name in model.state_dict()})
    if model_name == "sequence_deepfm":
        history = runner._causal_for(hp["sequence_length"])["valid"]
        scores = model.predict(Xvalid, history)
    else:
        scores = model.predict(Xvalid)
    return np.asarray(users, dtype=str), np.asarray(labels, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def _assert_aligned(
    reference: tuple[np.ndarray, np.ndarray, np.ndarray],
    candidate: tuple[np.ndarray, np.ndarray, np.ndarray],
    name: str,
) -> None:
    ref_users, ref_labels, _ = reference
    users, labels, _ = candidate
    if not np.array_equal(ref_users, users) or not np.array_equal(ref_labels, labels):
        raise ValueError(f"{name} is not aligned to the reference validation rows")


def _correlations(users: np.ndarray, scores: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    normalized = {name: within_user_zscore(users, values) for name, values in scores.items()}
    result: dict[str, dict[str, float]] = {}
    for left, right in itertools.combinations(scores, 2):
        result.setdefault(left, {})[right] = float(np.corrcoef(normalized[left], normalized[right])[0, 1])
        result.setdefault(right, {})[left] = result[left][right]
    return result


def _weights(size: int) -> Iterable[tuple[float, ...]]:
    if size == 1:
        yield (1.0,)
    elif size == 2:
        for right in (0.2, 0.4, 0.5, 0.6, 0.8):
            yield (1.0 - right, right)
    elif size == 3:
        yield (1 / 3, 1 / 3, 1 / 3)
        for dominant in range(3):
            weight = [0.2, 0.2, 0.2]
            weight[dominant] = 0.6
            yield tuple(weight)
        for order in itertools.permutations((0.2, 0.3, 0.5)):
            yield order
    else:
        raise ValueError("the predeclared search supports up to three members")


def _fuse(
    users: np.ndarray,
    values: list[np.ndarray],
    weights: tuple[float, ...],
    method: str,
) -> np.ndarray:
    if method == "zscore":
        normalized = [within_user_zscore(users, value) for value in values]
    elif method == "rank":
        normalized = [within_user_rank(users, value) for value in values]
    elif method == "zscore_rank":
        normalized = [within_user_zscore(users, values[0])]
        normalized.extend(within_user_rank(users, value) for value in values[1:])
    elif method == "rank_zscore":
        normalized = [within_user_rank(users, values[0])]
        normalized.extend(within_user_zscore(users, value) for value in values[1:])
    else:
        raise ValueError(f"unknown fusion method: {method}")
    return np.sum(np.asarray(weights, dtype=np.float32)[:, None] * np.asarray(normalized), axis=0)


def _fuse_normalized(
    normalized: dict[str, dict[str, np.ndarray]],
    members: tuple[str, ...],
    weights: tuple[float, ...],
    method: str,
) -> np.ndarray:
    """Fuse precomputed transforms; avoids an O(candidates × rows) re-pass."""
    values = [normalized[method][name] for name in members]
    return np.sum(
        np.asarray(weights, dtype=np.float32)[:, None] * np.asarray(values), axis=0
    )


def _evaluate(evaluator, users, labels, scores) -> dict[str, float | int]:
    metrics = evaluator(users, labels, scores)
    return {
        key: int(value) if key in {"users", "rows"} else float(value)
        for key, value in metrics.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/heterogeneous_ensemble_search")
    parser.add_argument("--max-members", type=int, choices=(2, 3), default=3)
    parser.add_argument("--include-weak", action="store_true", help="include ADT/LightGCN in subset search")
    parser.add_argument(
        "--cache", action="append", default=[], metavar="NAME=PATH",
        help="add another validation cache without changing this script (repeatable)",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    initial = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
    runner.prepare()
    pool: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    sources: dict[str, str] = {}
    cache_pool = dict(CACHE_POOL)
    for entry in args.cache:
        if "=" not in entry:
            raise ValueError(f"--cache must have NAME=PATH form, got {entry!r}")
        name, path = entry.split("=", 1)
        if not name or not path:
            raise ValueError(f"--cache must have NAME=PATH form, got {entry!r}")
        cache_pool[name] = path
    for name, relative in cache_pool.items():
        path = ROOT / relative
        if path.exists() and (args.include_weak or name not in {"adt", "lightgcn"}):
            pool[name] = _load_cache(path)
            sources[name] = str(path.relative_to(ROOT))
    for name, (model_name, relative) in CHECKPOINT_POOL.items():
        path = ROOT / relative
        if path.exists():
            cached = output_dir / f"{name}_validation.npz"
            if cached.exists():
                pool[name] = _load_cache(cached)
                sources[name] = str(cached.relative_to(ROOT))
            else:
                pool[name] = _checkpoint_scores(runner, initial, model_name, path)
                sources[name] = f"inference:{path.relative_to(ROOT)}"
                # Save a generic cache so subsequent searches do not need to
                # build sequence tensors or instantiate the model again.
                np.savez_compressed(
                    cached,
                    users=pool[name][0], labels=pool[name][1], scores=pool[name][2],
                )
    if len(pool) < 2:
        raise RuntimeError("fewer than two aligned validation components were found")

    reference = next(iter(pool.values()))
    for name, values in pool.items():
        _assert_aligned(reference, values, name)
    users, labels, _ = reference
    scores = {name: values[2] for name, values in pool.items()}
    standalone = {
        name: _evaluate(runner.evaluate_mod.evaluate, users, labels, values[2])
        for name, values in pool.items()
    }
    correlations = _correlations(users, scores)
    # The weak graph/tree baselines are retained in the report, but by default
    # are not allowed to dominate a huge subset sweep when their standalone
    # score is far below every useful candidate.
    search_names = list(pool)
    if not args.include_weak:
        search_names = [name for name in search_names if name not in {"adt", "lightgcn"}]
    results: list[dict] = []
    methods_for_size = {
        1: ("zscore",),
        2: ("zscore", "rank", "zscore_rank", "rank_zscore"),
        3: ("zscore", "rank", "zscore_rank"),
    }
    normalized = {
        "zscore": {name: within_user_zscore(users, value) for name, value in scores.items()},
        "rank": {name: within_user_rank(users, value) for name, value in scores.items()},
    }
    normalized["zscore_rank"] = {
        name: normalized["zscore"][name] for name in scores
    }
    normalized["rank_zscore"] = {
        name: normalized["rank"][name] for name in scores
    }
    for size in range(1, args.max_members + 1):
        for members in itertools.combinations(search_names, size):
            values = [scores[name] for name in members]
            for method in methods_for_size[size]:
                for weights in _weights(size):
                    # For asymmetric transforms, the first member uses the
                    # first transform and all remaining members use the second.
                    if method == "zscore_rank":
                        transformed = [normalized["zscore"][members[0]]]
                        transformed.extend(normalized["rank"][name] for name in members[1:])
                        fused = np.sum(
                            np.asarray(weights, dtype=np.float32)[:, None]
                            * np.asarray(transformed), axis=0
                        )
                    elif method == "rank_zscore":
                        transformed = [normalized["rank"][members[0]]]
                        transformed.extend(normalized["zscore"][name] for name in members[1:])
                        fused = np.sum(
                            np.asarray(weights, dtype=np.float32)[:, None]
                            * np.asarray(transformed), axis=0
                        )
                    else:
                        fused = _fuse_normalized(normalized, members, weights, method)
                    metrics = _evaluate(runner.evaluate_mod.evaluate, users, labels, fused)
                    results.append({
                        "members": list(members),
                        "method": method,
                        "weights": [float(value) for value in weights],
                        "metrics": metrics,
                        "source": [sources[name] for name in members],
                    })
    results.sort(key=lambda row: row["metrics"]["primary"], reverse=True)
    base_members = ("fm_bpr", "deepfm_bce")
    base = next(
        (row for row in results if tuple(row["members"]) == base_members and row["method"] == "zscore" and row["weights"] == [0.6, 0.4]),
        None,
    )
    champion = standalone["fm_bpr"]
    if base is not None:
        champion = base["metrics"]
    known_peak = None
    rank_summary = ROOT / "runs/rank_calibrated_ensemble/summary.json"
    if rank_summary.exists():
        rank_data = json.loads(rank_summary.read_text(encoding="utf-8"))
        known_peak = max(
            (float(row["primary"]) for row in rank_data.get("official_validation", {}).values()),
            default=None,
        )
    payload = {
        "test_labels_used": False,
        "search_policy": {
            "prediction_cache_only": True,
            "max_members": args.max_members,
            "include_weak": args.include_weak,
            "normalizations": methods_for_size,
            "weight_grid": "predeclared coarse grid; no fine validation sweep",
        },
        "pool_sources": sources,
        "standalone": standalone,
        "within_user_prediction_correlations": correlations,
        "reference": {
            "description": "0.6 FM+BPR + 0.4 DeepFM+BCE z-score blend when available",
            "metrics": champion,
        },
        "evaluated_candidates": len(results),
        "top_20": results[:20],
        "best_candidate": results[0],
        "best_delta_vs_reference": results[0]["metrics"]["primary"] - champion["primary"],
        "known_single_split_peak": known_peak,
        "best_delta_vs_known_single_split_peak": (
            None if known_peak is None else results[0]["metrics"]["primary"] - known_peak
        ),
        "selection_status": "research_only; no rolling or paired-seed confirmation",
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "components": standalone,
        "evaluated_candidates": len(results),
        "reference_primary": champion["primary"],
        "best_candidate": results[0],
        "best_delta_vs_reference": payload["best_delta_vs_reference"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
