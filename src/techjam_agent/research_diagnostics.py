from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable

import numpy as np

from .ensemble import within_user_zscore


def _plain_metrics(metrics: dict) -> dict:
    return {
        str(key): value.item() if hasattr(value, "item") else value
        for key, value in metrics.items()
    }


def strict_history_lengths(
    splits: dict[str, list[tuple]],
    event_times: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Count strictly earlier impressions per user without reading any labels.

    Splits are processed chronologically and target-free impressions carry into
    later splits. Rows sharing one timestamp are queried before any of them is
    appended, so same-time events cannot see each other.
    """
    if set(splits) != set(event_times):
        raise ValueError("splits and event_times must contain the same keys")
    ordered_splits = sorted(
        splits,
        key=lambda name: int(np.min(event_times[name])) if len(event_times[name]) else 0,
    )
    counts: Counter[object] = Counter()
    output: dict[str, np.ndarray] = {}
    for split in ordered_splits:
        rows = splits[split]
        times = np.asarray(event_times[split], dtype=np.int64)
        if len(rows) != len(times):
            raise ValueError(f"event-time length mismatch for {split}")
        values = np.zeros(len(rows), dtype=np.int32)
        order = np.argsort(times, kind="stable")
        cursor = 0
        while cursor < len(order):
            timestamp = int(times[order[cursor]])
            end = cursor + 1
            while end < len(order) and int(times[order[end]]) == timestamp:
                end += 1
            indices = order[cursor:end]
            for raw_index in indices:
                index = int(raw_index)
                values[index] = counts[rows[index][1]]
            for raw_index in indices:
                counts[rows[int(raw_index)][1]] += 1
            cursor = end
        output[split] = values
    return output


def history_availability_bucket(lengths: np.ndarray) -> np.ndarray:
    """Map history lengths to the predeclared cold/medium/rich slices."""
    values = np.asarray(lengths)
    result = np.empty(len(values), dtype=object)
    result[values <= 2] = "cold_0_2"
    result[(values >= 3) & (values <= 10)] = "medium_3_10"
    result[values > 10] = "rich_11_plus"
    return result


def build_slice_values(
    splits: dict[str, list[tuple]],
    history_lengths: dict[str, np.ndarray],
    split: str = "valid",
) -> dict[str, np.ndarray]:
    """Build target-free, predeclared evaluation slices for one split."""
    rows = splits[split]
    train_item_counts = Counter(row[2] for row in splits["train"])
    observed_counts = np.asarray(list(train_item_counts.values()), dtype=np.float64)
    if len(observed_counts):
        tail_edge, head_edge = np.quantile(observed_counts, [0.5, 0.9])
    else:
        tail_edge = head_edge = 0.0
    popularity = np.empty(len(rows), dtype=object)
    for index, row in enumerate(rows):
        count = train_item_counts.get(row[2], 0)
        if count == 0:
            popularity[index] = "unseen"
        elif count <= tail_edge:
            popularity[index] = "tail"
        elif count <= head_edge:
            popularity[index] = "middle"
        else:
            popularity[index] = "head"

    durations = np.asarray([float(row[5]) for row in rows])
    duration = np.empty(len(rows), dtype=object)
    duration[durations < 30_000] = "short_lt_30s"
    duration[(durations >= 30_000) & (durations < 120_000)] = "medium_30_120s"
    duration[durations >= 120_000] = "long_120s_plus"

    dates = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    unique_dates = np.unique(dates)
    boundary = unique_dates[(len(unique_dates) - 1) // 2] if len(unique_dates) else 0
    period = np.where(dates <= boundary, "early", "late").astype(object)
    return {
        "history": history_availability_bucket(history_lengths[split]),
        "item_popularity": popularity,
        "duration": duration,
        "period": period,
    }


def _finite_aligned(
    users, labels: np.ndarray, *score_arrays: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    user_values = np.asarray(users, dtype=object)
    label_values = np.asarray(labels, dtype=np.float32)
    scores = [np.asarray(values, dtype=np.float32) for values in score_arrays]
    expected = len(label_values)
    if len(user_values) != expected or any(len(values) != expected for values in scores):
        raise ValueError("users, labels, and scores must be aligned")
    if any(not np.all(np.isfinite(values)) for values in scores):
        raise ValueError("scores must be finite")
    return user_values, label_values, scores


def evaluate_slices(
    evaluate: Callable,
    users,
    labels: np.ndarray,
    scores: np.ndarray,
    slices: dict[str, np.ndarray],
    min_rows: int = 1,
) -> dict[str, dict]:
    """Apply the unchanged official evaluator overall and within fixed slices."""
    user_values, label_values, (score_values,) = _finite_aligned(
        users, labels, scores
    )
    result = {
        "overall": {
            **_plain_metrics(evaluate(user_values, label_values, score_values)),
            "positive_rate": float(np.mean(label_values)) if len(label_values) else 0.0,
        }
    }
    for dimension, values in slices.items():
        values = np.asarray(values, dtype=object)
        if len(values) != len(label_values):
            raise ValueError(f"slice {dimension} is not aligned")
        for value in sorted(set(values.tolist())):
            mask = values == value
            if int(mask.sum()) < min_rows:
                continue
            result[f"{dimension}={value}"] = {
                **_plain_metrics(
                    evaluate(user_values[mask], label_values[mask], score_values[mask])
                ),
                "positive_rate": float(np.mean(label_values[mask])),
            }
    return result


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    negatives = labels == 0
    if not np.any(positives) or not np.any(negatives):
        return 0.5
    differences = scores[positives, None] - scores[None, negatives]
    return float(np.mean((differences > 0) + 0.5 * (differences == 0)))


def _ndcg(labels: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    ranked = labels[np.argsort(-scores, kind="stable")][:k]
    ideal = np.sort(labels)[::-1][:k]
    discounts = np.log2(np.arange(len(ranked), dtype=np.float64) + 2.0)
    ideal_discounts = np.log2(np.arange(len(ideal), dtype=np.float64) + 2.0)
    dcg = float(np.sum(ranked / discounts))
    idcg = float(np.sum(ideal / ideal_discounts))
    return 0.0 if idcg == 0.0 else dcg / idcg


def _user_complementarity(
    users: np.ndarray,
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> dict[str, float | int]:
    groups: dict[object, list[int]] = defaultdict(list)
    for index, user in enumerate(users):
        groups[user].append(index)
    wins_a = wins_b = ties = recovered = errors_a = introduced = correct_a = 0
    deltas = []
    for indices in groups.values():
        selection = np.asarray(indices, dtype=np.int64)
        group_labels = labels[selection]
        group_a = scores_a[selection]
        group_b = scores_b[selection]
        primary_a = 0.5 * (
            _rank_auc(group_labels, group_a) + _ndcg(group_labels, group_a)
        )
        primary_b = 0.5 * (
            _rank_auc(group_labels, group_b) + _ndcg(group_labels, group_b)
        )
        delta = primary_b - primary_a
        deltas.append(delta)
        if delta > 1e-12:
            wins_b += 1
        elif delta < -1e-12:
            wins_a += 1
        else:
            ties += 1
        positive = group_labels == 1
        negative = group_labels == 0
        if np.any(positive) and np.any(negative):
            diff_a = group_a[positive, None] - group_a[None, negative]
            diff_b = group_b[positive, None] - group_b[None, negative]
            bad_a = diff_a <= 0
            good_a = diff_a > 0
            errors_a += int(np.sum(bad_a))
            correct_a += int(np.sum(good_a))
            recovered += int(np.sum(bad_a & (diff_b > 0)))
            introduced += int(np.sum(good_a & (diff_b <= 0)))
    total_users = len(groups)
    return {
        "users": total_users,
        "model_b_user_wins": wins_b,
        "model_a_user_wins": wins_a,
        "ties": ties,
        "model_b_user_win_rate": wins_b / total_users if total_users else 0.0,
        "mean_per_user_primary_delta": float(np.mean(deltas)) if deltas else 0.0,
        "model_a_pair_errors": errors_a,
        "model_b_recovered_a_errors": recovered,
        "pair_error_recovery_rate": recovered / errors_a if errors_a else 0.0,
        "model_b_new_pair_errors": introduced,
        "pair_error_introduction_rate": introduced / correct_a if correct_a else 0.0,
    }


def conditional_complementarity(
    evaluate: Callable,
    users,
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    slices: dict[str, np.ndarray],
    min_rows: int = 1,
) -> dict[str, dict]:
    """Explain where model B improves or hurts model A, overall and by slice."""
    user_values, label_values, score_arrays = _finite_aligned(
        users, labels, scores_a, scores_b
    )
    score_a, score_b = score_arrays
    normalized_a = within_user_zscore(user_values, score_a)
    normalized_b = within_user_zscore(user_values, score_b)
    scopes = {"overall": np.ones(len(label_values), dtype=bool)}
    for dimension, values in slices.items():
        values = np.asarray(values, dtype=object)
        if len(values) != len(label_values):
            raise ValueError(f"slice {dimension} is not aligned")
        for value in sorted(set(values.tolist())):
            scopes[f"{dimension}={value}"] = values == value
    result = {}
    for name, mask in scopes.items():
        if int(mask.sum()) < min_rows:
            continue
        metrics_a = _plain_metrics(
            evaluate(user_values[mask], label_values[mask], score_a[mask])
        )
        metrics_b = _plain_metrics(
            evaluate(user_values[mask], label_values[mask], score_b[mask])
        )
        if int(mask.sum()) > 1:
            correlation = float(np.corrcoef(normalized_a[mask], normalized_b[mask])[0, 1])
            if not np.isfinite(correlation):
                correlation = None
        else:
            correlation = 1.0
        result[name] = {
            "model_a": metrics_a,
            "model_b": metrics_b,
            "primary_delta_b_minus_a": float(metrics_b["primary"] - metrics_a["primary"]),
            "within_user_score_correlation": correlation,
            **_user_complementarity(
                user_values[mask], label_values[mask], score_a[mask], score_b[mask]
            ),
        }
    return result


def categorical_placebos(
    real_values: np.ndarray,
    cardinality: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Create matched constant, shuffled, and random categorical controls."""
    values = np.asarray(real_values, dtype=np.int32)
    if cardinality < 1:
        raise ValueError("cardinality must be positive")
    if len(values) and (int(values.min()) < 0 or int(values.max()) >= cardinality):
        raise ValueError("real_values fall outside the declared cardinality")
    rng = np.random.default_rng(seed)
    return {
        "real": values.copy(),
        "constant": np.zeros_like(values),
        "shuffled": rng.permutation(values),
        "random_same_cardinality": rng.integers(
            0, cardinality, size=len(values), dtype=np.int32
        ),
    }


def placebo_verdict(
    real_primary: float,
    placebo_primaries: dict[str, float],
    epsilon: float = 1e-5,
) -> dict[str, float | str]:
    """Require a real feature to beat every placebo before attributing its gain."""
    if not placebo_primaries:
        raise ValueError("at least one placebo score is required")
    strongest_name, strongest_score = max(
        placebo_primaries.items(), key=lambda item: item[1]
    )
    margin = float(real_primary - strongest_score)
    return {
        "verdict": "KEEP_CANDIDATE" if margin > epsilon else "REINTERPRET",
        "real_primary": float(real_primary),
        "strongest_placebo": strongest_name,
        "strongest_placebo_primary": float(strongest_score),
        "real_minus_strongest_placebo": margin,
    }
