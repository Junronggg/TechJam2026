from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np


def _bin_count(value: int) -> str:
    if value <= 2:
        return "1-2"
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    return "11+"


def build_error_slice_report(
    train_rows: list[tuple],
    validation_rows: list[tuple],
    scores: np.ndarray,
    evaluate: Callable[..., dict[str, Any]],
    *,
    min_rows: int = 200,
) -> dict[str, Any]:
    """Validation-only ranking diagnostics with no test-label access."""
    if len(validation_rows) != len(scores):
        raise ValueError("validation score count does not match validation rows")
    users = [row[1] for row in validation_rows]
    labels = np.asarray([row[6] for row in validation_rows], dtype=np.float32)
    train_history = Counter(row[1] for row in train_rows)
    candidate_counts = Counter(users)
    positive_counts = Counter()
    for row in validation_rows:
        positive_counts[row[1]] += int(row[6])

    dimensions: dict[str, list[str]] = {
        "candidate_count": [_bin_count(candidate_counts[row[1]]) for row in validation_rows],
        "positive_count": ["0" if positive_counts[row[1]] == 0 else _bin_count(positive_counts[row[1]]) for row in validation_rows],
        "train_history": [_bin_count(train_history[row[1]]) if train_history[row[1]] else "cold" for row in validation_rows],
        "tab": [str(row[4]) for row in validation_rows],
        "hour": [str(row[9]) if len(row) > 9 else "unknown" for row in validation_rows],
        "weekday": [str(row[10]) if len(row) > 10 else "unknown" for row in validation_rows],
        "upload_age": [
            "unknown" if len(row) <= 11 or row[11] < 0 else
            ("0-7d" if row[11] <= 7 else "8-14d" if row[11] <= 14 else "15d+")
            for row in validation_rows
        ],
        "video_type": [str(row[12]) if len(row) > 12 else "unknown" for row in validation_rows],
        "user_activity": [str(row[13]) if len(row) > 13 else "unknown" for row in validation_rows],
    }
    slices = []
    for dimension, values in dimensions.items():
        groups: dict[str, list[int]] = defaultdict(list)
        for index, value in enumerate(values):
            groups[value].append(index)
        for value, indices in groups.items():
            if len(indices) < min_rows:
                continue
            metrics = evaluate(
                [users[index] for index in indices], labels[indices], scores[indices]
            )
            slices.append({
                "dimension": dimension,
                "value": value,
                "rows": len(indices),
                "users": int(metrics["users"]),
                "positive_rate": float(labels[indices].mean()),
                "GAUC": float(metrics["GAUC"]),
                "nDCG@5": float(metrics["nDCG@5"]),
                "primary": float(metrics["primary"]),
            })
    overall = evaluate(users, labels, scores)
    zero_positive_users = sum(value == 0 for value in positive_counts.values())
    ranking_ceiling = 1.0 - zero_positive_users / max(1, len(positive_counts))
    comparable = [row for row in slices if row["dimension"] != "positive_count"]
    comparable.sort(key=lambda row: (row["primary"], -row["rows"]))
    return {
        "scope": "validation only; row slices recompute ranking within each slice",
        "overall": {key: float(overall[key]) for key in ("GAUC", "nDCG@5", "primary")},
        "structural": {
            "validation_users": len(positive_counts),
            "zero_positive_users": zero_positive_users,
            "zero_positive_user_rate": zero_positive_users / max(1, len(positive_counts)),
            "maximum_dataset_ndcg": ranking_ceiling,
        },
        "worst_slices": comparable[:8],
        "slices": slices,
    }
