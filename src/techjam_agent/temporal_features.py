from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime

import numpy as np


SPLIT_ORDER = ("train", "valid", "test")


def _ordinal(value: object) -> int:
    return datetime.strptime(str(int(value)), "%Y%m%d").date().toordinal()


def strict_past_window_counts(
    splits: dict[str, list[tuple]],
    key_index: int,
    window_days: int,
) -> dict[str, np.ndarray]:
    """Count prior-day exposures in a rolling window without using labels."""
    if window_days < 1:
        raise ValueError("window_days must be at least 1")
    result = {name: np.zeros(len(rows), dtype=np.float32)
              for name, rows in splits.items()}
    history: dict[object, Counter[int]] = defaultdict(Counter)
    for split in SPLIT_ORDER:
        if split not in splits:
            continue
        rows = splits[split]
        by_day: dict[int, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            by_day[_ordinal(row[0])].append(index)
        for day in sorted(by_day):
            indices = by_day[day]
            for index in indices:
                key = rows[index][key_index]
                result[split][index] = sum(
                    history[key][previous]
                    for previous in range(day - window_days, day)
                )
            for index in indices:
                history[rows[index][key_index]][day] += 1
    return result


def bucket_log_counts(
    counts: dict[str, np.ndarray],
    buckets: int = 10,
) -> tuple[dict[str, np.ndarray], int]:
    """Fit quantile bins on train log-counts and transform every split."""
    train_values = np.log1p(counts["train"])
    edges = np.unique(np.quantile(
        train_values, np.linspace(0, 1, buckets + 1)[1:-1]
    ))
    encoded = {
        split: np.searchsorted(edges, np.log1p(values)).astype(np.int32)
        for split, values in counts.items()
    }
    return encoded, len(edges) + 1
