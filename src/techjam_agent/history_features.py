from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Iterable


def aggregate(rows: Iterable[tuple], key_index: int) -> tuple[dict[Hashable, list[int]], float]:
    """Return per-key [positives, impressions] and the train-only global label rate."""
    stats: dict[Hashable, list[int]] = defaultdict(lambda: [0, 0])
    positives = impressions = 0
    for row in rows:
        label = int(row[6])
        record = stats[row[key_index]]
        record[0] += label
        record[1] += 1
        positives += label
        impressions += 1
    return dict(stats), positives / impressions if impressions else 0.0


def aggregate_pair(rows: Iterable[tuple], first_index: int, second_index: int):
    """Aggregate labels for a compound preference key such as (user_id, tab)."""
    stats: dict[tuple[Hashable, Hashable], list[int]] = defaultdict(lambda: [0, 0])
    positives = impressions = 0
    for row in rows:
        label = int(row[6])
        record = stats[(row[first_index], row[second_index])]
        record[0] += label
        record[1] += 1
        positives += label
        impressions += 1
    return dict(stats), positives / impressions if impressions else 0.0


def smoothed_rate_bucket(
    key: Hashable,
    stats: dict[Hashable, list[int]],
    global_rate: float,
    *,
    label_to_leave_out: int | None = None,
    prior: float = 20.0,
    buckets: int = 20,
) -> int:
    """Bucket a train-derived target rate; optionally remove the current train label."""
    positives, impressions = stats.get(key, [0, 0])
    if label_to_leave_out is not None:
        positives -= int(label_to_leave_out)
        impressions -= 1
    rate = (positives + prior * global_rate) / (impressions + prior)
    return min(buckets - 1, max(0, int(rate * buckets)))
