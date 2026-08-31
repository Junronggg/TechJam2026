from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Hashable, Iterable


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


RATE_FEATURES = {
    "user_long_view_rate",
    "item_long_view_rate",
    "author_long_view_rate",
    "user_tab_long_view_rate",
    "user_author_long_view_rate",
    "user_tag_long_view_rate",
}
COUNT_FEATURES = {
    "author_impression_count": "impressions",
    "author_long_view_count": "positives",
    "user_author_impression_count": "impressions",
    "user_author_long_view_count": "positives",
    "user_tag_impression_count": "impressions",
}
USER_TAG_FEATURES = {"user_tag_impression_count", "user_tag_long_view_rate"}
USER_AUTHOR_FEATURES = {
    "user_author_impression_count", "user_author_long_view_count",
    "user_author_long_view_rate",
}
AFFINITY_FEATURES = USER_TAG_FEATURES | USER_AUTHOR_FEATURES
TAG_INDEX = 7
EVENT_TIME_INDEX = 8


def _adjusted_counts(
    stats: dict[Hashable, list[int]],
    key: Hashable,
    label_to_leave_out: int | None,
) -> tuple[int, int]:
    positives, impressions = stats.get(key, [0, 0])
    if label_to_leave_out is not None:
        positives -= int(label_to_leave_out)
        impressions -= 1
    return max(0, positives), max(0, impressions)


def _smoothed_rate(positives: int, impressions: int, prior_rate: float,
                   prior: float = 20.0) -> float:
    return (positives + prior * prior_rate) / (impressions + prior)


def count_bucket(count: int, buckets: int = 20) -> int:
    """Fixed log bucket avoids learning bin boundaries from target values."""
    return min(buckets - 1, max(0, int(math.log2(max(0, count) + 1) * 2.0)))


@dataclass
class TrainHistoryStatistics:
    """Train-only aggregates used by FM and LightGBM historical features."""

    global_rate: float
    global_positives: int
    global_impressions: int
    groups: dict[str, dict[Hashable, list[int]]]

    @classmethod
    def build(cls, rows: Iterable[tuple], features: Iterable[str]) -> "TrainHistoryStatistics":
        materialized = rows if isinstance(rows, list) else list(rows)
        requested = set(features)
        positives = sum(int(row[6]) for row in materialized)
        global_rate = positives / len(materialized) if materialized else 0.0
        groups: dict[str, dict[Hashable, list[int]]] = {}
        if "user_long_view_rate" in requested or "continuous_history_stats" in requested:
            groups["user"], _ = aggregate(materialized, 1)
        if "item_long_view_rate" in requested or "continuous_history_stats" in requested:
            groups["item"], _ = aggregate(materialized, 2)
        if requested.intersection({
            "author_impression_count", "author_long_view_count", "author_long_view_rate",
            "user_author_impression_count", "user_author_long_view_count",
            "user_author_long_view_rate",
        }):
            groups["author"], _ = aggregate(materialized, 3)
        if "user_tab_long_view_rate" in requested:
            groups["user_tab"], _ = aggregate_pair(materialized, 1, 4)
        if requested.intersection(USER_TAG_FEATURES):
            groups["tag"], _ = aggregate(materialized, TAG_INDEX)
            groups["user_tag"], _ = aggregate_pair(materialized, 1, TAG_INDEX)
        if requested.intersection({
            "user_author_impression_count", "user_author_long_view_count",
            "user_author_long_view_rate",
        }):
            groups["user_author"], _ = aggregate_pair(materialized, 1, 3)
        return cls(global_rate, positives, len(materialized), groups)

    def global_prior(self, label_to_leave_out: int | None) -> float:
        if label_to_leave_out is None or self.global_impressions <= 1:
            return self.global_rate
        return (
            self.global_positives - int(label_to_leave_out)
        ) / (self.global_impressions - 1)

    def value(
        self,
        feature: str,
        row: tuple,
        *,
        leave_one_out: bool,
    ) -> float:
        label = int(row[6]) if leave_one_out else None
        global_prior = self.global_prior(label)
        if feature == "user_long_view_rate":
            positive, count = _adjusted_counts(self.groups["user"], row[1], label)
            return _smoothed_rate(positive, count, global_prior)
        if feature == "item_long_view_rate":
            positive, count = _adjusted_counts(self.groups["item"], row[2], label)
            return _smoothed_rate(positive, count, global_prior)
        if feature == "user_tab_long_view_rate":
            positive, count = _adjusted_counts(
                self.groups["user_tab"], (row[1], row[4]), label
            )
            return _smoothed_rate(positive, count, global_prior)
        if feature.startswith("user_tag_"):
            pair_positive, pair_count = _adjusted_counts(
                self.groups["user_tag"], (row[1], row[TAG_INDEX]), label
            )
            if feature == "user_tag_impression_count":
                return float(pair_count)
            if feature == "user_tag_long_view_rate":
                tag_positive, tag_count = _adjusted_counts(
                    self.groups["tag"], row[TAG_INDEX], label
                )
                tag_rate = _smoothed_rate(tag_positive, tag_count, global_prior)
                return _smoothed_rate(pair_positive, pair_count, tag_rate)
        if feature.startswith("author_"):
            positive, count = _adjusted_counts(self.groups["author"], row[3], label)
            if feature == "author_impression_count":
                return float(count)
            if feature == "author_long_view_count":
                return float(positive)
            if feature == "author_long_view_rate":
                return _smoothed_rate(positive, count, global_prior)
        if feature.startswith("user_author_"):
            pair_positive, pair_count = _adjusted_counts(
                self.groups["user_author"], (row[1], row[3]), label
            )
            if feature == "user_author_impression_count":
                return float(pair_count)
            if feature == "user_author_long_view_count":
                return float(pair_positive)
            if feature == "user_author_long_view_rate":
                author_positive, author_count = _adjusted_counts(
                    self.groups["author"], row[3], label
                )
                author_rate = _smoothed_rate(
                    author_positive, author_count, global_prior
                )
                return _smoothed_rate(pair_positive, pair_count, author_rate)
        raise ValueError(f"unsupported history feature: {feature}")

    def categorical_value(self, feature: str, row: tuple, *, leave_one_out: bool) -> int:
        value = self.value(feature, row, leave_one_out=leave_one_out)
        if feature in RATE_FEATURES:
            return min(19, max(0, int(value * 20)))
        if feature in COUNT_FEATURES:
            return count_bucket(int(value))
        raise ValueError(f"unsupported categorical history feature: {feature}")

    def numeric_value(self, feature: str, row: tuple, *, leave_one_out: bool) -> float:
        value = self.value(feature, row, leave_one_out=leave_one_out)
        return float(math.log1p(value)) if feature in COUNT_FEATURES else float(value)

    def chronological_user_tag_values(
        self,
        feature: str,
        rows: Iterable[tuple],
        *,
        categorical: bool,
    ) -> list[float | int]:
        """Encode train rows from strictly earlier events only.

        Rows sharing the same timestamp are encoded before any row in that
        timestamp group updates the history, preventing tie-order leakage.
        Validation and test rows use ``value`` against full training history.
        """
        if feature not in USER_TAG_FEATURES:
            raise ValueError(f"not a user-tag history feature: {feature}")
        materialized = rows if isinstance(rows, list) else list(rows)
        values: list[float | int] = [0] * len(materialized)
        pair_stats: dict[tuple[Hashable, Hashable], list[int]] = defaultdict(
            lambda: [0, 0]
        )
        tag_stats: dict[Hashable, list[int]] = defaultdict(lambda: [0, 0])
        global_positive = 0
        global_count = 0

        def event_time(index: int) -> int:
            row = materialized[index]
            return int(row[EVENT_TIME_INDEX]) if len(row) > EVENT_TIME_INDEX else index

        order = sorted(range(len(materialized)), key=lambda index: (event_time(index), index))
        start = 0
        while start < len(order):
            timestamp = event_time(order[start])
            end = start + 1
            while end < len(order) and event_time(order[end]) == timestamp:
                end += 1

            global_prior = global_positive / global_count if global_count else 0.5
            for index in order[start:end]:
                row = materialized[index]
                tag = row[TAG_INDEX]
                pair = (row[1], tag)
                pair_positive, pair_count = pair_stats[pair]
                if feature == "user_tag_impression_count":
                    raw_value = float(pair_count)
                else:
                    tag_positive, tag_count = tag_stats[tag]
                    tag_rate = _smoothed_rate(tag_positive, tag_count, global_prior)
                    raw_value = _smoothed_rate(pair_positive, pair_count, tag_rate)
                if categorical:
                    values[index] = (
                        count_bucket(int(raw_value))
                        if feature in COUNT_FEATURES
                        else min(19, max(0, int(raw_value * 20)))
                    )
                else:
                    values[index] = (
                        float(math.log1p(raw_value))
                        if feature in COUNT_FEATURES else float(raw_value)
                    )

            for index in order[start:end]:
                row = materialized[index]
                label = int(row[6])
                tag = row[TAG_INDEX]
                pair_record = pair_stats[(row[1], tag)]
                pair_record[0] += label
                pair_record[1] += 1
                tag_record = tag_stats[tag]
                tag_record[0] += label
                tag_record[1] += 1
                global_positive += label
                global_count += 1
            start = end
        return values

    def chronological_affinity_values(
        self, feature: str, rows: Iterable[tuple], *, categorical: bool,
    ) -> list[float | int]:
        """Strictly past-only hierarchical user-tag/user-author encoding.

        Candidate affinity is computed before the timestamp group is added,
        and backs off from the user-candidate pair to the candidate entity and
        finally the global rate. This prevents both current-label and tie-order
        leakage in training features.
        """
        if feature in USER_TAG_FEATURES:
            return self.chronological_user_tag_values(
                feature, rows, categorical=categorical
            )
        if feature not in USER_AUTHOR_FEATURES:
            raise ValueError(f"not an affinity feature: {feature}")
        materialized = rows if isinstance(rows, list) else list(rows)
        values: list[float | int] = [0] * len(materialized)
        pair_stats: dict[tuple[Hashable, Hashable], list[int]] = defaultdict(
            lambda: [0, 0]
        )
        entity_stats: dict[Hashable, list[int]] = defaultdict(lambda: [0, 0])
        global_positive = global_count = 0

        def event_time(index: int) -> int:
            row = materialized[index]
            return int(row[EVENT_TIME_INDEX]) if len(row) > EVENT_TIME_INDEX else index

        order = sorted(range(len(materialized)), key=lambda index: (event_time(index), index))
        start = 0
        while start < len(order):
            timestamp = event_time(order[start])
            end = start + 1
            while end < len(order) and event_time(order[end]) == timestamp:
                end += 1
            prior = global_positive / global_count if global_count else 0.5
            for index in order[start:end]:
                row = materialized[index]
                entity = row[3]
                pair_positive, pair_count = pair_stats[(row[1], entity)]
                if feature == "user_author_impression_count":
                    raw = float(pair_count)
                elif feature == "user_author_long_view_count":
                    raw = float(pair_positive)
                else:
                    entity_positive, entity_count = entity_stats[entity]
                    entity_rate = _smoothed_rate(entity_positive, entity_count, prior)
                    raw = _smoothed_rate(pair_positive, pair_count, entity_rate)
                if categorical:
                    values[index] = (
                        count_bucket(int(raw)) if feature in COUNT_FEATURES
                        else min(19, max(0, int(raw * 20)))
                    )
                else:
                    values[index] = (
                        float(math.log1p(raw)) if feature in COUNT_FEATURES else float(raw)
                    )
            for index in order[start:end]:
                row = materialized[index]
                label, entity = int(row[6]), row[3]
                pair_stats[(row[1], entity)][0] += label
                pair_stats[(row[1], entity)][1] += 1
                entity_stats[entity][0] += label
                entity_stats[entity][1] += 1
                global_positive += label
                global_count += 1
            start = end
        return values
