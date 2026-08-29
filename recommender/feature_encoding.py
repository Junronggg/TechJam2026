"""Leakage-safe categorical encoding for base and historical FM features."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


# Interaction tuple indices shared with official_fm.load_train_valid().
DATE, USER, VIDEO, AUTHOR, TAB, DURATION, LABEL, TAG, HOURMIN = range(9)
BASE_FEATURES = {"user_id", "video_id", "author_id", "tab", "dur_bucket"}
HISTORICAL_FEATURES = {
    "item_popularity",
    "user_activity",
    "item_long_view_rate",
    "user_tag_affinity",
}
SUPPORTED_FEATURES = BASE_FEATURES | HISTORICAL_FEATURES


@dataclass(frozen=True)
class HistoricalStatistics:
    global_rate: float
    item_count: collections.Counter[str]
    item_positive: collections.Counter[str]
    user_count: collections.Counter[str]
    user_tag_count: collections.Counter[tuple[str, str]]
    user_tag_positive: collections.Counter[tuple[str, str]]


def fit_historical_statistics(
    train_rows: Sequence[tuple[object, ...]],
) -> HistoricalStatistics:
    item_count: collections.Counter[str] = collections.Counter()
    item_positive: collections.Counter[str] = collections.Counter()
    user_count: collections.Counter[str] = collections.Counter()
    user_tag_count: collections.Counter[tuple[str, str]] = collections.Counter()
    user_tag_positive: collections.Counter[tuple[str, str]] = collections.Counter()
    positives = 0
    for row in train_rows:
        user, video, tag, label = str(row[USER]), str(row[VIDEO]), str(row[TAG]), int(row[LABEL])
        item_count[video] += 1
        item_positive[video] += label
        user_count[user] += 1
        user_tag_count[(user, tag)] += 1
        user_tag_positive[(user, tag)] += label
        positives += label
    return HistoricalStatistics(
        global_rate=positives / max(1, len(train_rows)),
        item_count=item_count,
        item_positive=item_positive,
        user_count=user_count,
        user_tag_count=user_tag_count,
        user_tag_positive=user_tag_positive,
    )


def _smoothed_rate(
    positives: int,
    count: int,
    global_rate: float,
    smoothing: float,
    own_label: int = 0,
    leave_one_out: bool = False,
) -> float:
    adjusted_count = count - (1 if leave_one_out else 0)
    adjusted_positives = positives - (own_label if leave_one_out else 0)
    return (adjusted_positives + smoothing * global_rate) / (
        adjusted_count + smoothing
    )


def _quantile_edges(values: Iterable[float], count: int, buckets: int) -> np.ndarray:
    array = np.fromiter(values, dtype=np.float64, count=count)
    return np.quantile(array, np.linspace(0, 1, buckets + 1)[1:-1])


class FeatureEncoder:
    """Fit vocabularies/buckets on train and transform validation without labels."""

    def __init__(
        self,
        features: Sequence[str],
        smoothing: float = 20.0,
        buckets: int = 20,
    ) -> None:
        self.features = tuple(features)
        self.smoothing = float(smoothing)
        self.buckets = int(buckets)
        unknown = set(self.features).difference(SUPPORTED_FEATURES)
        if unknown:
            raise ValueError(f"Unsupported features: {sorted(unknown)}")
        if not self.features:
            raise ValueError("At least one feature is required")

    def fit_transform(
        self, splits: dict[str, list[tuple[object, ...]]]
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, list[str]]], int]:
        train = splits["train"]
        statistics = fit_historical_statistics(train)
        edges = self._fit_edges(train, statistics)

        vocabularies: list[dict[str, int]] = [dict() for _ in self.features]
        for row in train:
            for index, value in enumerate(self._raw_values(row, statistics, edges, True)):
                if value not in vocabularies[index]:
                    vocabularies[index][value] = len(vocabularies[index])
        unknown_ids = [len(vocab) for vocab in vocabularies]
        dimensions = [len(vocab) + 1 for vocab in vocabularies]
        offsets = np.cumsum([0] + dimensions[:-1]).astype(np.int32)

        encoded: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
        for split_name, rows in splits.items():
            matrix = np.empty((len(rows), len(self.features)), dtype=np.int32)
            labels = np.empty(len(rows), dtype=np.float32)
            users: list[str] = []
            is_train = split_name == "train"
            for row_index, row in enumerate(rows):
                raw = self._raw_values(row, statistics, edges, is_train)
                for feature_index, value in enumerate(raw):
                    matrix[row_index, feature_index] = (
                        vocabularies[feature_index].get(value, unknown_ids[feature_index])
                        + offsets[feature_index]
                    )
                labels[row_index] = int(row[LABEL])
                users.append(str(row[USER]))
            encoded[split_name] = (matrix, labels, users)
        return encoded, int(sum(dimensions))

    def _fit_edges(
        self,
        train: Sequence[tuple[object, ...]],
        statistics: HistoricalStatistics,
    ) -> dict[str, np.ndarray]:
        edges: dict[str, np.ndarray] = {}
        for feature in self.features:
            if feature == "dur_bucket":
                edges[feature] = _quantile_edges(
                    (float(row[DURATION]) for row in train), len(train), 10
                )
            elif feature in HISTORICAL_FEATURES:
                edges[feature] = _quantile_edges(
                    (
                        self._historical_value(feature, row, statistics, True)
                        for row in train
                    ),
                    len(train),
                    self.buckets,
                )
        return edges

    def _raw_values(
        self,
        row: tuple[object, ...],
        statistics: HistoricalStatistics,
        edges: dict[str, np.ndarray],
        is_train: bool,
    ) -> list[str]:
        values: list[str] = []
        for feature in self.features:
            if feature == "user_id":
                value = str(row[USER])
            elif feature == "video_id":
                value = str(row[VIDEO])
            elif feature == "author_id":
                value = str(row[AUTHOR])
            elif feature == "tab":
                value = str(row[TAB])
            elif feature == "dur_bucket":
                value = str(int(np.searchsorted(edges[feature], float(row[DURATION]))))
            else:
                numeric = self._historical_value(feature, row, statistics, is_train)
                value = str(int(np.searchsorted(edges[feature], numeric)))
            values.append(value)
        return values

    def _historical_value(
        self,
        feature: str,
        row: tuple[object, ...],
        statistics: HistoricalStatistics,
        is_train: bool,
    ) -> float:
        user, video, tag, label = str(row[USER]), str(row[VIDEO]), str(row[TAG]), int(row[LABEL])
        if feature == "item_popularity":
            return float(np.log1p(statistics.item_count[video]))
        if feature == "user_activity":
            return float(np.log1p(statistics.user_count[user]))
        if feature == "item_long_view_rate":
            return _smoothed_rate(
                statistics.item_positive[video],
                statistics.item_count[video],
                statistics.global_rate,
                self.smoothing,
                own_label=label,
                leave_one_out=is_train,
            )
        if feature == "user_tag_affinity":
            key = (user, tag)
            return _smoothed_rate(
                statistics.user_tag_positive[key],
                statistics.user_tag_count[key],
                statistics.global_rate,
                self.smoothing,
                own_label=label,
                leave_one_out=is_train,
            )
        raise ValueError(f"Unknown historical feature: {feature}")

