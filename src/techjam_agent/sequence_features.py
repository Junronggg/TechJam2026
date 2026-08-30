from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


SEQUENCE_FEATURE_DIMS = {
    "prior_video_positive": 2,
    "author_positive_recency": 6,
    "prior_video_count": 6,
    "previous_author_same": 2,
}
LOG_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)


def _recency_bucket(delta_ms: int | None) -> int:
    if delta_ms is None:
        return 0
    hour = 60 * 60 * 1000
    day = 24 * hour
    if delta_ms <= hour:
        return 1
    if delta_ms <= day:
        return 2
    if delta_ms <= 3 * day:
        return 3
    if delta_ms <= 7 * day:
        return 4
    return 5


def _count_bucket(value: int) -> int:
    if value == 0:
        return 0
    if value == 1:
        return 1
    if value == 2:
        return 2
    if value <= 4:
        return 3
    if value <= 9:
        return 4
    return 5


def _raw_key(row: dict[str, str]) -> tuple:
    return (
        int(row["date"]),
        row["user_id"],
        row["video_id"],
        row["tab"],
        float(row["duration_ms"]),
    )


def _starter_key(row: tuple) -> tuple:
    return (int(row[0]), row[1], row[2], row[4], float(row[5]))


def align_event_times(
    data_dir: Path,
    splits: dict[str, list[tuple]],
) -> dict[str, np.ndarray]:
    """Align raw millisecond timestamps to the starter kit's preserved row order."""
    date_to_split: dict[int, str] = {}
    for split, rows in splits.items():
        for date in {int(row[0]) for row in rows}:
            if date in date_to_split:
                raise ValueError(f"date {date} appears in multiple splits")
            date_to_split[date] = split
    aligned = {
        split: np.empty(len(rows), dtype=np.int64)
        for split, rows in splits.items()
    }
    positions = {split: 0 for split in splits}
    for filename in LOG_FILES:
        with (data_dir / filename).open(encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                split = date_to_split.get(int(raw["date"]))
                if split is None:
                    continue
                index = positions[split]
                if index >= len(splits[split]) or _raw_key(raw) != _starter_key(
                    splits[split][index]
                ):
                    raise ValueError(f"event-time alignment failed for {split} row {index}")
                aligned[split][index] = int(raw["time_ms"])
                positions[split] += 1
    for split, rows in splits.items():
        if positions[split] != len(rows):
            raise ValueError(
                f"missing event times for {split}: {positions[split]}/{len(rows)}"
            )
    return aligned


def strict_sequence_categories(
    splits: dict[str, list[tuple]],
    event_times: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Build causal candidate-history categories.

    Train rows use only labels at strictly earlier timestamps. Validation and
    test start from the final train state and never update positive history with
    their own labels. Output arrays retain the starter kit's original row order.
    """
    train = splits["train"]
    prior_positive_videos: set[tuple[object, object]] = set()
    last_positive_author_time: dict[tuple[object, object], int] = {}
    video_exposure_counts: dict[tuple[object, object], int] = defaultdict(int)
    previous_author: dict[object, object] = {}
    result: dict[str, dict[str, np.ndarray]] = {}

    train_prior = np.zeros(len(train), dtype=np.int32)
    train_recency = np.zeros(len(train), dtype=np.int32)
    train_video_count = np.zeros(len(train), dtype=np.int32)
    train_previous_author = np.zeros(len(train), dtype=np.int32)
    order = np.argsort(event_times["train"], kind="stable")
    cursor = 0
    while cursor < len(order):
        timestamp = int(event_times["train"][order[cursor]])
        end = cursor + 1
        while end < len(order) and int(event_times["train"][order[end]]) == timestamp:
            end += 1
        indices = order[cursor:end]
        for raw_index in indices:
            index = int(raw_index)
            row = train[index]
            user, video, author = row[1], row[2], row[3]
            train_prior[index] = int((user, video) in prior_positive_videos)
            previous = last_positive_author_time.get((user, author))
            train_recency[index] = _recency_bucket(
                None if previous is None else timestamp - previous
            )
            train_video_count[index] = _count_bucket(
                video_exposure_counts[(user, video)]
            )
            train_previous_author[index] = int(previous_author.get(user) == author)
        for raw_index in indices:
            row = train[int(raw_index)]
            if int(row[6]) == 1:
                prior_positive_videos.add((row[1], row[2]))
                last_positive_author_time[(row[1], row[3])] = timestamp
        for raw_index in indices:
            row = train[int(raw_index)]
            video_exposure_counts[(row[1], row[2])] += 1
            previous_author[row[1]] = row[3]
        cursor = end
    result["train"] = {
        "prior_video_positive": train_prior,
        "author_positive_recency": train_recency,
        "prior_video_count": train_video_count,
        "previous_author_same": train_previous_author,
    }

    for split, rows in splits.items():
        if split == "train":
            continue
        prior = np.zeros(len(rows), dtype=np.int32)
        recency = np.zeros(len(rows), dtype=np.int32)
        video_count = np.zeros(len(rows), dtype=np.int32)
        same_author = np.zeros(len(rows), dtype=np.int32)
        split_video_counts = defaultdict(int, video_exposure_counts)
        split_previous_author = dict(previous_author)
        order = np.argsort(event_times[split], kind="stable")
        cursor = 0
        while cursor < len(order):
            timestamp = int(event_times[split][order[cursor]])
            end = cursor + 1
            while end < len(order) and int(event_times[split][order[end]]) == timestamp:
                end += 1
            indices = order[cursor:end]
            for raw_index in indices:
                index = int(raw_index)
                row = rows[index]
                user, video, author = row[1], row[2], row[3]
                prior[index] = int((user, video) in prior_positive_videos)
                previous = last_positive_author_time.get((user, author))
                recency[index] = _recency_bucket(
                    None if previous is None else max(0, timestamp - previous)
                )
                video_count[index] = _count_bucket(
                    split_video_counts[(user, video)]
                )
                same_author[index] = int(
                    split_previous_author.get(user) == author
                )
            for raw_index in indices:
                row = rows[int(raw_index)]
                split_video_counts[(row[1], row[2])] += 1
                split_previous_author[row[1]] = row[3]
            cursor = end
        result[split] = {
            "prior_video_positive": prior,
            "author_positive_recency": recency,
            "prior_video_count": video_count,
            "previous_author_same": same_author,
        }
    return result
