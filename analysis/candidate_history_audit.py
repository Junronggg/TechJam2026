"""Audit leakage-safe candidate-history signals on the official validation dates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "kuairand-starter-kit"))

from evaluate import evaluate
from techjam_agent.feasibility_producers import (
    COVERAGE_SUMMARY_PATH,
    coverage_summary_from_signals,
    write_versioned_json,
)


TRAIN_END = 20220421
VALID_END = 20220428
LOG_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)


def count_bucket(value: int) -> int:
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


def grouped_indices(times: np.ndarray):
    order = np.argsort(times, kind="stable")
    cursor = 0
    while cursor < len(order):
        timestamp = int(times[order[cursor]])
        end = cursor + 1
        while end < len(order) and int(times[order[end]]) == timestamp:
            end += 1
        yield timestamp, order[cursor:end]
        cursor = end


def binary_summary(values: np.ndarray, labels: np.ndarray) -> dict:
    result = {}
    for value in (0, 1):
        selected = values == value
        result[str(value)] = {
            "rows": int(selected.sum()),
            "long_view_rate": float(labels[selected].mean()) if selected.any() else None,
        }
    result["coverage"] = float(values.mean())
    return result


def bucket_summary(values: np.ndarray, labels: np.ndarray) -> dict:
    result = {}
    for value in range(6):
        selected = values == value
        result[str(value)] = {
            "rows": int(selected.sum()),
            "long_view_rate": float(labels[selected].mean()) if selected.any() else None,
        }
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = ROOT / "data/KuaiRand-Pure/data"
    video_to_author = {}
    with (data_dir / "video_features_basic_pure.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            video_to_author[row["video_id"]] = row["author_id"]

    records = {"train": [], "valid": []}
    for filename in LOG_FILES:
        with (data_dir / filename).open(encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                date = int(raw["date"])
                if date > VALID_END:
                    continue
                split = "train" if date <= TRAIN_END else "valid"
                video = raw["video_id"]
                records[split].append(
                    (
                        int(raw["time_ms"]),
                        raw["user_id"],
                        video,
                        video_to_author.get(video, "UNK"),
                        int(raw["long_view"] != "0"),
                    )
                )

    video_counts: dict[tuple[str, str], int] = defaultdict(int)
    author_counts: dict[tuple[str, str], int] = defaultdict(int)
    positive_videos: set[tuple[str, str]] = set()
    positive_authors: set[tuple[str, str]] = set()
    last_positive_author_time: dict[tuple[str, str], int] = {}
    last_video: dict[str, str] = {}
    last_author: dict[str, str] = {}

    train = records["train"]
    train_times = np.asarray([row[0] for row in train], dtype=np.int64)
    for timestamp, indices in grouped_indices(train_times):
        for raw_index in indices:
            _, user, video, author, label = train[int(raw_index)]
            if label:
                positive_videos.add((user, video))
                positive_authors.add((user, author))
                last_positive_author_time[(user, author)] = timestamp
        for raw_index in indices:
            _, user, video, author, _ = train[int(raw_index)]
            video_counts[(user, video)] += 1
            author_counts[(user, author)] += 1
            last_video[user] = video
            last_author[user] = author

    valid = records["valid"]
    valid_times = np.asarray([row[0] for row in valid], dtype=np.int64)
    users = [row[1] for row in valid]
    labels = np.asarray([row[4] for row in valid], dtype=np.float32)
    prior_video_positive = np.zeros(len(valid), dtype=np.int32)
    prior_video_count = np.zeros(len(valid), dtype=np.int32)
    prior_author_count = np.zeros(len(valid), dtype=np.int32)
    prior_author_positive = np.zeros(len(valid), dtype=np.int32)
    previous_video_same = np.zeros(len(valid), dtype=np.int32)
    previous_author_same = np.zeros(len(valid), dtype=np.int32)
    author_positive_recency = np.full(len(valid), -1.0, dtype=np.float32)

    for timestamp, indices in grouped_indices(valid_times):
        for raw_index in indices:
            index = int(raw_index)
            _, user, video, author, _ = valid[index]
            prior_video_positive[index] = int((user, video) in positive_videos)
            prior_video_count[index] = count_bucket(video_counts[(user, video)])
            prior_author_count[index] = count_bucket(author_counts[(user, author)])
            prior_author_positive[index] = int((user, author) in positive_authors)
            previous_video_same[index] = int(last_video.get(user) == video)
            previous_author_same[index] = int(last_author.get(user) == author)
            previous = last_positive_author_time.get((user, author))
            if previous is not None:
                author_positive_recency[index] = -np.log1p(max(0, timestamp - previous))
        # Update only target-free exposure state with validation inputs.
        for raw_index in indices:
            _, user, video, author, _ = valid[int(raw_index)]
            video_counts[(user, video)] += 1
            author_counts[(user, author)] += 1
            last_video[user] = video
            last_author[user] = author

    signals = {
        "prior_video_positive": prior_video_positive.astype(np.float32),
        "prior_video_count": prior_video_count.astype(np.float32),
        "prior_author_count": prior_author_count.astype(np.float32),
        "prior_author_positive": prior_author_positive.astype(np.float32),
        "previous_video_same": previous_video_same.astype(np.float32),
        "previous_author_same": previous_author_same.astype(np.float32),
        "author_positive_recency": author_positive_recency,
    }
    payload = {
        "policy": "train labels plus validation input history; no rows after 2022-04-28",
        "rows": {name: len(rows) for name, rows in records.items()},
        "validation_long_view_rate": float(labels.mean()),
        "feature_only_metrics": {
            name: {key: float(value) if key not in ("users", "rows") else int(value)
                   for key, value in evaluate(users, labels, scores).items()}
            for name, scores in signals.items()
        },
        "binary_associations": {
            "prior_video_positive": binary_summary(prior_video_positive, labels),
            "prior_author_positive": binary_summary(prior_author_positive, labels),
            "previous_video_same": binary_summary(previous_video_same, labels),
            "previous_author_same": binary_summary(previous_author_same, labels),
        },
        "count_associations": {
            "prior_video_count": bucket_summary(prior_video_count, labels),
            "prior_author_count": bucket_summary(prior_author_count, labels),
        },
    }
    print(json.dumps(payload, indent=2))
    if args.write_summary:
        write_versioned_json(
            coverage_summary_from_signals(signals, test_labels_used=False),
            args.output,
        )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit candidate-history coverage on official validation dates"
    )
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help="Write versioned validation-only coverage JSON after printing stdout",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / COVERAGE_SUMMARY_PATH,
        help="Coverage JSON path used only with --write-summary",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
