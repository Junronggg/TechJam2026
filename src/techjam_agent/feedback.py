from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


AUXILIARY_COLUMNS = ("is_click", "is_like", "completion", "log_capped_watch")
AUXILIARY_SELECTIONS = {
    "click": (0,),
    "like": (1,),
    "completion": (2,),
    "click_like": (0, 1),
    "click_like_completion": (0, 1, 2),
    "log_watch": (3,),
    "censored_watch": (),
}
LOG_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)


def _raw_key(row: dict[str, str]) -> tuple:
    return (
        int(row["date"]),
        row["user_id"],
        row["video_id"],
        row["tab"],
        float(row["duration_ms"]),
        1 if row["long_view"] != "0" else 0,
    )


def _starter_key(row: tuple) -> tuple:
    return (int(row[0]), row[1], row[2], row[4], float(row[5]), int(row[6]))


def auxiliary_task_count(selection: str) -> int:
    if selection == "censored_watch":
        return 1
    try:
        return len(AUXILIARY_SELECTIONS[selection])
    except KeyError as exc:
        raise ValueError(f"unknown auxiliary signal selection: {selection}") from exc


def align_auxiliary_feedback(
    data_dir: Path,
    splits: dict[str, list[tuple]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Align training-only click, like, and censored completion targets.

    Completion is min(play_time, duration) / duration. Rows with non-positive
    duration are masked because their completion ratio is undefined.
    """
    date_to_split = {}
    for split, rows in splits.items():
        for date in {int(row[0]) for row in rows}:
            if date in date_to_split:
                raise ValueError(f"date {date} appears in multiple splits")
            date_to_split[date] = split
    aligned = {
        split: np.empty((len(rows), len(AUXILIARY_COLUMNS)), dtype=np.float32)
        for split, rows in splits.items()
    }
    masks = {
        split: np.ones((len(rows), len(AUXILIARY_COLUMNS)), dtype=np.float32)
        for split, rows in splits.items()
    }
    positions = {split: 0 for split in splits}
    for filename in LOG_FILES:
        with (data_dir / filename).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                split = date_to_split.get(int(row["date"]))
                if split is None:
                    continue
                index = positions[split]
                if index >= len(splits[split]) or _raw_key(row) != _starter_key(
                    splits[split][index]
                ):
                    raise ValueError(f"auxiliary label alignment failed for {split} row {index}")
                duration = float(row["duration_ms"])
                play_time = max(0.0, float(row["play_time_ms"]))
                if duration > 0:
                    completion = min(play_time, duration) / duration
                else:
                    completion = 0.0
                    masks[split][index, 2:] = 0.0
                log_capped_watch = np.log1p(min(play_time, max(0.0, duration)))
                aligned[split][index] = (
                    float(row["is_click"]),
                    float(row["is_like"]),
                    completion,
                    log_capped_watch,
                )
                positions[split] += 1
    for split, rows in splits.items():
        if positions[split] != len(rows):
            raise ValueError(
                f"missing auxiliary labels for {split}: {positions[split]}/{len(rows)}"
            )
    train_observed = masks["train"][:, 3] > 0
    if not np.any(train_observed):
        raise ValueError("log-watch auxiliary target has no positive-duration train rows")
    scale = float(np.quantile(aligned["train"][train_observed, 3], 0.99))
    scale = max(scale, 1e-6)
    for split in aligned:
        aligned[split][:, 3] = np.clip(aligned[split][:, 3] / scale, 0.0, 1.0)
    return aligned, masks


def align_censored_watch_feedback(
    data_dir: Path,
    splits: dict[str, list[tuple]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build a right-censored log-watch target from strictly aligned feedback.

    An incomplete play observes its log watch time exactly. A completed play only
    establishes the lower bound log(duration): predictions above that threshold are
    not penalized. The returned censor indicator is consumed by the one-sided loss.
    """
    date_to_split: dict[int, str] = {}
    for split, rows in splits.items():
        for date in {int(row[0]) for row in rows}:
            if date in date_to_split:
                raise ValueError(f"date {date} appears in multiple splits")
            date_to_split[date] = split
    targets = {
        split: np.zeros((len(rows), 1), dtype=np.float32)
        for split, rows in splits.items()
    }
    masks = {
        split: np.ones((len(rows), 1), dtype=np.float32)
        for split, rows in splits.items()
    }
    censored = {
        split: np.zeros((len(rows), 1), dtype=np.float32)
        for split, rows in splits.items()
    }
    positions = {split: 0 for split in splits}
    for filename in LOG_FILES:
        with (data_dir / filename).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                split = date_to_split.get(int(row["date"]))
                if split is None:
                    continue
                index = positions[split]
                if index >= len(splits[split]) or _raw_key(row) != _starter_key(
                    splits[split][index]
                ):
                    raise ValueError(
                        f"censored watch alignment failed for {split} row {index}"
                    )
                duration = float(row["duration_ms"])
                play_time = max(0.0, float(row["play_time_ms"]))
                if duration <= 0:
                    masks[split][index, 0] = 0.0
                else:
                    is_censored = play_time >= duration
                    observed_or_bound = duration if is_censored else play_time
                    targets[split][index, 0] = np.log1p(observed_or_bound)
                    censored[split][index, 0] = float(is_censored)
                positions[split] += 1
    for split, rows in splits.items():
        if positions[split] != len(rows):
            raise ValueError(
                f"missing censored watch labels for {split}: "
                f"{positions[split]}/{len(rows)}"
            )
    observed = masks["train"][:, 0] > 0
    if not np.any(observed):
        raise ValueError("censored watch target has no positive-duration train rows")
    scale = max(float(np.quantile(targets["train"][observed, 0], 0.99)), 1e-6)
    for split in targets:
        targets[split][:, 0] = np.clip(targets[split][:, 0] / scale, 0.0, 1.0)
    return targets, masks, censored


def select_auxiliary_feedback(
    labels: np.ndarray,
    masks: np.ndarray,
    selection: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        indices = AUXILIARY_SELECTIONS[selection]
    except KeyError as exc:
        raise ValueError(f"unknown auxiliary signal selection: {selection}") from exc
    return labels[:, indices], masks[:, indices]


def align_auxiliary_labels(
    data_dir: Path,
    splits: dict[str, list[tuple]],
) -> dict[str, np.ndarray]:
    """Compatibility wrapper returning all auxiliary target columns."""
    return align_auxiliary_feedback(data_dir, splits)[0]
