from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


AUXILIARY_COLUMNS = ("is_click", "is_like")
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


def align_auxiliary_labels(
    data_dir: Path,
    splits: dict[str, list[tuple]],
) -> dict[str, np.ndarray]:
    """Align click/like labels to starter rows without changing organizer code."""
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
                aligned[split][index] = tuple(
                    float(row[column]) for column in AUXILIARY_COLUMNS
                )
                positions[split] += 1
    for split, rows in splits.items():
        if positions[split] != len(rows):
            raise ValueError(
                f"missing auxiliary labels for {split}: {positions[split]}/{len(rows)}"
            )
    return aligned
