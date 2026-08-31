from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


USER_INDEX = 1
LABEL_INDEX = 6
TIMESTAMP_INDEX = 8
ITEM_FIELD = 1
USER_FIELD = 0


@dataclass(frozen=True)
class SequenceContext:
    """Strictly historical previous-positive context for candidate scoring."""

    users: dict[str, np.ndarray]
    items: dict[str, np.ndarray]
    previous_items: dict[str, np.ndarray]
    user_count: int
    item_count: int
    padding_item: int


def _local_field_ids(
    encoded: dict[str, tuple[np.ndarray, np.ndarray, Any]], field: int
) -> tuple[dict[str, np.ndarray], int]:
    train = np.asarray(encoded["train"][0])
    if train.ndim != 2 or train.shape[1] <= field:
        raise ValueError(f"encoded train matrix does not contain field {field}")
    offset = int(train[:, field].min())
    local = {
        split: (np.asarray(values[0])[:, field] - offset).astype(np.int32)
        for split, values in encoded.items()
    }
    width = max((int(values.max()) if len(values) else -1) for values in local.values()) + 1
    return local, width


def build_previous_positive_context(
    splits: dict[str, list[tuple]],
    encoded: dict[str, tuple[np.ndarray, np.ndarray, Any]],
) -> SequenceContext:
    """Build the last positive item visible before each interaction.

    Training rows use only strictly earlier training labels. All rows sharing a
    timestamp are encoded before any of their labels update history. Validation
    and test receive the final training state only: validation/test outcomes do
    not feed back into features inside the offline research benchmark.
    """

    if "train" not in splits or "train" not in encoded:
        raise ValueError("train split is required for sequential context")
    for split, rows in splits.items():
        if split not in encoded or len(rows) != len(encoded[split][0]):
            raise ValueError(f"raw and encoded row counts differ for split={split!r}")

    users, user_count = _local_field_ids(encoded, USER_FIELD)
    items, item_count = _local_field_ids(encoded, ITEM_FIELD)
    padding_item = item_count
    previous = {
        split: np.full(len(rows), padding_item, dtype=np.int32)
        for split, rows in splits.items()
    }

    by_user: dict[str, list[int]] = {}
    for index, row in enumerate(splits["train"]):
        by_user.setdefault(str(row[USER_INDEX]), []).append(index)

    final_positive: dict[str, int] = {}
    for user, indices in by_user.items():
        ordered = sorted(indices, key=lambda index: (int(splits["train"][index][TIMESTAMP_INDEX]), index))
        last_item = padding_item
        start = 0
        while start < len(ordered):
            timestamp = int(splits["train"][ordered[start]][TIMESTAMP_INDEX])
            end = start + 1
            while (
                end < len(ordered)
                and int(splits["train"][ordered[end]][TIMESTAMP_INDEX]) == timestamp
            ):
                end += 1
            tied = ordered[start:end]
            previous["train"][tied] = last_item
            for index in tied:
                if int(splits["train"][index][LABEL_INDEX]) == 1:
                    last_item = int(items["train"][index])
            start = end
        final_positive[user] = last_item

    for split in splits:
        if split == "train":
            continue
        previous[split] = np.fromiter(
            (
                final_positive.get(str(row[USER_INDEX]), padding_item)
                for row in splits[split]
            ),
            dtype=np.int32,
            count=len(splits[split]),
        )

    return SequenceContext(
        users=users,
        items=items,
        previous_items=previous,
        user_count=user_count,
        item_count=item_count,
        padding_item=padding_item,
    )
