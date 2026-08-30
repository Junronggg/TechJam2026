from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


def _time_gap_bucket(delta_ms: int) -> int:
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


def strict_past_sequences(
    splits: dict[str, list[tuple]],
    event_times: dict[str, np.ndarray],
    encoded: dict[str, tuple[np.ndarray, np.ndarray, list]],
    max_length: int = 16,
) -> dict[str, dict[str, np.ndarray]]:
    """Build model-ready, leakage-safe last-K user sequences.

    Each history event stores encoded video/author IDs, a behavior state, and a
    time-gap bucket. Train history may use strictly earlier train long_view
    labels. Validation/test labels are never appended: their past events enter
    later history only as observed exposures. Same-timestamp rows cannot interact.
    """
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if set(splits) != set(event_times) or set(splits) != set(encoded):
        raise ValueError("splits, event_times, and encoded must contain the same keys")
    ordered_splits = sorted(
        splits,
        key=lambda name: int(np.min(event_times[name])) if len(event_times[name]) else 0,
    )
    # Stored tuple: encoded video, encoded author, behavior state, event time.
    histories: dict[object, deque[tuple[int, int, int, int]]] = defaultdict(
        lambda: deque(maxlen=max_length)
    )
    output: dict[str, dict[str, np.ndarray]] = {}
    for split in ordered_splits:
        rows = splits[split]
        X, labels, _ = encoded[split]
        times = np.asarray(event_times[split], dtype=np.int64)
        if len(rows) != len(times) or len(rows) != len(X):
            raise ValueError(f"sequence alignment mismatch for {split}")
        video = np.zeros((len(rows), max_length), dtype=np.int32)
        author = np.zeros((len(rows), max_length), dtype=np.int32)
        behavior = np.zeros((len(rows), max_length), dtype=np.int8)
        time_gap = np.zeros((len(rows), max_length), dtype=np.int8)
        mask = np.zeros((len(rows), max_length), dtype=np.float32)
        length = np.zeros(len(rows), dtype=np.int32)
        order = np.argsort(times, kind="stable")
        cursor = 0
        while cursor < len(order):
            timestamp = int(times[order[cursor]])
            end = cursor + 1
            while end < len(order) and int(times[order[end]]) == timestamp:
                end += 1
            indices = order[cursor:end]
            for raw_index in indices:
                index = int(raw_index)
                events = list(histories[rows[index][1]])
                count = min(len(events), max_length)
                length[index] = count
                start = max_length - count
                for position, event in enumerate(events[-count:], start=start):
                    event_video, event_author, event_behavior, event_time = event
                    video[index, position] = event_video
                    author[index, position] = event_author
                    behavior[index, position] = event_behavior
                    time_gap[index, position] = _time_gap_bucket(
                        max(0, timestamp - event_time)
                    )
                    mask[index, position] = 1.0
            # Only after all same-time rows have queried their histories may the
            # current impressions become visible to later timestamps.
            for raw_index in indices:
                index = int(raw_index)
                behavior_state = (
                    2 if split == "train" and int(labels[index]) == 1 else 1
                )
                histories[rows[index][1]].append(
                    (int(X[index, 1]), int(X[index, 2]), behavior_state, timestamp)
                )
            cursor = end
        output[split] = {
            "video_id": video,
            "author_id": author,
            "behavior": behavior,
            "time_gap": time_gap,
            "mask": mask,
            "length": length,
        }
    return output
