from __future__ import annotations

from collections import defaultdict

import numpy as np


def within_user_zscore(users, scores: np.ndarray) -> np.ndarray:
    """Normalize scores per user so differently scaled models can be blended."""
    groups: dict[object, list[int]] = defaultdict(list)
    for index, user in enumerate(users):
        groups[user].append(index)
    normalized = np.empty(len(scores), dtype=np.float32)
    for indices in groups.values():
        selection = np.asarray(indices, dtype=np.int64)
        values = np.asarray(scores[selection], dtype=np.float32)
        normalized[selection] = (values - values.mean()) / (values.std() + 1e-6)
    return normalized


def blend_scores(
    users,
    fm_scores: np.ndarray,
    deepfm_scores: np.ndarray,
    deepfm_weight: float,
) -> np.ndarray:
    if not 0.0 <= deepfm_weight <= 1.0:
        raise ValueError("deepfm_weight must be between 0 and 1")
    fm = within_user_zscore(users, fm_scores)
    deepfm = within_user_zscore(users, deepfm_scores)
    return (1.0 - deepfm_weight) * fm + deepfm_weight * deepfm
