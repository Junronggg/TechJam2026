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


def within_user_rank(users, scores: np.ndarray) -> np.ndarray:
    """Center each user's scores by within-user rank.

    The official metrics only depend on the ordering within a user.  A rank
    transform is therefore a useful calibration control when two models have
    different score distributions.  Ties use a stable order, matching the
    evaluator's deterministic tie behavior.
    """
    groups: dict[object, list[int]] = defaultdict(list)
    for index, user in enumerate(users):
        groups[user].append(index)
    normalized = np.empty(len(scores), dtype=np.float32)
    for indices in groups.values():
        selection = np.asarray(indices, dtype=np.int64)
        values = np.asarray(scores[selection], dtype=np.float32)
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(selection), dtype=np.float32)
        ranks[order] = np.arange(len(selection), dtype=np.float32)
        normalized[selection] = (
            ranks - (len(selection) - 1.0) / 2.0
        ) / max(1, len(selection))
    return normalized


def blend_scores(
    users,
    fm_scores: np.ndarray,
    deepfm_scores: np.ndarray,
    deepfm_weight: float,
    normalization: str = "zscore",
) -> np.ndarray:
    if not 0.0 <= deepfm_weight <= 1.0:
        raise ValueError("deepfm_weight must be between 0 and 1")
    if normalization == "zscore":
        fm = within_user_zscore(users, fm_scores)
        deepfm = within_user_zscore(users, deepfm_scores)
    elif normalization == "fm_zscore_deepfm_rank":
        fm = within_user_zscore(users, fm_scores)
        deepfm = within_user_rank(users, deepfm_scores)
    elif normalization == "fm_rank_deepfm_zscore":
        fm = within_user_rank(users, fm_scores)
        deepfm = within_user_zscore(users, deepfm_scores)
    else:
        raise ValueError(
            "normalization must be one of zscore, fm_zscore_deepfm_rank, "
            "fm_rank_deepfm_zscore"
        )
    return (1.0 - deepfm_weight) * fm + deepfm_weight * deepfm
