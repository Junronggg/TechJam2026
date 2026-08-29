"""Ranking-aligned objectives for the NumPy FM parameters."""

from __future__ import annotations

import collections
from typing import Sequence

import numpy as np


def build_within_user_pairs(
    users: Sequence[str],
    labels: np.ndarray,
    seed: int,
    pairs_per_positive: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample deterministic positive/negative pairs from each training user."""
    grouped: dict[str, tuple[list[int], list[int]]] = collections.defaultdict(
        lambda: ([], [])
    )
    for index, (user, label) in enumerate(zip(users, labels)):
        grouped[user][0 if label > 0 else 1].append(index)

    random = np.random.default_rng(seed)
    positive_parts: list[np.ndarray] = []
    negative_parts: list[np.ndarray] = []
    for positives, negatives in grouped.values():
        if not positives or not negatives:
            continue
        positive_array = np.repeat(np.asarray(positives, dtype=np.int64), pairs_per_positive)
        negative_array = random.choice(
            np.asarray(negatives, dtype=np.int64), size=len(positive_array), replace=True
        )
        positive_parts.append(positive_array)
        negative_parts.append(negative_array)
    if not positive_parts:
        raise ValueError("No users contain both positive and negative training labels")
    return np.concatenate(positive_parts), np.concatenate(negative_parts)


def pairwise_bpr_step(model: object, positive_x: np.ndarray, negative_x: np.ndarray) -> float:
    """One Adam update on -log(sigmoid(score_positive - score_negative))."""
    batch_size = len(positive_x)
    positive_score, positive_embeddings, positive_sum = model.logits(positive_x)
    negative_score, negative_embeddings, negative_sum = model.logits(negative_x)
    difference = positive_score - negative_score
    sigmoid_difference = 1.0 / (1.0 + np.exp(-np.clip(difference, -30, 30)))
    gradient = ((sigmoid_difference - 1.0) / batch_size).astype(np.float32)

    gradient_v = np.zeros_like(model.V)
    gradient_w = np.zeros_like(model.W)
    np.add.at(gradient_w, positive_x, gradient[:, None])
    np.add.at(gradient_w, negative_x, -gradient[:, None])
    np.add.at(
        gradient_v,
        positive_x,
        gradient[:, None, None] * (positive_sum[:, None, :] - positive_embeddings),
    )
    np.add.at(
        gradient_v,
        negative_x,
        -gradient[:, None, None] * (negative_sum[:, None, :] - negative_embeddings),
    )
    gradient_v += model.l2 * model.V
    gradient_w += model.l2 * model.W

    model.t += 1
    beta_one, beta_two, epsilon = 0.9, 0.999, 1e-8
    for parameter, parameter_gradient, momentum, variance in (
        (model.V, gradient_v, model.mV, model.vV),
        (model.W, gradient_w, model.mW, model.vW),
    ):
        momentum *= beta_one
        momentum += (1 - beta_one) * parameter_gradient
        variance *= beta_two
        variance += (1 - beta_two) * (parameter_gradient * parameter_gradient)
        parameter -= model.lr * (momentum / (1 - beta_one**model.t)) / (
            np.sqrt(variance / (1 - beta_two**model.t)) + epsilon
        )
    return float(np.mean(np.logaddexp(0.0, -difference)))

