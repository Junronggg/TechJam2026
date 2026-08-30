from __future__ import annotations

from collections import defaultdict

import numpy as np


def build_pair_indices(
    users,
    labels,
    rng: np.random.Generator,
    pairs_per_positive: int = 1,
    *,
    negative_scores=None,
    hard_negative_candidates: int = 2,
):
    """Sample same-user negatives for every positive belonging to a mixed user."""
    if pairs_per_positive < 1:
        raise ValueError("pairs_per_positive must be at least 1")
    if negative_scores is not None:
        negative_scores = np.asarray(negative_scores)
        if len(negative_scores) != len(labels):
            raise ValueError("negative_scores and labels must have the same length")
        if hard_negative_candidates < 2:
            raise ValueError("hard_negative_candidates must be at least 2")
    groups = defaultdict(lambda: [[], []])
    for index, (user, label) in enumerate(zip(users, labels)):
        groups[user][int(label)].append(index)
    positive_parts, negative_parts = [], []
    for negatives, positives in groups.values():
        if not positives or not negatives:
            continue
        positive = np.repeat(np.asarray(positives, dtype=np.int64), pairs_per_positive)
        negative_pool = np.asarray(negatives, dtype=np.int64)
        if negative_scores is None:
            negative = rng.choice(negative_pool, size=len(positive), replace=True)
        else:
            candidates = rng.choice(
                negative_pool,
                size=(len(positive), hard_negative_candidates),
                replace=True,
            )
            hardest = np.argmax(negative_scores[candidates], axis=1)
            negative = candidates[np.arange(len(positive)), hardest]
        positive_parts.append(positive); negative_parts.append(negative)
    if not positive_parts:
        raise ValueError("no users contain both positive and negative training examples")
    positive = np.concatenate(positive_parts)
    negative = np.concatenate(negative_parts)
    order = rng.permutation(len(positive))
    return positive[order], negative[order]


def bpr_step(model, positive_x, negative_x) -> float:
    """Apply one Adam update for -log(sigmoid(score_pos-score_neg))."""
    positive_z, positive_e, positive_s = model.logits(positive_x)
    negative_z, negative_e, negative_s = model.logits(negative_x)
    difference = positive_z - negative_z
    pair_probability = 1.0 / (1.0 + np.exp(-np.clip(difference, -30, 30)))
    positive_g = ((pair_probability - 1.0) / len(difference)).astype(np.float32)
    negative_g = -positive_g
    gradient_v = np.zeros_like(model.V)
    gradient_w = np.zeros_like(model.W)
    np.add.at(gradient_w, positive_x, positive_g[:, None])
    np.add.at(gradient_w, negative_x, negative_g[:, None])
    np.add.at(gradient_v, positive_x,
              positive_g[:, None, None] * (positive_s[:, None, :] - positive_e))
    np.add.at(gradient_v, negative_x,
              negative_g[:, None, None] * (negative_s[:, None, :] - negative_e))
    gradient_v += model.l2 * model.V
    gradient_w += model.l2 * model.W
    model.t += 1
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for parameter, gradient, momentum, variance in (
        (model.V, gradient_v, model.mV, model.vV),
        (model.W, gradient_w, model.mW, model.vW),
    ):
        momentum *= beta1; momentum += (1 - beta1) * gradient
        variance *= beta2; variance += (1 - beta2) * (gradient * gradient)
        parameter -= model.lr * (momentum / (1 - beta1 ** model.t)) / (
            np.sqrt(variance / (1 - beta2 ** model.t)) + epsilon)
    return float(-np.mean(np.log(pair_probability + 1e-9)))


def hybrid_step(model, positive_x, negative_x, bpr_weight: float) -> float:
    """One Adam update for a weighted BPR and pointwise BCE objective."""
    if not 0.0 <= bpr_weight <= 1.0:
        raise ValueError("bpr_weight must be between 0 and 1")
    positive_z, positive_e, positive_s = model.logits(positive_x)
    negative_z, negative_e, negative_s = model.logits(negative_x)
    pair_probability = 1.0 / (
        1.0 + np.exp(-np.clip(positive_z - negative_z, -30, 30))
    )
    positive_probability = 1.0 / (1.0 + np.exp(-np.clip(positive_z, -30, 30)))
    negative_probability = 1.0 / (1.0 + np.exp(-np.clip(negative_z, -30, 30)))
    count = len(positive_z)
    positive_g = (
        bpr_weight * (pair_probability - 1.0) / count
        + (1.0 - bpr_weight) * (positive_probability - 1.0) / (2 * count)
    ).astype(np.float32)
    negative_g = (
        bpr_weight * (1.0 - pair_probability) / count
        + (1.0 - bpr_weight) * negative_probability / (2 * count)
    ).astype(np.float32)
    gradient_v = np.zeros_like(model.V)
    gradient_w = np.zeros_like(model.W)
    np.add.at(gradient_w, positive_x, positive_g[:, None])
    np.add.at(gradient_w, negative_x, negative_g[:, None])
    np.add.at(
        gradient_v,
        positive_x,
        positive_g[:, None, None] * (positive_s[:, None, :] - positive_e),
    )
    np.add.at(
        gradient_v,
        negative_x,
        negative_g[:, None, None] * (negative_s[:, None, :] - negative_e),
    )
    gradient_v += model.l2 * model.V
    gradient_w += model.l2 * model.W
    model.t += 1
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for parameter, gradient, momentum, variance in (
        (model.V, gradient_v, model.mV, model.vV),
        (model.W, gradient_w, model.mW, model.vW),
    ):
        momentum *= beta1
        momentum += (1 - beta1) * gradient
        variance *= beta2
        variance += (1 - beta2) * gradient * gradient
        parameter -= model.lr * (momentum / (1 - beta1**model.t)) / (
            np.sqrt(variance / (1 - beta2**model.t)) + epsilon
        )
    model.b -= model.lr * (positive_g.sum() + negative_g.sum())
    bpr_loss = -np.mean(np.log(pair_probability + 1e-9))
    bce_loss = -0.5 * np.mean(
        np.log(positive_probability + 1e-9) + np.log(1.0 - negative_probability + 1e-9)
    )
    return float(bpr_weight * bpr_loss + (1.0 - bpr_weight) * bce_loss)
