from __future__ import annotations

from collections import defaultdict

import numpy as np


def build_pair_indices(users, labels, rng: np.random.Generator):
    """Sample one same-user negative for every positive belonging to a mixed user."""
    groups = defaultdict(lambda: [[], []])
    for index, (user, label) in enumerate(zip(users, labels)):
        groups[user][int(label)].append(index)
    positive_parts, negative_parts = [], []
    for negatives, positives in groups.values():
        if not positives or not negatives:
            continue
        positive = np.asarray(positives, dtype=np.int64)
        negative = rng.choice(np.asarray(negatives, dtype=np.int64), size=len(positive), replace=True)
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
