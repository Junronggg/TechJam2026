from __future__ import annotations

import numpy as np


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


class FieldAwareFM:
    """Small NumPy field-aware factorization machine.

    Each categorical value owns a different embedding for every field it can
    interact with. This is a genuinely different model family from the starter
    FM while retaining the same deterministic optimizer and checkpoint format.
    """

    def __init__(self, dim: int, fields: int, k: int = 16, lr: float = 0.001,
                 l2: float = 1e-6, seed: int = 0) -> None:
        if fields < 2:
            raise ValueError("FFM requires at least two fields")
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0.0, 0.01, (dim, fields, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.fields = int(fields)
        self.lr, self.l2 = float(lr), float(l2)
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2 or X.shape[1] != self.fields:
            raise ValueError(f"FFM expected {self.fields} fields, got shape {X.shape}")
        score = self.b + self.W[X].sum(axis=1)
        for left in range(self.fields - 1):
            left_ids = X[:, left]
            for right in range(left + 1, self.fields):
                left_embedding = self.V[left_ids, right]
                right_embedding = self.V[X[:, right], left]
                score = score + np.sum(left_embedding * right_embedding, axis=1)
        return score

    def _accumulate_score_gradient(
        self,
        X: np.ndarray,
        score_gradient: np.ndarray,
        gradient_v: np.ndarray,
        gradient_w: np.ndarray,
    ) -> None:
        np.add.at(gradient_w, X, score_gradient[:, None])
        batch = len(X)
        for left in range(self.fields - 1):
            left_ids = X[:, left]
            for right in range(left + 1, self.fields):
                right_ids = X[:, right]
                left_embedding = self.V[left_ids, right].copy()
                right_embedding = self.V[right_ids, left].copy()
                np.add.at(
                    gradient_v,
                    (left_ids, np.full(batch, right, dtype=np.intp)),
                    score_gradient[:, None] * right_embedding,
                )
                np.add.at(
                    gradient_v,
                    (right_ids, np.full(batch, left, dtype=np.intp)),
                    score_gradient[:, None] * left_embedding,
                )

    def _apply(self, gradient_v: np.ndarray, gradient_w: np.ndarray) -> None:
        gradient_v += self.l2 * self.V
        gradient_w += self.l2 * self.W
        self.t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, gradient, momentum, variance in (
            (self.V, gradient_v, self.mV, self.vV),
            (self.W, gradient_w, self.mW, self.vW),
        ):
            momentum *= beta1
            momentum += (1.0 - beta1) * gradient
            variance *= beta2
            variance += (1.0 - beta2) * (gradient * gradient)
            parameter -= self.lr * (momentum / (1.0 - beta1 ** self.t)) / (
                np.sqrt(variance / (1.0 - beta2 ** self.t)) + epsilon
            )

    def step(self, X: np.ndarray, labels: np.ndarray) -> float:
        logits = self.logits(X)
        probabilities = _sigmoid(logits)
        score_gradient = ((probabilities - labels) / len(labels)).astype(np.float32)
        gradient_v = np.zeros_like(self.V)
        gradient_w = np.zeros_like(self.W)
        self._accumulate_score_gradient(X, score_gradient, gradient_v, gradient_w)
        self._apply(gradient_v, gradient_w)
        self.b -= self.lr * score_gradient.sum()
        return float(-np.mean(
            labels * np.log(probabilities + 1e-9)
            + (1.0 - labels) * np.log(1.0 - probabilities + 1e-9)
        ))

    def bpr_step(self, positive_x: np.ndarray, negative_x: np.ndarray) -> float:
        difference = self.logits(positive_x) - self.logits(negative_x)
        probability = _sigmoid(difference)
        positive_gradient = ((probability - 1.0) / len(difference)).astype(np.float32)
        gradient_v = np.zeros_like(self.V)
        gradient_w = np.zeros_like(self.W)
        self._accumulate_score_gradient(
            positive_x, positive_gradient, gradient_v, gradient_w
        )
        self._accumulate_score_gradient(
            negative_x, -positive_gradient, gradient_v, gradient_w
        )
        self._apply(gradient_v, gradient_w)
        return float(-np.mean(np.log(probability + 1e-9)))

    def predict(self, X: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        return np.concatenate([
            self.logits(X[start:start + batch_size])
            for start in range(0, len(X), batch_size)
        ])
