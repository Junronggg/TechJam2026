from __future__ import annotations

from pathlib import Path

import numpy as np


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


class FPMC:
    """Factorizing Personalized Markov Chains with a pairwise BPR update.

    The score combines long-term user-item preference with a first-order
    transition from the user's most recent positive item to the candidate.
    This small NumPy model is a low-cost sequence-value scout, not a claim that
    a first-order chain is equivalent to SASRec.
    """

    STATE_KEYS = (
        "user_factors", "item_factors", "previous_factors", "next_factors",
        "item_bias",
    )

    def __init__(
        self,
        user_count: int,
        item_count: int,
        embedding_dim: int = 16,
        learning_rate: float = 0.001,
        l2: float = 1e-6,
        seed: int = 0,
    ) -> None:
        if user_count < 1 or item_count < 1:
            raise ValueError("FPMC requires at least one user and one item")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        rng = np.random.default_rng(seed)
        shape_user = (int(user_count), int(embedding_dim))
        shape_item = (int(item_count), int(embedding_dim))
        self.user_factors = rng.normal(0.0, 0.01, shape_user).astype(np.float32)
        self.item_factors = rng.normal(0.0, 0.01, shape_item).astype(np.float32)
        # One additional row is the no-history padding state.
        self.previous_factors = rng.normal(
            0.0, 0.01, (int(item_count) + 1, int(embedding_dim))
        ).astype(np.float32)
        self.previous_factors[-1] = 0.0
        self.next_factors = rng.normal(0.0, 0.01, shape_item).astype(np.float32)
        self.item_bias = np.zeros(int(item_count), dtype=np.float32)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.t = 0
        for name in self.STATE_KEYS:
            value = getattr(self, name)
            setattr(self, f"m_{name}", np.zeros_like(value))
            setattr(self, f"v_{name}", np.zeros_like(value))

    def score(
        self, users: np.ndarray, items: np.ndarray, previous_items: np.ndarray
    ) -> np.ndarray:
        users = np.asarray(users, dtype=np.int32)
        items = np.asarray(items, dtype=np.int32)
        previous_items = np.asarray(previous_items, dtype=np.int32)
        if not (users.shape == items.shape == previous_items.shape):
            raise ValueError("users, items, and previous_items must share a shape")
        preference = np.sum(self.user_factors[users] * self.item_factors[items], axis=1)
        transition = np.sum(
            self.previous_factors[previous_items] * self.next_factors[items], axis=1
        )
        return self.item_bias[items] + preference + transition

    def _adam(self, name: str, gradient: np.ndarray) -> None:
        parameter = getattr(self, name)
        momentum = getattr(self, f"m_{name}")
        variance = getattr(self, f"v_{name}")
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        momentum *= beta1
        momentum += (1.0 - beta1) * gradient
        variance *= beta2
        variance += (1.0 - beta2) * gradient * gradient
        parameter -= self.learning_rate * (momentum / (1.0 - beta1 ** self.t)) / (
            np.sqrt(variance / (1.0 - beta2 ** self.t)) + epsilon
        )

    def bpr_step(
        self,
        users: np.ndarray,
        positive_items: np.ndarray,
        negative_items: np.ndarray,
        previous_items: np.ndarray,
    ) -> float:
        if not len(users):
            raise ValueError("FPMC BPR batch must not be empty")
        positive_score = self.score(users, positive_items, previous_items)
        negative_score = self.score(users, negative_items, previous_items)
        probability = _sigmoid(positive_score - negative_score)
        score_gradient = ((probability - 1.0) / len(users)).astype(np.float32)

        gradients = {
            name: np.zeros_like(getattr(self, name)) for name in self.STATE_KEYS
        }
        user_values = self.user_factors[users].copy()
        positive_values = self.item_factors[positive_items].copy()
        negative_values = self.item_factors[negative_items].copy()
        previous_values = self.previous_factors[previous_items].copy()
        positive_next = self.next_factors[positive_items].copy()
        negative_next = self.next_factors[negative_items].copy()

        np.add.at(
            gradients["user_factors"], users,
            score_gradient[:, None] * (positive_values - negative_values),
        )
        np.add.at(
            gradients["item_factors"], positive_items,
            score_gradient[:, None] * user_values,
        )
        np.add.at(
            gradients["item_factors"], negative_items,
            -score_gradient[:, None] * user_values,
        )
        np.add.at(
            gradients["previous_factors"], previous_items,
            score_gradient[:, None] * (positive_next - negative_next),
        )
        np.add.at(
            gradients["next_factors"], positive_items,
            score_gradient[:, None] * previous_values,
        )
        np.add.at(
            gradients["next_factors"], negative_items,
            -score_gradient[:, None] * previous_values,
        )
        np.add.at(gradients["item_bias"], positive_items, score_gradient)
        np.add.at(gradients["item_bias"], negative_items, -score_gradient)

        for name in self.STATE_KEYS:
            gradients[name] += self.l2 * getattr(self, name)
        # Padding represents no sequence history and must remain neutral.
        gradients["previous_factors"][-1] = 0.0
        self.t += 1
        for name in self.STATE_KEYS:
            self._adam(name, gradients[name])
        self.previous_factors[-1] = 0.0
        return float(-np.mean(np.log(probability + 1e-9)))

    def predict(
        self,
        users: np.ndarray,
        items: np.ndarray,
        previous_items: np.ndarray,
        batch_size: int = 200_000,
    ) -> np.ndarray:
        return np.concatenate([
            self.score(
                users[start:start + batch_size],
                items[start:start + batch_size],
                previous_items[start:start + batch_size],
            )
            for start in range(0, len(users), batch_size)
        ])

    def save(self, path: Path, *, best_epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            **{name: getattr(self, name) for name in self.STATE_KEYS},
            best_epoch=np.asarray(best_epoch),
        )

    def load(self, path: Path) -> int:
        with np.load(path) as state:
            for name in self.STATE_KEYS:
                value = np.asarray(state[name])
                expected = getattr(self, name).shape
                if value.shape != expected:
                    raise ValueError(
                        f"FPMC checkpoint {name} shape {value.shape} != expected {expected}"
                    )
                setattr(self, name, value.copy())
            best_epoch = int(state["best_epoch"])
        return best_epoch


class SequentialFM:
    """FM over request fields plus a recent-positive item transition term."""

    STATE_KEYS = ("V", "W", "previous_factors", "next_factors")

    def __init__(
        self,
        categorical_dim: int,
        item_count: int,
        embedding_dim: int = 16,
        learning_rate: float = 0.001,
        l2: float = 1e-6,
        seed: int = 0,
    ) -> None:
        if categorical_dim < 1 or item_count < 1 or embedding_dim < 1:
            raise ValueError("SequentialFM dimensions must be positive")
        rng = np.random.default_rng(seed)
        self.V = rng.normal(
            0.0, 0.01, (int(categorical_dim), int(embedding_dim))
        ).astype(np.float32)
        self.W = np.zeros(int(categorical_dim), dtype=np.float32)
        self.previous_factors = rng.normal(
            0.0, 0.01, (int(item_count) + 1, int(embedding_dim))
        ).astype(np.float32)
        self.previous_factors[-1] = 0.0
        self.next_factors = rng.normal(
            0.0, 0.01, (int(item_count), int(embedding_dim))
        ).astype(np.float32)
        self.b = np.float32(0.0)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.t = 0
        for name in self.STATE_KEYS:
            value = getattr(self, name)
            setattr(self, f"m_{name}", np.zeros_like(value))
            setattr(self, f"v_{name}", np.zeros_like(value))

    def _components(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        embeddings = self.V[X]
        summed = embeddings.sum(axis=1)
        interactions = 0.5 * (
            (summed * summed).sum(axis=1) - (embeddings * embeddings).sum(axis=(1, 2))
        )
        logits = self.b + self.W[X].sum(axis=1) + interactions
        return logits, embeddings, summed

    def score(
        self, X: np.ndarray, items: np.ndarray, previous_items: np.ndarray
    ) -> np.ndarray:
        logits = self._components(X)[0]
        transition = np.sum(
            self.previous_factors[previous_items] * self.next_factors[items], axis=1
        )
        return logits + transition

    def _adam(self, name: str, gradient: np.ndarray) -> None:
        parameter = getattr(self, name)
        momentum = getattr(self, f"m_{name}")
        variance = getattr(self, f"v_{name}")
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        momentum *= beta1
        momentum += (1.0 - beta1) * gradient
        variance *= beta2
        variance += (1.0 - beta2) * gradient * gradient
        parameter -= self.learning_rate * (momentum / (1.0 - beta1 ** self.t)) / (
            np.sqrt(variance / (1.0 - beta2 ** self.t)) + epsilon
        )

    def bpr_step(
        self,
        positive_x: np.ndarray,
        negative_x: np.ndarray,
        positive_items: np.ndarray,
        negative_items: np.ndarray,
        previous_items: np.ndarray,
    ) -> float:
        if not len(positive_x):
            raise ValueError("SequentialFM BPR batch must not be empty")
        positive_logits, positive_e, positive_sum = self._components(positive_x)
        negative_logits, negative_e, negative_sum = self._components(negative_x)
        previous_values = self.previous_factors[previous_items].copy()
        positive_next = self.next_factors[positive_items].copy()
        negative_next = self.next_factors[negative_items].copy()
        difference = positive_logits - negative_logits + np.sum(
            previous_values * (positive_next - negative_next), axis=1
        )
        probability = _sigmoid(difference)
        gradient = ((probability - 1.0) / len(difference)).astype(np.float32)

        gradient_v = np.zeros_like(self.V)
        gradient_w = np.zeros_like(self.W)
        gradient_previous = np.zeros_like(self.previous_factors)
        gradient_next = np.zeros_like(self.next_factors)
        np.add.at(gradient_w, positive_x, gradient[:, None])
        np.add.at(gradient_w, negative_x, -gradient[:, None])
        np.add.at(
            gradient_v, positive_x,
            gradient[:, None, None] * (positive_sum[:, None, :] - positive_e),
        )
        np.add.at(
            gradient_v, negative_x,
            -gradient[:, None, None] * (negative_sum[:, None, :] - negative_e),
        )
        np.add.at(
            gradient_previous, previous_items,
            gradient[:, None] * (positive_next - negative_next),
        )
        np.add.at(
            gradient_next, positive_items, gradient[:, None] * previous_values,
        )
        np.add.at(
            gradient_next, negative_items, -gradient[:, None] * previous_values,
        )
        gradient_v += self.l2 * self.V
        gradient_w += self.l2 * self.W
        gradient_previous += self.l2 * self.previous_factors
        gradient_next += self.l2 * self.next_factors
        gradient_previous[-1] = 0.0
        self.t += 1
        for name, value in (
            ("V", gradient_v),
            ("W", gradient_w),
            ("previous_factors", gradient_previous),
            ("next_factors", gradient_next),
        ):
            self._adam(name, value)
        self.previous_factors[-1] = 0.0
        return float(-np.mean(np.log(probability + 1e-9)))

    def predict(
        self,
        X: np.ndarray,
        items: np.ndarray,
        previous_items: np.ndarray,
        batch_size: int = 200_000,
    ) -> np.ndarray:
        return np.concatenate([
            self.score(
                X[start:start + batch_size],
                items[start:start + batch_size],
                previous_items[start:start + batch_size],
            )
            for start in range(0, len(X), batch_size)
        ])

    def save(self, path: Path, *, best_epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            **{name: getattr(self, name) for name in self.STATE_KEYS},
            b=np.asarray(self.b),
            best_epoch=np.asarray(best_epoch),
        )

    def load(self, path: Path) -> int:
        with np.load(path) as state:
            for name in self.STATE_KEYS:
                value = np.asarray(state[name])
                expected = getattr(self, name).shape
                if value.shape != expected:
                    raise ValueError(
                        f"SequentialFM checkpoint {name} shape {value.shape} != {expected}"
                    )
                setattr(self, name, value.copy())
            self.b = np.float32(state["b"])
            best_epoch = int(state["best_epoch"])
        return best_epoch
