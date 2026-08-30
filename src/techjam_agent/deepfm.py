from __future__ import annotations

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


class DeepFM:
    """Small NumPy DeepFM sharing one embedding table across FM and MLP paths."""

    def __init__(
        self,
        dimension: int,
        fields: int,
        embedding_dim: int = 16,
        hidden_dim: int = 32,
        learning_rate: float = 0.0003,
        l2: float = 1e-6,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dimension, embedding_dim)).astype(np.float32)
        self.W = np.zeros(dimension, dtype=np.float32)
        self.b = np.asarray(0.0, dtype=np.float32)
        self.H = rng.normal(
            0, np.sqrt(2.0 / (fields * embedding_dim)),
            (fields * embedding_dim, hidden_dim),
        ).astype(np.float32)
        self.hb = np.zeros(hidden_dim, dtype=np.float32)
        self.O = rng.normal(0, 0.01, hidden_dim).astype(np.float32)
        self.ob = np.asarray(0.0, dtype=np.float32)
        self.lr = float(learning_rate)
        self.l2 = float(l2)
        self.t = 0
        self._parameters = {
            "V": self.V, "W": self.W, "b": self.b, "H": self.H,
            "hb": self.hb, "O": self.O, "ob": self.ob,
        }
        self._regularized_parameters = ("V", "W", "H", "O")
        self._momentum = {name: np.zeros_like(value) for name, value in self._parameters.items()}
        self._variance = {name: np.zeros_like(value) for name, value in self._parameters.items()}

    def _forward(self, X: np.ndarray):
        embeddings = self.V[X]
        summed = embeddings.sum(axis=1)
        interaction = 0.5 * (
            (summed * summed).sum(axis=1) - (embeddings * embeddings).sum(axis=(1, 2))
        )
        flattened = embeddings.reshape(len(X), -1)
        hidden_pre = flattened @ self.H + self.hb
        hidden = np.maximum(hidden_pre, 0.0)
        logits = self.b + self.W[X].sum(axis=1) + interaction + hidden @ self.O + self.ob
        return logits, (X, embeddings, summed, flattened, hidden_pre, hidden)

    def _gradients(self, cache, output_gradient: np.ndarray) -> dict[str, np.ndarray]:
        X, embeddings, summed, flattened, hidden_pre, hidden = cache
        gradient_o = hidden.T @ output_gradient
        gradient_ob = np.asarray(output_gradient.sum(), dtype=np.float32)
        gradient_hidden = output_gradient[:, None] * self.O[None, :]
        gradient_pre = gradient_hidden * (hidden_pre > 0)
        gradient_h = flattened.T @ gradient_pre
        gradient_hb = gradient_pre.sum(axis=0)
        gradient_flat = gradient_pre @ self.H.T
        gradient_embeddings = gradient_flat.reshape(embeddings.shape)
        gradient_embeddings += output_gradient[:, None, None] * (
            summed[:, None, :] - embeddings
        )
        gradient_v = np.zeros_like(self.V)
        gradient_w = np.zeros_like(self.W)
        np.add.at(gradient_v, X, gradient_embeddings)
        np.add.at(gradient_w, X, output_gradient[:, None])
        return {
            "V": gradient_v,
            "W": gradient_w,
            "b": np.asarray(output_gradient.sum(), dtype=np.float32),
            "H": gradient_h,
            "hb": gradient_hb,
            "O": gradient_o,
            "ob": gradient_ob,
        }

    def _apply(self, gradients: dict[str, np.ndarray]) -> None:
        for name in self._regularized_parameters:
            gradients[name] += self.l2 * self._parameters[name]
        self.t += 1
        beta_one, beta_two, epsilon = 0.9, 0.999, 1e-8
        for name, parameter in self._parameters.items():
            gradient = gradients[name].astype(np.float32, copy=False)
            momentum = self._momentum[name]
            variance = self._variance[name]
            momentum *= beta_one
            momentum += (1 - beta_one) * gradient
            variance *= beta_two
            variance += (1 - beta_two) * gradient * gradient
            parameter -= self.lr * (momentum / (1 - beta_one**self.t)) / (
                np.sqrt(variance / (1 - beta_two**self.t)) + epsilon
            )

    def step(self, X: np.ndarray, labels: np.ndarray) -> float:
        logits, cache = self._forward(X)
        probabilities = _sigmoid(logits)
        output_gradient = ((probabilities - labels) / len(labels)).astype(np.float32)
        self._apply(self._gradients(cache, output_gradient))
        return float(-np.mean(
            labels * np.log(probabilities + 1e-9)
            + (1 - labels) * np.log(1 - probabilities + 1e-9)
        ))

    def bpr_step(self, positive_x: np.ndarray, negative_x: np.ndarray) -> float:
        positive_logits, positive_cache = self._forward(positive_x)
        negative_logits, negative_cache = self._forward(negative_x)
        difference = positive_logits - negative_logits
        probability = _sigmoid(difference)
        positive_gradient = ((probability - 1.0) / len(difference)).astype(np.float32)
        positive = self._gradients(positive_cache, positive_gradient)
        negative = self._gradients(negative_cache, -positive_gradient)
        self._apply({name: positive[name] + negative[name] for name in positive})
        return float(np.mean(np.logaddexp(0.0, -difference)))

    def hybrid_step(
        self,
        positive_x: np.ndarray,
        negative_x: np.ndarray,
        bpr_weight: float,
    ) -> float:
        if not 0.0 <= bpr_weight <= 1.0:
            raise ValueError("bpr_weight must be between 0 and 1")
        positive_logits, positive_cache = self._forward(positive_x)
        negative_logits, negative_cache = self._forward(negative_x)
        pair_probability = _sigmoid(positive_logits - negative_logits)
        positive_probability = _sigmoid(positive_logits)
        negative_probability = _sigmoid(negative_logits)
        count = len(positive_logits)
        positive_gradient = (
            bpr_weight * (pair_probability - 1.0) / count
            + (1.0 - bpr_weight) * (positive_probability - 1.0) / (2 * count)
        ).astype(np.float32)
        negative_gradient = (
            bpr_weight * (1.0 - pair_probability) / count
            + (1.0 - bpr_weight) * negative_probability / (2 * count)
        ).astype(np.float32)
        positive = self._gradients(positive_cache, positive_gradient)
        negative = self._gradients(negative_cache, negative_gradient)
        self._apply({name: positive[name] + negative[name] for name in positive})
        bpr_loss = -np.mean(np.log(pair_probability + 1e-9))
        bce_loss = -0.5 * np.mean(
            np.log(positive_probability + 1e-9)
            + np.log(1.0 - negative_probability + 1e-9)
        )
        return float(bpr_weight * bpr_loss + (1.0 - bpr_weight) * bce_loss)

    def predict(self, X: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        return np.concatenate([
            self._forward(X[start:start + batch_size])[0]
            for start in range(0, len(X), batch_size)
        ])

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._parameters.items()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for name, parameter in self._parameters.items():
            parameter[...] = state[name]


class MultiTaskDeepFM(DeepFM):
    """DeepFM with shared representations and configurable auxiliary heads."""

    def __init__(self, *args, auxiliary_tasks: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        rng = np.random.default_rng(int(kwargs.get("seed", 0)) + 10_007)
        hidden_dim = self.H.shape[1]
        self.A = rng.normal(0, 0.01, (hidden_dim, auxiliary_tasks)).astype(np.float32)
        self.ab = np.zeros(auxiliary_tasks, dtype=np.float32)
        self._parameters.update({"A": self.A, "ab": self.ab})
        self._momentum.update({"A": np.zeros_like(self.A), "ab": np.zeros_like(self.ab)})
        self._variance.update({"A": np.zeros_like(self.A), "ab": np.zeros_like(self.ab)})
        self._regularized_parameters = (*self._regularized_parameters, "A")

    def multitask_step(
        self,
        X: np.ndarray,
        long_view_labels: np.ndarray,
        auxiliary_labels: np.ndarray,
        auxiliary_weight: float,
        auxiliary_mask: np.ndarray | None = None,
        auxiliary_loss: str = "bce",
    ) -> float:
        if auxiliary_labels.ndim != 2 or auxiliary_labels.shape[1] != self.A.shape[1]:
            raise ValueError("auxiliary_labels has the wrong shape")
        if auxiliary_mask is None:
            auxiliary_mask = np.ones_like(auxiliary_labels, dtype=np.float32)
        if auxiliary_mask.shape != auxiliary_labels.shape:
            raise ValueError("auxiliary_mask has the wrong shape")
        observed = max(1.0, float(auxiliary_mask.sum()))
        logits, cache = self._forward(X)
        _, _, _, flattened, hidden_pre, hidden = cache
        main_probabilities = _sigmoid(logits)
        main_gradient = (
            (main_probabilities - long_view_labels) / len(long_view_labels)
        ).astype(np.float32)
        gradients = self._gradients(cache, main_gradient)

        auxiliary_logits = hidden @ self.A + self.ab
        auxiliary_probabilities = _sigmoid(auxiliary_logits)
        if auxiliary_loss == "bce":
            loss_gradient = auxiliary_probabilities - auxiliary_labels
            element_loss = -(
                auxiliary_labels * np.log(auxiliary_probabilities + 1e-9)
                + (1 - auxiliary_labels) * np.log(1 - auxiliary_probabilities + 1e-9)
            )
        elif auxiliary_loss == "mse":
            difference = auxiliary_probabilities - auxiliary_labels
            loss_gradient = 2.0 * difference * auxiliary_probabilities * (
                1.0 - auxiliary_probabilities
            )
            element_loss = difference * difference
        else:
            raise ValueError("auxiliary_loss must be 'bce' or 'mse'")
        auxiliary_gradient = (
            auxiliary_weight * loss_gradient * auxiliary_mask / observed
        ).astype(np.float32)
        gradients["A"] = hidden.T @ auxiliary_gradient
        gradients["ab"] = auxiliary_gradient.sum(axis=0)
        hidden_gradient = auxiliary_gradient @ self.A.T
        pre_gradient = hidden_gradient * (hidden_pre > 0)
        gradients["H"] += flattened.T @ pre_gradient
        gradients["hb"] += pre_gradient.sum(axis=0)
        embedding_gradient = (pre_gradient @ self.H.T).reshape(
            len(X), X.shape[1], self.V.shape[1]
        )
        auxiliary_v = np.zeros_like(self.V)
        np.add.at(auxiliary_v, X, embedding_gradient)
        gradients["V"] += auxiliary_v
        self._apply(gradients)

        main_loss = -np.mean(
            long_view_labels * np.log(main_probabilities + 1e-9)
            + (1 - long_view_labels) * np.log(1 - main_probabilities + 1e-9)
        )
        auxiliary_loss = float((element_loss * auxiliary_mask).sum() / observed)
        return float(main_loss + auxiliary_weight * auxiliary_loss)

    def pairwise_multitask_step(
        self,
        positive_x: np.ndarray,
        negative_x: np.ndarray,
        positive_auxiliary: np.ndarray,
        negative_auxiliary: np.ndarray,
        auxiliary_weight: float,
        positive_mask: np.ndarray | None = None,
        negative_mask: np.ndarray | None = None,
        auxiliary_loss: str = "bce",
    ) -> float:
        """Optimize BPR for long-view ranking plus pointwise auxiliary feedback."""
        expected = (len(positive_x), self.A.shape[1])
        if positive_auxiliary.shape != expected or negative_auxiliary.shape != expected:
            raise ValueError("pairwise auxiliary labels have the wrong shape")
        if positive_mask is None:
            positive_mask = np.ones_like(positive_auxiliary, dtype=np.float32)
        if negative_mask is None:
            negative_mask = np.ones_like(negative_auxiliary, dtype=np.float32)
        if positive_mask.shape != expected or negative_mask.shape != expected:
            raise ValueError("pairwise auxiliary masks have the wrong shape")

        positive_logits, positive_cache = self._forward(positive_x)
        negative_logits, negative_cache = self._forward(negative_x)
        difference = positive_logits - negative_logits
        pair_probability = _sigmoid(difference)
        positive_gradient = (
            (pair_probability - 1.0) / len(difference)
        ).astype(np.float32)
        positive_gradients = self._gradients(positive_cache, positive_gradient)
        negative_gradients = self._gradients(negative_cache, -positive_gradient)
        gradients = {
            name: positive_gradients[name] + negative_gradients[name]
            for name in positive_gradients
        }
        gradients["A"] = np.zeros_like(self.A)
        gradients["ab"] = np.zeros_like(self.ab)

        observed = max(1.0, float(positive_mask.sum() + negative_mask.sum()))
        auxiliary_loss_sum = 0.0
        for cache, labels, mask in (
            (positive_cache, positive_auxiliary, positive_mask),
            (negative_cache, negative_auxiliary, negative_mask),
        ):
            X, _, _, flattened, hidden_pre, hidden = cache
            auxiliary_logits = hidden @ self.A + self.ab
            probabilities = _sigmoid(auxiliary_logits)
            if auxiliary_loss == "bce":
                loss_gradient = probabilities - labels
                element_loss = -(
                    labels * np.log(probabilities + 1e-9)
                    + (1 - labels) * np.log(1 - probabilities + 1e-9)
                )
            elif auxiliary_loss == "mse":
                difference_aux = probabilities - labels
                loss_gradient = 2.0 * difference_aux * probabilities * (
                    1.0 - probabilities
                )
                element_loss = difference_aux * difference_aux
            else:
                raise ValueError("auxiliary_loss must be 'bce' or 'mse'")
            auxiliary_gradient = (
                auxiliary_weight * loss_gradient * mask / observed
            ).astype(np.float32)
            gradients["A"] += hidden.T @ auxiliary_gradient
            gradients["ab"] += auxiliary_gradient.sum(axis=0)
            hidden_gradient = auxiliary_gradient @ self.A.T
            pre_gradient = hidden_gradient * (hidden_pre > 0)
            gradients["H"] += flattened.T @ pre_gradient
            gradients["hb"] += pre_gradient.sum(axis=0)
            embedding_gradient = (pre_gradient @ self.H.T).reshape(
                len(X), X.shape[1], self.V.shape[1]
            )
            auxiliary_v = np.zeros_like(self.V)
            np.add.at(auxiliary_v, X, embedding_gradient)
            gradients["V"] += auxiliary_v
            auxiliary_loss_sum += float((element_loss * mask).sum())

        self._apply(gradients)
        bpr_loss = float(np.mean(np.logaddexp(0.0, -difference)))
        return bpr_loss + auxiliary_weight * auxiliary_loss_sum / observed
