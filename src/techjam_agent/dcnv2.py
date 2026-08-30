from __future__ import annotations

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


class DCNv2:
    """Small low-rank DCNv2 with a parallel one-layer deep path."""

    def __init__(
        self,
        dimension: int,
        fields: int,
        embedding_dim: int = 16,
        hidden_dim: int = 32,
        cross_layers: int = 2,
        cross_rank: int = 16,
        learning_rate: float = 0.001,
        l2: float = 1e-6,
        seed: int = 0,
    ) -> None:
        if cross_layers < 1 or cross_rank < 1:
            raise ValueError("cross_layers and cross_rank must be positive")
        rng = np.random.default_rng(seed)
        input_dim = fields * embedding_dim
        self.V = rng.normal(0, 0.01, (dimension, embedding_dim)).astype(np.float32)
        self.W = np.zeros(dimension, dtype=np.float32)
        self.b = np.asarray(0.0, dtype=np.float32)
        self.H = rng.normal(
            0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim)
        ).astype(np.float32)
        self.hb = np.zeros(hidden_dim, dtype=np.float32)
        self.U = rng.normal(
            0, 0.01, (cross_layers, input_dim, cross_rank)
        ).astype(np.float32)
        self.R = rng.normal(
            0, 0.01, (cross_layers, cross_rank, input_dim)
        ).astype(np.float32)
        self.cb = np.zeros((cross_layers, input_dim), dtype=np.float32)
        self.Oc = rng.normal(0, 0.01, input_dim).astype(np.float32)
        self.Od = rng.normal(0, 0.01, hidden_dim).astype(np.float32)
        self.ob = np.asarray(0.0, dtype=np.float32)
        self.lr = float(learning_rate)
        self.l2 = float(l2)
        self.t = 0
        self._parameters = {
            "V": self.V,
            "W": self.W,
            "b": self.b,
            "H": self.H,
            "hb": self.hb,
            "U": self.U,
            "R": self.R,
            "cb": self.cb,
            "Oc": self.Oc,
            "Od": self.Od,
            "ob": self.ob,
        }
        self._regularized_parameters = ("V", "W", "H", "U", "R", "Oc", "Od")
        self._momentum = {
            name: np.zeros_like(value) for name, value in self._parameters.items()
        }
        self._variance = {
            name: np.zeros_like(value) for name, value in self._parameters.items()
        }

    def _forward(self, X: np.ndarray):
        embeddings = self.V[X]
        x0 = embeddings.reshape(len(X), -1)
        hidden_pre = x0 @ self.H + self.hb
        hidden = np.maximum(hidden_pre, 0.0)

        cross_states = [x0]
        projections = []
        cross_values = []
        cross = x0
        for layer in range(len(self.U)):
            projection = cross @ self.U[layer]
            value = projection @ self.R[layer] + self.cb[layer]
            cross = x0 * value + cross
            projections.append(projection)
            cross_values.append(value)
            cross_states.append(cross)
        logits = (
            self.b
            + self.W[X].sum(axis=1)
            + cross @ self.Oc
            + hidden @ self.Od
            + self.ob
        )
        cache = (
            X,
            embeddings,
            x0,
            hidden_pre,
            hidden,
            cross_states,
            projections,
            cross_values,
        )
        return logits, cache

    def _gradients(self, cache, output_gradient: np.ndarray) -> dict[str, np.ndarray]:
        (
            X,
            embeddings,
            x0,
            hidden_pre,
            hidden,
            cross_states,
            projections,
            cross_values,
        ) = cache
        gradient_od = hidden.T @ output_gradient
        hidden_gradient = output_gradient[:, None] * self.Od[None, :]
        hidden_pre_gradient = hidden_gradient * (hidden_pre > 0)
        gradient_h = x0.T @ hidden_pre_gradient
        gradient_hb = hidden_pre_gradient.sum(axis=0)
        x0_gradient = hidden_pre_gradient @ self.H.T

        gradient_oc = cross_states[-1].T @ output_gradient
        cross_gradient = output_gradient[:, None] * self.Oc[None, :]
        gradient_u = np.zeros_like(self.U)
        gradient_r = np.zeros_like(self.R)
        gradient_cb = np.zeros_like(self.cb)
        for layer in range(len(self.U) - 1, -1, -1):
            value = cross_values[layer]
            previous_cross = cross_states[layer]
            projection = projections[layer]
            x0_gradient += cross_gradient * value
            value_gradient = cross_gradient * x0
            gradient_r[layer] = projection.T @ value_gradient
            gradient_cb[layer] = value_gradient.sum(axis=0)
            projection_gradient = value_gradient @ self.R[layer].T
            gradient_u[layer] = previous_cross.T @ projection_gradient
            cross_gradient = cross_gradient + projection_gradient @ self.U[layer].T
        x0_gradient += cross_gradient

        embedding_gradient = x0_gradient.reshape(embeddings.shape)
        gradient_v = np.zeros_like(self.V)
        gradient_w = np.zeros_like(self.W)
        np.add.at(gradient_v, X, embedding_gradient)
        np.add.at(gradient_w, X, output_gradient[:, None])
        return {
            "V": gradient_v,
            "W": gradient_w,
            "b": np.asarray(output_gradient.sum(), dtype=np.float32),
            "H": gradient_h,
            "hb": gradient_hb,
            "U": gradient_u,
            "R": gradient_r,
            "cb": gradient_cb,
            "Oc": gradient_oc,
            "Od": gradient_od,
            "ob": np.asarray(output_gradient.sum(), dtype=np.float32),
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
