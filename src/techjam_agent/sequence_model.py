from __future__ import annotations

import numpy as np

from .deepfm import DeepFM, _sigmoid


class LightweightSequenceDeepFM(DeepFM):
    """DeepFM plus single-head candidate-conditioned attention over last-K events."""

    def __init__(self, *args, sequence_length: int = 16, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        rng = np.random.default_rng(int(kwargs.get("seed", 0)) + 20_011)
        embedding_dim = self.V.shape[1]
        scale = np.sqrt(2.0 / embedding_dim)
        self.SQ = rng.normal(0, scale, (embedding_dim, embedding_dim)).astype(np.float32)
        self.SK = rng.normal(0, scale, (embedding_dim, embedding_dim)).astype(np.float32)
        self.SV = rng.normal(0, scale, (embedding_dim, embedding_dim)).astype(np.float32)
        self.behavior_embedding = rng.normal(0, 0.01, (3, embedding_dim)).astype(np.float32)
        self.time_gap_embedding = rng.normal(0, 0.01, (6, embedding_dim)).astype(np.float32)
        self.position_embedding = rng.normal(
            0, 0.01, (sequence_length, embedding_dim)
        ).astype(np.float32)
        self.sequence_scale = np.asarray(0.1, dtype=np.float32)
        self.sequence_length = int(sequence_length)
        additions = {
            "SQ": self.SQ,
            "SK": self.SK,
            "SV": self.SV,
            "behavior_embedding": self.behavior_embedding,
            "time_gap_embedding": self.time_gap_embedding,
            "position_embedding": self.position_embedding,
            "sequence_scale": self.sequence_scale,
        }
        self._parameters.update(additions)
        self._momentum.update({name: np.zeros_like(value) for name, value in additions.items()})
        self._variance.update({name: np.zeros_like(value) for name, value in additions.items()})
        self._regularized_parameters = (
            *self._regularized_parameters,
            "SQ", "SK", "SV", "behavior_embedding", "time_gap_embedding",
            "position_embedding",
        )

    def _sequence_forward(self, X: np.ndarray, history: dict[str, np.ndarray]):
        mask = np.asarray(history["mask"], dtype=np.float32)
        if mask.shape != (len(X), self.sequence_length):
            raise ValueError("history mask has the wrong shape")
        video_ids = np.asarray(history["video_id"], dtype=np.int32)
        author_ids = np.asarray(history["author_id"], dtype=np.int32)
        behavior_ids = np.asarray(history["behavior"], dtype=np.int32)
        time_gap_ids = np.asarray(history["time_gap"], dtype=np.int32)
        candidate = self.V[X[:, 1]] + self.V[X[:, 2]]
        history_input = (
            self.V[video_ids]
            + self.V[author_ids]
            + self.behavior_embedding[behavior_ids]
            + self.time_gap_embedding[time_gap_ids]
            + self.position_embedding[None, :, :]
        ) * mask[:, :, None]
        query = candidate @ self.SQ
        keys = history_input @ self.SK
        values = history_input @ self.SV
        normalizer = np.sqrt(float(self.V.shape[1]))
        attention_logits = np.einsum("bd,bkd->bk", query, keys) / normalizer
        masked_logits = np.where(mask > 0, attention_logits, -1e9)
        maxima = np.max(masked_logits, axis=1, keepdims=True)
        exponentials = np.exp(np.clip(masked_logits - maxima, -30, 0)) * mask
        weights = exponentials / (exponentials.sum(axis=1, keepdims=True) + 1e-9)
        context = np.einsum("bk,bkd->bd", weights, values)
        raw_score = np.sum(context * query, axis=1) / normalizer
        score = self.sequence_scale * raw_score
        cache = (
            X, video_ids, author_ids, behavior_ids, time_gap_ids, mask,
            candidate, history_input, query, keys, values, weights, context,
            raw_score, normalizer,
        )
        return score, cache

    def _sequence_gradients(
        self,
        cache,
        output_gradient: np.ndarray,
    ) -> dict[str, np.ndarray]:
        (
            X, video_ids, author_ids, behavior_ids, time_gap_ids, mask,
            candidate, history_input, query, keys, values, weights, context,
            raw_score, normalizer,
        ) = cache
        scaled_gradient = output_gradient * self.sequence_scale
        gradient_context = scaled_gradient[:, None] * query / normalizer
        gradient_query = scaled_gradient[:, None] * context / normalizer
        gradient_values = weights[:, :, None] * gradient_context[:, None, :]
        gradient_weights = np.einsum("bd,bkd->bk", gradient_context, values)
        centered = gradient_weights - np.sum(
            gradient_weights * weights, axis=1, keepdims=True
        )
        gradient_attention = weights * centered * mask
        gradient_query += np.einsum(
            "bk,bkd->bd", gradient_attention, keys
        ) / normalizer
        gradient_keys = (
            gradient_attention[:, :, None] * query[:, None, :] / normalizer
        )
        gradient_sq = candidate.T @ gradient_query
        gradient_candidate = gradient_query @ self.SQ.T
        gradient_sk = np.einsum("bki,bkj->ij", history_input, gradient_keys)
        gradient_history = gradient_keys @ self.SK.T
        gradient_sv = np.einsum("bki,bkj->ij", history_input, gradient_values)
        gradient_history += gradient_values @ self.SV.T
        gradient_history *= mask[:, :, None]

        gradient_v = np.zeros_like(self.V)
        np.add.at(gradient_v, X[:, 1], gradient_candidate)
        np.add.at(gradient_v, X[:, 2], gradient_candidate)
        np.add.at(gradient_v, video_ids, gradient_history)
        np.add.at(gradient_v, author_ids, gradient_history)
        gradient_behavior = np.zeros_like(self.behavior_embedding)
        gradient_time = np.zeros_like(self.time_gap_embedding)
        np.add.at(gradient_behavior, behavior_ids, gradient_history)
        np.add.at(gradient_time, time_gap_ids, gradient_history)
        return {
            "V": gradient_v,
            "SQ": gradient_sq,
            "SK": gradient_sk,
            "SV": gradient_sv,
            "behavior_embedding": gradient_behavior,
            "time_gap_embedding": gradient_time,
            "position_embedding": gradient_history.sum(axis=0),
            "sequence_scale": np.asarray(
                np.sum(output_gradient * raw_score), dtype=np.float32
            ),
        }

    def step(
        self,
        X: np.ndarray,
        history: dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> float:
        base_logits, base_cache = self._forward(X)
        sequence_logits, sequence_cache = self._sequence_forward(X, history)
        probabilities = _sigmoid(base_logits + sequence_logits)
        output_gradient = ((probabilities - labels) / len(labels)).astype(np.float32)
        gradients = self._gradients(base_cache, output_gradient)
        sequence_gradients = self._sequence_gradients(sequence_cache, output_gradient)
        for name, value in sequence_gradients.items():
            if name in gradients:
                gradients[name] += value
            else:
                gradients[name] = value
        self._apply(gradients)
        return float(-np.mean(
            labels * np.log(probabilities + 1e-9)
            + (1 - labels) * np.log(1 - probabilities + 1e-9)
        ))

    def predict(
        self,
        X: np.ndarray,
        history: dict[str, np.ndarray],
        batch_size: int = 100_000,
    ) -> np.ndarray:
        scores = []
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            history_batch = {
                name: values[start:end]
                for name, values in history.items()
                if name != "length"
            }
            base, _ = self._forward(X[start:end])
            sequence, _ = self._sequence_forward(X[start:end], history_batch)
            scores.append(base + sequence)
        return np.concatenate(scores)
