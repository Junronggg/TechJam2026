"""Controlled FM+BPR watch-time auxiliary ablation.

This is intentionally a small research script rather than a new fixed action
menu.  It tests the missing comparison exposed by the external run:
FM+BPR + censored watch-time supervision, with and without the two label-free
history fields (prior exposure and author recency).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.bpr import build_pair_indices
from techjam_agent.feedback import align_censored_watch_feedback
from techjam_agent.runner import ExperimentRunner


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


class FMWatchtime:
    """FM+BPR with a censored-watch head on the summed field embedding."""

    def __init__(self, dimension: int, fields: int, *, lr: float, l2: float,
                 seed: int, aux_weight: float) -> None:
        del fields  # The auxiliary head consumes the summed k-dimensional embedding.
        base = np.random.default_rng(seed)
        self.V = base.normal(0.0, 0.01, (dimension, 16)).astype(np.float32)
        self.W = np.zeros(dimension, dtype=np.float32)
        self.b = np.float32(0.0)
        self.A = base.normal(0.0, 0.01, 16).astype(np.float32)
        self.ab = np.float32(0.0)
        self.lr, self.l2, self.aux_weight = float(lr), float(l2), float(aux_weight)
        self.mV, self.vV = np.zeros_like(self.V), np.zeros_like(self.V)
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mA, self.vA = np.zeros_like(self.A), np.zeros_like(self.A)
        self.mab = np.float32(0.0)
        self.vab = np.float32(0.0)
        self.t = 0

    def _fm(self, X: np.ndarray):
        E = self.V[X]
        S = E.sum(axis=1)
        interaction = 0.5 * ((S * S).sum(axis=1) - (E * E).sum(axis=(1, 2)))
        return self.b + self.W[X].sum(axis=1) + interaction, E, S

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.concatenate([
            self._fm(X[start:start + 200_000])[0]
            for start in range(0, len(X), 200_000)
        ])

    def _apply(self, gradients: dict[str, np.ndarray | np.float32]) -> None:
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        params = (("V", self.V, self.mV, self.vV), ("W", self.W, self.mW, self.vW),
                  ("A", self.A, self.mA, self.vA))
        for name, parameter, momentum, variance in params:
            gradient = np.asarray(gradients[name], dtype=np.float32)
            if name in ("V", "W", "A"):
                gradient = gradient + self.l2 * parameter
            momentum *= beta1
            momentum += (1.0 - beta1) * gradient
            variance *= beta2
            variance += (1.0 - beta2) * gradient * gradient
            parameter -= self.lr * (momentum / (1.0 - beta1 ** self.t)) / (
                np.sqrt(variance / (1.0 - beta2 ** self.t)) + eps
            )
        grad_ab = np.float32(gradients["ab"])
        self.mab = beta1 * self.mab + (1.0 - beta1) * grad_ab
        self.vab = beta2 * self.vab + (1.0 - beta2) * grad_ab * grad_ab
        self.ab -= self.lr * (self.mab / (1.0 - beta1 ** self.t)) / (
            np.sqrt(self.vab / (1.0 - beta2 ** self.t)) + eps
        )
        self.b -= self.lr * np.float32(gradients["b"])

    def step(
        self,
        positive_x: np.ndarray,
        negative_x: np.ndarray,
        positive_target: np.ndarray,
        negative_target: np.ndarray,
        positive_mask: np.ndarray,
        negative_mask: np.ndarray,
        positive_censored: np.ndarray,
        negative_censored: np.ndarray,
    ) -> float:
        positive_z, positive_e, positive_s = self._fm(positive_x)
        negative_z, negative_e, negative_s = self._fm(negative_x)
        difference = positive_z - negative_z
        pair_probability = sigmoid(difference)
        positive_g = ((pair_probability - 1.0) / len(difference)).astype(np.float32)
        negative_g = -positive_g
        gradient_v = np.zeros_like(self.V)
        gradient_w = np.zeros_like(self.W)
        np.add.at(gradient_w, positive_x, positive_g[:, None])
        np.add.at(gradient_w, negative_x, negative_g[:, None])
        np.add.at(gradient_v, positive_x,
                  positive_g[:, None, None] * (positive_s[:, None, :] - positive_e))
        np.add.at(gradient_v, negative_x,
                  negative_g[:, None, None] * (negative_s[:, None, :] - negative_e))

        aux_loss = 0.0
        head_A = np.zeros_like(self.A)
        head_ab = np.float32(0.0)
        observed = max(1.0, float(positive_mask.sum() + negative_mask.sum()))
        for current_x, S, target, mask, censored in (
            (positive_x, positive_s, positive_target, positive_mask, positive_censored),
            (negative_x, negative_s, negative_target, negative_mask, negative_censored),
        ):
            target = np.asarray(target, dtype=np.float32).reshape(-1)
            mask = np.asarray(mask, dtype=np.float32).reshape(-1)
            censored = np.asarray(censored, dtype=np.float32).reshape(-1)
            probability = sigmoid(S @ self.A + self.ab)
            difference_aux = probability - target
            active = np.where((censored > 0) & (difference_aux >= 0), 0.0, difference_aux)
            grad_probability = 2.0 * active * probability * (1.0 - probability)
            aux_gradient = (self.aux_weight * grad_probability * mask / observed).astype(np.float32)
            gradient_A = S.T @ aux_gradient
            # S is a sum, so every field embedding receives the same auxiliary gradient.
            updates = np.broadcast_to(
                aux_gradient[:, None, None] * self.A[None, None, :],
                (len(current_x), current_x.shape[1], self.V.shape[1]),
            )
            np.add.at(gradient_v, current_x, updates)
            head_A += gradient_A
            head_ab += np.float32(aux_gradient.sum())
            aux_loss += float((active * active * mask).sum())

        self._apply({
            "V": gradient_v,
            "W": gradient_w,
            "A": head_A,
            "ab": head_ab,
            "b": np.float32(positive_g.sum() + negative_g.sum()),
        })
        return float(np.mean(np.logaddexp(0.0, -difference)) + aux_loss / observed)


def run_one(
    runner: ExperimentRunner,
    config: dict,
    aux_weight: float,
    checkpoint: Path | None = None,
) -> dict:
    encoded, dimension = runner._encoded_for(config)
    Xtr, ytr, utr = encoded["train"]
    Xva, yva, uva = encoded["valid"]
    targets, masks, censored = align_censored_watch_feedback(
        runner.data_dir, {"train": runner._splits["train"]}
    )
    target, mask, censor = targets["train"], masks["train"], censored["train"]
    hp = config["hyperparameters"]
    model = FMWatchtime(
        dimension, Xtr.shape[1], lr=hp["learning_rate"], l2=hp["l2"],
        seed=hp["seed"], aux_weight=aux_weight,
    )
    rng = np.random.default_rng(hp["seed"])
    best = -1.0
    best_state = None
    best_epoch = 0
    bad = 0
    started = time.monotonic()
    for epoch in range(hp["epochs"]):
        positive, negative = build_pair_indices(
            utr, ytr, rng, hp["pairs_per_positive"]
        )
        for start in range(0, len(positive), hp["batch_size"]):
            selection = slice(start, start + hp["batch_size"])
            model.step(
                Xtr[positive[selection]], Xtr[negative[selection]],
                target[positive[selection]], target[negative[selection]],
                mask[positive[selection]], mask[negative[selection]],
                censor[positive[selection]], censor[negative[selection]],
            )
        metrics = runner.evaluate_mod.evaluate(uva, yva, model.predict(Xva))
        score = float(metrics["primary"])
        print(f"    epoch {epoch + 1:02d} | primary={score:.6f} | best={max(best, score):.6f}", flush=True)
        if score > best + 1e-5:
            best, bad = score, 0
            best_epoch = epoch + 1
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b),
                          model.A.copy(), np.float32(model.ab))
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    if best_state is None:
        raise RuntimeError("watch-time FM did not produce a checkpoint")
    model.V, model.W, model.b, model.A, model.ab = best_state
    valid_scores = model.predict(Xva)
    valid = runner.evaluate_mod.evaluate(uva, yva, valid_scores)
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            checkpoint,
            V=model.V,
            W=model.W,
            b=model.b,
            A=model.A,
            ab=model.ab,
            best_epoch=np.asarray(best_epoch),
        )
        # The generic validation artifact is what offline ensemble search
        # consumes; it contains no test rows or labels.
        runner._save_validation_artifact(checkpoint, uva, yva, valid_scores)
    return {**{key: float(value) for key, value in valid.items()},
            "runtime_seconds": float(time.monotonic() - started),
            "auxiliary_loss_weight": aux_weight,
            "best_epoch": int(best_epoch)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/fm_watchtime_ablation")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    initial = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
    runner.prepare()
    results = {}
    for name, features, weight in (
        ("fm_watchtime", {}, 0.3),
        ("fm_watchtime_7field", {"prior_video_exposure": True, "author_recency": True}, 0.3),
    ):
        config = copy.deepcopy(initial)
        config["model"] = "fm"
        config["training_objective"] = "bpr"
        config["hyperparameters"]["learning_rate"] = 0.0003
        config["features"].update(features)
        print(f"\n=== {name} ===", flush=True)
        results[name] = {
            "features": features,
            "metrics": run_one(runner, config, weight, output_dir / f"{name}.npz"),
        }
        print(json.dumps(results[name], indent=2), flush=True)
    baseline = results["fm_watchtime"]["metrics"]["primary"]
    for row in results.values():
        row["delta_vs_watchtime_base"] = row["metrics"]["primary"] - baseline
    payload = {
        "selection_split": "validation only (2022-04-22 through 2022-04-28)",
        "test_labels_used": False,
        "objective": "FM+BPR plus one-sided censored watch-time auxiliary",
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
