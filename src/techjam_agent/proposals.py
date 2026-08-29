from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import ALLOWED_VALUES, FEATURE_KEYS, MODELS, OBJECTIVES, apply_changes, experiment_key


@dataclass(frozen=True)
class Proposal:
    hypothesis: str
    reason: str
    changes: dict[str, Any]
    source: str

    @classmethod
    def parse(cls, value: dict[str, Any], source: str) -> "Proposal":
        if not isinstance(value.get("hypothesis"), str) or not value["hypothesis"].strip():
            raise ValueError("proposal requires a non-empty hypothesis")
        if not isinstance(value.get("reason"), str) or not value["reason"].strip():
            raise ValueError("proposal requires a non-empty reason")
        changes = value.get("changes")
        if not isinstance(changes, dict) or not 1 <= len(changes) <= 3:
            raise ValueError("proposal must contain one atomic action (at most three config fields)")
        return cls(value["hypothesis"].strip(), value["reason"].strip(), changes, source)

    def as_dict(self) -> dict[str, Any]:
        return {"hypothesis": self.hypothesis, "reason": self.reason,
                "changes": self.changes, "source": self.source}


class DeterministicResearcher:
    """Safe offline policy; also provides a fallback when an LLM is unavailable."""

    def propose(self, best: dict[str, Any], history: list[dict[str, Any]]) -> Proposal:
        tried = {experiment_key(item["config"]) for item in history if "config" in item}
        hp = best["hyperparameters"]
        if best["model"] == "fm" and best["training_objective"] == "bce":
            candidate = apply_changes(best, {"training_objective": "bpr"})
            if experiment_key(candidate) not in tried:
                return Proposal(
                    "Align FM training with within-user ranking by replacing BCE with pairwise BPR.",
                    "GAUC and nDCG reward positive items ranking above negatives, not calibrated classification.",
                    {"training_objective": "bpr"}, "deterministic",
                )
        if best["model"] == "fm":
            model_change = ({"model": "lightgbm", "training_objective": "bce"}
                            if best["training_objective"] == "bpr" else {"model": "lightgbm"})
            candidate = apply_changes(best, model_change)
            if experiment_key(candidate) not in tried:
                return Proposal(
                    "Test whether LightGBM outperforms pointwise FM on the original fields.",
                    "This isolates model choice before adding continuous history statistics.",
                    model_change, "deterministic",
                )
            user_tab_changes = {**model_change, "user_tab_long_view_rate": True}
            user_tab = apply_changes(best, user_tab_changes)
            if experiment_key(user_tab) not in tried:
                return Proposal(
                    "Test whether smoothed user-by-tab long-view preference improves ranking.",
                    "Category-specific preference may be useful even when global user propensity is not.",
                    user_tab_changes, "deterministic",
                )
            stats_changes = {**model_change, "continuous_history_stats": True}
            with_stats = apply_changes(best, stats_changes)
            if experiment_key(with_stats) not in tried:
                return Proposal(
                    "Test continuous train-only user/item rates and log-counts with LightGBM.",
                    "This follows the pure LightGBM control even when that branch was not the global best.",
                    stats_changes, "deterministic",
                )
        if best["model"] == "lightgbm" and not best["features"]["continuous_history_stats"]:
            if not best["features"]["user_tab_long_view_rate"]:
                candidate = apply_changes(best, {"user_tab_long_view_rate": True})
                if experiment_key(candidate) not in tried:
                    return Proposal(
                        "Test whether smoothed user-by-tab long-view preference improves ranking.",
                        "Category-specific preference may be useful even when global user propensity is not.",
                        {"user_tab_long_view_rate": True}, "deterministic",
                    )
            candidate = apply_changes(best, {"continuous_history_stats": True})
            if experiment_key(candidate) not in tried:
                return Proposal(
                    "Test continuous train-only user/item rates and log-counts with LightGBM.",
                    "This distinguishes weak statistics from an unsuitable categorical-bucket representation.",
                    {"continuous_history_stats": True}, "deterministic",
                )
        for key in FEATURE_KEYS:
            if key in ("continuous_history_stats", "user_tab_long_view_rate"):
                continue
            if not best["features"][key]:
                candidate = apply_changes(best, {key: True})
                if experiment_key(candidate) not in tried:
                    return Proposal(
                        f"Test whether train-only {key} improves within-user ranking.",
                        "Target-rate history may expose behavioral propensity unavailable to the base fields.",
                        {key: True}, "deterministic",
                    )
        order = ("learning_rate", "l2", "embedding_dim", "epochs", "batch_size", "patience")
        for key in order:
            for value in ALLOWED_VALUES[key]:
                if value == hp[key]:
                    continue
                candidate = apply_changes(best, {key: value})
                if experiment_key(candidate) not in tried:
                    return Proposal(
                        f"Test whether {key}={value} improves ranking quality.",
                        "A controlled one-variable experiment preserves attribution and reproducibility.",
                        {key: value}, "deterministic",
                    )
        raise StopIteration("the configured FM experiment space is exhausted")


class OpenAICompatibleResearcher:
    """Small OpenAI-compatible JSON client using only the Python standard library."""

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for --researcher llm")

    def propose(self, best: dict[str, Any], history: list[dict[str, Any]]) -> Proposal:
        compact_history = [{"iteration": x.get("iteration"), "changes": x.get("changes"),
                            "metrics": x.get("metrics"), "decision": x.get("decision")}
                           for x in history[-20:]]
        prompt = {
            "objective": "Propose exactly one experiment maximizing mean(GAUC,nDCG@5).",
            "rules": ["Return JSON only", "Change exactly one allowed hyperparameter",
                      "Do not repeat an experiment", "Prefer a scientific hypothesis"],
            "allowed_values": {"model": MODELS, "training_objective": OBJECTIVES, **ALLOWED_VALUES,
                               **{key: (False, True) for key in FEATURE_KEYS}},
            "current_best": best,
            "history": compact_history,
            "schema": {"hypothesis": "string", "reason": "string", "changes": {"one_key": "value"}},
        }
        body = json.dumps({"model": self.model, "messages": [
            {"role": "system", "content": "You are a cautious autonomous ML researcher."},
            {"role": "user", "content": json.dumps(prompt)}],
            "response_format": {"type": "json_object"}, "temperature": 0.2}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM proposal failed: {exc}") from exc
        raw = json.loads(payload["choices"][0]["message"]["content"])
        proposal = Proposal.parse(raw, "llm")
        candidate = apply_changes(best, proposal.changes)
        if experiment_key(candidate) in {experiment_key(x["config"]) for x in history if "config" in x}:
            raise ValueError("LLM repeated a previous experiment")
        return proposal
