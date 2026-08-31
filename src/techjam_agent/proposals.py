from __future__ import annotations

import json
import hashlib
import itertools
import math
import copy
import os
import re
import socket
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable

from .config import ALLOWED_VALUES, FEATURE_KEYS, MODELS, OBJECTIVES, apply_changes, experiment_key
from .autonomous_branch import MAX_SOURCE_CHARS, validate_source
from .code_evolution import EVOLUTION_RECIPES
from .memory import build_memory_summary, collect_tried_keys
from .operator_registry import (
    MODEL_SPECS,
    OPERATORS,
    branch_for_changes,
    planner_model_registry,
    planner_registry,
)
from .research import STRATEGIES, build_research_context
from .tree import branch_name, node_id_for, select_parent

# One experiment normally changes one field. Up to four fields are allowed only
# so a model-family switch can also reset ensemble-only state and its objective.
MAX_CHANGE_FIELDS = 4
AUTONOMOUS_COMPOSITE_LIMIT = 64
ALLOWED_CHANGE_KEYS = set(ALLOWED_VALUES) | set(FEATURE_KEYS) | {
    "model", "training_objective", "code_branch"
}
PROPOSAL_RESPONSE_KEYS = frozenset({
    "proposal_type", "candidate_id", "observation", "diagnosis", "hypothesis", "evidence_ids",
    "reason", "changes", "expected_effect", "risk", "estimated_cost",
    "success_condition",
})
OPTIONAL_PROPOSAL_KEYS = frozenset({"code_branch"})
ZERO_TOKEN_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
OFFICIAL_BASELINE_PRIMARY = 0.6016
CONVERGENCE_EPSILON = 0.002
HTTP_TIMEOUT_SECONDS = 60
MAX_LLM_ATTEMPTS = 3
RECENT_HISTORY = 20
VALIDATION_METRIC_KEYS = ("GAUC", "nDCG@5", "primary")
EXPECTED_EFFECT_VALUES = ("increase", "decrease", "neutral", "uncertain")
CHANGE_RULE = (
    "Select exactly one entry from legal_candidates and copy its candidate_id and changes. "
    "The catalog contains only compatible, non-duplicate experiments with between 1 and "
    f"{MAX_CHANGE_FIELDS} allow-listed fields. "
    "Your changes are applied to expansion_parent, which may differ from global_best. "
    "Do not supply lineage fields such as parent_id; the Controller owns lineage. "
    "For candidate_kind=code_branch, set proposal_type='code_branch' and include a compact "
    "code_branch object with branch_name and source; set base_model and description to a "
    "string or null. "
    "The source must implement the stated fit_validate/finalize contract; do not access files, "
    "the network, subprocesses, or environment variables. "
    "Use validation metrics only; never use or request test metrics. The official promotion score "
    "is validation Primary, while a candidate may set validation_metric='nDCG@5' to retain the "
    "checkpoint or blend that best serves top-five ranking."
)
RESEARCHER_SYSTEM_PROMPT = (
    "You are the Planner in a validation-only autonomous recommender-systems research loop. "
    "Propose exactly one controlled, legal configuration experiment from the Controller-selected "
    "parent. Ground every claim in the supplied evidence IDs and distinguish observed facts from "
    "hypotheses. Optimize validation Primary, the mean of GAUC and nDCG@5, while explaining the "
    "expected direction of both component metrics. Only a strict improvement over the global "
    "validation incumbent is positive evidence; branch-local gains and weak nodes are failures for "
    "allocation purposes. Never request test metrics, choose lineage, alter budgets, "
    "or claim significance from one seed. A nDCG-focused candidate may choose the legal "
    "validation_metric='nDCG@5' control with group/listwise ranking; this does not change "
    "the official Primary promotion rule. Consider "
    "unexplored model families when they test a "
    "meaningfully different hypothesis instead of repeatedly tuning the incumbent family. "
    "Prefer one changed field for ordinary registry candidates; in autonomous mode, use a "
    "bounded automatically composed candidate when it tests one coherent feature interaction "
    "or optimizer bundle. Use up to four fields only when the Controller supplies that bundle "
    "or a compatibility-safe model/objective switch. "
    "When a candidate_kind=code_branch row is present, you may use it to explore a new model or "
    "feature implementation. Keep generated source compact and deterministic, implement exactly "
    "fit_validate(runner, config, checkpoint) and finalize(runner, config, checkpoint, output), "
    "and use runner.autonomous_* helpers or pure numpy/torch operations only. "
    "Select a legal candidate_id, cite only IDs from "
    "valid_evidence_ids, and return only the structured proposal requested by the API."
)
REPAIR_INSTRUCTION = (
    "Your previous proposal was rejected by deterministic validation. Correct the stated error, "
    "select a different ID from legal_candidates, cite only valid_evidence_ids, and copy that "
    "candidate's changes exactly in the required schema. If using code_branch, keep the source "
    "within the stated import/entry-point contract."
)


def proposal_json_schema(
    valid_evidence_ids: list[str] | tuple[str, ...] | None = None,
    valid_candidate_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Strict API response schema; semantic config validation still happens locally."""
    change_value = {
        "anyOf": [
            {"type": "string"},
            {"type": "number"},
            {"type": "integer"},
            {"type": "boolean"},
        ]
    }
    return {
        "type": "object",
        "additionalProperties": False,
        # OpenAI-compatible strict JSON-schema mode requires every declared
        # property to be required. ``null`` represents an ordinary config
        # proposal; a code-branch proposal supplies the object.
        "required": sorted(PROPOSAL_RESPONSE_KEYS | OPTIONAL_PROPOSAL_KEYS),
        "properties": {
            "proposal_type": {"type": "string", "enum": ["config", "code_branch"]},
            "candidate_id": {
                "type": "string",
                **({"enum": list(valid_candidate_ids)} if valid_candidate_ids else {}),
            },
            "observation": {"type": "string"},
            "diagnosis": {"type": "string"},
            "hypothesis": {"type": "string"},
            "evidence_ids": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "string",
                    **({"enum": list(valid_evidence_ids)} if valid_evidence_ids else {}),
                },
            },
            "reason": {"type": "string"},
            "changes": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_CHANGE_FIELDS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["field", "value"],
                    "properties": {
                        "field": {"type": "string", "enum": sorted(ALLOWED_CHANGE_KEYS)},
                        "value": change_value,
                    },
                },
            },
            "expected_effect": {
                "type": "object",
                "additionalProperties": False,
                "required": list(VALIDATION_METRIC_KEYS),
                "properties": {
                    key: {"type": "string", "enum": list(EXPECTED_EFFECT_VALUES)}
                    for key in VALIDATION_METRIC_KEYS
                },
            },
            "risk": {"type": "string"},
            "estimated_cost": {"type": "string", "enum": ["low", "medium", "high"]},
            "success_condition": {"type": "string"},
            "code_branch": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["branch_name", "source", "base_model", "description"],
                        "properties": {
                            "branch_name": {"type": "string", "minLength": 1},
                            "source": {"type": "string", "minLength": 1},
                            "base_model": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "description": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        },
                    },
                ],
            },
        },
    }


@dataclass(frozen=True)
class Proposal:
    hypothesis: str
    reason: str
    changes: dict[str, Any]
    source: str
    token_usage: dict[str, int] | None = None
    llm_attempts: int = 0
    proposal_type: str = "config"
    candidate_id: str | None = None
    observation: str | None = None
    diagnosis: str | None = None
    evidence_ids: tuple[str, ...] = ()
    expected_effect: dict[str, str] | None = None
    risk: str | None = None
    estimated_cost: str | None = None
    success_condition: str | None = None
    llm_call_ids: tuple[str, ...] = ()
    fallback: dict[str, Any] | None = None
    code_branch: dict[str, Any] | None = None

    @classmethod
    def parse(cls, value: dict[str, Any], source: str) -> "Proposal":
        if not isinstance(value, dict):
            raise ValueError("proposal response must be a JSON object")
        known_keys = PROPOSAL_RESPONSE_KEYS | OPTIONAL_PROPOSAL_KEYS
        unknown = set(value) - known_keys
        if unknown:
            raise ValueError(f"unsupported proposal fields: {sorted(unknown)}")
        missing = PROPOSAL_RESPONSE_KEYS - set(value)
        if missing:
            raise ValueError(f"proposal response is missing fields: {sorted(missing)}")
        proposal_type = value.get("proposal_type")
        if proposal_type not in {"config", "code_branch"}:
            raise ValueError("proposal_type must be 'config' or 'code_branch'")
        candidate_id = value.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("proposal requires a non-empty candidate_id")
        for key in ("observation", "diagnosis", "hypothesis", "reason", "risk",
                    "success_condition"):
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise ValueError(f"proposal requires a non-empty {key}")
        evidence_ids = value.get("evidence_ids")
        if (not isinstance(evidence_ids, list) or not evidence_ids or
                any(not isinstance(item, str) or not item.strip() for item in evidence_ids)):
            raise ValueError("proposal requires one or more non-empty evidence_ids")
        changes = _normalize_changes(value.get("changes"))
        if not isinstance(changes, dict) or not 1 <= len(changes) <= MAX_CHANGE_FIELDS:
            raise ValueError(
                f"proposal must contain one atomic action (at most {MAX_CHANGE_FIELDS} config fields)"
            )
        illegal = set(changes) - ALLOWED_CHANGE_KEYS
        if illegal:
            raise ValueError(f"unsupported proposal keys: {sorted(illegal)}")
        expected = value.get("expected_effect")
        if not isinstance(expected, dict) or set(expected) != set(VALIDATION_METRIC_KEYS):
            raise ValueError("expected_effect must contain GAUC, nDCG@5, and primary")
        if any(expected[key] not in EXPECTED_EFFECT_VALUES for key in VALIDATION_METRIC_KEYS):
            raise ValueError("expected_effect contains an unsupported direction")
        if value.get("estimated_cost") not in {"low", "medium", "high"}:
            raise ValueError("estimated_cost must be low, medium, or high")
        code_branch = value.get("code_branch")
        if proposal_type == "code_branch":
            if not isinstance(code_branch, dict):
                raise ValueError("code_branch proposals require a code_branch object")
            unknown_branch_fields = set(code_branch) - {
                "branch_name", "source", "base_model", "description"
            }
            if unknown_branch_fields:
                raise ValueError(
                    f"unsupported code_branch fields: {sorted(unknown_branch_fields)}"
                )
            if not isinstance(code_branch.get("branch_name"), str) or not code_branch["branch_name"].strip():
                raise ValueError("code_branch.branch_name must be a non-empty string")
            if not isinstance(code_branch.get("source"), str) or not code_branch["source"].strip():
                raise ValueError("code_branch.source must be a non-empty string")
            if "base_model" in code_branch and code_branch["base_model"] is not None and not isinstance(code_branch["base_model"], str):
                raise ValueError("code_branch.base_model must be a string or null")
            if "description" in code_branch and code_branch["description"] is not None and not isinstance(code_branch["description"], str):
                raise ValueError("code_branch.description must be a string or null")
        elif code_branch is not None:
            raise ValueError("code_branch is only valid for proposal_type='code_branch'")
        return cls(
            value["hypothesis"].strip(), value["reason"].strip(), changes, source,
            proposal_type=proposal_type,
            candidate_id=candidate_id.strip(),
            observation=value["observation"].strip(),
            diagnosis=value["diagnosis"].strip(),
            evidence_ids=tuple(item.strip() for item in evidence_ids),
            expected_effect={key: str(expected[key]) for key in VALIDATION_METRIC_KEYS},
            risk=value["risk"].strip(),
            estimated_cost=value["estimated_cost"],
            success_condition=value["success_condition"].strip(),
            code_branch=copy.deepcopy(code_branch) if isinstance(code_branch, dict) else None,
        )

    def as_dict(self) -> dict[str, Any]:
        usage = ZERO_TOKEN_USAGE if self.token_usage is None else self.token_usage
        result = {
            "proposal_type": self.proposal_type,
            "candidate_id": self.candidate_id,
            "observation": self.observation,
            "diagnosis": self.diagnosis,
            "hypothesis": self.hypothesis,
            "evidence_ids": list(self.evidence_ids),
            "reason": self.reason,
            "changes": self.changes,
            "expected_effect": self.expected_effect,
            "risk": self.risk,
            "estimated_cost": self.estimated_cost,
            "success_condition": self.success_condition,
            "source": self.source,
            "llm_call_ids": list(self.llm_call_ids),
            "fallback": self.fallback,
            "token_usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            },
            "llm_attempts": int(self.llm_attempts),
        }
        if self.code_branch is not None:
            result["code_branch"] = copy.deepcopy(self.code_branch)
        return result


def _normalize_changes(value: Any) -> dict[str, Any]:
    """Accept the strict field/value list and a legacy object from compatible providers."""
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, list):
        raise ValueError("changes must be a field/value list")
    changes: dict[str, Any] = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != {"field", "value"}:
            raise ValueError("each change must contain exactly field and value")
        field = item.get("field")
        if not isinstance(field, str) or not field:
            raise ValueError("each change field must be a non-empty string")
        if field in changes:
            raise ValueError(f"duplicate change field: {field}")
        changes[field] = item.get("value")
    return changes


def empty_token_usage() -> dict[str, int]:
    return dict(ZERO_TOKEN_USAGE)


def extract_token_usage(payload: dict[str, Any] | None) -> dict[str, int]:
    """Read OpenAI-compatible usage; missing fields become zero and never crash."""
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return empty_token_usage()
    prompt = _nonneg_int(usage.get("prompt_tokens"))
    completion = _nonneg_int(usage.get("completion_tokens"))
    total = _nonneg_int(usage.get("total_tokens"))
    if total == 0:
        total = prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def add_token_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {
        "prompt_tokens": left["prompt_tokens"] + right["prompt_tokens"],
        "completion_tokens": left["completion_tokens"] + right["completion_tokens"],
        "total_tokens": left["total_tokens"] + right["total_tokens"],
    }


def _nonneg_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def validation_metrics_only(metrics: Any) -> dict[str, Any] | None:
    """Keep validation ranking metrics only. Drop test split numbers if present."""
    if not isinstance(metrics, dict):
        return None
    cleaned: dict[str, float] = {}
    for key in VALIDATION_METRIC_KEYS:
        try:
            value = float(metrics[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            cleaned[key] = value
    return cleaned or None


def validation_critique_only(critique: Any) -> dict[str, Any] | None:
    if not isinstance(critique, dict):
        return None
    allowed = (
        "observation", "interpretation", "confidence", "verdict", "delta",
        "meaningful_improvement", "next_test", "reasons", "hypothesis_status",
        "evidence_strength", "seed_count",
        "bottleneck", "recommended_strategy_ids", "failure_category",
    )
    cleaned = {key: critique[key] for key in allowed if key in critique}
    deltas = validation_metrics_only(critique.get("metric_deltas"))
    if deltas is not None:
        cleaned["metric_deltas"] = deltas
    return cleaned or None


def compact_history_for_planner(history: list[dict[str, Any]], recent: int = RECENT_HISTORY) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in history[-recent:]:
        iteration = item.get("iteration")
        compact.append({
            "evidence_id": _evidence_id(iteration, item.get("evidence_id")),
            "iteration": iteration,
            "parent_id": item.get("parent_id"),
            "hypothesis": item.get("hypothesis"),
            "critique": validation_critique_only(item.get("critique")),
            "changes": item.get("changes"),
            "decision": item.get("decision"),
            "source": item.get("source"),
            "metrics": validation_metrics_only(item.get("metrics")),
        })
    return compact


def _evidence_id(iteration: Any, supplied: Any = None) -> str:
    if isinstance(supplied, str) and supplied:
        return supplied
    return f"iteration_{iteration:03d}" if isinstance(iteration, int) else "historical_record"


def _config_value(config: dict[str, Any], field: str) -> Any:
    spec = OPERATORS[field]
    return config.get(field) if spec.target == "root" else config[spec.target].get(field)


def candidate_id_for(config: dict[str, Any]) -> str:
    """Return a stable public ID for one fully resolved candidate configuration."""
    key = experiment_key(config)
    return f"candidate_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}"


def _epoch_cap_is_execution_equivalent(
    candidate: dict[str, Any], history: list[dict[str, Any]] | None,
) -> bool:
    """Skip deterministic epoch caps that end after an already observed early stop."""
    candidate_epochs = candidate["hyperparameters"]["epochs"]
    candidate_signature = copy.deepcopy(candidate)
    candidate_signature["hyperparameters"]["epochs"] = 0
    candidate_key = experiment_key(candidate_signature)
    for item in history or []:
        config = item.get("config") if isinstance(item, dict) else None
        metrics = item.get("metrics") if isinstance(item, dict) else None
        if not isinstance(config, dict) or not isinstance(metrics, dict):
            continue
        try:
            best_epoch = int(metrics["best_epoch"])
            patience = int(config["hyperparameters"]["patience"])
        except (KeyError, TypeError, ValueError):
            continue
        previous_signature = copy.deepcopy(config)
        previous_signature["hyperparameters"]["epochs"] = 0
        if (
            experiment_key(previous_signature) == candidate_key
            and candidate_epochs >= best_epoch + patience
        ):
            return True
    return False


def _model_switches(parent: dict[str, Any]) -> list[dict[str, Any]]:
    current_model = parent["model"]
    current_objective = parent["training_objective"]
    switches: list[dict[str, Any]] = []
    for model in MODELS:
        # ``custom`` is represented by the dedicated generated-code candidate
        # below. It needs a source payload and must not appear as a bare model
        # switch in the normal menu.
        if model in {current_model, "custom"}:
            continue
        for objective in MODEL_SPECS[model].objectives:
            changes: dict[str, Any] = {"model": model}
            if objective != current_objective:
                changes["training_objective"] = objective
            if current_model == "fm_ensemble":
                changes.update({
                    "ensemble_size": 1,
                    "ensemble_seed_set": "sequential",
                })
            if model == "fm_ensemble":
                changes.update({"ensemble_size": 2, "ensemble_seed_set": "sequential"})
            if len(changes) <= MAX_CHANGE_FIELDS:
                switches.append(changes)
            if model == "fm_ensemble":
                selected = {**changes, "ensemble_seed_set": "3,4"}
                if len(selected) <= MAX_CHANGE_FIELDS:
                    switches.append(selected)
    return switches


def _autonomous_change_sets(parent: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate coherent multi-field actions for open-ended search.

    The normal catalog deliberately favors one-field experiments.  That is
    useful for reproducibility, but it makes the Planner dependent on humans
    having anticipated every useful feature interaction.  Autonomous mode
    adds a bounded set of candidate-varying bundles and optimizer bundles.
    Every bundle still goes through ``apply_changes`` and the normal
    compatibility/duplicate checks; this expands search, not the execution
    safety boundary.
    """
    model = parent.get("model")
    objective = parent.get("training_objective")
    feature_fields = [
        field for field, spec in OPERATORS.items()
        if spec.target == "features"
        and not _config_value(parent, field)
        and model in spec.models
        and objective in spec.objectives
    ]

    # These groups encode natural mechanisms rather than arbitrary Cartesian
    # products.  The pairwise expansion below still lets the agent discover
    # cross-group interactions without flooding the prompt.
    groups = (
        ("hour", "weekday", "upload_age_bucket", "freshness_decay"),
        ("time_decay_item_popularity", "time_decay_author_popularity",
         "time_decay_tag_popularity"),
        ("user_tag_impression_count", "user_tag_long_view_rate",
         "user_author_impression_count", "user_author_long_view_count",
         "user_author_long_view_rate"),
        ("tag", "video_type", "duration_fine_bucket"),
        ("recent_history_similarity", "user_activity", "user_tab_long_view_rate"),
    )
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(fields: tuple[str, ...]) -> None:
        changes = {field: True for field in fields if field in feature_fields}
        if not 2 <= len(changes) <= MAX_CHANGE_FIELDS:
            return
        signature = json.dumps(changes, sort_keys=True, separators=(",", ":"))
        if signature not in seen:
            seen.add(signature)
            results.append(changes)

    # Prefer mechanism-complete bundles, then their pairwise interactions.
    for group in groups:
        available = tuple(field for field in group if field in feature_fields)
        if len(available) >= 2:
            add(available[:MAX_CHANGE_FIELDS])
            for size in (2, 3):
                for combo in itertools.combinations(available, size):
                    add(combo)
    for combo in itertools.combinations(feature_fields, 2):
        add(combo)
        if len(results) >= AUTONOMOUS_COMPOSITE_LIMIT:
            return results[:AUTONOMOUS_COMPOSITE_LIMIT]

    # A small number of optimizer/objective bundles are useful only when the
    # current family supports them.  Values are chosen deterministically; the
    # Planner may still choose any individual value from the regular catalog.
    optimizer_fields = (
        "learning_rate", "l2", "batch_size", "patience", "dropout",
    )
    optimizer_alternatives: dict[str, Any] = {}
    for field in optimizer_fields:
        spec = OPERATORS[field]
        if model not in spec.models or objective not in spec.objectives:
            continue
        current = _config_value(parent, field)
        alternative = next((value for value in spec.values if value != current), None)
        if alternative is not None:
            optimizer_alternatives[field] = alternative
    for combo in itertools.combinations(tuple(optimizer_alternatives), 2):
        results.append({field: optimizer_alternatives[field] for field in combo})
        if len(results) >= AUTONOMOUS_COMPOSITE_LIMIT:
            break

    # Top-k ranking is a different optimization target from pairwise GAUC.
    # Expose a few legal nDCG-focused bundles so autonomous search can test
    # listwise loss/negative sampling and checkpoint selection together.  These
    # are proposals, not forced defaults; Controller still promotes by the
    # official Primary score.
    if "validation_metric" in OPERATORS and _config_value(parent, "validation_metric") != "nDCG@5":
        add_changes = {"validation_metric": "nDCG@5"}
        if objective in MODEL_SPECS.get(model, MODEL_SPECS["custom"]).objectives:
            if objective != "group_softmax" and "group_softmax" in MODEL_SPECS[model].objectives:
                add_changes["training_objective"] = "group_softmax"
                if "negatives_per_positive" in OPERATORS:
                    add_changes["negatives_per_positive"] = 4
            elif objective == "bpr" and "negatives_per_positive" in OPERATORS:
                add_changes["negatives_per_positive"] = 4
        try:
            # Apply the same compatibility validation as every other bundle.
            apply_changes(parent, add_changes)
        except (KeyError, TypeError, ValueError):
            add_changes = {"validation_metric": "nDCG@5"}
        if add_changes not in results:
            results.append(add_changes)
    if model in {"hybrid_blend", "lightgcn_hybrid"} and _config_value(parent, "blend_mode") != "zscore":
        try:
            apply_changes(parent, {"blend_mode": "zscore", "validation_metric": "nDCG@5"})
        except (KeyError, TypeError, ValueError):
            pass
        else:
            results.append({"blend_mode": "zscore", "validation_metric": "nDCG@5"})
    return results[:AUTONOMOUS_COMPOSITE_LIMIT]


def legal_candidate_catalog(
    parent: dict[str, Any], history: list[dict[str, Any]] | None,
    *, autonomous: bool = False,
) -> list[dict[str, Any]]:
    """Enumerate exact legal, non-duplicate actions for the selected parent.

    The Planner chooses from this controller-generated catalog. It never needs
    to infer compatibility resets or guess which configurations are hidden in
    truncated research memory.
    """
    tried = set(collect_tried_keys(history))
    proposed_changes: list[dict[str, Any]] = []
    autonomous_signatures: set[str] = set()
    recipe_for: dict[str, str] = {}
    recipe_purpose: dict[str, str] = {}
    for field, spec in OPERATORS.items():
        if field in {"model", "code_branch"}:
            continue
        current = _config_value(parent, field)
        for value in spec.values:
            if value == current:
                continue
            proposed_changes.append({field: value})

    # Custom two-seed incumbents need an atomic reset before their size changes.
    if parent["model"] == "fm_ensemble":
        current_size = parent["hyperparameters"]["ensemble_size"]
        for size in OPERATORS["ensemble_size"].values:
            if size > 1 and size != current_size:
                proposed_changes.append({
                    "ensemble_size": size,
                    "ensemble_seed_set": "sequential",
                })
    proposed_changes.extend(_model_switches(parent))

    # AIDE/MLEvolve-style constrained code branches. Each recipe activates an
    # already tested implementation path and one coherent mechanism bundle;
    # the LLM selects branches but cannot inject arbitrary executable code.
    for recipe in EVOLUTION_RECIPES:
        changes = {
            field: value for field, value in recipe.targets.items()
            if _config_value(parent, field) != value
        }
        if 1 <= len(changes) <= MAX_CHANGE_FIELDS:
            proposed_changes.append(changes)
            signature = json.dumps(changes, sort_keys=True, separators=(",", ":"))
            recipe_for[signature] = recipe.recipe_id
            recipe_purpose[recipe.recipe_id] = recipe.purpose

    if autonomous:
        # Keep the generated bundles after recipes so the shortlist remains
        # anchored by known mechanisms, while still exposing genuinely new
        # combinations when those mechanisms have been exhausted.
        for changes in _autonomous_change_sets(parent):
            proposed_changes.append(changes)
            autonomous_signatures.add(
                json.dumps(changes, sort_keys=True, separators=(",", ":"))
            )

    # Open-ended AIDE/MLEvolve branch. The source is supplied in the proposal
    # payload and materialized by the Controller; the catalog only exposes a
    # stable, compatibility-checked configuration transition.
    generated_changes: dict[str, Any] = {
        "model": "custom",
        "code_branch": "__generated__",
    }
    if parent.get("training_objective") != "bpr":
        generated_changes["training_objective"] = "bpr"
    if parent.get("model") == "fm_ensemble":
        generated_changes.update({"ensemble_size": 1, "ensemble_seed_set": "sequential"})
    if len(generated_changes) <= MAX_CHANGE_FIELDS:
        proposed_changes.append(generated_changes)

    catalog: list[dict[str, Any]] = []
    seen: set[str] = set()
    for changes in proposed_changes:
        if not 1 <= len(changes) <= MAX_CHANGE_FIELDS:
            continue
        try:
            candidate = apply_changes(parent, changes)
        except (KeyError, TypeError, ValueError):
            continue
        key = experiment_key(candidate)
        if key in tried or key in seen:
            continue
        if set(changes) == {"epochs"} and _epoch_cap_is_execution_equivalent(
            candidate, history
        ):
            continue
        seen.add(key)
        specs = [OPERATORS[field] for field in changes]
        cost_order = {"low": 0, "medium": 1, "high": 2}
        costs = [spec.cost for spec in specs]
        if "model" in changes:
            costs.append(MODEL_SPECS[candidate["model"]].estimated_cost)
        cost = max(costs, key=cost_order.__getitem__)
        row = {
            "candidate_id": candidate_id_for(candidate),
            "changes": changes,
            "branch": branch_for_changes(changes),
            "cost": cost,
        }
        recipe_id = recipe_for.get(json.dumps(changes, sort_keys=True, separators=(",", ":")))
        if recipe_id:
            row.update({
                "candidate_kind": "evolution_recipe",
                "evolution_recipe": recipe_id,
                "evolution_purpose": recipe_purpose[recipe_id],
            })
        signature = json.dumps(changes, sort_keys=True, separators=(",", ":"))
        if signature in autonomous_signatures and "candidate_kind" not in row:
            row.update({
                "candidate_kind": "autonomous_composite",
                "autonomy_generated": True,
                "evolution_purpose": (
                    "Automatically composed compatible feature or optimizer bundle; "
                    "validate the interaction rather than assuming each component helps."
                ),
            })
        catalog.append(row)
    # Mark the generated branch after normal candidates have been resolved so
    # deterministic/offline planners can explicitly skip it when no source is
    # available, while LLM planners can opt into it.
    generated_signature = json.dumps(generated_changes, sort_keys=True, separators=(",", ":"))
    for row in catalog:
        if json.dumps(row.get("changes"), sort_keys=True, separators=(",", ":")) == generated_signature:
            row.update({
                "candidate_kind": "code_branch",
                "evolution_purpose": (
                    "Generate and evaluate a new model or feature implementation under the "
                    "validated fit_validate/finalize contract."
                ),
                "code_contract": (
                    "source must define fit_validate(runner, config, checkpoint) and "
                    "finalize(runner, config, checkpoint, output); only math/numpy/torch/"
                    "typing/collections imports are allowed. Use runner.autonomous_encoded or "
                    "runner.autonomous_dense_matrices (split='train_valid' or 'test'; test labels "
                    "are redacted), runner.autonomous_evaluate, "
                    "runner.autonomous_write_validation_slices, runner.autonomous_save_checkpoint, "
                    "runner.autonomous_load_checkpoint, runner.autonomous_write_submission, or "
                    "runner.autonomous_run_builtin to compose existing scorers. fit_validate "
                    "must save a checkpoint; finalize must write the submission. Keep source "
                    "under 24,000 characters and deterministic."
                ),
            })
    return catalog


def standardize_proposal(
    proposal: Proposal,
    parent: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    autonomous: bool = False,
) -> Proposal:
    """Normalize every researcher through the same legal proposal contract.

    LLM proposals already contain this metadata. Deterministic and fallback
    proposals historically did not, which made their logs impossible to audit
    with the same candidate/evidence interface. The Controller calls this
    function before resolving any candidate.
    """

    catalog = legal_candidate_catalog(parent, history, autonomous=autonomous)
    selected = next((item for item in catalog if item["changes"] == proposal.changes), None)
    if selected is None:
        # Preserve the Controller's distinct duplicate-configuration outcome:
        # a legacy researcher may intentionally return the same action again.
        candidate = apply_changes(parent, proposal.changes)
        if experiment_key(candidate) not in set(collect_tried_keys(history)):
            raise ValueError("proposal changes do not match an available legal candidate")
        specs = [OPERATORS[field] for field in proposal.changes]
        cost_order = {"low": 0, "medium": 1, "high": 2}
        costs = [spec.cost for spec in specs]
        if "model" in proposal.changes:
            costs.append(MODEL_SPECS[candidate["model"]].estimated_cost)
        selected = {
            "candidate_id": candidate_id_for(candidate),
            "changes": proposal.changes,
            "cost": max(costs, key=cost_order.__getitem__),
        }
    if proposal.candidate_id not in (None, selected["candidate_id"]):
        raise ValueError("proposal candidate_id does not match its resolved configuration")
    is_code_candidate = selected.get("candidate_kind") == "code_branch"
    if is_code_candidate and (proposal.proposal_type != "code_branch" or not proposal.code_branch):
        raise ValueError("the generated-code candidate requires proposal_type='code_branch' and source")
    if not is_code_candidate and proposal.proposal_type == "code_branch":
        raise ValueError("code_branch proposal must select the generated-code candidate")

    supplied = tuple(proposal.evidence_ids)
    if not supplied:
        relevant = []
        change_fields = set(proposal.changes)
        for item in reversed(history):
            previous = item.get("changes") if isinstance(item.get("changes"), dict) else {}
            if change_fields.intersection(previous):
                relevant.append(_evidence_id(item.get("iteration"), item.get("evidence_id")))
                break
        supplied = tuple(relevant or ["benchmark_reference"])

    incumbent = global_best_view(history, parent)
    incumbent_primary = (incumbent.get("validation_metrics") or {}).get("primary")
    observation = proposal.observation or (
        "The selected parent is being compared with the current validation evidence."
        if incumbent_primary is None else
        f"Current best observed validation Primary={float(incumbent_primary):.6f}."
    )
    diagnosis = proposal.diagnosis or (
        "A controlled candidate from the legal registry can test this mechanism without "
        "changing evaluation semantics."
    )
    expected_effect = proposal.expected_effect or {
        "GAUC": "uncertain", "nDCG@5": "uncertain", "primary": "uncertain"
    }
    risk = proposal.risk or (
        "The candidate may add cost without incremental ranking signal; validation decides."
    )
    success_condition = proposal.success_condition or (
        "Validation Primary exceeds the current global incumbent by more than epsilon; "
        "a smaller gain is non-improving evidence."
    )
    return replace(
        proposal,
        candidate_id=selected["candidate_id"],
        observation=observation,
        diagnosis=diagnosis,
        evidence_ids=supplied,
        expected_effect=expected_effect,
        risk=risk,
        estimated_cost=proposal.estimated_cost or selected["cost"],
        success_condition=success_condition,
    )


def evidence_catalog(
    history: list[dict[str, Any]], data_profile: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build a compact catalog whose IDs exactly match local validation."""
    catalog: list[dict[str, Any]] = [{
        "evidence_id": "benchmark_reference",
        "kind": "benchmark",
        "observation": f"Official validation Primary={OFFICIAL_BASELINE_PRIMARY:.4f}.",
    }]
    seen = {"benchmark_reference"}
    for strategy in STRATEGIES:
        catalog.append({
            "evidence_id": strategy.strategy_id,
            "kind": "research_strategy",
            "observation": strategy.purpose,
            "success_signal": strategy.success_signal,
        })
        seen.add(strategy.strategy_id)
    for item in history:
        if not isinstance(item, dict):
            continue
        evidence_id = _evidence_id(item.get("iteration"), item.get("evidence_id"))
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        critique = validation_critique_only(item.get("critique")) or {}
        metrics = validation_metrics_only(item.get("metrics")) or {}
        catalog.append({
            "evidence_id": evidence_id,
            "kind": "experiment_result",
            "changes": item.get("changes") if isinstance(item.get("changes"), dict) else {},
            "primary": metrics.get("primary"),
            "verdict": critique.get("verdict"),
        })
    if isinstance(data_profile, dict):
        summary_id = data_profile.get("evidence_id")
        if isinstance(summary_id, str) and summary_id and summary_id not in seen:
            seen.add(summary_id)
            catalog.append({
                "evidence_id": summary_id,
                "kind": "data_profile",
                "observation": "Validation-safe dataset profile summary.",
            })
        for finding in data_profile.get("key_findings", []):
            if not isinstance(finding, dict):
                continue
            evidence_id = finding.get("evidence_id")
            if not isinstance(evidence_id, str) or not evidence_id or evidence_id in seen:
                continue
            seen.add(evidence_id)
            catalog.append({
                "evidence_id": evidence_id,
                "kind": "data_profile_finding",
                "observation": finding.get("observation"),
                "candidate_direction": finding.get("candidate_direction"),
                "caveat": finding.get("caveat"),
            })
    return catalog


def compact_memory_for_planner(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain useful explored-path lessons without repeating every archived row."""
    summary = build_memory_summary(history)
    return {
        "counts": summary["counts"],
        "research_findings": summary["research_findings"],
        "confirmed_insights": summary["confirmed_insights"],
        # RecMind-inspired self-inspiring memory: rejected and uncertain paths
        # remain available as concise evidence rather than disappearing when a
        # different branch becomes the incumbent.
        "explored_path_lessons": {
            "promising": summary["promising"],
            "uncertain": summary["uncertain"],
            "negative": summary["negative"],
            "failed": summary["failed"],
        },
    }


def research_family_status(
    history: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compact hierarchical view of explored and available research families."""
    available: dict[str, int] = {}
    for candidate in candidates:
        family = str(candidate["branch"])
        available[family] = available.get(family, 0) + 1
    stats: dict[str, dict[str, Any]] = {}
    for item in history:
        if not isinstance(item, dict) or not item.get("changes"):
            continue
        family = branch_name(item["changes"])
        record = stats.setdefault(family, {
            "research_family": family,
            "experiments": 0,
            "failures": 0,
            "best_delta_vs_parent": None,
        })
        record["experiments"] += 1
        if item.get("status") != "success":
            record["failures"] += 1
        delta = item.get("delta_from_parent")
        if isinstance(delta, (int, float)) and math.isfinite(float(delta)):
            current = record["best_delta_vs_parent"]
            record["best_delta_vs_parent"] = (
                float(delta) if current is None else max(float(current), float(delta))
            )
    families = sorted(set(available) | set(stats))
    result = []
    for family in families:
        record = dict(stats.get(family, {
            "research_family": family,
            "experiments": 0,
            "failures": 0,
            "best_delta_vs_parent": None,
        }))
        record["available_candidates"] = available.get(family, 0)
        record["status"] = "unexplored" if record["experiments"] == 0 else "explored"
        result.append(record)
    return result


def global_best_view(
    history: list[dict[str, Any]], fallback_config: dict[str, Any],
) -> dict[str, Any]:
    """Keep incumbent config, metrics, and evidence ID from the same record."""
    candidates: list[dict[str, Any]] = []
    current_run = [
        item for item in history
        if isinstance(item, dict) and not item.get("historical")
    ]
    pools = (
        current_run,
        [item for item in history if isinstance(item, dict)
         and item.get("source") == "validated_incumbent"],
        [item for item in history if isinstance(item, dict)],
    )
    for pool in pools:
        candidates = []
        for item in pool:
            config = item.get("config")
            metrics = validation_metrics_only(item.get("metrics"))
            if (not isinstance(config, dict) or metrics is None or
                    item.get("status") not in (None, "success", "ok")):
                continue
            candidates.append({
                "config": config,
                "validation_metrics": metrics,
                "evidence_id": _evidence_id(item.get("iteration"), item.get("evidence_id")),
            })
        if candidates:
            return max(candidates, key=lambda item: item["validation_metrics"]["primary"])
    return {
        "config": fallback_config,
        "validation_metrics": latest_best_validation_metrics(history),
        "evidence_id": None,
    }


def remaining_search_space(
    best_config: dict[str, Any], *, autonomous: bool = False,
) -> dict[str, Any]:
    features = best_config.get("features") or {}
    result = {
        "models": [model for model in MODELS if model != best_config.get("model")],
        "training_objectives": [obj for obj in OBJECTIVES if obj != best_config.get("training_objective")],
        "features_still_off": [key for key in FEATURE_KEYS if not features.get(key)],
    }
    if autonomous:
        result["autonomous_composite_candidates"] = len(_autonomous_change_sets(best_config))
    return result


def latest_best_validation_metrics(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in reversed(history):
        if item.get("decision") == "KEEP":
            cleaned = validation_metrics_only(item.get("metrics"))
            if cleaned is not None:
                return cleaned
    return None


def default_allowed_values() -> dict[str, Any]:
    registry = planner_registry()
    all_models, all_objectives = tuple(MODELS), tuple(OBJECTIVES)
    return {
        "max_change_fields": MAX_CHANGE_FIELDS,
        "model_objectives": {
            model_id: list(spec.objectives) for model_id, spec in MODEL_SPECS.items()
        },
        "model_capabilities": planner_model_registry(),
        "model_switch_rules": {
            "leaving_fm_ensemble": {
                "ensemble_size": 1,
                "ensemble_seed_set": "sequential",
            },
            "entering_fm_ensemble": {
                "training_objective": "bpr",
                "ensemble_size": "greater than 1",
            },
            "entering_lightgbm": {
                "training_objective": "bce or lambdarank",
            },
        },
        "operators": {
            key: {
                "values": value["values"],
                "description": value["description"],
                "cost": value["cost"],
                **({"models": value["models"]}
                   if tuple(value["models"]) != all_models else {}),
                **({"objectives": value["objectives"]}
                   if tuple(value["objectives"]) != all_objectives else {}),
                **({"requires": value["requires"]} if value["requires"] else {}),
            }
            for key, value in registry.items()
        },
    }


def expansion_parent_view(
    history: list[dict[str, Any]],
    global_best_config: dict[str, Any],
) -> dict[str, Any]:
    """Describe the node being expanded, and whether it is the global best.

    The parent config is sent only when it differs from the global best, so the
    single-branch case costs no extra tokens.
    """
    parent = select_parent(history)
    if parent is None:
        return {"node_id": None, "iteration": None, "branch": None,
                "validation_primary": None, "validation_metrics": None,
                "same_as_global_best": True}
    parent_metrics = None
    for item in history:
        if item.get("iteration") == parent.iteration:
            parent_metrics = validation_metrics_only(item.get("metrics"))
            break
    view = {"node_id": parent.node_id, "iteration": parent.iteration,
            "branch": parent.branch, "validation_primary": parent.primary,
            "validation_metrics": parent_metrics,
            "same_as_global_best": parent.config == global_best_config}
    if not view["same_as_global_best"]:
        view["config"] = parent.config
    return view


def expansion_parent_view_for_config(
    history: list[dict[str, Any]],
    parent_config: dict[str, Any],
    global_best_config: dict[str, Any],
) -> dict[str, Any]:
    """Build a lineage view for a tree-selected config, including weaker parents."""
    parent_key = experiment_key(parent_config)
    for item in reversed(history):
        config = item.get("config") if isinstance(item, dict) else None
        iteration = item.get("iteration") if isinstance(item, dict) else None
        metrics = item.get("metrics") if isinstance(item, dict) else None
        if (not isinstance(config, dict) or not isinstance(iteration, int) or
                experiment_key(config) != parent_key):
            continue
        primary = validation_metrics_only(metrics)
        changes = item.get("changes") if isinstance(item.get("changes"), dict) else {}
        view = {
            "node_id": node_id_for(iteration),
            "iteration": iteration,
            "branch": branch_name(changes),
            "validation_primary": None if primary is None else primary.get("primary"),
            "validation_metrics": primary,
            "same_as_global_best": parent_config == global_best_config,
        }
        if not view["same_as_global_best"]:
            view["config"] = parent_config
        return view
    return {
        "node_id": None,
        "iteration": None,
        "branch": None,
        "validation_primary": None,
        "validation_metrics": None,
        "same_as_global_best": parent_config == global_best_config,
        **({} if parent_config == global_best_config else {"config": parent_config}),
    }


def build_planner_prompt(
    best_config: dict[str, Any],
    history: list[dict[str, Any]],
    allowed_values: dict[str, Any] | None = None,
    *,
    official_baseline_primary: float = OFFICIAL_BASELINE_PRIMARY,
    epsilon: float = CONVERGENCE_EPSILON,
    recent: int = RECENT_HISTORY,
    expansion_parent: dict[str, Any] | None = None,
    expansion_config: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
    data_profile: dict[str, Any] | None = None,
    research_context: dict[str, Any] | None = None,
    allow_code_branches: bool = False,
    autonomous: bool = False,
) -> dict[str, Any]:
    """Build the planner context. Does not call an HTTP API."""
    selected_parent = expansion_config or best_config
    all_candidates = legal_candidate_catalog(
        selected_parent, history, autonomous=autonomous
    )
    if not allow_code_branches:
        all_candidates = [
            row for row in all_candidates if row.get("candidate_kind") != "code_branch"
        ]
    research = research_context or build_research_context(
        selected_parent,
        history,
        all_candidates,
        iteration=int((budget or {}).get("iteration") or 1),
        total_iterations=int((budget or {}).get("total_iterations") or 10),
        parent_id=(expansion_parent or {}).get("node_id"),
        remaining_seconds=(budget or {}).get("remaining_seconds"),
    )
    candidates = research.get("ranked_candidates") or all_candidates
    if not allow_code_branches:
        candidates = [
            row for row in candidates if row.get("candidate_kind") != "code_branch"
        ]
    elif not any(row.get("candidate_kind") == "code_branch" for row in candidates):
        # Keep the open-ended option visible even when deterministic utility
        # ranking prefers five cheaper registry actions. This is the only
        # deliberate sixth row in the compact prompt and makes autonomy an
        # actual choice rather than a dead code path.
        generated = next(
            (row for row in all_candidates if row.get("candidate_kind") == "code_branch"),
            None,
        )
        if generated is not None:
            candidates = [*list(candidates)[:4], generated]
    incumbent = global_best_view(history, best_config)
    retrieved = research.get("retrieved_experiences") or {"successes": [], "failures": []}
    successes = list(retrieved.get("successes") or [])[:2]
    failures = list(retrieved.get("failures") or [])[:3]
    strategy_rows = list(research.get("retrieved_strategies") or [])[:3]
    candidate_rows = []
    for row in list(candidates)[:5]:
        candidate_rows.append({
            key: row[key] for key in (
                "candidate_id", "changes", "strategy_id", "predicted_utility",
                "estimated_runtime_seconds", "cost", "hypothesis", "risk",
                "evidence_ids", "candidate_kind", "evolution_recipe",
                "evolution_purpose",
                "code_contract",
                "autonomy_generated",
            ) if key in row
        })

    # EDA is selected for the mechanisms currently under consideration. The
    # complete profile remains an artifact on disk; sending all of it every
    # turn previously dominated the prompt and obscured the decision.
    candidate_tokens = set(re.findall(
        r"[a-z0-9]+", json.dumps([row.get("changes") for row in candidate_rows]).lower()
    ))
    eda_rows: list[dict[str, Any]] = []
    if isinstance(data_profile, dict):
        for finding in data_profile.get("key_findings", []):
            if not isinstance(finding, dict):
                continue
            text = json.dumps(finding).lower()
            tokens = set(re.findall(r"[a-z0-9]+", text))
            overlap = len(candidate_tokens & tokens)
            eda_rows.append({**finding, "_relevance": overlap})
        eda_rows.sort(key=lambda row: (-row.pop("_relevance"), str(row.get("evidence_id", ""))))
        eda_rows = eda_rows[:5]

    evidence = [{
        "evidence_id": "benchmark_reference", "kind": "benchmark",
        "observation": f"Official validation Primary={official_baseline_primary:.4f}.",
    }]
    seen_evidence = {"benchmark_reference"}
    for row in successes + failures + eda_rows:
        evidence_id = row.get("evidence_id") if isinstance(row, dict) else None
        if isinstance(evidence_id, str) and evidence_id not in seen_evidence:
            seen_evidence.add(evidence_id)
            metrics = row.get("metrics") if isinstance(row, dict) else None
            if isinstance(metrics, dict):
                evidence_row = {
                    "evidence_id": evidence_id,
                    "kind": "experiment_result",
                    "outcome": row.get("outcome"),
                    "primary": metrics.get("primary"),
                }
            else:
                evidence_row = {
                    "evidence_id": evidence_id,
                    "kind": "eda_finding",
                }
            evidence.append(evidence_row)
    for row in strategy_rows:
        strategy_id = row.get("strategy_id")
        if isinstance(strategy_id, str) and strategy_id not in seen_evidence:
            seen_evidence.add(strategy_id)
            evidence.append({
                "evidence_id": strategy_id,
                "kind": "research_strategy",
                "observation": row.get("purpose"),
                "success_signal": row.get("success_signal"),
            })
    if incumbent.get("evidence_id") and incumbent["evidence_id"] not in seen_evidence:
        seen_evidence.add(incumbent["evidence_id"])
        evidence.append({
            "evidence_id": incumbent["evidence_id"],
            "kind": "incumbent",
            "primary": (incumbent.get("validation_metrics") or {}).get("primary"),
        })

    compact_budget = {
        key: (budget or {}).get(key) for key in (
            "iteration", "total_iterations", "remaining_iterations", "remaining_seconds"
        )
    }
    memory_summary = build_memory_summary(history)
    compact_findings = []
    for finding in list(memory_summary.get("research_findings") or [])[-3:]:
        compact_findings.append({
            key: finding.get(key) for key in (
                "mechanism", "evidence_status", "best_delta", "seed_count"
            ) if key in finding
        })
    compact_memory = {
        "counts": memory_summary.get("counts", {}),
        "research_findings": compact_findings,
    }
    compact_profile = None
    if isinstance(data_profile, dict):
        compact_profile = {
            **({"evidence_id": data_profile["evidence_id"]}
               if isinstance(data_profile.get("evidence_id"), str) else {}),
            "key_findings": eda_rows,
        }
    return {
        "objective": (
            "Propose exactly one experiment that can improve validation Primary "
            "(mean of validation GAUC and nDCG@5)."
        ),
        "change_rule": CHANGE_RULE,
        "decision_semantics": (
            "Positive reward requires Primary above the global incumbent by more than epsilon. "
            "Only the baseline root and epsilon-clearing global improvements are parents; "
            "all other valid results are non-improving evidence."
        ),
        "official_baseline_primary": official_baseline_primary,
        "benchmark_reference": {
            "evidence_id": "benchmark_reference",
            "official_validation_primary": official_baseline_primary,
        },
        "epsilon": epsilon,
        "autonomy": {
            "autonomous_mode": bool(autonomous),
            "code_branches_enabled": bool(allow_code_branches),
            "source_limit_characters": MAX_SOURCE_CHARS,
            "candidate_generation": (
                "controller-generated legal catalog plus bounded automatically composed bundles"
                if autonomous else "registered legal catalog"
            ),
        },
        "budget": compact_budget,
        "remaining": remaining_search_space(selected_parent, autonomous=autonomous),
        "memory": compact_memory,
        "research_search": {
            "phase": research.get("phase"),
            "expansion_mode": research.get("expansion_mode"),
            "diagnoses": list(research.get("diagnoses") or [])[:3],
            "retrieved_strategies": strategy_rows,
            "retrieved_experiences": {"successes": successes, "failures": failures},
            "ranked_candidates": [
                {key: row.get(key) for key in ("candidate_id", "predicted_utility")}
                for row in candidate_rows
            ],
        },
        "global_best": incumbent,
        "expansion_parent": (expansion_parent if expansion_parent is not None
                             else expansion_parent_view(history, incumbent["config"])),
        "data_profile": compact_profile,
        "valid_evidence_ids": [item["evidence_id"] for item in evidence],
        "evidence_catalog": evidence,
        "legal_candidates": candidate_rows,
        "response_contract": {
            "proposal_type": (
                "return 'config' for registry candidates; return 'code_branch' only for "
                "candidate_kind=code_branch"
            ),
            "candidate_id": "select exactly one ID from legal_candidates",
            "evidence_ids": "cite one or more IDs from valid_evidence_ids only",
            "changes": "copy the selected legal candidate's changes exactly",
            "expected_effect": "direction for GAUC, nDCG@5, and primary",
            "estimated_cost": "advisory only; the Controller enforces the real budget",
            "code_branch": (
                "for code_branch proposals, provide branch_name and source defining "
                "fit_validate(runner, config, checkpoint) and finalize(runner, config, checkpoint, output); "
                "base_model and description are strings or null"
            ),
            "research_process": (
                "Choose using only the compact incumbent, relevant failures/successes, EDA, "
                "diagnoses, and five-candidate shortlist supplied here."
            ),
        },
    }


def data_profile_evidence_ids(data_profile: dict[str, Any] | None) -> set[str]:
    """Collect the evidence IDs that an LLM is allowed to cite from the profile."""
    if not isinstance(data_profile, dict):
        return set()
    result = set()
    supplied = data_profile.get("evidence_id")
    if isinstance(supplied, str) and supplied:
        result.add(supplied)
    findings = data_profile.get("key_findings")
    if isinstance(findings, list):
        for finding in findings:
            evidence_id = finding.get("evidence_id") if isinstance(finding, dict) else None
            if isinstance(evidence_id, str) and evidence_id:
                result.add(evidence_id)
    return result


class DeterministicResearcher:
    """Offline researcher using the same diagnosis/retrieval ranking as the LLM."""

    def __init__(self, *, autonomous_mode: bool = False) -> None:
        self._run_context: dict[str, Any] = {}
        self.autonomous_mode = bool(autonomous_mode)

    def set_run_context(self, context: dict[str, Any]) -> None:
        self._run_context = dict(context)

    @staticmethod
    def _proposal(
        candidate: dict[str, Any],
        *,
        hypothesis: str,
        reason: str,
        observation: str = (
            "This candidate is available in the Controller-generated legal experiment catalog."
        ),
        diagnosis: str = (
            "A controlled validation experiment can test whether this mechanism adds "
            "incremental within-user ranking signal."
        ),
        expected_effect: dict[str, str] | None = None,
        risk: str = (
            "The added model capacity may increase runtime without improving GAUC or nDCG@5."
        ),
        success_condition: str = (
            "Validation Primary exceeds the current global incumbent by more than epsilon; "
            "smaller gains remain non-improving evidence."
        ),
    ) -> Proposal:
        return Proposal(
            hypothesis=hypothesis,
            reason=reason,
            changes=dict(candidate["changes"]),
            source="deterministic",
            token_usage=empty_token_usage(),
            candidate_id=str(candidate["candidate_id"]),
            observation=observation,
            diagnosis=diagnosis,
            evidence_ids=tuple(dict.fromkeys([
                *candidate.get("evidence_ids", ["benchmark_reference"]),
                *([candidate["strategy_id"]] if candidate.get("strategy_id") else []),
            ])),
            expected_effect=expected_effect or {
                "GAUC": "uncertain",
                "nDCG@5": "uncertain",
                "primary": "increase",
            },
            risk=risk,
            estimated_cost=str(candidate["cost"]),
            success_condition=success_condition,
        )

    def propose(self, best: dict[str, Any], history: list[dict[str, Any]]) -> Proposal:
        catalog = legal_candidate_catalog(
            best, history, autonomous=self.autonomous_mode
        )
        if not catalog:
            raise StopIteration("the legal non-duplicate experiment space is exhausted")
        research = self._run_context.get("research")
        if not isinstance(research, dict):
            research = build_research_context(
                best,
                history,
                catalog,
                iteration=int(self._run_context.get("iteration") or 1),
                total_iterations=int(self._run_context.get("total_iterations") or 10),
                parent_id=self._run_context.get("parent_id"),
                remaining_seconds=self._run_context.get("remaining_seconds"),
            )
        ranked = research.get("ranked_candidates") if isinstance(research, dict) else None
        # Offline mode has no source generator. Keep the open-ended candidate
        # visible to the LLM, but never emit a source-less custom config.
        usable_catalog = [
            item for item in catalog if item.get("candidate_kind") != "code_branch"
        ]
        if not usable_catalog:
            raise StopIteration("only generated-code candidates remain; use --researcher llm --open-ended")
        available = {item["candidate_id"]: item for item in usable_catalog}
        candidate = next(
            (item for item in (ranked or []) if item.get("candidate_id") in available),
            usable_catalog[0],
        )
        return self._proposal(
            candidate,
            hypothesis=str(candidate.get("hypothesis") or "Test the highest-utility legal research candidate."),
            reason=str(candidate.get("reason") or "Diagnosis, retrieved experience, novelty, and cost rank this candidate highest."),
            diagnosis=str(candidate.get("diagnosis") or "The progressive research policy selected this candidate."),
            risk=str(candidate.get("risk") or "The hypothesis may not improve validation ranking."),
        )


class OpenAICompatibleResearcher:
    """Small OpenAI-compatible JSON client using only the Python standard library."""

    is_llm = True

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        *,
        api_key: str | None = None,
        urlopen: Callable[..., Any] | None = None,
        timeout: float = HTTP_TIMEOUT_SECONDS,
        retry_backoff_seconds: float = 0.5,
        sleeper: Callable[[float], None] = time.sleep,
        data_profile: dict[str, Any] | None = None,
        allow_code_branches: bool = False,
        autonomous_mode: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for --researcher llm")
        self._urlopen = urlopen or urllib.request.urlopen
        self.timeout = timeout
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._sleep = sleeper
        self.last_token_usage = empty_token_usage()
        self.last_attempts = 0
        self.last_call_records: list[dict[str, Any]] = []
        self.last_error_category: str | None = None
        self.last_proposal_episode_id: str | None = None
        self._run_context: dict[str, Any] = {}
        self._last_http_meta: dict[str, Any] = {}
        self.data_profile = data_profile
        self.allow_code_branches = bool(allow_code_branches)
        self.autonomous_mode = bool(autonomous_mode)

    def set_run_context(self, context: dict[str, Any]) -> None:
        """Receive deterministic budget context from the Controller before planning."""
        self._run_context = {
            "iteration": context.get("iteration"),
            "parent_id": context.get("parent_id"),
            "remaining_iterations": context.get("remaining_iterations"),
            "remaining_seconds": context.get("remaining_seconds"),
            "estimated_next_experiment_seconds": context.get(
                "estimated_next_experiment_seconds"
            ),
            "total_iterations": context.get("total_iterations"),
            "research": context.get("research"),
            "allow_code_branches": self.allow_code_branches,
            "autonomous_mode": self.autonomous_mode,
        }

    def propose(self, best: dict[str, Any], history: list[dict[str, Any]]) -> Proposal:
        incumbent = global_best_view(history, best)
        global_best_config = incumbent["config"]
        parent_view = expansion_parent_view_for_config(history, best, global_best_config)
        prompt = build_planner_prompt(
            global_best_config,
            history,
            expansion_parent=parent_view,
            expansion_config=best,
            budget=self._run_context,
            data_profile=self.data_profile,
            research_context=(
                self._run_context.get("research")
                if isinstance(self._run_context.get("research"), dict) else None
            ),
            allow_code_branches=self.allow_code_branches,
            autonomous=self.autonomous_mode,
        )
        legal_candidates = prompt["legal_candidates"]
        if not legal_candidates:
            raise StopIteration("the legal non-duplicate experiment space is exhausted")
        candidates_by_id = {item["candidate_id"]: item for item in legal_candidates}
        valid_evidence_ids = set(prompt["valid_evidence_ids"])
        accumulated = empty_token_usage()
        self.last_token_usage = empty_token_usage()
        self.last_attempts = 0
        self.last_call_records = []
        self.last_error_category = None
        episode_id = f"proposal_{uuid.uuid4().hex}"
        self.last_proposal_episode_id = episode_id
        last_error = "unknown error"
        repair_error: str | None = None
        prompt_text = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
        prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
            self.last_attempts = attempt
            call_id = f"llm_{uuid.uuid4().hex}"
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.perf_counter()
            record = {
                "call_id": call_id,
                "proposal_episode_id": episode_id,
                "request_type": "planner_proposal",
                "iteration": self._run_context.get("iteration"),
                "parent_id": self._run_context.get("parent_id"),
                "attempt": attempt,
                "started_at": started_at,
                "model": self.model,
                "endpoint": f"{self.base_url}/chat/completions",
                "repair": attempt > 1,
                "system_prompt": RESEARCHER_SYSTEM_PROMPT,
                "repair_instruction": None if attempt == 1 else _redact_text(
                    f"{REPAIR_INSTRUCTION} Previous rejection: {repair_error or last_error}",
                    self.api_key,
                ),
                "prompt_hash": prompt_hash,
                "prompt_characters": len(prompt_text),
                "prompt_section_characters": {
                    str(key): len(json.dumps(value, ensure_ascii=False, sort_keys=True))
                    for key, value in prompt.items()
                },
                "prompt": _redact_data(prompt, self.api_key),
                "usage": empty_token_usage(),
                "http_status": None,
                "provider_request_id": None,
                "response": None,
                "result": "failure",
                "error": None,
            }
            try:
                payload = self._chat(
                    prompt,
                    repair_error=repair_error,
                    valid_evidence_ids=sorted(valid_evidence_ids),
                    valid_candidate_ids=sorted(candidates_by_id),
                )
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                    socket.timeout, ValueError) as exc:
                category, retryable, status = _classify_call_error(exc)
                last_error = f"{type(exc).__name__}: {exc}"
                repair_error = last_error
                self.last_error_category = category
                record.update(self._last_http_meta)
                record["http_status"] = status if status is not None else record.get("http_status")
                if isinstance(exc, urllib.error.HTTPError):
                    record["provider_request_id"] = _header_value(exc, "x-request-id")
                    body = _http_error_body(exc)
                    if body:
                        record["response"] = {
                            "body": _redact_text(body[:4000], self.api_key)
                        }
                record["latency_seconds"] = time.perf_counter() - started
                record["error"] = {
                    "category": category,
                    "type": type(exc).__name__,
                    "message": _redact_text(str(exc), self.api_key),
                    "retryable": retryable,
                }
                self.last_call_records.append(record)
                if not retryable or attempt >= MAX_LLM_ATTEMPTS:
                    raise LLMProposalFailure(category, attempt, self.last_call_records) from exc
                self._backoff(attempt)
                continue
            usage = extract_token_usage(payload)
            accumulated = add_token_usage(accumulated, usage)
            self.last_token_usage = accumulated
            record.update(self._last_http_meta)
            record["usage"] = usage
            record["provider_request_id"] = (
                record.get("provider_request_id") or payload.get("id")
            )
            content = _message_content(payload)
            record["response"] = {"content": _redact_text(content or "", self.api_key)}
            try:
                if not content:
                    refusal = _message_refusal(payload)
                    if refusal:
                        record["response"] = {
                            "refusal": _redact_text(refusal, self.api_key)
                        }
                        raise LLMResponseError("model_refusal", refusal, retryable=False)
                    raise LLMResponseError("schema_validation", "LLM response content is empty")
                try:
                    raw = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise LLMResponseError("invalid_json", str(exc)) from exc
                if not isinstance(raw, dict):
                    raise LLMResponseError(
                        "schema_validation", "LLM response content must be a JSON object"
                    )
                proposal = Proposal.parse(raw, "llm")
                selected = candidates_by_id.get(proposal.candidate_id)
                if selected is None:
                    raise LLMResponseError(
                        "candidate_validation",
                        f"unknown or unavailable candidate_id: {proposal.candidate_id!r}",
                    )
                unknown_evidence = set(proposal.evidence_ids) - valid_evidence_ids
                if unknown_evidence:
                    raise LLMResponseError(
                        "schema_validation",
                        f"proposal cited unknown evidence IDs: {sorted(unknown_evidence)}",
                    )
                if proposal.changes != selected["changes"]:
                    raise LLMResponseError(
                        "candidate_validation",
                        "proposal changes do not match the selected legal candidate",
                    )
                if selected.get("candidate_kind") == "code_branch":
                    if proposal.proposal_type != "code_branch" or not proposal.code_branch:
                        raise LLMResponseError(
                            "candidate_validation",
                            "generated-code candidate requires proposal_type='code_branch' and source",
                        )
                    try:
                        validate_source(str(proposal.code_branch.get("source", "")))
                    except ValueError as exc:
                        raise LLMResponseError(
                            "code_validation", str(exc), retryable=True
                        ) from exc
                elif proposal.proposal_type != "config":
                    raise LLMResponseError(
                        "candidate_validation",
                        "built-in candidates require proposal_type='config'",
                    )
                try:
                    candidate = apply_changes(best, proposal.changes)
                except (KeyError, TypeError, ValueError) as exc:
                    raise LLMResponseError("invalid_config", str(exc)) from exc
            except LLMResponseError as exc:
                category, retryable = exc.category, exc.retryable
                last_error = f"{type(exc).__name__}: {exc}"
                repair_error = last_error
                self.last_error_category = category
                record["latency_seconds"] = time.perf_counter() - started
                record["error"] = {
                    "category": category,
                    "type": type(exc).__name__,
                    "message": _redact_text(str(exc), self.api_key),
                    "retryable": retryable,
                }
                self.last_call_records.append(record)
                if not retryable or attempt >= MAX_LLM_ATTEMPTS:
                    raise LLMProposalFailure(category, attempt, self.last_call_records) from exc
                continue
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                category = "schema_validation"
                last_error = f"{type(exc).__name__}: {exc}"
                repair_error = last_error
                self.last_error_category = category
                record["latency_seconds"] = time.perf_counter() - started
                record["error"] = {
                    "category": category,
                    "type": type(exc).__name__,
                    "message": _redact_text(str(exc), self.api_key),
                    "retryable": True,
                }
                self.last_call_records.append(record)
                if attempt >= MAX_LLM_ATTEMPTS:
                    raise LLMProposalFailure(category, attempt, self.last_call_records) from exc
                continue
            record["result"] = "success"
            record["error"] = None
            record["latency_seconds"] = time.perf_counter() - started
            self.last_call_records.append(record)
            self.last_error_category = None
            return replace(
                proposal,
                token_usage=dict(accumulated),
                llm_attempts=attempt,
                llm_call_ids=tuple(row["call_id"] for row in self.last_call_records),
            )
        raise LLMProposalFailure(
            self.last_error_category or "unknown", MAX_LLM_ATTEMPTS, self.last_call_records
        )

    def _backoff(self, attempt: int) -> None:
        if self.retry_backoff_seconds:
            self._sleep(self.retry_backoff_seconds * (2 ** (attempt - 1)))

    def _chat(
        self,
        prompt: dict[str, Any],
        *,
        repair_error: str | None,
        valid_evidence_ids: list[str],
        valid_candidate_ids: list[str],
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        if repair_error:
            messages.append({
                "role": "user",
                "content": f"{REPAIR_INSTRUCTION} Previous rejection: {repair_error}",
            })
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "research_experiment_proposal",
                    "strict": True,
                    "schema": proposal_json_schema(
                        valid_evidence_ids=valid_evidence_ids,
                        valid_candidate_ids=valid_candidate_ids,
                    ),
                },
            },
            "temperature": 0.2,
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        self._last_http_meta = {}
        with self._urlopen(request, timeout=self.timeout) as response:
            self._last_http_meta = {
                "http_status": _response_status(response),
                "provider_request_id": _header_value(response, "x-request-id"),
            }
            raw = response.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        self._last_http_meta["response_characters"] = len(raw)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._last_http_meta["response"] = {
                "body": _redact_text(raw[:4000], self.api_key)
            }
            raise ValueError("LLM HTTP body was not JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("LLM HTTP body must be a JSON object")
        return payload


class LLMResponseError(ValueError):
    def __init__(self, category: str, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable


class LLMProposalFailure(RuntimeError):
    def __init__(self, category: str, attempts: int,
                 records: list[dict[str, Any]] | None = None) -> None:
        super().__init__(f"LLM proposal failed after {attempts} attempts: {category}")
        self.category = category
        self.attempts = attempts
        self.call_ids = tuple(
            str(item.get("call_id")) for item in (records or []) if item.get("call_id")
        )


def _classify_call_error(exc: Exception) -> tuple[str, bool, int | None]:
    if isinstance(exc, urllib.error.HTTPError):
        status = int(exc.code)
        if status in {401, 403}:
            return "authentication", False, status
        if status == 429:
            return "rate_limit", True, status
        if status in {408, 504}:
            return "timeout", True, status
        if status >= 500:
            return "provider_5xx", True, status
        if status == 400 and "context" in str(exc).lower():
            return "context_limit", False, status
        return "provider_4xx", False, status
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "timeout", True, None
    if isinstance(exc, urllib.error.URLError):
        if isinstance(getattr(exc, "reason", None), (TimeoutError, socket.timeout)):
            return "timeout", True, None
        return "network", True, None
    if isinstance(exc, ValueError):
        return "invalid_json", True, None
    return "unknown", False, None


def _response_status(response: Any) -> int | None:
    value = getattr(response, "status", getattr(response, "code", None))
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _header_value(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    value = getter(name)
    return str(value) if value else None


def _http_error_body(error: urllib.error.HTTPError) -> str:
    reader = getattr(error, "read", None)
    if not callable(reader):
        return ""
    try:
        value = reader()
    except Exception:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value) if value is not None else ""


def _redact_text(value: str, api_key: str | None = None) -> str:
    text = value
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    return re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED]", text)


def _redact_data(value: Any, api_key: str | None = None) -> Any:
    if isinstance(value, str):
        return _redact_text(value, api_key)
    if isinstance(value, dict):
        return {str(key): _redact_data(item, api_key) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_data(item, api_key) for item in value]
    return value


def _message_content(payload: dict[str, Any]) -> str | None:
    try:
        content = payload["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return content if isinstance(content, str) else None


def _message_refusal(payload: dict[str, Any]) -> str | None:
    try:
        refusal = payload["choices"][0]["message"].get("refusal")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return refusal if isinstance(refusal, str) and refusal.strip() else None


def _message_json(payload: dict[str, Any]) -> dict[str, Any]:
    content = _message_content(payload)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response content is empty")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response content must be a JSON object")
    return parsed
