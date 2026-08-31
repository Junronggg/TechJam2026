from __future__ import annotations

from typing import Any


RESEARCH_PRINCIPLES = (
    "Maximize validation Primary while reducing redundant experiments and human intervention.",
    "Use train and validation evidence only; never request or infer test results.",
    "Prefer a materially new information source over fine-grained tuning of a saturated family.",
    "Do not repeat a STOP_DIRECTION family unless the mechanism or evidence source is materially different.",
    "Treat a small gain as a discovery that requires confirmation; prefer evidence from multiple rolling folds over a single split.",
    "Require matched placebo evidence when a new categorical history feature may change model structure.",
    "Check prediction diversity and conditional error recovery before recommending an ensemble.",
    "Separate observation from interpretation and state uncertainty explicitly.",
)


def controller_guards(project: dict[str, Any] | None = None) -> dict[str, Any]:
    project = project or {}
    limits = project.get("run_limits", {})
    return {
        "test_labels_allowed_during_research": False,
        "max_iterations": limits.get("max_iterations", 50),
        "max_wall_clock_hours": limits.get("max_wall_clock_hours", 6),
        "official_evaluator_only": True,
        "isolated_subprocess_timeout_seconds": project.get(
            "experiment_timeout_seconds", 900
        ),
        "llm_repository_edits_allowed": False,
    }


def decision_record_contract() -> dict[str, str]:
    """Fields the Controller derives and audits after one candidate is selected."""
    return {
        "hypothesis": "falsifiable research claim",
        "mechanism_basis": "why this mechanism may change ranking quality",
        "family": "experiment family selected from candidate_ranking",
        "proposed_action": "registered primary skill_id",
        "expected_gain": "planner prior or observed estimate",
        "novelty": "difference from previously tested information sources",
        "risk": "main evidence, redundancy, or compute risk",
        "required_confirmation": "registered evidence skill_ids",
    }


def capability_policy() -> dict[str, Any]:
    return {
        "unregistered_capability_action": "report_gap_and_do_not_execute",
        "capability_builder_enabled": False,
        "known_gaps": ["train_graph", "build_new_model_family"],
    }


def system_prompt() -> str:
    principles = "\n".join(f"- {principle}" for principle in RESEARCH_PRINCIPLES)
    return (
        "You are an Autonomous ML Researcher. Apply these research principles:\n"
        f"{principles}\n"
        "Treat prior_evidence as authoritative experimental memory, dataset_facts as "
        "training-only observations, and method_reference as conditional guidance rather "
        "than proof. "
        "The skill catalog describes what can actually be executed. Select exactly one "
        "ranked candidate and never invent a skill, model, feature, or configuration. "
        "Controller guards are hard constraints, not suggestions. Your hypothesis and "
        "reason explain the decision; the Controller derives the audited decision record "
        "from the selected candidate and registry."
    )
