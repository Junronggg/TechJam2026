from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SKILL_REGISTRY_VERSION = "v1"


@dataclass(frozen=True)
class ResearchSkill:
    """A reusable executable capability, not a research conclusion."""

    skill_id: str
    category: str
    owner: str
    handler: str
    description: str
    status: str = "available"
    test_labels_allowed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SkillRegistry:
    def __init__(self, skills: tuple[ResearchSkill, ...]) -> None:
        self._skills = {skill.skill_id: skill for skill in skills}
        if len(self._skills) != len(skills):
            raise ValueError("research skill ids must be unique")

    def require(self, skill_id: str) -> ResearchSkill:
        skill = self._skills.get(skill_id)
        if skill is None:
            raise ValueError(f"unregistered research skill: {skill_id}")
        if skill.status != "available":
            raise ValueError(f"research skill is not executable: {skill_id}")
        if skill.test_labels_allowed:
            raise ValueError(f"research skill may not access test labels: {skill_id}")
        return skill

    def catalog(self) -> list[dict[str, Any]]:
        return [
            self._skills[key].as_dict()
            for key in sorted(self._skills)
        ]

    def primary_for_candidate(
        self, family: str, changes: dict[str, Any]
    ) -> str:
        if family in {
            "multitask", "pairwise_multitask", "censored_watchtime",
            "pairwise_censored_watchtime",
        }:
            return "train_with_auxiliary_loss"
        if any(key in CONTROLLED_FEATURE_KEYS or key in OTHER_FEATURE_KEYS
               for key in changes):
            return "build_feature"
        return "train_model"

    def evidence_for_candidate(
        self, family: str, changes: dict[str, Any]
    ) -> tuple[str, ...]:
        required: list[str] = []
        if any(key in CONTROLLED_FEATURE_KEYS for key in changes):
            required.append("run_placebo")
        required.extend(("run_rolling", "run_paired_seeds"))
        if family in MODEL_DIVERSITY_FAMILIES:
            required.append("analyze_prediction_diversity")
        for skill_id in required:
            self.require(skill_id)
        return tuple(required)


CONTROLLED_FEATURE_KEYS = {
    "prior_video_positive",
    "author_positive_recency",
    "prior_video_count",
    "previous_author_same",
}
OTHER_FEATURE_KEYS = {
    "user_long_view_rate",
    "item_long_view_rate",
    "continuous_history_stats",
    "user_tab_long_view_rate",
    "user_tab_cross",
    "user_author_cross",
    "user_recent_3d_activity",
    "item_recent_3d_exposure",
    "global_context",
}
MODEL_DIVERSITY_FAMILIES = {
    "heterogeneous_ensemble",
    "multitask",
    "pairwise_multitask",
    "censored_watchtime",
    "pairwise_censored_watchtime",
    "cross_network",
    "sequence_model",
    "tree_model",
}


DEFAULT_SKILLS = (
    ResearchSkill(
        "read_research_memory", "research_memory", "planner",
        "techjam_agent.memory.build_structured_research_memory",
        "Read validation-only hypotheses and distilled family policies.",
    ),
    ResearchSkill(
        "profile_candidate", "discovery", "runner",
        "techjam_agent.research_diagnostics",
        "Measure coverage, slices, and candidate diagnostics without test labels.",
    ),
    ResearchSkill(
        "train_model", "training", "runner",
        "techjam_agent.runner.ExperimentRunner.run",
        "Train one registered model/objective configuration and score validation.",
    ),
    ResearchSkill(
        "train_with_auxiliary_loss", "training", "runner",
        "techjam_agent.runner.ExperimentRunner.run",
        "Train a registered auxiliary or multi-task objective.",
    ),
    ResearchSkill(
        "build_feature", "discovery", "runner",
        "techjam_agent.runner.ExperimentRunner.run",
        "Apply one registered leakage-safe feature transformation.",
    ),
    ResearchSkill(
        "run_placebo", "evidence", "controller",
        "techjam_agent.controller.Controller._maybe_schedule_placebos",
        "Compare a feature with constant, shuffled, and cardinality-matched controls.",
    ),
    ResearchSkill(
        "run_rolling", "evidence", "controller",
        "techjam_agent.confirmation.run_rolling_confirmation",
        "Run expanding-window reference/candidate validation folds.",
    ),
    ResearchSkill(
        "run_paired_seeds", "evidence", "controller",
        "techjam_agent.confirmation.run_paired_seed_confirmation",
        "Run predeclared paired optimization seeds.",
    ),
    ResearchSkill(
        "analyze_prediction_diversity", "evidence", "runner",
        "techjam_agent.research_diagnostics.compute_diagnostics",
        "Measure prediction correlation, slices, and conditional error recovery.",
    ),
    ResearchSkill(
        "update_research_memory", "research_memory", "controller",
        "techjam_agent.memory.build_structured_research_memory",
        "Write reflected validation evidence and reusable family policy.",
    ),
)


def default_skill_registry() -> SkillRegistry:
    return SkillRegistry(DEFAULT_SKILLS)
