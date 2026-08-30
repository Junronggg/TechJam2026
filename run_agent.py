"""Single entry point; defaults to the active src/techjam_agent autonomous loop."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from agent.critic import GroundedCritic
from agent.manager import ResearchManager
from agent.planner import DeterministicPlanner, ExperimentTemplate
from agent.tree import TreePolicyConfig, TreeSearchPolicy
from experiment.logger import ExperimentLogger
from experiment.real_backend import OfficialFMSubprocessBackend
from experiment.runner import DryRunBackend, ExperimentRunner
from experiment.schemas import (
    ExperimentStatus,
    MetricBundle,
    ModelConfig,
    Operation,
    RunBudget,
)
from experiment.validator import ExperimentValidator


ROOT = Path(__file__).resolve().parent


def _run_active_agent() -> int:
    path = ROOT / "scripts" / "run_agent.py"
    spec = importlib.util.spec_from_file_location("techjam_active_run_agent", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load active agent entry point: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main())


def load_agent_config() -> dict[str, object]:
    return json.loads((ROOT / "configs" / "agent.json").read_text(encoding="utf-8"))


def baseline_config() -> ModelConfig:
    return ModelConfig(
        model="fm",
        features=("user_id", "video_id", "author_id", "tab", "dur_bucket"),
        hyperparameters={"k": 16, "lr": 0.001, "epochs": 40, "l2": 1e-6},
        seed=0,
    )


def dry_run_templates() -> tuple[ExperimentTemplate, ...]:
    return (
        ExperimentTemplate(
            branch="capacity",
            hypothesis="A larger FM latent dimension may capture richer interactions.",
            operation=Operation.CHANGE_HYPERPARAMETER,
            parameters={"name": "k", "value": 32},
            expected_effect={"gauc": "small increase", "ndcg": "uncertain"},
        ),
        ExperimentTemplate(
            branch="optimization",
            hypothesis="A higher learning rate may reach a better solution within the epoch budget.",
            operation=Operation.CHANGE_HYPERPARAMETER,
            parameters={"name": "lr", "value": 0.003},
            expected_effect={"gauc": "increase", "ndcg": "increase"},
        ),
        ExperimentTemplate(
            branch="feature_ablation",
            hypothesis="Removing duration buckets tests whether they add ranking signal.",
            operation=Operation.REMOVE_FEATURE,
            parameters={"feature": "dur_bucket"},
            expected_effect={"gauc": "uncertain", "ndcg": "uncertain"},
        ),
        ExperimentTemplate(
            branch="item_history",
            hypothesis="Smoothed item history may add ranking signal.",
            operation=Operation.ADD_FEATURE,
            parameters={"feature": "item_long_view_rate"},
            expected_effect={"gauc": "increase", "ndcg": "uncertain"},
        ),
        ExperimentTemplate(
            branch="personalization",
            hypothesis="User-tag affinity may add personalized ranking signal.",
            operation=Operation.ADD_FEATURE,
            parameters={"feature": "user_tag_affinity"},
            expected_effect={"gauc": "increase", "ndcg": "increase"},
        ),
    )


def real_run_templates() -> tuple[ExperimentTemplate, ...]:
    return (
        ExperimentTemplate(
            branch="ranking_objective",
            hypothesis="Within-user BPR should align FM training with ranking metrics.",
            operation=Operation.CHANGE_OBJECTIVE,
            parameters={"objective": "bpr"},
            expected_effect={"gauc": "increase", "ndcg": "increase"},
            evidence="The benchmark scores within-user ranking, while the baseline uses pointwise loss.",
        ),
        ExperimentTemplate(
            branch="item_history",
            hypothesis="Smoothed historical item relevance may improve ranking.",
            operation=Operation.ADD_FEATURE,
            parameters={"feature": "item_long_view_rate"},
            expected_effect={"gauc": "increase", "ndcg": "increase"},
            evidence="Uses leave-one-out train encoding and train-only validation transforms.",
        ),
        ExperimentTemplate(
            branch="personalization",
            hypothesis="Smoothed user-tag relevance history may improve personalization.",
            operation=Operation.ADD_FEATURE,
            parameters={"feature": "user_tag_affinity"},
            expected_effect={"gauc": "increase", "ndcg": "increase"},
            evidence="Uses only train labels with leave-one-out encoding for training rows.",
        ),
        ExperimentTemplate(
            branch="item_exposure",
            hypothesis="Item exposure popularity may add a stable prior.",
            operation=Operation.ADD_FEATURE,
            parameters={"feature": "item_popularity"},
            expected_effect={"gauc": "uncertain", "ndcg": "increase"},
            evidence="Counts training impressions only.",
        ),
        ExperimentTemplate(
            branch="user_activity",
            hypothesis="User activity buckets may interact with item-side FM fields.",
            operation=Operation.ADD_FEATURE,
            parameters={"feature": "user_activity"},
            expected_effect={"gauc": "uncertain", "ndcg": "uncertain"},
            evidence="Counts training interactions only; useful through FM cross terms.",
        ),
    )


def build_manager(
    output_dir: Path,
    iterations: int,
    templates: tuple[ExperimentTemplate, ...],
    backend: object,
) -> ResearchManager:
    settings = load_agent_config()
    budget_settings = dict(settings["budget"])
    tree_settings = dict(settings["tree_search"])
    budget_settings["max_iterations"] = iterations
    return ResearchManager(
        planner=DeterministicPlanner(templates[:iterations]),
        critic=GroundedCritic(float(budget_settings["convergence_epsilon"])),
        runner=ExperimentRunner(
            runs_dir=output_dir / "runs",
            backend=backend,
            validator=ExperimentValidator(),
        ),
        logger=ExperimentLogger(output_dir / "experiment_history.jsonl"),
        budget=RunBudget(**budget_settings),
        tree_policy=TreeSearchPolicy(TreePolicyConfig(**tree_settings)),
    )


def build_real_backend(data_dir: Path) -> OfficialFMSubprocessBackend:
    settings = load_agent_config()
    backend_settings = dict(settings["real_backend"])
    return OfficialFMSubprocessBackend(
        project_root=ROOT,
        starter_dir=ROOT / "kuairand-starter-kit",
        data_dir=data_dir,
        evaluator_sha256=str(settings["official_evaluator_sha256"]),
        timeout_seconds=float(backend_settings["experiment_timeout_seconds"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TechJam autonomous ML research agent")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="exercise control flow with deterministic fake metrics and no training",
    )
    mode.add_argument(
        "--real-run",
        action="store_true",
        help="run real FM experiments on the KuaiRand-Pure validation split",
    )
    parser.add_argument("--iterations", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "KuaiRand-Pure" / "data",
    )
    return parser.parse_args()


def print_summary(manager: ResearchManager, summary: object, real_run: bool) -> None:
    print(f"Iterations: {summary.completed_iterations}")
    print(f"Stop reason: {summary.stop_reason}")
    print(f"Best node: {summary.best_node_id}")
    label = "Best validation metrics" if real_run else "Best simulated metrics"
    print(
        f"{label}: GAUC {summary.best_metrics.gauc:.4f} | "
        f"nDCG@5 {summary.best_metrics.ndcg_at_5:.4f} | "
        f"Primary {summary.best_metrics.primary:.4f}"
    )
    if not real_run:
        return
    official_primary = (0.6674 + 0.5357) / 2.0
    print(f"Delta vs official validation baseline: {summary.best_metrics.primary - official_primary:+.4f}")
    print("Experiment tree:")
    for node in manager.memory.nodes.values():
        if node.result.metrics is None:
            print(f"  {node.node_id} <- {node.parent_id}: {node.result.status.value}")
            continue
        decision = node.critic.decision.value if node.critic else "root"
        print(
            f"  {node.node_id} <- {node.parent_id}: {node.branch} | "
            f"Primary {node.result.metrics.primary:.4f} | {decision}"
        )


def main() -> int:
    if not any(flag in sys.argv[1:] for flag in ("--dry-run", "--real-run")):
        return _run_active_agent()
    args = parse_args()
    if not args.dry_run and not args.real_run:
        print(
            "Choose --dry-run for simulated control-flow checks or --real-run "
            "for real KuaiRand-Pure validation experiments."
        )
        return 2

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config = baseline_config()
    if args.dry_run:
        output_dir = args.output_dir or ROOT / "artifacts" / "architecture-smoke" / run_stamp
        manager = build_manager(
            output_dir, args.iterations, dry_run_templates(), DryRunBackend()
        )
        summary = manager.run(config, MetricBundle(gauc=0.6674, ndcg_at_5=0.5357))
        print("Architecture dry run only: no model was trained and no LLM was called.")
    else:
        output_dir = args.output_dir or ROOT / "artifacts" / "real-runs" / run_stamp
        backend = build_real_backend(args.data_dir)
        print("Reproducing the official FM baseline on validation only...", flush=True)
        reproduced = backend.run_config("baseline", config, output_dir / "baseline")
        if reproduced.status is not ExperimentStatus.SUCCESS or reproduced.metrics is None:
            print(reproduced.error or f"Baseline failed with status={reproduced.status.value}")
            return 1
        print(
            f"Baseline reproduced: GAUC {reproduced.metrics.gauc:.4f} | "
            f"nDCG@5 {reproduced.metrics.ndcg_at_5:.4f} | "
            f"Primary {reproduced.metrics.primary:.4f}",
            flush=True,
        )
        manager = build_manager(output_dir, args.iterations, real_run_templates(), backend)
        summary = manager.run(
            config,
            reproduced.metrics,
            baseline_result=reproduced,
            initial_elapsed_seconds=reproduced.runtime_seconds,
        )
        print("Real validation run: deterministic Planner, no LLM calls, no test-label feedback.")

    print_summary(manager, summary, args.real_run)
    print(f"Evidence: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
