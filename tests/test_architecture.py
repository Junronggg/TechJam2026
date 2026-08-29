from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agent.critic import GroundedCritic
from agent.manager import ResearchManager
from agent.memory import ResearchMemory
from agent.planner import DeterministicPlanner, ExperimentTemplate
from agent.tree import TreePolicyConfig, TreeSearchPolicy
from experiment.logger import ExperimentLogger
from experiment.executor import SubprocessExecutor
from experiment.runner import DryRunBackend, ExperimentRunner
from experiment.schemas import (
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
    MetricBundle,
    ModelConfig,
    Operation,
    RunBudget,
)
from experiment.validator import (
    ExperimentValidationError,
    ExperimentValidator,
    ValidationPolicy,
)
from recommender.config import apply_experiment
from recommender.feature_encoding import FeatureEncoder
from recommender.objectives import build_within_user_pairs


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "test-temp"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def baseline_config() -> ModelConfig:
    return ModelConfig(
        model="fm",
        features=("user_id", "video_id", "author_id", "tab", "dur_bucket"),
        hyperparameters={"k": 16, "lr": 0.001, "epochs": 40, "l2": 1e-6},
        seed=0,
    )


class SchemaAndValidationTests(unittest.TestCase):
    def test_primary_is_metric_mean(self) -> None:
        metrics = MetricBundle(0.6674, 0.5357)
        self.assertAlmostEqual(metrics.primary, 0.60155)

    def test_operator_does_not_mutate_parent(self) -> None:
        parent = baseline_config()
        spec = ExperimentSpec(
            "exp_001",
            "baseline",
            "capacity",
            "Increase capacity",
            Operation.CHANGE_HYPERPARAMETER,
            {"name": "k", "value": 32},
        )
        child = apply_experiment(parent, spec)
        self.assertEqual(parent.hyperparameters["k"], 16)
        self.assertEqual(child.hyperparameters["k"], 32)
        self.assertNotEqual(parent.signature(), child.signature())

    def test_change_objective_is_explicit_and_immutable(self) -> None:
        parent = baseline_config()
        spec = ExperimentSpec(
            "exp_bpr",
            "baseline",
            "ranking_objective",
            "Use BPR",
            Operation.CHANGE_OBJECTIVE,
            {"objective": "bpr"},
        )
        child = apply_experiment(parent, spec)
        self.assertEqual(parent.objective, "pointwise")
        self.assertEqual(child.objective, "bpr")

    def test_historical_validation_encoding_does_not_depend_on_validation_labels(self) -> None:
        train = [
            (20220408, "u1", "v1", "a1", "1", 10.0, 1, "t1", 1200),
            (20220408, "u1", "v2", "a2", "1", 20.0, 0, "t2", 1201),
            (20220408, "u2", "v1", "a1", "1", 10.0, 0, "t1", 1202),
        ]
        valid_zero = [(20220422, "u1", "v1", "a1", "1", 10.0, 0, "t1", 1200)]
        valid_one = [(20220422, "u1", "v1", "a1", "1", 10.0, 1, "t1", 1200)]
        features = ("user_id", "video_id", "item_long_view_rate", "user_tag_affinity")
        encoded_zero, _ = FeatureEncoder(features, buckets=3).fit_transform(
            {"train": train, "valid": valid_zero}
        )
        encoded_one, _ = FeatureEncoder(features, buckets=3).fit_transform(
            {"train": train, "valid": valid_one}
        )
        self.assertTrue(
            (encoded_zero["valid"][0] == encoded_one["valid"][0]).all()
        )

    def test_pair_sampling_stays_within_user(self) -> None:
        users = ["u1", "u1", "u2", "u2", "u3"]
        labels = np.asarray([1, 0, 0, 1, 1], dtype=float)
        positives, negatives = build_within_user_pairs(users, labels, seed=0)
        self.assertTrue(all(users[p] == users[n] for p, n in zip(positives, negatives)))
        self.assertTrue(all(labels[p] == 1 and labels[n] == 0 for p, n in zip(positives, negatives)))

    def test_validator_rejects_non_finite_value(self) -> None:
        validator = ExperimentValidator()
        config = ModelConfig("fm", baseline_config().features, {"lr": math.nan})
        with self.assertRaises(ExperimentValidationError):
            validator.validate_config(config)

    def test_validator_protects_official_evaluator(self) -> None:
        policy = ValidationPolicy(
            allowed_operations=frozenset({Operation.NOVEL_PATCH}),
            allowed_features=frozenset(),
            allowed_models=frozenset(),
        )
        validator = ExperimentValidator(policy)
        spec = ExperimentSpec(
            "exp_bad",
            "baseline",
            "patch",
            "Modify evaluator",
            Operation.NOVEL_PATCH,
            {"target_path": "kuairand-starter-kit/evaluate.py"},
        )
        with self.assertRaises(ExperimentValidationError):
            validator.validate_spec(spec)

    def test_executor_returns_timeout_without_stalling(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as raw_dir:
            root = Path(raw_dir)
            result = SubprocessExecutor().run(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                cwd=root,
                run_dir=root / "run",
                timeout_seconds=0.05,
            )
            self.assertTrue(result.timed_out)
            self.assertIsNone(result.return_code)
            self.assertTrue(result.stdout_path.is_file())


class TreeAndManagerTests(unittest.TestCase):
    def test_rejected_node_is_not_expandable(self) -> None:
        memory = ResearchMemory()
        base_metrics = MetricBundle(0.66, 0.54)
        memory.add_root(
            baseline_config(),
            ExperimentResult("baseline", ExperimentStatus.SUCCESS, base_metrics),
        )
        spec = ExperimentSpec(
            "exp_bad",
            "baseline",
            "bad_branch",
            "Bad change",
            Operation.CHANGE_HYPERPARAMETER,
            {"name": "k", "value": 32},
        )
        config = apply_experiment(baseline_config(), spec)
        result = ExperimentResult(
            "exp_bad", ExperimentStatus.SUCCESS, MetricBundle(0.64, 0.52)
        )
        critic = GroundedCritic().review(base_metrics, result, spec)
        memory.add_child("baseline", config, spec, result, critic)
        frontier = TreeSearchPolicy().frontier(memory)
        self.assertEqual([node.node_id for node in frontier], ["baseline"])

    def test_tree_keeps_multiple_branch_frontier(self) -> None:
        memory = ResearchMemory()
        base_metrics = MetricBundle(0.66, 0.54)
        memory.add_root(
            baseline_config(),
            ExperimentResult("baseline", ExperimentStatus.SUCCESS, base_metrics),
        )
        for number, branch, delta in ((1, "features", 0.01), (2, "models", 0.008)):
            spec = ExperimentSpec(
                f"exp_{number:03d}",
                "baseline",
                branch,
                branch,
                Operation.CHANGE_HYPERPARAMETER,
                {"name": "k", "value": 16 + number},
            )
            config = apply_experiment(baseline_config(), spec)
            result = ExperimentResult(
                spec.experiment_id,
                ExperimentStatus.SUCCESS,
                MetricBundle(base_metrics.gauc + delta, base_metrics.ndcg_at_5 + delta),
            )
            critic = GroundedCritic().review(base_metrics, result, spec)
            memory.add_child("baseline", config, spec, result, critic)
        frontier = TreeSearchPolicy(TreePolicyConfig(max_active_branches=3)).frontier(memory)
        self.assertEqual({node.branch for node in frontier}, {"baseline", "features", "models"})

    def test_dry_run_exercises_full_loop_and_writes_evidence(self) -> None:
        templates = (
            ExperimentTemplate(
                "capacity",
                "Increase capacity",
                Operation.CHANGE_HYPERPARAMETER,
                {"name": "k", "value": 32},
                {"gauc": "increase"},
            ),
            ExperimentTemplate(
                "optimization",
                "Tune learning rate",
                Operation.CHANGE_HYPERPARAMETER,
                {"name": "lr", "value": 0.003},
                {"primary": "increase"},
            ),
            ExperimentTemplate(
                "feature_ablation",
                "Test duration signal",
                Operation.REMOVE_FEATURE,
                {"feature": "dur_bucket"},
                {"primary": "uncertain"},
            ),
        )
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as raw_dir:
            root = Path(raw_dir)
            logger = ExperimentLogger(root / "experiment_history.jsonl")
            manager = ResearchManager(
                planner=DeterministicPlanner(templates),
                critic=GroundedCritic(),
                runner=ExperimentRunner(root / "runs", DryRunBackend()),
                logger=logger,
                budget=RunBudget(max_iterations=3),
            )
            summary = manager.run(baseline_config(), MetricBundle(0.6674, 0.5357))
            self.assertEqual(summary.completed_iterations, 3)
            self.assertEqual(len(manager.memory.nodes), 4)
            self.assertTrue((root / "final_summary.json").is_file())
            self.assertTrue((root / "tree_snapshot.json").is_file())
            records = [json.loads(line) for line in logger.log_path.read_text().splitlines()]
            self.assertEqual(len(records), 4)
            self.assertEqual(records[0]["event"], "baseline")


if __name__ == "__main__":
    unittest.main()
