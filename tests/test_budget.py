"""Stopping and budget behavior. Fake runner, fake researcher, controlled clock only."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import ALLOWED_VALUES, apply_changes
from techjam_agent.controller import Controller
from techjam_agent.proposals import DeterministicResearcher, Proposal, empty_token_usage
from techjam_agent.tree import ExperimentParent, branch_name, select_parent

STOP_REASONS = {
    "max_iterations",
    "converged",
    "wall_clock_exhausted",
    "insufficient_time_for_next_experiment",
    "search_exhausted",
    "duplicate_configuration",
    "baseline_failed",
}
ACCOUNTING_FIELDS = (
    "stop_reason", "total_experiments", "candidate_experiments", "elapsed_seconds",
    "remaining_seconds", "convergence_streak", "limits",
)


def load_config() -> dict:
    return json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))


def load_project() -> dict:
    return json.loads((ROOT / "configs" / "project.json").read_text(encoding="utf-8"))


class FakeClock:
    """Monotonic, test-controlled. Advances only when the harness says so."""

    def __init__(self, start: float = 0.0, step: float = 0.0) -> None:
        self.now = start
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ScriptedRunner:
    """Returns queued validation metrics; never trains and never sleeps."""

    def __init__(self, primaries, clock: FakeClock | None = None, cost: float = 0.0) -> None:
        self.primaries = list(primaries)
        self.clock = clock
        self.cost = cost
        self.configs: list[dict] = []

    @property
    def calls(self) -> int:
        return len(self.configs)

    def run(self, config, checkpoint):
        self.configs.append(copy.deepcopy(config))
        if self.clock is not None and self.cost:
            self.clock.advance(self.cost)
        value = self.primaries.pop(0) if self.primaries else 0.6015
        if isinstance(value, Exception):
            raise value
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        if value is None:
            return {"GAUC": 0.66, "nDCG@5": 0.53}
        return {"GAUC": value, "nDCG@5": value, "primary": value,
                "test_GAUC": 0.9999, "test_primary": 0.9999}

    def finalize(self, config, checkpoint, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("row_id,user_id,video_id,score\n", encoding="utf-8")
        return {"GAUC": 0.62, "nDCG@5": 0.58, "primary": 0.5953}


class SweepResearcher:
    """Walks the allowed learning-rate/epoch grid so proposals never repeat."""

    def __init__(self) -> None:
        self.grid = [{"learning_rate": value} for value in ALLOWED_VALUES["learning_rate"]]
        self.grid += [{"epochs": value} for value in ALLOWED_VALUES["epochs"]]
        self.index = 0

    def propose(self, best, history):
        while self.index < len(self.grid):
            changes = self.grid[self.index]
            self.index += 1
            try:
                apply_changes(best, changes)
            except ValueError:
                continue
            return Proposal(f"Sweep {changes}.", "Controlled single-field sweep.",
                            changes, "fake", empty_token_usage())
        raise StopIteration("fake sweep exhausted")


class GlobalBestParentController(Controller):
    """Pins expansion to the global-best node so convergence accounting is what is measured."""

    def _select_parent(self):
        return select_parent(self.history)


class WeakParentController(Controller):
    """Always expands the baseline, which stops being the global best after the first KEEP."""

    def _select_parent(self):
        for row in self.history:
            if row["iteration"] == 0 and row.get("metrics"):
                return ExperimentParent("baseline", 0, row["config"],
                                        row["metrics"]["primary"], branch_name(row["changes"]))
        return None


def build(runner, researcher, tmp: Path, clock=None, project=None, controller_cls=Controller):
    return controller_cls(runner, researcher, load_config(), project or load_project(),
                          tmp / "logs", tmp / "artifacts", tmp / "submissions",
                          **({} if clock is None else {"clock": clock}))


def run_controller(runner, researcher, max_iterations=None, clock=None, project=None,
                   controller_cls=Controller, research_after_convergence=False):
    with tempfile.TemporaryDirectory() as tmp:
        controller = build(runner, researcher, Path(tmp), clock, project, controller_cls)
        with patch("sys.stdout", new=io.StringIO()):
            summary = controller.run(
                max_iterations,
                research_after_convergence=research_after_convergence,
            )
        return controller, summary


class IterationCapTests(unittest.TestCase):
    def test_one_iteration_runs_baseline_only(self) -> None:
        runner = ScriptedRunner([0.6015, 0.7000])
        controller, summary = run_controller(runner, SweepResearcher(), max_iterations=1)
        self.assertEqual(runner.calls, 1)
        self.assertEqual(len(controller.history), 1)
        self.assertEqual(controller.history[0]["iteration"], 0)
        self.assertEqual(summary["total_experiments"], 1)
        self.assertEqual(summary["candidate_experiments"], 0)
        self.assertEqual(summary["stop_reason"], "max_iterations")

    def test_three_iterations_run_at_most_three_experiments(self) -> None:
        runner = ScriptedRunner([0.6015, 0.6100, 0.6200, 0.6300, 0.6400])
        controller, summary = run_controller(runner, SweepResearcher(), max_iterations=3)
        self.assertEqual(runner.calls, 3)
        self.assertEqual(len(controller.history), 3)
        self.assertEqual(summary["total_experiments"], 3)
        self.assertEqual(summary["candidate_experiments"], 2)
        self.assertEqual([row["iteration"] for row in controller.history], [0, 1, 2])

    def test_request_above_official_maximum_is_clamped_to_fifty(self) -> None:
        runner = ScriptedRunner([0.6015 + index * 0.01 for index in range(80)])
        controller, summary = run_controller(runner, SweepResearcher(), max_iterations=500)
        self.assertEqual(summary["limits"]["max_total_experiments"], 50)
        self.assertLessEqual(runner.calls, 50)
        self.assertLessEqual(len(controller.history), 50)

    def test_default_cap_never_exceeds_fifty_total_executions(self) -> None:
        project = load_project()
        project["run_limits"]["max_iterations"] = 4
        runner = ScriptedRunner([0.6015 + index * 0.01 for index in range(20)])
        controller, summary = run_controller(runner, SweepResearcher(), project=project)
        self.assertEqual(runner.calls, 4)
        self.assertEqual(summary["total_experiments"], 4)
        self.assertEqual(summary["candidate_experiments"], 3)


class ConvergenceTests(unittest.TestCase):
    def test_baseline_does_not_count_toward_the_streak(self) -> None:
        runner = ScriptedRunner([0.6015])
        controller, summary = run_controller(runner, SweepResearcher(), max_iterations=1)
        self.assertEqual(controller.convergence_streak, 0)
        self.assertEqual(summary["convergence_streak"], 0)

    def test_three_consecutive_small_deltas_stop_the_run(self) -> None:
        runner = ScriptedRunner([0.6015, 0.6016, 0.6016, 0.6016, 0.7000])
        controller, summary = run_controller(runner, SweepResearcher(), max_iterations=10,
                                             controller_cls=GlobalBestParentController)
        self.assertEqual(summary["stop_reason"], "converged")
        self.assertEqual(summary["convergence_streak"], 3)
        self.assertEqual(runner.calls, 4)
        self.assertEqual(summary["candidate_experiments"], 3)
        self.assertTrue(all(row["expanded_global_best"] for row in controller.history[1:]))

    def test_research_mode_records_competition_convergence_but_keeps_exploring(self) -> None:
        runner = ScriptedRunner([0.6015, 0.6016, 0.6016, 0.6016, 0.6017])
        controller, summary = run_controller(
            runner,
            SweepResearcher(),
            max_iterations=5,
            controller_cls=GlobalBestParentController,
            research_after_convergence=True,
        )
        self.assertEqual(runner.calls, 5)
        self.assertEqual(summary["stop_reason"], "max_iterations")
        self.assertTrue(summary["competition_converged"])
        self.assertEqual(summary["competition_converged_at"], 3)
        self.assertTrue(summary["research_after_convergence"])
        self.assertEqual(summary["competition_best_at_convergence"]["iteration"], 1)

    def test_convergence_reached_on_final_iteration_is_still_recorded(self) -> None:
        runner = ScriptedRunner([0.6015, 0.6016, 0.6016, 0.6016])
        _, summary = run_controller(
            runner,
            SweepResearcher(),
            max_iterations=4,
            controller_cls=GlobalBestParentController,
            research_after_convergence=True,
        )
        self.assertEqual(summary["stop_reason"], "max_iterations")
        self.assertTrue(summary["competition_converged"])
        self.assertEqual(summary["competition_converged_at"], 3)

    def test_summary_writes_separate_submission_candidate_pool(self) -> None:
        runner = ScriptedRunner([0.6015, 0.6018])
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            controller = build(runner, SweepResearcher(), base)
            with patch("sys.stdout", new=io.StringIO()):
                summary = controller.run(max_iterations=2)
            candidates = json.loads(
                Path(summary["submission_candidates"]).read_text(encoding="utf-8")
            )
        self.assertEqual(summary["submission_candidate_count"], 2)
        self.assertEqual(candidates[0]["submission_status"], "ALLOW")
        self.assertIn("scientific_status", candidates[0])
        self.assertIn("competition_status", candidates[0])

    def test_unscoped_rejected_experiment_is_not_a_submission_candidate(self) -> None:
        runner = ScriptedRunner([0.6015, 0.5900])
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            controller = build(runner, SweepResearcher(), base)
            with patch("sys.stdout", new=io.StringIO()):
                summary = controller.run(max_iterations=2)
            candidates = json.loads(
                Path(summary["submission_candidates"]).read_text(encoding="utf-8")
            )
        self.assertEqual(summary["submission_candidate_count"], 1)
        by_iteration = {row["iteration"]: row for row in candidates}
        self.assertEqual(by_iteration[0]["submission_status"], "ALLOW")
        self.assertEqual(by_iteration[1]["submission_status"], "EXCLUDE")

    def test_meaningful_delta_resets_the_streak(self) -> None:
        runner = ScriptedRunner([0.6015, 0.6016, 0.6016, 0.6500, 0.6501])
        controller, summary = run_controller(runner, SweepResearcher(), max_iterations=5,
                                             controller_cls=GlobalBestParentController)
        self.assertEqual(runner.calls, 5)
        self.assertEqual(summary["stop_reason"], "max_iterations")
        self.assertEqual(summary["convergence_streak"], 1)

    def test_epsilon_boundary_value_counts_toward_the_streak(self) -> None:
        runner = ScriptedRunner([0.6000, 0.6020, 0.6040, 0.6060, 0.7000])
        controller, summary = run_controller(runner, SweepResearcher(), max_iterations=10)
        self.assertEqual(summary["stop_reason"], "converged")
        self.assertEqual(summary["convergence_streak"], 3)

    def test_failures_are_not_convergence_evidence(self) -> None:
        runner = ScriptedRunner([
            0.6015,
            RuntimeError("experiment exceeded 900s timeout"),
            float("nan"),
            None,
            0.6016,
        ])
        controller, summary = run_controller(runner, SweepResearcher(), max_iterations=5)
        self.assertEqual(runner.calls, 5)
        self.assertEqual(summary["stop_reason"], "max_iterations")
        self.assertEqual(summary["convergence_streak"], 1)
        verdicts = [row["critique"]["verdict"] for row in controller.history[1:]]
        self.assertEqual(verdicts, ["failed", "failed", "failed", "noise"])

    def test_keep_stays_independent_from_critic_verdict(self) -> None:
        runner = ScriptedRunner([0.601470, 0.603396])
        controller, _ = run_controller(runner, SweepResearcher(), max_iterations=2)
        candidate = controller.history[1]
        self.assertEqual(candidate["decision"], "KEEP")
        self.assertEqual(candidate["critique"]["verdict"], "noise")
        self.assertFalse(candidate["critique"]["meaningful_improvement"])
        self.assertEqual(controller.convergence_streak, 1)

    def test_weak_parent_exploration_never_signals_convergence(self) -> None:
        """Deliberate exploration of a losing branch is not evidence the leader has stalled."""
        runner = ScriptedRunner([0.6015, 0.6300, 0.6100, 0.6101, 0.6102])
        controller, summary = run_controller(runner, SweepResearcher(), max_iterations=5,
                                             controller_cls=WeakParentController)
        self.assertEqual(runner.calls, 5)
        self.assertEqual(summary["stop_reason"], "max_iterations")
        self.assertEqual(summary["convergence_streak"], 0)
        explorations = controller.history[2:]
        self.assertEqual([row["parent_id"] for row in explorations], ["baseline"] * 3)
        self.assertFalse(any(row["expanded_global_best"] for row in explorations))
        self.assertTrue(all(row["delta_from_best"] < 0 for row in explorations))

    def test_global_best_parent_small_deltas_still_converge(self) -> None:
        runner = ScriptedRunner([0.6015, 0.6016, 0.6017, 0.6018, 0.7000])
        controller, summary = run_controller(runner, SweepResearcher(), max_iterations=10,
                                             controller_cls=GlobalBestParentController)
        self.assertEqual(summary["stop_reason"], "converged")
        self.assertEqual(summary["convergence_streak"], 3)
        self.assertEqual(runner.calls, 4)
        self.assertEqual([row["parent_id"] for row in controller.history],
                         [None, "baseline", "exp_001", "exp_002"])

    def test_meaningful_improvement_from_global_best_resets_the_streak(self) -> None:
        runner = ScriptedRunner([0.6015, 0.6016, 0.6017, 0.6500])
        controller, summary = run_controller(runner, SweepResearcher(), max_iterations=4,
                                             controller_cls=GlobalBestParentController)
        self.assertEqual(summary["stop_reason"], "max_iterations")
        self.assertEqual(summary["convergence_streak"], 0)
        self.assertTrue(controller.history[3]["expanded_global_best"])
        self.assertAlmostEqual(controller.history[3]["delta_from_best"], 0.0483, places=6)

    def test_weak_parent_breakthrough_resets_the_streak(self) -> None:
        """A new global best is progress even when an explored side branch produced it."""

        class PlannedParentController(Controller):
            """Expands the global best except at iteration 3, which explores the baseline."""

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.streaks: list[int] = []

            def _select_parent(self):
                if len(self.history) == 3:
                    row = self.history[0]
                    return ExperimentParent("baseline", 0, row["config"],
                                            row["metrics"]["primary"],
                                            branch_name(row["changes"]))
                return select_parent(self.history)

            def _record(self, item, parent_id):
                super()._record(item, parent_id)
                self.streaks.append(self.convergence_streak)

        runner = ScriptedRunner([0.6015, 0.6016, 0.6017, 0.6500, 0.6501])
        controller, summary = run_controller(runner, SweepResearcher(), max_iterations=5,
                                             controller_cls=PlannedParentController)
        breakthrough = controller.history[3]
        self.assertEqual(breakthrough["parent_id"], "baseline")
        self.assertEqual(breakthrough["global_best_node_id_before"], "exp_002")
        self.assertFalse(breakthrough["expanded_global_best"])
        self.assertEqual(breakthrough["decision"], "KEEP")
        self.assertEqual(controller.streaks, [0, 1, 2, 0, 1])
        self.assertEqual(summary["stop_reason"], "max_iterations")
        self.assertEqual(summary["convergence_streak"], 1)
        self.assertEqual(runner.calls, 5)

    def test_failed_or_non_finite_exploration_leaves_the_streak_untouched(self) -> None:
        runner = ScriptedRunner([
            0.6015,
            0.6016,
            RuntimeError("experiment exceeded 900s timeout"),
            float("nan"),
            None,
        ])
        controller, summary = run_controller(runner, SweepResearcher(), max_iterations=5,
                                             controller_cls=WeakParentController)
        self.assertEqual(runner.calls, 5)
        self.assertEqual(summary["stop_reason"], "max_iterations")
        self.assertEqual(summary["convergence_streak"], 1)
        self.assertEqual([row["critique"]["verdict"] for row in controller.history[2:]],
                         ["failed", "failed", "failed"])

    def test_test_metrics_never_affect_stopping(self) -> None:
        runner = ScriptedRunner([0.6015, 0.6016, 0.6016, 0.6016])
        controller, summary = run_controller(runner, SweepResearcher(), max_iterations=10)
        self.assertEqual(summary["stop_reason"], "converged")
        self.assertEqual(summary["best_primary"], 0.6016)
        for row in controller.history:
            self.assertEqual(row["metrics"]["test_primary"], 0.9999)
        self.assertNotIn("0.9999", json.dumps(summary))


class WallClockTests(unittest.TestCase):
    def test_budget_starts_before_baseline_execution(self) -> None:
        clock = FakeClock()
        project = load_project()
        project["experiment_timeout_seconds"] = 0
        project["run_limits"]["max_wall_clock_hours"] = 1.0
        runner = ScriptedRunner([0.6015, 0.7000], clock=clock, cost=3600.0)
        controller, summary = run_controller(runner, SweepResearcher(), max_iterations=5,
                                             clock=clock, project=project)
        self.assertEqual(runner.calls, 1)
        self.assertEqual(len(controller.history), 1)
        self.assertEqual(summary["stop_reason"], "wall_clock_exhausted")
        self.assertEqual(summary["remaining_seconds"], 0.0)

    def test_no_experiment_starts_after_budget_exhaustion(self) -> None:
        clock = FakeClock()
        project = load_project()
        project["experiment_timeout_seconds"] = 0
        project["run_limits"]["max_wall_clock_hours"] = 6
        runner = ScriptedRunner([0.6015, 0.6100, 0.6200], clock=clock, cost=10_000.0)
        controller, summary = run_controller(runner, SweepResearcher(), max_iterations=10,
                                             clock=clock, project=project)
        self.assertEqual(runner.calls, 3)
        self.assertEqual(summary["stop_reason"], "wall_clock_exhausted")
        self.assertEqual(summary["total_experiments"], 3)

    def test_insufficient_remaining_time_stops_before_the_runner(self) -> None:
        clock = FakeClock()
        project = load_project()
        runner = ScriptedRunner([0.6015, 0.6100], clock=clock, cost=21_000.0)
        controller, summary = run_controller(runner, SweepResearcher(), max_iterations=10,
                                             clock=clock, project=project)
        self.assertEqual(runner.calls, 1)
        self.assertEqual(summary["stop_reason"], "insufficient_time_for_next_experiment")
        self.assertGreater(summary["remaining_seconds"], 0.0)
        self.assertLess(summary["remaining_seconds"],
                        summary["limits"]["experiment_cost_seconds"])

    def test_reasoning_time_counts_against_the_global_budget(self) -> None:
        clock = FakeClock()
        project = load_project()
        project["experiment_timeout_seconds"] = 0
        project["run_limits"]["max_wall_clock_hours"] = 1.0

        class SlowThinkingResearcher(SweepResearcher):
            def propose(self, best, history):
                clock.advance(1800.0)
                return super().propose(best, history)

        runner = ScriptedRunner([0.6015, 0.6100, 0.6200], clock=clock, cost=0.0)
        controller, summary = run_controller(runner, SlowThinkingResearcher(), max_iterations=10,
                                             clock=clock, project=project)
        self.assertEqual(runner.calls, 3)
        self.assertEqual(summary["stop_reason"], "wall_clock_exhausted")
        self.assertEqual(summary["elapsed_seconds"], 3600.0)


class AccountingTests(unittest.TestCase):
    def test_summary_always_reports_stop_reason_and_accounting(self) -> None:
        runner = ScriptedRunner([0.6015, 0.6100])
        _, summary = run_controller(runner, SweepResearcher(), max_iterations=2)
        for field in ACCOUNTING_FIELDS:
            self.assertIn(field, summary)
        self.assertIn(summary["stop_reason"], STOP_REASONS)
        self.assertEqual(summary["limits"]["convergence_epsilon"], 0.002)
        self.assertEqual(summary["limits"]["convergence_rounds"], 3)
        self.assertEqual(summary["limits"]["official_max_iterations"], 50)
        self.assertEqual(summary["limits"]["experiment_cost_seconds"], 900.0)
        json.dumps(summary)

    def test_baseline_failure_reports_baseline_failed(self) -> None:
        runner = ScriptedRunner([RuntimeError("baseline crashed")])
        controller, summary = run_controller(runner, SweepResearcher(), max_iterations=5)
        self.assertEqual(summary["stop_reason"], "baseline_failed")
        self.assertEqual(summary["total_experiments"], 1)
        self.assertEqual(summary["candidate_experiments"], 0)
        self.assertIsNone(summary["best_primary"])
        self.assertIsNone(summary["final_test_metrics"])
        self.assertEqual(controller.convergence_streak, 0)

    def test_exhausted_search_space_reports_search_exhausted(self) -> None:
        class ExhaustedResearcher(DeterministicResearcher):
            def propose(self, best, history):
                raise StopIteration("the configured FM experiment space is exhausted")

        runner = ScriptedRunner([0.6015])
        _, summary = run_controller(runner, ExhaustedResearcher(), max_iterations=5)
        self.assertEqual(summary["stop_reason"], "search_exhausted")
        self.assertIn("exhausted", summary["stop_detail"])
        self.assertEqual(summary["total_experiments"], 1)

    def test_summary_is_written_to_disk_with_accounting(self) -> None:
        runner = ScriptedRunner([0.6015, 0.6100])
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            controller = build(runner, SweepResearcher(), base)
            with patch("sys.stdout", new=io.StringIO()):
                controller.run(max_iterations=2)
            stored = json.loads((base / "logs" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["total_experiments"], 2)
        self.assertEqual(stored["limits"]["max_total_experiments"], 2)
        self.assertIn("remaining_seconds", stored)

    def test_old_history_records_remain_readable(self) -> None:
        runner = ScriptedRunner([0.6015, 0.6100])
        controller, _ = run_controller(runner, SweepResearcher(), max_iterations=2)
        for row in controller.history:
            for key in ("iteration", "hypothesis", "changes", "config", "decision",
                        "status", "metrics", "critique", "delta_from_best"):
                self.assertIn(key, row)


if __name__ == "__main__":
    unittest.main()
