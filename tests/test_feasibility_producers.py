"""Fixture tests for validation-only feasibility producers.

No KuaiRand load, no training, no API calls, no Markdown reads.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "analysis"))

from techjam_agent.config import FEATURE_KEYS
from techjam_agent.evidence import (
    attach_feasibility_evidence,
    build_feasibility_evidence,
    collect_artifact_evidence,
)
from techjam_agent.experiment_planner import (
    ActionType,
    admissible_candidates,
    rank_candidates,
)
from techjam_agent.feasibility_producers import (
    COVERAGE_FEATURE_KEYS,
    COVERAGE_SCHEMA_VERSION,
    COVERAGE_SUMMARY_PATH,
    LEAKAGE_REGISTRY_PATH,
    MARKDOWN_FORBIDDEN,
    coverage_manifest_sources,
    coverage_summary_from_signals,
    correlation_from_scores,
    correlation_pair,
    correlation_summary_from_pairs,
    format_missing_correlation_inputs,
    leakage_manifest_sources,
    leakage_rows_requiring_confirmation,
    build_leakage_registry,
    missing_correlation_inputs,
    write_versioned_json,
)

from test_action_filters import bpr_config, row_for


class ProducerSchemaTests(unittest.TestCase):
    def test_coverage_schema_for_computed_supported_features(self) -> None:
        signals = {
            "prior_video_positive": np.array([0, 1, 0, 1, 0], dtype=np.int32),
            "author_positive_recency": np.array([-1.0, -0.2, -1.0, 0.0, -1.0]),
            "prior_video_count": np.array([0, 1, 2, 0, 3], dtype=np.int32),
            "previous_author_same": np.array([0, 0, 1, 0, 0], dtype=np.int32),
            "prior_author_positive": np.array([1, 1, 1, 1, 1], dtype=np.int32),
        }
        first = coverage_summary_from_signals(signals)
        second = coverage_summary_from_signals(signals)
        self.assertEqual(first, second)
        self.assertEqual(first["version"], COVERAGE_SCHEMA_VERSION)
        self.assertIs(first["test_labels_used"], False)
        self.assertEqual(set(first["features"]), set(COVERAGE_FEATURE_KEYS))
        self.assertNotIn("prior_author_positive", first["features"])
        video = first["features"]["prior_video_positive"]
        self.assertEqual(video, {
            "feature": "prior_video_positive",
            "split": "validation",
            "coverage": 0.4,
            "eligible_rows": 2,
            "total_rows": 5,
        })
        recency = first["features"]["author_positive_recency"]
        self.assertEqual(recency["eligible_rows"], 2)
        self.assertEqual(recency["split"], "validation")
        counts = first["features"]["prior_video_count"]
        self.assertEqual(counts["eligible_rows"], 3)
        self.assertAlmostEqual(counts["coverage"], 0.6)

    def test_coverage_omits_supported_features_that_were_not_computed(self) -> None:
        summary = coverage_summary_from_signals({
            "prior_video_positive": np.array([1, 0, 0]),
        })
        self.assertEqual(list(summary["features"]), ["prior_video_positive"])

    def test_correlation_schema_from_existing_scores(self) -> None:
        left = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        right = np.array([0.11, 0.19, 0.31, 0.39], dtype=np.float64)
        pair = correlation_from_scores("fm", "deepfm", left, right)
        self.assertEqual(pair["models"], ["fm", "deepfm"])
        self.assertEqual(pair["split"], "validation")
        self.assertIs(pair["test_labels_used"], False)
        self.assertTrue(-1.0 <= pair["correlation"] <= 1.0)
        summary = correlation_summary_from_pairs([pair])
        self.assertEqual(summary["pairs"]["fm_deepfm"], pair)
        self.assertIs(summary["test_labels_used"], False)

    def test_leakage_registry_covers_every_feature_key(self) -> None:
        registry = build_leakage_registry()
        self.assertEqual(set(registry["features"]), set(FEATURE_KEYS))
        self.assertIs(registry["test_labels_used"], False)
        for name, row in registry["features"].items():
            self.assertEqual(row["feature"], name)
            self.assertIn(row["status"], {"safe", "unsafe", "uncertain"})
            self.assertIsInstance(row["leakage_safe"], bool)
            self.assertIsInstance(row["strict_past"], bool)
            self.assertTrue(row["implementation_source"])
            self.assertTrue(row["rationale"])
            if row["status"] == "safe":
                self.assertTrue(row["leakage_safe"])
                self.assertTrue(row["strict_past"])

    def test_producers_reject_test_labels_used(self) -> None:
        with self.assertRaisesRegex(ValueError, "test_labels_used=true"):
            coverage_summary_from_signals({"prior_video_positive": [1]}, test_labels_used=True)
        with self.assertRaisesRegex(ValueError, "test_labels_used=true"):
            correlation_pair("fm", "deepfm", 0.2, test_labels_used=True)
        with self.assertRaisesRegex(ValueError, "test_labels_used=true"):
            build_leakage_registry(test_labels_used=True)


class ManifestRoutingTests(unittest.TestCase):
    def _manifest(self, sources):
        return {
            "version": 1,
            "task": "long_view",
            "feature_schema": "v3",
            "feasibility_schema_version": 1,
            "sources": sources,
        }

    def test_validation_only_manifest_routing_and_exact_feature_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "TRY.md").write_text(
                "prior_video_positive coverage 0.0304\n", encoding="utf-8"
            )
            coverage = coverage_summary_from_signals({
                "prior_video_positive": np.array([1, 0, 0, 0]),
                "author_positive_recency": np.array([-1.0, 0.1, 0.2, -1.0]),
            })
            write_versioned_json(coverage, root / "coverage.json")
            write_versioned_json(
                build_leakage_registry(),
                root / "leakage.json",
            )
            sources = [
                {
                    "id": "coverage_prior_video_positive_v1",
                    "family": "candidate_history",
                    "kind": "feature_coverage",
                    "path": "coverage.json",
                    "pointer": ["features", "prior_video_positive"],
                    "validation_only": True,
                    "applies_to": {"features": {"prior_video_positive": True}},
                },
                {
                    "id": "coverage_author_positive_recency_v1",
                    "family": "candidate_history",
                    "kind": "feature_coverage",
                    "path": "coverage.json",
                    "pointer": ["features", "author_positive_recency"],
                    "validation_only": True,
                    "applies_to": {"features": {"author_positive_recency": True}},
                },
                {
                    "id": "leakage_prior_video_positive_v1",
                    "family": "candidate_history",
                    "kind": "leakage_status",
                    "path": "leakage.json",
                    "pointer": ["features", "prior_video_positive"],
                    "validation_only": True,
                    "applies_to": {"features": {"prior_video_positive": True}},
                },
            ]
            records = collect_artifact_evidence(root, self._manifest(sources))
        by_id = {row["source_id"]: row for row in records}
        video = by_id["coverage_prior_video_positive_v1"]
        recency = by_id["coverage_author_positive_recency_v1"]
        self.assertEqual(video["applies_to"]["features"], {"prior_video_positive": True})
        self.assertEqual(video["result"]["eligible_rows"], 1)
        self.assertEqual(video["result"]["total_rows"], 4)
        self.assertAlmostEqual(video["result"]["coverage"], 0.25)
        self.assertEqual(recency["result"]["eligible_rows"], 2)
        self.assertNotEqual(video["result"]["coverage"], recency["result"]["coverage"])
        self.assertEqual(
            by_id["leakage_prior_video_positive_v1"]["result"]["status"], "safe"
        )

    def test_loader_rejects_test_labels_used_on_generated_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = coverage_summary_from_signals({
                "prior_video_positive": np.array([1, 0]),
            })
            payload["test_labels_used"] = True
            (root / "coverage.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "used test labels"):
                collect_artifact_evidence(root, self._manifest([{
                    "id": "bad",
                    "family": "candidate_history",
                    "kind": "feature_coverage",
                    "path": "coverage.json",
                    "pointer": ["features", "prior_video_positive"],
                    "validation_only": True,
                    "applies_to": {"features": {"prior_video_positive": True}},
                }]))

    def test_markdown_is_never_read_by_producers_or_loader(self) -> None:
        opened: list[str] = []
        original = Path.read_text

        def guarded(self, *args, **kwargs):
            opened.append(Path(self).name)
            if Path(self).name in MARKDOWN_FORBIDDEN:
                raise AssertionError("producers must not read markdown research logs")
            return original(self, *args, **kwargs)

        signals = {"prior_video_positive": np.array([0, 1, 0])}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "TRY.md").write_text("coverage=0.0304\n", encoding="utf-8")
            with patch.object(Path, "read_text", guarded):
                summary = coverage_summary_from_signals(signals)
                write_versioned_json(summary, root / "coverage.json")
                collect_artifact_evidence(root, self._manifest([{
                    "id": "coverage_prior_video_positive_v1",
                    "family": "candidate_history",
                    "kind": "feature_coverage",
                    "path": "coverage.json",
                    "pointer": ["features", "prior_video_positive"],
                    "validation_only": True,
                    "applies_to": {"features": {"prior_video_positive": True}},
                }]))
        self.assertNotIn("TRY.md", opened)
        self.assertNotIn("AGENT-TRY.md", opened)

    def test_deterministic_json_bytes(self) -> None:
        signals = {
            "prior_video_positive": np.array([1, 0, 1]),
            "previous_author_same": np.array([0, 0, 1]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.json"
            second = Path(tmp) / "b.json"
            write_versioned_json(coverage_summary_from_signals(signals), first)
            write_versioned_json(coverage_summary_from_signals(signals), second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            write_versioned_json(build_leakage_registry(), first)
            write_versioned_json(build_leakage_registry(), second)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_production_manifest_routes_exact_features(self) -> None:
        manifest = json.loads(
            (ROOT / "configs" / "evidence_manifest.json").read_text(encoding="utf-8")
        )
        coverage = [
            source for source in manifest["sources"]
            if source["kind"] == "feature_coverage"
        ]
        leakage = [
            source for source in manifest["sources"]
            if source["kind"] == "leakage_status"
        ]
        self.assertEqual(coverage, coverage_manifest_sources())
        self.assertEqual(leakage, leakage_manifest_sources())
        for source in coverage + leakage:
            self.assertIs(source["validation_only"], True)
            features = source["applies_to"]["features"]
            self.assertEqual(len(features), 1)
            feature = next(iter(features))
            self.assertEqual(source["pointer"], ["features", feature])
        self.assertEqual(
            {next(iter(row["applies_to"]["features"])) for row in leakage},
            set(FEATURE_KEYS),
        )
        self.assertEqual(
            {next(iter(row["applies_to"]["features"])) for row in coverage},
            set(COVERAGE_FEATURE_KEYS),
        )
        self.assertTrue(all(row.get("optional") is True for row in coverage))
        self.assertEqual(
            {row["path"] for row in coverage}, {COVERAGE_SUMMARY_PATH}
        )
        self.assertEqual(
            {row["path"] for row in leakage}, {LEAKAGE_REGISTRY_PATH}
        )


class CorrelationInputTests(unittest.TestCase):
    def test_missing_correlation_inputs_are_reported_not_fabricated(self) -> None:
        from prediction_correlation import emit_from_existing_summary

        # Keep this contract test independent of optional local checkpoints
        # produced by earlier experiments.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = missing_correlation_inputs(root)
            self.assertTrue(missing["official_validation_checkpoints"])
            self.assertTrue(missing["existing_correlation_summaries"])
            report = format_missing_correlation_inputs(missing)
            self.assertIn("Person 1 must supply", report)
            for path in missing["official_validation_checkpoints"]:
                self.assertIn(path, report)
            output = root / "correlation.json"
            self.assertFalse(output.exists())
            existing = root / "summary.json"
            existing.write_text(json.dumps({
                "test_labels_used": False,
                "official_validation": {
                    "prediction_correlations": {"fm_deepfm": 0.42, "fm_dcnv2": 0.11},
                },
            }), encoding="utf-8")
            extracted = emit_from_existing_summary(existing)
        self.assertEqual(extracted["pairs"]["fm_deepfm"]["models"], ["fm", "deepfm"])
        self.assertEqual(extracted["pairs"]["fm_deepfm"]["split"], "validation")
        self.assertIs(extracted["test_labels_used"], False)


class PlannerConsumesFixturesTests(unittest.TestCase):
    def test_planner_consumes_generated_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_versioned_json(
                coverage_summary_from_signals({
                    "prior_video_positive": np.array([1] + [0] * 199),
                    "author_positive_recency": np.full(200, -1.0),
                    "previous_author_same": np.ones(200, dtype=np.int32),
                }),
                root / "coverage.json",
            )
            write_versioned_json(
                correlation_summary_from_pairs([
                    correlation_from_scores(
                        "fm",
                        "deepfm",
                        np.linspace(0.0, 1.0, 16),
                        np.linspace(0.0, 1.0, 16),
                    )
                ]),
                root / "correlation.json",
            )
            write_versioned_json(build_leakage_registry(), root / "leakage.json")
            sources = [
                {
                    "id": "coverage_prior_video_positive_v1",
                    "family": "candidate_history",
                    "kind": "feature_coverage",
                    "path": "coverage.json",
                    "pointer": ["features", "prior_video_positive"],
                    "validation_only": True,
                    "applies_to": {"features": {"prior_video_positive": True}},
                },
                {
                    "id": "coverage_author_positive_recency_v1",
                    "family": "candidate_history",
                    "kind": "feature_coverage",
                    "path": "coverage.json",
                    "pointer": ["features", "author_positive_recency"],
                    "validation_only": True,
                    "applies_to": {"features": {"author_positive_recency": True}},
                },
                {
                    "id": "coverage_previous_author_same_v1",
                    "family": "candidate_history",
                    "kind": "feature_coverage",
                    "path": "coverage.json",
                    "pointer": ["features", "previous_author_same"],
                    "validation_only": True,
                    "applies_to": {"features": {"previous_author_same": True}},
                },
                {
                    "id": "corr_fm_deepfm_v1",
                    "family": "heterogeneous_ensemble",
                    "kind": "prediction_correlation",
                    "path": "correlation.json",
                    "pointer": ["pairs", "fm_deepfm"],
                    "validation_only": True,
                    "applies_to": {"models": ["ensemble"]},
                },
                {
                    "id": "leakage_prior_video_positive_v1",
                    "family": "candidate_history",
                    "kind": "leakage_status",
                    "path": "leakage.json",
                    "pointer": ["features", "prior_video_positive"],
                    "validation_only": True,
                    "applies_to": {"features": {"prior_video_positive": True}},
                },
                {
                    "id": "leakage_author_positive_recency_v1",
                    "family": "candidate_history",
                    "kind": "leakage_status",
                    "path": "leakage.json",
                    "pointer": ["features", "author_positive_recency"],
                    "validation_only": True,
                    "applies_to": {"features": {"author_positive_recency": True}},
                },
                {
                    "id": "leakage_previous_author_same_v1",
                    "family": "candidate_history",
                    "kind": "leakage_status",
                    "path": "leakage.json",
                    "pointer": ["features", "previous_author_same"],
                    "validation_only": True,
                    "applies_to": {"features": {"previous_author_same": True}},
                },
                {
                    "id": "leakage_global_context_v1",
                    "family": "global_context",
                    "kind": "leakage_status",
                    "path": "leakage.json",
                    "pointer": ["features", "global_context"],
                    "validation_only": True,
                    "applies_to": {"features": {"global_context": True}},
                },
                {
                    "id": "leakage_user_long_view_rate_v1",
                    "family": "global_target_statistics",
                    "kind": "leakage_status",
                    "path": "leakage.json",
                    "pointer": ["features", "user_long_view_rate"],
                    "validation_only": True,
                    "applies_to": {"features": {"user_long_view_rate": True}},
                },
            ]
            feasibility = build_feasibility_evidence(root, {
                "version": 1,
                "task": "long_view",
                "feature_schema": "v3",
                "feasibility_schema_version": 1,
                "sources": sources,
            })
        evidence = attach_feasibility_evidence({}, feasibility)
        ranked = rank_candidates(bpr_config(), [], prior_evidence=evidence)
        video = row_for(ranked, prior_video_positive=True)
        recency = row_for(ranked, author_positive_recency=True)
        previous = row_for(ranked, previous_author_same=True)
        context = row_for(ranked, global_context=True)
        target = row_for(ranked, user_long_view_rate=True)
        self.assertFalse(video.hard_blocked)
        self.assertTrue(video.soft_stopped)
        self.assertIn(video, admissible_candidates(ranked, relax_soft=True))
        self.assertTrue(recency.hard_blocked)
        self.assertFalse(recency.soft_stopped)
        self.assertFalse(previous.hard_blocked)
        self.assertFalse(previous.soft_stopped)
        self.assertFalse(context.hard_blocked)
        self.assertFalse(context.soft_stopped)
        self.assertTrue(target.soft_stopped)
        self.assertIn("uncertain_leakage_evidence", target.evidence_reasons)
        ensemble = next(
            row for row in ranked if row.candidate.action_type == ActionType.TRY_ENSEMBLE
        )
        self.assertFalse(ensemble.hard_blocked)
        self.assertTrue(ensemble.soft_stopped)
        self.assertIn("high_prediction_correlation", ensemble.evidence_reasons)


class TrackedLeakageFileTests(unittest.TestCase):
    def test_committed_registry_matches_builder(self) -> None:
        path = ROOT / LEAKAGE_REGISTRY_PATH
        self.assertTrue(path.is_file())
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, build_leakage_registry())
        self.assertEqual(
            leakage_rows_requiring_confirmation(),
            (
                "user_long_view_rate",
                "item_long_view_rate",
                "continuous_history_stats",
                "user_tab_long_view_rate",
            ),
        )


if __name__ == "__main__":
    unittest.main()
