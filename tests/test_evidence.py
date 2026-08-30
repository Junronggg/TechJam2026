from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from techjam_agent.config import FEATURE_KEYS
from techjam_agent.evidence import (
    DEFAULT_FEASIBILITY_THRESHOLDS,
    FEASIBILITY_KINDS,
    FEASIBILITY_SCHEMA_VERSION,
    POLICY_KINDS,
    build_feasibility_evidence,
    build_generated_family_policies,
    collect_artifact_evidence,
    merge_generated_policies,
)


ROOT = Path(__file__).resolve().parents[1]


class GeneratedEvidenceTests(unittest.TestCase):
    def _manifest(self, sources):
        return {
            "version": 1,
            "task": "long_view",
            "feature_schema": "v3",
            "noise_threshold": 0.0002,
            "sources": sources,
        }

    def test_rolling_failure_generates_scoped_stop_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "rolling.json"
            artifact.write_text(json.dumps({
                "aggregate": {"mean_delta": -0.0003, "wins": 1, "folds": 3}
            }), encoding="utf-8")
            generated = build_generated_family_policies(root, self._manifest([{
                "id": "temporal_v1",
                "family": "temporal_counts",
                "kind": "rolling_aggregate",
                "path": "rolling.json",
                "pointer": ["aggregate"],
                "validation_only": True,
                "applies_to": {"models": ["ensemble"]},
            }]))
        policy = generated["family_policies"][0]
        self.assertEqual(policy["policy"], "stop_direction")
        self.assertEqual(policy["scientific_verdict"], "REJECTED")
        self.assertEqual(policy["competition_status"], "NOT_ELIGIBLE")
        self.assertEqual(policy["applies_to"]["models"], ["ensemble"])
        self.assertEqual(len(policy["created_from"][0]["sha256"]), 64)
        self.assertFalse(generated["test_metrics_included"])

    def test_positive_rolling_plus_uncertain_seeds_requires_more_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rolling.json").write_text(json.dumps({
                "aggregate": {"mean_delta": 0.0008, "wins": 3, "folds": 3}
            }), encoding="utf-8")
            (root / "seeds.json").write_text(json.dumps({
                "aggregate": {
                    "paired_mean_delta": 0.0003,
                    "wins": 3,
                    "seeds": 4,
                    "approx_95pct_interval": [-0.0004, 0.0010],
                }
            }), encoding="utf-8")
            common = {
                "family": "global_context",
                "validation_only": True,
                "applies_to": {"models": ["fm"]},
                "pointer": ["aggregate"],
            }
            generated = build_generated_family_policies(root, self._manifest([
                {**common, "id": "rolling", "kind": "rolling_aggregate",
                 "path": "rolling.json"},
                {**common, "id": "seeds", "kind": "paired_seed",
                 "path": "seeds.json"},
            ]))
        policy = generated["family_policies"][0]
        self.assertEqual(policy["policy"], "gather_evidence")
        self.assertEqual(policy["scientific_verdict"], "UNCERTAIN")
        self.assertEqual(policy["competition_status"], "ELIGIBLE")

    def test_test_label_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bad.json").write_text(json.dumps({
                "test_labels_used": True,
                "aggregate": {"mean_delta": 1.0, "wins": 3, "folds": 3},
            }), encoding="utf-8")
            manifest = self._manifest([{
                "id": "bad",
                "family": "temporal_counts",
                "kind": "rolling_aggregate",
                "path": "bad.json",
                "pointer": ["aggregate"],
                "validation_only": True,
            }])
            with self.assertRaisesRegex(ValueError, "used test labels"):
                collect_artifact_evidence(root, manifest)

    def test_generated_policy_overrides_manual_policy_for_same_family(self):
        base = {"family_policies": [
            {"family": "temporal_counts", "policy": "gather_evidence"},
            {"family": "ranking_objective", "policy": "exploit_with_confirmation"},
        ]}
        generated = {
            "version": 1,
            "source": "generated_from_validation_artifacts",
            "test_metrics_included": False,
            "task": "long_view",
            "feature_schema": "v3",
            "family_policies": [
                {"family": "temporal_counts", "policy": "stop_direction"}
            ],
        }
        merged = merge_generated_policies(base, generated)
        by_family = {row["family"]: row for row in merged["family_policies"]}
        self.assertEqual(by_family["temporal_counts"]["policy"], "stop_direction")
        self.assertIn("ranking_objective", by_family)

    def test_tracked_policy_snapshot_matches_current_artifacts(self):
        manifest = json.loads(
            (ROOT / "configs" / "evidence_manifest.json").read_text(encoding="utf-8")
        )
        expected = json.loads(
            (ROOT / "configs" / "generated_family_policies.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(build_generated_family_policies(ROOT, manifest), expected)
        policies = expected["family_policies"]
        multitask = next(row for row in policies if row["family"] == "multitask")
        self.assertEqual(multitask["policy"], "exploit_with_confirmation")
        global_context = [
            row for row in policies if row["family"] == "global_context"
        ]
        self.assertEqual(len(global_context), 2)
        temporal = next(row for row in policies if row["family"] == "temporal_counts")
        self.assertEqual(temporal["applies_to"]["features"], {
            "item_recent_3d_exposure": True,
            "user_recent_3d_activity": True,
        })
        pairwise = [
            row for row in policies if row["family"] == "pairwise_multitask"
        ]
        self.assertEqual(len(pairwise), 2)
        self.assertTrue(all(
            row["policy"] == "stop_direction" for row in pairwise
        ))

    def test_feasibility_sources_do_not_change_family_policy_snapshot(self) -> None:
        manifest = json.loads(
            (ROOT / "configs" / "evidence_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["feasibility_schema_version"], FEASIBILITY_SCHEMA_VERSION)
        self.assertEqual(
            manifest["feasibility_thresholds"]["low_coverage"],
            DEFAULT_FEASIBILITY_THRESHOLDS["low_coverage"],
        )
        expected = json.loads(
            (ROOT / "configs" / "generated_family_policies.json").read_text(
                encoding="utf-8"
            )
        )
        generated = build_generated_family_policies(ROOT, manifest)
        self.assertEqual(generated, expected)
        feasibility = build_feasibility_evidence(ROOT, manifest)
        kinds = {record["kind"] for record in feasibility["records"]}
        self.assertEqual(kinds, {"family_runtime", "leakage_status", "feature_coverage"})
        self.assertTrue(all(len(record["sha256"]) == 64 for record in feasibility["records"]))
        self.assertNotIn("prediction_correlation", kinds)
        leakage = [
            record for record in feasibility["records"]
            if record["kind"] == "leakage_status"
        ]
        self.assertEqual(
            {next(iter(record["applies_to"]["features"])) for record in leakage},
            set(FEATURE_KEYS),
        )
        coverage = [
            record for record in feasibility["records"]
            if record["kind"] == "feature_coverage"
        ]
        self.assertTrue(coverage)
        self.assertTrue(all(
            record["applies_to"]["features"] and record["kind"] in FEASIBILITY_KINDS
            for record in coverage
        ))
        policy_kinds = {
            source["kind"]
            for policy in generated["family_policies"]
            for source in policy["created_from"]
        }
        self.assertTrue(policy_kinds)
        self.assertTrue(policy_kinds <= POLICY_KINDS)
        self.assertNotIn("feature_coverage", policy_kinds)
        self.assertFalse(policy_kinds & FEASIBILITY_KINDS)

    def test_markdown_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "markdown"):
                collect_artifact_evidence(root, self._manifest([{
                    "id": "from_try",
                    "family": "candidate_history",
                    "kind": "feature_coverage",
                    "path": "TRY.md",
                    "pointer": ["coverage"],
                    "validation_only": True,
                    "applies_to": {"features": {"prior_video_positive": True}},
                }]))

    def test_test_labelled_coverage_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "coverage.json").write_text(json.dumps({
                "test_labels_used": True,
                "coverage": 0.003,
                "eligible_rows": 4,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "used test labels"):
                collect_artifact_evidence(root, self._manifest([{
                    "id": "bad_coverage",
                    "family": "candidate_history",
                    "kind": "feature_coverage",
                    "path": "coverage.json",
                    "pointer": [],
                    "validation_only": True,
                    "applies_to": {"features": {"prior_video_positive": True}},
                }]))

    def test_test_split_correlation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "corr.json").write_text(json.dumps({
                "correlation": 0.995,
                "models": ["fm", "deepfm"],
                "split": "test",
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation predictions"):
                collect_artifact_evidence(root, self._manifest([{
                    "id": "bad_corr",
                    "family": "heterogeneous_ensemble",
                    "kind": "prediction_correlation",
                    "path": "corr.json",
                    "pointer": [],
                    "validation_only": True,
                }]))

    def test_loader_reads_coverage_correlation_runtime_and_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "coverage.json").write_text(json.dumps({
                "test_labels_used": False,
                "coverage": 0.003,
                "eligible_rows": 4,
                "total_rows": 124909,
            }), encoding="utf-8")
            (root / "corr.json").write_text(json.dumps({
                "correlation": 0.995,
                "models": ["fm", "deepfm"],
                "split": "validation",
            }), encoding="utf-8")
            (root / "runtime.json").write_text(json.dumps({
                "test_labels_used": False,
                "folds": {
                    "fold_1": {"seq": {"runtime_seconds": 800.0}},
                    "fold_2": {"seq": {"runtime_seconds": 820.0}},
                },
            }), encoding="utf-8")
            (root / "leakage.json").write_text(json.dumps({
                "status": "unsafe",
                "leakage_safe": False,
                "strict_past": False,
            }), encoding="utf-8")
            common = {"validation_only": True}
            feasibility = build_feasibility_evidence(root, self._manifest([
                {**common, "id": "cov", "family": "candidate_history",
                 "kind": "feature_coverage", "path": "coverage.json",
                 "pointer": [], "applies_to": {"features": {"prior_video_positive": True}}},
                {**common, "id": "corr", "family": "heterogeneous_ensemble",
                 "kind": "prediction_correlation", "path": "corr.json", "pointer": []},
                {**common, "id": "rt", "family": "sequence_model",
                 "kind": "family_runtime", "path": "runtime.json", "pointer": ["folds"],
                 "runtime_model": "seq", "applies_to": {"models": ["sequence_deepfm"]}},
                {**common, "id": "leak", "family": "explicit_crosses",
                 "kind": "leakage_status", "path": "leakage.json", "pointer": [],
                 "applies_to": {"features": {"user_tab_cross": True}}},
            ]))
        by_kind = {row["kind"]: row for row in feasibility["records"]}
        self.assertEqual(by_kind["feature_coverage"]["result"]["coverage"], 0.003)
        self.assertEqual(by_kind["feature_coverage"]["result"]["eligible_rows"], 4)
        self.assertEqual(by_kind["prediction_correlation"]["result"]["models"],
                         ["deepfm", "fm"])
        self.assertEqual(by_kind["family_runtime"]["result"]["median_runtime_seconds"], 810.0)
        self.assertEqual(by_kind["leakage_status"]["result"]["status"], "unsafe")
        self.assertEqual(feasibility["version"], FEASIBILITY_SCHEMA_VERSION)
        self.assertFalse(feasibility["test_metrics_included"])


if __name__ == "__main__":
    unittest.main()
