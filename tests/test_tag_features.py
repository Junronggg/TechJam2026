"""Leakage and encoding tests for raw tag and user-tag feature engineering."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import apply_changes, normalize_config, validate_config
from techjam_agent.history_features import TrainHistoryStatistics
from techjam_agent.runner import ExperimentRunner


def load_config() -> dict:
    return json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))


def row(timestamp: int, user: str, video: str, tag: str, label: int) -> tuple:
    return (20220408, user, video, "author", "tab", 10.0, label, tag, timestamp)


class UserTagLeakageTests(unittest.TestCase):
    def test_new_feature_flags_are_legal(self) -> None:
        config = apply_changes(load_config(), {
            "tag": True,
            "user_tag_impression_count": True,
            "user_tag_long_view_rate": True,
        })
        validate_config(config)
        self.assertTrue(config["features"]["tag"])

    def test_archived_configs_get_false_defaults(self) -> None:
        old = load_config()
        for feature in ("tag", "user_tag_impression_count", "user_tag_long_view_rate"):
            old["features"].pop(feature)
        normalized = normalize_config(old)
        validate_config(normalized)
        self.assertFalse(normalized["features"]["tag"])

    def test_train_encoding_uses_only_strictly_earlier_events(self) -> None:
        rows = [
            row(100, "u", "v1", "sports", 1),
            row(200, "u", "v2", "sports", 0),
            row(300, "u", "v3", "sports", 1),
        ]
        statistics = TrainHistoryStatistics.build(rows, [
            "user_tag_impression_count", "user_tag_long_view_rate",
        ])
        counts = statistics.chronological_user_tag_values(
            "user_tag_impression_count", rows, categorical=False
        )
        rates = statistics.chronological_user_tag_values(
            "user_tag_long_view_rate", rows, categorical=False
        )
        self.assertEqual(counts, [0.0, np.log1p(1.0), np.log1p(2.0)])
        self.assertAlmostEqual(rates[0], 0.5)
        self.assertAlmostEqual(rates[1], 1.0)

        changed_future = [*rows[:2], row(300, "u", "v3", "sports", 0)]
        changed = statistics.chronological_user_tag_values(
            "user_tag_long_view_rate", changed_future, categorical=False
        )
        self.assertEqual(rates[:2], changed[:2])

    def test_equal_timestamp_rows_do_not_leak_between_each_other(self) -> None:
        rows = [
            row(100, "u", "v1", "sports", 1),
            row(100, "u", "v2", "sports", 0),
        ]
        statistics = TrainHistoryStatistics.build(rows, ["user_tag_long_view_rate"])
        values = statistics.chronological_user_tag_values(
            "user_tag_long_view_rate", rows, categorical=False
        )
        self.assertEqual(values, [0.5, 0.5])

    def test_validation_feature_never_reads_validation_label(self) -> None:
        train = [
            row(100, "u", "v1", "sports", 1),
            row(200, "u", "v2", "sports", 0),
        ]
        zero = row(300, "u", "v3", "sports", 0)
        one = row(300, "u", "v3", "sports", 1)
        statistics = TrainHistoryStatistics.build(train, ["user_tag_long_view_rate"])
        self.assertEqual(
            statistics.numeric_value("user_tag_long_view_rate", zero, leave_one_out=False),
            statistics.numeric_value("user_tag_long_view_rate", one, leave_one_out=False),
        )


class RawTagEncodingTests(unittest.TestCase):
    def test_tag_vocabulary_is_fitted_on_train_with_unknown_bucket(self) -> None:
        runner = ExperimentRunner.__new__(ExperimentRunner)
        runner._splits = {
            "train": [row(100, "u1", "v1", "sports", 1),
                      row(200, "u2", "v2", "music", 0)],
            "valid": [row(300, "u3", "v3", "unseen", 0)],
            "test": [row(400, "u4", "v4", "sports", 1)],
        }
        runner._categorical_cache = {}
        encoded, width = runner._raw_categorical("tag")
        self.assertEqual(width, 3)
        self.assertEqual(encoded["train"].tolist(), [0, 1])
        self.assertEqual(encoded["valid"].tolist(), [2])
        self.assertEqual(encoded["test"].tolist(), [0])

    def test_lightgbm_marks_raw_tag_as_categorical(self) -> None:
        tagged = apply_changes(load_config(), {"model": "lightgbm", "tag": True})
        self.assertEqual(
            ExperimentRunner._lightgbm_categorical_columns(tagged),
            [0, 1, 2, 3, 4, 5],
        )


if __name__ == "__main__":
    unittest.main()
