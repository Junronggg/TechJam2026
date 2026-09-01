from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.data_profile import (
    EVALUATION_LOG,
    TRAIN_LOG,
    USER_FILE,
    VIDEO_FILE,
    build_profile,
    load_planner_context,
    write_profile,
)
from techjam_agent.proposals import build_planner_prompt, data_profile_evidence_ids


LOG_FIELDS = [
    "user_id", "video_id", "date", "hourmin", "time_ms", "is_click", "is_like",
    "is_follow", "is_comment", "is_forward", "is_hate", "long_view", "play_time_ms",
    "duration_ms", "profile_stay_time", "comment_stay_time", "is_profile_enter",
    "is_rand", "tab",
]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def log_row(user: str, video: str, event_date: int, label: object, tab: str = "1") -> dict[str, object]:
    return {
        "user_id": user, "video_id": video, "date": event_date, "hourmin": 900,
        "time_ms": 1, "is_click": 0, "is_like": 0, "is_follow": 0,
        "is_comment": 0, "is_forward": 0, "is_hate": 0, "long_view": label,
        "play_time_ms": 0, "duration_ms": 20000, "profile_stay_time": 0,
        "comment_stay_time": 0, "is_profile_enter": 0, "is_rand": 0, "tab": tab,
    }


class DataProfileTests(unittest.TestCase):
    def fixture(self, root: Path) -> None:
        write_csv(
            root / USER_FILE,
            ["user_id", "user_active_degree", "is_lowactive_period", "is_live_streamer",
             "is_video_author", "follow_user_num_range", "fans_user_num_range",
             "friend_user_num_range", "register_days_range"],
            [
                {"user_id": "u1", "user_active_degree": "full_active",
                 "is_lowactive_period": 0, "is_live_streamer": 0, "is_video_author": 0,
                 "follow_user_num_range": "0", "fans_user_num_range": "0",
                 "friend_user_num_range": "0", "register_days_range": "30+"},
                {"user_id": "u2", "user_active_degree": "low_active",
                 "is_lowactive_period": 1, "is_live_streamer": 0, "is_video_author": 0,
                 "follow_user_num_range": "0", "fans_user_num_range": "0",
                 "friend_user_num_range": "0", "register_days_range": "30+"},
            ],
        )
        write_csv(
            root / VIDEO_FILE,
            ["video_id", "author_id", "video_type", "upload_dt", "upload_type",
             "visible_status", "video_duration", "server_width", "server_height",
             "music_id", "music_type", "tag"],
            [
                {"video_id": "v1", "author_id": "a1", "video_type": "NORMAL",
                 "upload_dt": "2022-04-10", "upload_type": "Web", "visible_status": 0,
                 "video_duration": 20000, "server_width": 720, "server_height": 1280,
                 "music_id": "m1", "music_type": "1", "tag": "t1"},
                {"video_id": "v2", "author_id": "a2", "video_type": "AD",
                 "upload_dt": "2022-04-11", "upload_type": "Web", "visible_status": 0,
                 "video_duration": 20000, "server_width": 720, "server_height": 1280,
                 "music_id": "m2", "music_type": "2", "tag": "t2"},
            ],
        )
        write_csv(
            root / TRAIN_LOG, LOG_FIELDS,
            [log_row("u1", "v1", 20220411, 1), log_row("u1", "v2", 20220412, 0)],
        )
        write_csv(
            root / EVALUATION_LOG, LOG_FIELDS,
            [
                log_row("u1", "v1", 20220422, 0),
                log_row("u2", "v2", 20220423, 1, "2"),
                # This deliberately invalid value proves the test-period label is never read.
                log_row("u2", "v2", 20220429, "DO_NOT_READ_TEST_LABEL"),
            ],
        )

    def test_profile_skips_test_period_before_reading_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            profile, planner = build_profile(root, top_k=3)
        self.assertEqual(profile["split_summary"]["train"]["rows"], 2)
        self.assertEqual(profile["split_summary"]["validation"]["rows"], 2)
        self.assertFalse(profile["research_boundary"]["test_labels_accessed"])
        self.assertEqual(
            profile["research_boundary"]["skipped_rows_by_date"]
            ["evaluation_source_outside_validation"]["20220429"],
            1,
        )
        self.assertEqual(planner["scope"],
                         "Train labels and validation labels/features only; no test labels.")

    def test_writes_readable_and_planner_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            self.fixture(data)
            profile, planner = build_profile(data)
            output = root / "output"
            write_profile(output, profile, planner)
            loaded = load_planner_context(output / "planner_context.json")
            markdown = (output / "profile.md").read_text(encoding="utf-8")
        self.assertIn("data_profile_summary", data_profile_evidence_ids(loaded))
        self.assertIn("profile_affinity_coverage", data_profile_evidence_ids(loaded))
        self.assertIn("Test-period labels were not accessed", markdown)

    def test_planner_prompt_receives_profile_as_grounded_evidence(self) -> None:
        config = json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))
        profile = {
            "evidence_id": "data_profile_summary",
            "key_findings": [{"evidence_id": "profile_user_tag", "observation": "measured"}],
        }
        prompt = build_planner_prompt(config, [], data_profile=profile)
        self.assertEqual(prompt["data_profile"], profile)
        self.assertEqual(
            data_profile_evidence_ids(profile), {"data_profile_summary", "profile_user_tag"}
        )

    def test_loader_rejects_test_metrics_even_with_valid_scope(self) -> None:
        required = {
            "schema_version": "1.0", "evidence_id": "data_profile_summary",
            "scope": "Train labels and validation labels/features only; no test labels.",
            "split_summary": {}, "validation_ranking_context": {}, "cold_start": {},
            "affinity_coverage": {}, "train_validation_drift": {}, "key_findings": [],
            "leakage_constraints": {}, "final_test_metrics": {"primary": 1.0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(required), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbidden"):
                load_planner_context(path)


if __name__ == "__main__":
    unittest.main()
