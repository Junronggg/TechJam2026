from __future__ import annotations

import csv
import hashlib
import json
import math
from array import array
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


TRAIN_RANGE = (20220408, 20220421)
VALID_RANGE = (20220422, 20220428)
TRAIN_LOG = "log_standard_4_08_to_4_21_pure.csv"
EVALUATION_LOG = "log_standard_4_22_to_5_08_pure.csv"
USER_FILE = "user_features_pure.csv"
VIDEO_FILE = "video_features_basic_pure.csv"
VIDEO_STATISTICS_FILE = "video_features_statistic_pure.csv"
RANDOM_LOG = "log_random_4_22_to_5_08_pure.csv"

OUTCOME_COLUMNS = (
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "play_time_ms", "profile_stay_time", "comment_stay_time",
    "is_profile_enter",
)
PAIR_NAMES = ("user_author", "user_tag", "user_music", "user_tab")
LOW_CARDINALITY_GROUPS = {
    "date", "hour", "weekday", "tab", "user_active_degree", "is_lowactive_period",
    "register_days_range", "video_type", "upload_type", "music_type", "duration_band",
    "upload_age_band",
}
MASK64 = (1 << 64) - 1


def _clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "__MISSING__" if not text or text.lower() in {"nan", "null", "none"} else text


def _number_id(value: Any) -> str:
    """Canonicalize CSV IDs that are sometimes written as integer-valued floats."""
    text = _clean(value)
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if math.isfinite(number) and number.is_integer() else text


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _date_from_int(value: int) -> date:
    return datetime.strptime(str(value), "%Y%m%d").date()


def _duration_band(duration_ms: float | None) -> str:
    if duration_ms is None:
        return "missing"
    seconds = duration_ms / 1000.0
    for edge, label in ((10, "<10s"), (20, "10-20s"), (30, "20-30s"),
                        (60, "30-60s"), (120, "60-120s")):
        if seconds < edge:
            return label
    return "120s+"


def _upload_age_band(days: int | None) -> str:
    if days is None:
        return "missing"
    if days < 0:
        return "future_upload"
    for edge, label in ((3, "0-2d"), (8, "3-7d"), (15, "8-14d"), (31, "15-30d")):
        if days < edge:
            return label
    return "31d+"


@lru_cache(maxsize=None)
def _token64(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "little")


def _pair_code(left: str, right: str) -> int:
    # A stable 64-bit hash keeps million-row pair coverage compact. Collision risk
    # is negligible for profiling, but the report labels the result as approximate.
    x, y = _token64(left), _token64(right)
    return (x ^ (y + 0x9E3779B97F4A7C15 + ((x << 6) & MASK64) + (x >> 2))) & MASK64


def _quantiles(values: array) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p25": None, "p50": None, "p75": None, "p90": None,
                "p99": None, "max": None, "mean": None}
    data = np.frombuffer(values, dtype=np.float64)
    result = np.quantile(data, [0, .25, .5, .75, .9, .99, 1])
    return {
        "min": float(result[0]), "p25": float(result[1]), "p50": float(result[2]),
        "p75": float(result[3]), "p90": float(result[4]), "p99": float(result[5]),
        "max": float(result[6]), "mean": float(data.mean()),
    }


def _integer_quantiles(values: list[int]) -> dict[str, float | None]:
    packed = array("d", (float(value) for value in values))
    return _quantiles(packed)


class SplitAccumulator:
    def __init__(self, name: str, train_entities: dict[str, set[str]] | None = None) -> None:
        self.name = name
        self.rows = 0
        self.positives = 0
        self.groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        self.entities = {name: set() for name in ("user", "video", "author", "tag", "music")}
        self.users: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.durations = array("d")
        self.upload_ages = array("d")
        self.pairs = {name: array("Q") for name in PAIR_NAMES}
        self.missing_metadata = Counter()
        self.train_entities = train_entities
        self.unseen_rows = Counter()
        self.unseen_unique = {name: set() for name in self.entities}

    def add_group(self, name: str, value: Any, label: int) -> None:
        slot = self.groups[name][_clean(value)]
        slot[0] += 1
        slot[1] += label

    def add(
        self,
        row: dict[str, str],
        event_date: int,
        label: int,
        user_meta: dict[str, dict[str, str]],
        video_meta: dict[str, dict[str, str]],
    ) -> None:
        user, video = _clean(row.get("user_id")), _clean(row.get("video_id"))
        user_info, video_info = user_meta.get(user), video_meta.get(video)
        if user_info is None:
            self.missing_metadata["user"] += 1
            user_info = {}
        if video_info is None:
            self.missing_metadata["video"] += 1
            video_info = {}

        author = _clean(video_info.get("author_id"))
        tag = _clean(video_info.get("tag"))
        music = _number_id(video_info.get("music_id"))
        tab = _clean(row.get("tab"))
        entity_values = {
            "user": user, "video": video, "author": author, "tag": tag, "music": music,
        }
        self.rows += 1
        self.positives += label
        self.users[user][0] += 1
        self.users[user][1] += label
        for name, value in entity_values.items():
            self.entities[name].add(value)
            if self.train_entities is not None and value not in self.train_entities[name]:
                self.unseen_rows[name] += 1
                self.unseen_unique[name].add(value)

        hourmin = int(_float(row.get("hourmin")) or 0)
        event_day = _date_from_int(event_date)
        duration = _float(row.get("duration_ms"))
        if duration is not None:
            self.durations.append(duration)
        upload_age = None
        try:
            upload_date = datetime.strptime(_clean(video_info.get("upload_dt")), "%Y-%m-%d").date()
            upload_age = (event_day - upload_date).days
            self.upload_ages.append(float(upload_age))
        except ValueError:
            self.missing_metadata["upload_dt"] += 1

        group_values = {
            "date": str(event_date), "hour": f"{hourmin // 100:02d}",
            "weekday": event_day.strftime("%a"), "tab": tab,
            "user_active_degree": user_info.get("user_active_degree"),
            "is_lowactive_period": user_info.get("is_lowactive_period"),
            "register_days_range": user_info.get("register_days_range"),
            "video_type": video_info.get("video_type"),
            "upload_type": video_info.get("upload_type"),
            "music_type": video_info.get("music_type"), "tag": tag,
            "duration_band": _duration_band(duration),
            "upload_age_band": _upload_age_band(upload_age),
        }
        for name, value in group_values.items():
            self.add_group(name, value, label)

        for name, value in {
            "user_author": author, "user_tag": tag, "user_music": music, "user_tab": tab,
        }.items():
            self.pairs[name].append(_pair_code(user, value))


def _load_metadata(data_dir: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    user_columns = (
        "user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author",
        "follow_user_num_range", "fans_user_num_range", "friend_user_num_range",
        "register_days_range",
    )
    video_columns = (
        "author_id", "video_type", "upload_dt", "upload_type", "visible_status",
        "video_duration", "server_width", "server_height", "music_id", "music_type", "tag",
    )
    users: dict[str, dict[str, str]] = {}
    with (data_dir / USER_FILE).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            users[_clean(row.get("user_id"))] = {key: _clean(row.get(key)) for key in user_columns}
    videos: dict[str, dict[str, str]] = {}
    with (data_dir / VIDEO_FILE).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            videos[_clean(row.get("video_id"))] = {key: _clean(row.get(key)) for key in video_columns}
    return users, videos


def _consume_log(
    path: Path,
    split_range: tuple[int, int],
    accumulator: SplitAccumulator,
    users: dict[str, dict[str, str]],
    videos: dict[str, dict[str, str]],
) -> dict[str, int]:
    skipped = Counter()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"user_id", "video_id", "date", "hourmin", "duration_ms", "tab", "long_view"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        for row in reader:
            event_date = int(row["date"])
            if not split_range[0] <= event_date <= split_range[1]:
                # Most importantly, do not read long_view for test-period rows.
                skipped[str(event_date)] += 1
                continue
            label = 0 if row["long_view"] == "0" else 1
            accumulator.add(row, event_date, label, users, videos)
    return dict(sorted(skipped.items()))


def _split_summary(split: SplitAccumulator) -> dict[str, Any]:
    return {
        "rows": split.rows,
        "positives": split.positives,
        "long_view_rate": split.positives / split.rows if split.rows else None,
        "unique": {name: len(values) for name, values in split.entities.items()},
        "duration_ms": _quantiles(split.durations),
        "upload_age_days": _quantiles(split.upload_ages),
    }


def _ranking_summary(split: SplitAccumulator) -> dict[str, Any]:
    rows_per_user = [values[0] for values in split.users.values()]
    positives_per_user = [values[1] for values in split.users.values()]
    zero = sum(positive == 0 for positive in positives_per_user)
    all_positive = sum(positive == rows for rows, positive in split.users.values())
    mixed = len(split.users) - zero - all_positive
    return {
        "users": len(split.users),
        "zero_positive_users": zero,
        "all_positive_users": all_positive,
        "mixed_label_users_gauc_eligible": mixed,
        "rows_per_user": _integer_quantiles(rows_per_user),
        "positives_per_user": _integer_quantiles(positives_per_user),
        "metric_note": "GAUC uses mixed-label users; nDCG@5 includes zero-positive users as 0.",
    }


def _segment_table(split: SplitAccumulator, name: str, top_k: int) -> list[dict[str, Any]]:
    values = split.groups.get(name, {})
    limit = len(values) if name in LOW_CARDINALITY_GROUPS else top_k
    ordered = sorted(values.items(), key=lambda item: (-item[1][0], item[0]))[:limit]
    overall = split.positives / split.rows if split.rows else 0.0
    return [
        {
            "value": value, "rows": counts[0], "positives": counts[1],
            "long_view_rate": counts[1] / counts[0],
            "rate_delta_vs_split": counts[1] / counts[0] - overall,
        }
        for value, counts in ordered
    ]


def _total_variation(left: SplitAccumulator, right: SplitAccumulator, group: str) -> float:
    a, b = left.groups.get(group, {}), right.groups.get(group, {})
    if not left.rows or not right.rows:
        return 0.0
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(key, [0])[0] / left.rows - b.get(key, [0])[0] / right.rows)
                     for key in keys)


def _cold_start(train: SplitAccumulator, valid: SplitAccumulator) -> dict[str, Any]:
    result = {}
    for name in train.entities:
        unique_total = len(valid.entities[name])
        result[name] = {
            "unseen_unique": len(valid.unseen_unique[name]),
            "unseen_unique_rate": (len(valid.unseen_unique[name]) / unique_total
                                   if unique_total else 0.0),
            "rows_with_unseen": valid.unseen_rows[name],
            "row_rate": valid.unseen_rows[name] / valid.rows if valid.rows else 0.0,
        }
    return result


def _pair_coverage(train: SplitAccumulator, valid: SplitAccumulator) -> dict[str, Any]:
    result = {}
    for name in PAIR_NAMES:
        train_unique = np.unique(np.frombuffer(train.pairs[name], dtype=np.uint64))
        valid_values = np.frombuffer(valid.pairs[name], dtype=np.uint64)
        valid_unique = np.unique(valid_values)
        row_seen = np.isin(valid_values, train_unique, assume_unique=False)
        unique_seen = np.isin(valid_unique, train_unique, assume_unique=True)
        result[name] = {
            "validation_rows_seen_in_train": int(row_seen.sum()),
            "validation_row_coverage": float(row_seen.mean()) if len(row_seen) else 0.0,
            "validation_unique_pairs": int(len(valid_unique)),
            "validation_unique_pair_coverage": float(unique_seen.mean()) if len(unique_seen) else 0.0,
            "method": "stable_64bit_hash_approximation",
        }
    return result


def _metadata_coverage(split: SplitAccumulator) -> dict[str, Any]:
    return {
        name: {"missing_rows": int(count), "missing_row_rate": count / split.rows if split.rows else 0.0}
        for name, count in sorted(split.missing_metadata.items())
    }


def _rate_spread(split: SplitAccumulator, group: str, minimum_rows: int) -> tuple[float, str, str] | None:
    eligible = [(value, counts[1] / counts[0]) for value, counts in split.groups[group].items()
                if counts[0] >= minimum_rows]
    if len(eligible) < 2:
        return None
    low, high = min(eligible, key=lambda item: item[1]), max(eligible, key=lambda item: item[1])
    return high[1] - low[1], low[0], high[0]


def _opportunities(
    train: SplitAccumulator,
    valid: SplitAccumulator,
    cold: dict[str, Any],
    pairs: dict[str, Any],
    drift: dict[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    specs = (
        ("hour", "time_context", "hour/weekday context"),
        ("tab", "tab_context", "tab interactions"),
        ("user_active_degree", "user_activity", "user activity interactions"),
        ("video_type", "video_type", "video-type interactions"),
        ("tag", "tag_metadata", "tag and user-tag affinity"),
        ("upload_age_band", "content_recency", "upload-age or freshness interactions"),
    )
    minimum = max(100, valid.rows // 1000)
    for group, evidence_suffix, recommendation in specs:
        spread = _rate_spread(valid, group, minimum)
        if spread is None:
            continue
        delta, low, high = spread
        result.append({
            "evidence_id": f"profile_{evidence_suffix}",
            "observation": (
                f"Validation long-view rate differs by {delta:.4f} between sufficiently large "
                f"{group} groups ({low} vs {high})."
            ),
            "candidate_direction": recommendation,
            "caveat": "Descriptive association only; validate causally useful ranking signal by experiment.",
        })
    if cold["video"]["row_rate"] > 0.01:
        result.append({
            "evidence_id": "profile_video_cold_start",
            "observation": (
                f"{cold['video']['row_rate']:.2%} of validation rows contain videos unseen in train."
            ),
            "candidate_direction": "content metadata backoff using tag, music, type, author, and upload age",
            "caveat": "Static metadata is safe; target aggregates must be fitted on train only.",
        })
    best_pair = max(pairs, key=lambda key: pairs[key]["validation_row_coverage"])
    result.append({
        "evidence_id": "profile_affinity_coverage",
        "observation": (
            f"{best_pair} has the highest train-to-validation row coverage at "
            f"{pairs[best_pair]['validation_row_coverage']:.2%}."
        ),
        "candidate_direction": "prior-only affinity features with hierarchical backoff",
        "caveat": "For each event use history strictly before that event; never use the current label.",
    })
    if abs(drift["long_view_rate_delta"]) > 0.005:
        result.append({
            "evidence_id": "profile_temporal_label_drift",
            "observation": (
                f"Validation long-view rate changes by {drift['long_view_rate_delta']:+.4f} from train."
            ),
            "candidate_direction": "recency weighting and time-aware validation-safe history features",
            "caveat": "Tune recency only on validation; do not inspect test labels.",
        })
    return result


def _source_manifest(data_dir: Path) -> dict[str, Any]:
    result = {}
    for name in (TRAIN_LOG, EVALUATION_LOG, USER_FILE, VIDEO_FILE,
                 VIDEO_STATISTICS_FILE, RANDOM_LOG):
        path = data_dir / name
        if not path.is_file():
            result[name] = {"present": False}
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle), [])
        result[name] = {"present": True, "size_bytes": path.stat().st_size, "columns": header}
    return result


def build_planner_context(profile: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, evidence-addressable subset suitable for the LLM prompt."""
    boundary = profile["research_boundary"]
    if boundary.get("test_labels_accessed") is not False:
        raise ValueError("data profile is not approved for validation-only planner use")
    return {
        "schema_version": profile["schema_version"],
        "evidence_id": "data_profile_summary",
        "scope": "Train labels and validation labels/features only; no test labels.",
        "split_summary": profile["split_summary"],
        "validation_ranking_context": profile["ranking_diagnostics"]["validation"],
        "cold_start": profile["cold_start"],
        "affinity_coverage": profile["pair_coverage"],
        "train_validation_drift": profile["drift"],
        "key_findings": profile["feature_opportunities"],
        "leakage_constraints": profile["leakage_audit"],
    }


def load_planner_context(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("scope") != "Train labels and validation labels/features only; no test labels.":
        raise ValueError(f"invalid validation-only planner data profile: {path}")
    required = {
        "schema_version", "evidence_id", "scope", "split_summary",
        "validation_ranking_context", "cold_start", "affinity_coverage",
        "train_validation_drift", "key_findings", "leakage_constraints",
    }
    if not required <= set(value):
        raise ValueError(f"planner data profile is missing required sections: {path}")

    def reject_test_keys(item: Any, location: str = "root") -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = str(key).lower()
                if (normalized == "test" or normalized.startswith("test_") or
                        "_test_" in normalized or normalized.endswith("_test")):
                    raise ValueError(
                        f"test-derived section {location}.{key} is forbidden in planner context"
                    )
                reject_test_keys(child, f"{location}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                reject_test_keys(child, f"{location}[{index}]")

    reject_test_keys(value)
    return value


def build_profile(data_dir: Path, *, top_k: int = 15) -> tuple[dict[str, Any], dict[str, Any]]:
    required = (TRAIN_LOG, EVALUATION_LOG, USER_FILE, VIDEO_FILE)
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"dataset is missing from {data_dir}: {', '.join(missing)}")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    users, videos = _load_metadata(data_dir)
    train = SplitAccumulator("train")
    train_skipped = _consume_log(data_dir / TRAIN_LOG, TRAIN_RANGE, train, users, videos)
    valid = SplitAccumulator("validation", train.entities)
    evaluation_skipped = _consume_log(
        data_dir / EVALUATION_LOG, VALID_RANGE, valid, users, videos
    )

    cold = _cold_start(train, valid)
    pairs = _pair_coverage(train, valid)
    drift = {
        "long_view_rate_delta": valid.positives / valid.rows - train.positives / train.rows,
        "categorical_total_variation": {
            group: _total_variation(train, valid, group)
            for group in ("hour", "weekday", "tab", "user_active_degree", "video_type", "tag",
                          "upload_age_band")
        },
        "interpretation": "Distribution shift is descriptive; it does not establish predictive value.",
    }
    groups = sorted(set(train.groups) | set(valid.groups))
    profile = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "KuaiRand-Pure",
        "research_boundary": {
            "train_date_range": list(TRAIN_RANGE),
            "validation_date_range": list(VALID_RANGE),
            "test_labels_accessed": False,
            "skipped_rows_by_date": {
                "train_source_outside_train": train_skipped,
                "evaluation_source_outside_validation": evaluation_skipped,
            },
            "outcome_columns_excluded_from_candidate_features": list(OUTCOME_COLUMNS),
        },
        "source_manifest": _source_manifest(data_dir),
        "split_summary": {"train": _split_summary(train), "validation": _split_summary(valid)},
        "ranking_diagnostics": {"train": _ranking_summary(train), "validation": _ranking_summary(valid)},
        "cold_start": cold,
        "pair_coverage": pairs,
        "metadata_coverage": {
            "train": _metadata_coverage(train), "validation": _metadata_coverage(valid),
        },
        "segments": {
            "train": {group: _segment_table(train, group, top_k) for group in groups},
            "validation": {group: _segment_table(valid, group, top_k) for group in groups},
        },
        "drift": drift,
        "leakage_audit": {
            "safe_request_time_sources": {
                "interaction_context": ["date", "hourmin", "tab"],
                "user_metadata": [USER_FILE],
                "video_metadata": [VIDEO_FILE],
            },
            "train_only_derived": [
                "long_view target statistics fitted only on train",
                "user/item/author/tag/music histories using events strictly before the scored event",
            ],
            "forbidden_current_row_outcomes": list(OUTCOME_COLUMNS),
            "quarantined_until_provenance_or_rules_are_verified": {
                VIDEO_STATISTICS_FILE: "Aggregates may include validation/test future information.",
                RANDOM_LOG: "May be useful for debiasing, but contains post-train dates and needs rule review/date slicing.",
            },
        },
    }
    profile["feature_opportunities"] = _opportunities(train, valid, cold, pairs, drift)
    return profile, build_planner_context(profile)


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"


def render_markdown(profile: dict[str, Any]) -> str:
    train, valid = profile["split_summary"]["train"], profile["split_summary"]["validation"]
    lines = [
        "# KuaiRand-Pure data profile", "",
        f"Generated: {profile['generated_at']}", "",
        "> Research boundary: training and validation only. Test-period labels were not accessed.", "",
        "## Split summary", "",
        "| Split | Rows | Users | Videos | Positive rate |",
        "|---|---:|---:|---:|---:|",
        f"| Train | {train['rows']:,} | {train['unique']['user']:,} | {train['unique']['video']:,} | {_pct(train['long_view_rate'])} |",
        f"| Validation | {valid['rows']:,} | {valid['unique']['user']:,} | {valid['unique']['video']:,} | {_pct(valid['long_view_rate'])} |",
        "", "## Ranking diagnostics", "",
    ]
    for name in ("train", "validation"):
        ranking = profile["ranking_diagnostics"][name]
        lines.append(
            f"- {name.title()}: {ranking['users']:,} users; "
            f"{ranking['mixed_label_users_gauc_eligible']:,} mixed-label GAUC-eligible; "
            f"{ranking['zero_positive_users']:,} zero-positive users."
        )
    lines += ["", "## Validation cold start", "",
              "| Entity | Unseen unique | Unseen rows |", "|---|---:|---:|"]
    for name, values in profile["cold_start"].items():
        lines.append(
            f"| {name} | {_pct(values['unseen_unique_rate'])} | {_pct(values['row_rate'])} |"
        )
    lines += ["", "## Prior-affinity coverage", "",
              "| Pair | Validation rows seen in train | Unique-pair coverage |",
              "|---|---:|---:|"]
    for name, values in profile["pair_coverage"].items():
        lines.append(
            f"| {name} | {_pct(values['validation_row_coverage'])} | "
            f"{_pct(values['validation_unique_pair_coverage'])} |"
        )
    lines += ["", "## Measured feature opportunities", ""]
    for item in profile["feature_opportunities"]:
        lines.append(
            f"- `{item['evidence_id']}` — {item['observation']} Candidate: "
            f"{item['candidate_direction']}"
        )
    lines += [
        "", "## Leakage rules", "",
        "- Safe at request time: date/hour/tab plus static user and video metadata.",
        "- Label-derived aggregates must use training or strictly prior history only.",
        "- Never use current-row engagement outcomes (including play time, clicks, likes, comments, or long_view) as features.",
        f"- `{VIDEO_STATISTICS_FILE}` is quarantined until its aggregation window is proven not to include future labels.",
        f"- `{RANDOM_LOG}` is quarantined until competition rules and date-safe usage are verified.",
        "", "The JSON artifact contains complete segment tables and drift measurements.", "",
    ]
    return "\n".join(lines)


def write_profile(output_dir: Path, profile: dict[str, Any], planner_context: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "planner_context.json").write_text(
        json.dumps(planner_context, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "profile.md").write_text(render_markdown(profile), encoding="utf-8")
