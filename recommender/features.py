"""Feature metadata registry with explicit leakage boundaries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    required_columns: tuple[str, ...]
    fit_split: str
    leakage_rule: str
    implemented: bool = False


FEATURE_REGISTRY: dict[str, FeatureDefinition] = {
    "user_id": FeatureDefinition("user_id", ("user_id",), "none", "raw identifier", True),
    "video_id": FeatureDefinition("video_id", ("video_id",), "none", "raw identifier", True),
    "author_id": FeatureDefinition("author_id", ("author_id",), "none", "raw item metadata", True),
    "tab": FeatureDefinition("tab", ("tab",), "none", "raw impression context", True),
    "dur_bucket": FeatureDefinition(
        "dur_bucket", ("duration_ms",), "train", "quantiles must be fit on train only", True
    ),
    "item_popularity": FeatureDefinition(
        "item_popularity", ("video_id",), "train", "aggregate train interactions only"
    ),
    "user_activity": FeatureDefinition(
        "user_activity", ("user_id",), "train", "aggregate train interactions only"
    ),
    "item_long_view_rate": FeatureDefinition(
        "item_long_view_rate",
        ("video_id", "long_view"),
        "train",
        "never fit with validation labels",
    ),
    "user_category_affinity": FeatureDefinition(
        "user_category_affinity",
        ("user_id", "category", "long_view"),
        "train",
        "use historical train rows only with smoothing",
    ),
}


def implemented_features() -> set[str]:
    return {name for name, definition in FEATURE_REGISTRY.items() if definition.implemented}

