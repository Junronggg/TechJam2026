from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvolutionRecipe:
    """A fast, reproducible branch assembled from registered operators.

    The catalog also exposes a separate open-ended generated-code candidate;
    recipes remain the low-risk fallback when the Planner cannot justify code.
    """

    recipe_id: str
    purpose: str
    targets: dict[str, Any]


EVOLUTION_RECIPES = (
    EvolutionRecipe(
        "evolve_temporal_dense_cross",
        "Cross request time with upload age and continuous freshness in dense DCNv2.",
        {"model": "dcnv2_dense", "hour": True, "upload_age_bucket": True,
         "freshness_decay": True},
    ),
    EvolutionRecipe(
        "evolve_causal_popularity_cross",
        "Combine causal short-term item, author, and tag popularity in dense DCNv2.",
        {"model": "dcnv2_dense", "time_decay_item_popularity": True,
         "time_decay_author_popularity": True, "time_decay_tag_popularity": True},
    ),
    EvolutionRecipe(
        "evolve_hierarchical_two_tower",
        "Combine hierarchical affinities and recent candidate similarity in two towers.",
        {"model": "two_tower", "user_tag_long_view_rate": True,
         "user_author_long_view_rate": True, "recent_history_similarity": True},
    ),
    EvolutionRecipe(
        "evolve_group_ranker",
        "Train dense DCNv2 against one positive and four same-user negatives.",
        {"model": "dcnv2_dense", "training_objective": "group_softmax",
         "negatives_per_positive": 4},
    ),
    EvolutionRecipe(
        "evolve_heterogeneous_blend",
        "Blend independently trained DCNv2, FM-BPR, and LambdaRank predictions.",
        {"model": "hybrid_blend", "training_objective": "bpr"},
    ),
    EvolutionRecipe(
        "evolve_candidate_aware_din",
        "Attend to metadata-rich positive history conditioned on each candidate.",
        {"model": "din", "training_objective": "group_softmax",
         "negatives_per_positive": 4, "sequence_length": 20},
    ),
    EvolutionRecipe(
        "evolve_metadata_sequence",
        "Encode item, author, tag and duration history with a causal Transformer.",
        {"model": "sasrec_meta", "training_objective": "bpr",
         "sequence_length": 20},
    ),
    EvolutionRecipe(
        "evolve_din_context_bundle",
        "Condition DIN on request time, content type, and freshness metadata.",
        {"model": "din", "tag": True, "hour": True, "video_type": True},
    ),
    EvolutionRecipe(
        "evolve_ranking_multitask",
        "Optimize long-view listwise ranking with click and like as auxiliary targets.",
        {"model": "multitask", "training_objective": "group_softmax",
         "negatives_per_positive": 4, "auxiliary_weight": 0.1},
    ),
    EvolutionRecipe(
        "evolve_lightgcn_hybrid",
        "Blend graph collaborative propagation with an independently trained FM ranker.",
        {"model": "lightgcn_hybrid", "training_objective": "bpr",
         "graph_layers": 2},
    ),
    EvolutionRecipe(
        "evolve_hard_negative_din",
        "Train candidate-aware DIN against model-mined same-user hard negatives.",
        {"model": "din", "training_objective": "group_softmax",
         "negatives_per_positive": 4, "hard_negative_pool_size": 16},
    ),
)
