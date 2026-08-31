from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """Planner-visible capabilities for one executable recommender family."""

    model_id: str
    family: str
    objectives: tuple[str, ...]
    supports_sequence: bool = False
    supports_numeric_features: bool = False
    supports_engineered_features: bool = True
    estimated_cost: str = "medium"

    def prompt_view(self) -> dict[str, Any]:
        return asdict(self)


MODEL_SPECS: dict[str, ModelSpec] = {
    # ``custom`` is the open-ended escape hatch for an LLM-generated branch.
    # It deliberately advertises the broadest capabilities; the generated
    # source is still validated and run in the isolated worker before it can
    # affect the incumbent.
    "custom": ModelSpec(
        "custom", "generated_code", ("bce", "bpr", "group_softmax", "lambdarank"),
        supports_sequence=True, supports_numeric_features=True,
        supports_engineered_features=True, estimated_cost="high",
    ),
    "linear": ModelSpec("linear", "linear_control", ("bce", "bpr"), estimated_cost="low"),
    "fm": ModelSpec("fm", "factorization", ("bce", "bpr")),
    "ffm": ModelSpec("ffm", "field_aware_factorization", ("bce", "bpr")),
    "fm_ensemble": ModelSpec(
        "fm_ensemble", "factorization_ensemble", ("bpr",), estimated_cost="high"
    ),
    "lightgbm": ModelSpec(
        "lightgbm", "tree", ("bce", "lambdarank"),
        supports_numeric_features=True, estimated_cost="high",
    ),
    "seq_fm": ModelSpec(
        "seq_fm", "sequential_hybrid", ("bpr",), supports_sequence=True,
        supports_engineered_features=False, estimated_cost="medium",
    ),
    # FPMC is deliberately the first sequence scout: it tests whether a user's
    # most recent positive item adds signal before we pay the dependency and
    # runtime cost of a Transformer-based SASRec implementation.
    "fpmc": ModelSpec(
        "fpmc", "sequential", ("bpr",), supports_sequence=True,
        supports_engineered_features=False, estimated_cost="medium",
    ),
    "deepfm": ModelSpec(
        "deepfm",
        "deep_factorization",
        ("bce", "bpr", "group_softmax"),
        estimated_cost="high",
    ),

    "dcnv2": ModelSpec(
        "dcnv2",
        "feature_cross",
        ("bce", "bpr", "group_softmax"),
        estimated_cost="high",
    ),
    "dcnv2_dense": ModelSpec(
        "dcnv2_dense", "feature_cross_dense", ("bce", "bpr", "group_softmax"),
        supports_numeric_features=True, estimated_cost="high",
    ),
    "two_tower": ModelSpec(
        "two_tower", "retrieval_ranking", ("bce", "bpr", "group_softmax"),
        supports_sequence=True, supports_numeric_features=True,
        estimated_cost="high",
    ),
    "hybrid_blend": ModelSpec(
        "hybrid_blend", "heterogeneous_ensemble", ("bpr",),
        supports_numeric_features=True, estimated_cost="high",
    ),

    "sasrec": ModelSpec(
        "sasrec",
        "sequential",
        ("bce", "bpr"),
        supports_sequence=True,
        supports_engineered_features=False,
        estimated_cost="high",
    ),
    "din": ModelSpec(
        "din", "candidate_attention", ("bce", "bpr", "group_softmax"),
        supports_sequence=True, supports_numeric_features=False,
        estimated_cost="high",
    ),
    "sasrec_meta": ModelSpec(
        "sasrec_meta", "metadata_sequence", ("bce", "bpr", "group_softmax"),
        supports_sequence=True, supports_numeric_features=False,
        estimated_cost="high",
    ),
    "lightgcn": ModelSpec(
        "lightgcn", "collaborative_graph", ("bpr",),
        supports_sequence=False, supports_engineered_features=False,
        estimated_cost="high",
    ),
    "lightgcn_hybrid": ModelSpec(
        "lightgcn_hybrid", "graph_factorization_blend", ("bpr",),
        supports_sequence=False, supports_engineered_features=False,
        estimated_cost="high",
    ),

    "multitask": ModelSpec(
        "multitask",
        "multitask",
        ("bce", "bpr", "group_softmax"),
        estimated_cost="high",
    ),
}

MODEL_FAMILIES = tuple(MODEL_SPECS)
TRAINING_OBJECTIVES = ("bce", "bpr", "group_softmax", "lambdarank")
NUMPY_PAIRWISE_MODELS = (
    "linear", "fm", "ffm", "fm_ensemble", "seq_fm", "fpmc",
)
NEURAL_MODELS = (
    "custom", "deepfm", "dcnv2", "dcnv2_dense", "two_tower", "hybrid_blend",
    "sasrec", "din", "sasrec_meta", "lightgcn", "lightgcn_hybrid", "multitask",
)
OPTIMIZABLE_MODELS = (*NUMPY_PAIRWISE_MODELS, *NEURAL_MODELS)
PAIRWISE_MODELS = (
    "custom", *NUMPY_PAIRWISE_MODELS, "deepfm", "dcnv2", "dcnv2_dense", "two_tower",
    "hybrid_blend", "sasrec", "din", "sasrec_meta", "lightgcn",
    "lightgcn_hybrid", "multitask",
)
EMBEDDING_MODELS = (
    "fm", "ffm", "fm_ensemble", "seq_fm", "fpmc", *NEURAL_MODELS,
)
ENGINEERED_FEATURE_MODELS = (
    "custom", "linear", "fm", "ffm", "fm_ensemble", "lightgbm",
    "deepfm", "dcnv2", "dcnv2_dense", "two_tower", "hybrid_blend",
    "din", "sasrec_meta", "multitask",
)


@dataclass(frozen=True)
class OperatorSpec:
    field: str
    values: tuple[Any, ...]
    target: str
    branch: str
    description: str
    cost: str = "low"
    models: tuple[str, ...] = MODEL_FAMILIES
    objectives: tuple[str, ...] = TRAINING_OBJECTIVES
    requires: tuple[tuple[str, Any], ...] = ()
    default: Any = None

    def prompt_view(self) -> dict[str, Any]:
        value = asdict(self)
        value["requires"] = dict(self.requires)
        return value


OPERATORS: dict[str, OperatorSpec] = {
    "model": OperatorSpec(
        "model", MODEL_FAMILIES, "root", "model",
        "Choose among linear, FM, FFM, FM ensemble, LightGBM, "
        "DeepFM, categorical/dense DCNv2, a candidate/history two-tower, a DCN+FM rank blend, "
        "SASRec, candidate-aware DIN, metadata SASRec, LightGCN/hybrid, multi-task neural recommendation, "
        "sequential FM hybrid, the FPMC sequential control, or an LLM-generated custom branch.",
        "medium",
    ),
    "code_branch": OperatorSpec(
        "code_branch", ("__generated__",), "root", "model",
        "Activate an LLM-generated model/feature implementation. The Controller replaces the "
        "sentinel with a validated source path before execution.",
        "high", models=("custom",), objectives=TRAINING_OBJECTIVES,
    ),
    "training_objective": OperatorSpec(
        "training_objective", TRAINING_OBJECTIVES, "root", "ranking_objective",
        "Change between pointwise, pairwise, and grouped listwise ranking.", "medium",
    ),
    "validation_metric": OperatorSpec(
        "validation_metric", ("primary", "nDCG@5", "GAUC"), "hyperparameters", "ranking_objective",
        "Choose the validation metric used for epoch stopping and ensemble/blend selection. "
        "The official promotion score remains Primary=(GAUC+nDCG@5)/2.", "low",
        models=MODEL_FAMILIES, default="primary",
    ),
    "blend_mode": OperatorSpec(
        "blend_mode", ("rank", "zscore"), "hyperparameters", "model",
        "For heterogeneous blends, combine per-user percentile ranks or per-user z-scored "
        "component predictions before selecting weights.", "medium",
        models=("hybrid_blend", "lightgcn_hybrid"), objectives=("bpr",), default="rank",
    ),
    "embedding_dim": OperatorSpec(
        "embedding_dim", (8, 16, 32, 64, 128), "hyperparameters", "optimization",
        "Change FM/FFM/FPMC embedding capacity.", "medium",
        models=EMBEDDING_MODELS,
    ),
    "learning_rate": OperatorSpec(
        "learning_rate", (0.0002, 0.0005, 0.001, 0.002, 0.005), "hyperparameters", "optimization",
        "Change the model optimizer learning rate.", "low",
        models=OPTIMIZABLE_MODELS,
    ),
    "epochs": OperatorSpec(
        "epochs", (5, 10, 20, 30, 40, 50), "hyperparameters", "optimization",
        "Change the maximum training epochs.", "medium",
        models=OPTIMIZABLE_MODELS,
    ),
    "l2": OperatorSpec(
        "l2", (0.0, 1e-6, 1e-5, 1e-4), "hyperparameters", "optimization",
        "Change model L2 regularization.", "low",
        models=OPTIMIZABLE_MODELS,
    ),
    "batch_size": OperatorSpec(
        "batch_size", (1024, 2048, 4096, 8192, 16384, 32768), "hyperparameters", "optimization",
        "Change the model update batch size.", "low",
        models=OPTIMIZABLE_MODELS,
    ),
    "patience": OperatorSpec(
        "patience", (2, 3, 4, 5, 7), "hyperparameters", "optimization",
        "Change validation early-stopping patience.", "low",
        models=OPTIMIZABLE_MODELS,
    ),
    "seed": OperatorSpec(
        "seed", (0, 1, 2, 3, 4), "hyperparameters", "replication",
        "Repeat a configuration with a controlled random seed.", "medium",
    ),
    "ensemble_size": OperatorSpec(
        "ensemble_size", (1, 2, 3, 4, 5), "hyperparameters", "model",
        "Average independently trained FM+BPR checkpoints from fixed seeds.", "high",
        models=("fm_ensemble",), objectives=("bpr",), default=1,
    ),
    "ensemble_seed_set": OperatorSpec(
        "ensemble_seed_set", ("sequential", "3,4"), "hyperparameters", "model",
        "Choose a validated fixed seed set for an FM+BPR ensemble.", "high",
        models=("fm_ensemble",), objectives=("bpr",), default="sequential",
    ),
    "negatives_per_positive": OperatorSpec(
        "negatives_per_positive", (1, 2, 4, 8), "hyperparameters", "ranking_objective",
        "Sample multiple same-user negatives for each positive BPR interaction.", "medium",
        models=PAIRWISE_MODELS, objectives=("bpr", "group_softmax"), default=1,
    ),
    "negative_sampling_strategy": OperatorSpec(
        "negative_sampling_strategy", ("random", "same_tab", "same_author"),
        "hyperparameters", "ranking_objective",
        "Choose random or context-matched same-user BPR negatives.", "medium",
        models=PAIRWISE_MODELS, objectives=("bpr", "group_softmax"), default="random",
    ),
    "hard_negative_pool_size": OperatorSpec(
        "hard_negative_pool_size", (0, 4, 8, 16), "hyperparameters", "ranking_objective",
        "Mine the highest-scoring same-user negatives from a larger sampled pool.", "high",
        models=("custom", "deepfm", "dcnv2", "dcnv2_dense", "two_tower", "din", "sasrec_meta", "multitask"),
        objectives=("bpr", "group_softmax"), default=0,
    ),
    "dropout": OperatorSpec(
        "dropout", (0.0, 0.1, 0.2, 0.3), "hyperparameters", "optimization",
        "Tune neural regularization.", "low", models=NEURAL_MODELS, default=0.1,
    ),
    "sequence_length": OperatorSpec(
        "sequence_length", (10, 20, 50), "hyperparameters", "sequential",
        "Tune the number of strictly previous positive events encoded.", "high",
        models=("custom", "sasrec", "din", "sasrec_meta"), default=20,
    ),
    "auxiliary_weight": OperatorSpec(
        "auxiliary_weight", (0.05, 0.1, 0.2, 0.3), "hyperparameters", "ranking_objective",
        "Weight click/like auxiliary losses relative to long-view ranking.", "low",
        models=("multitask",), default=0.2,
    ),
    "graph_layers": OperatorSpec(
        "graph_layers", (1, 2, 3), "hyperparameters", "model",
        "Tune LightGCN neighborhood propagation depth.", "high",
        models=("lightgcn", "lightgcn_hybrid"), objectives=("bpr",), default=2,
    ),
    "user_long_view_rate": OperatorSpec(
        "user_long_view_rate", (False, True), "features", "features",
        "Add a leave-one-out train-only user rate; it is user-constant and needs interactions.", "medium",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "item_long_view_rate": OperatorSpec(
        "item_long_view_rate", (False, True), "features", "features",
        "Add a leave-one-out train-only smoothed item long-view-rate bucket.", "medium",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "continuous_history_stats": OperatorSpec(
        "continuous_history_stats", (False, True), "features", "features",
        "Add continuous user/item rates and counts.", "medium",
        models=("lightgbm",), requires=(("model", "lightgbm"),),
    ),
    "user_tab_long_view_rate": OperatorSpec(
        "user_tab_long_view_rate", (False, True), "features", "features",
        "Add leave-one-out smoothed user-by-tab preference from training history.", "medium",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "duration_fine_bucket": OperatorSpec(
        "duration_fine_bucket", (False, True), "features", "features",
        "Add a 50-bin duration feature fitted on train durations only.", "low",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "tag": OperatorSpec(
        "tag", (False, True), "features", "features",
        "Add the raw video tag as a train-vocabulary categorical field with UNK handling.",
        "low",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "hour": OperatorSpec(
        "hour", (False, True), "features", "features",
        "Add event hour as candidate-varying request context.", "low",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "weekday": OperatorSpec(
        "weekday", (False, True), "features", "features",
        "Add event weekday as candidate-varying request context.", "low",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "upload_age_bucket": OperatorSpec(
        "upload_age_bucket", (False, True), "features", "features",
        "Add fixed upload-age buckets computed at each impression time.", "low",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "freshness_decay": OperatorSpec(
        "freshness_decay", (False, True), "features", "features",
        "Add a continuous/bucketed exp(-upload_age/30d) freshness signal.", "low",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "user_activity": OperatorSpec(
        "user_activity", (False, True), "features", "features",
        "Add user activity class for interaction models; it is constant within a user group.",
        "low", models=ENGINEERED_FEATURE_MODELS,
    ),
    "video_type": OperatorSpec(
        "video_type", (False, True), "features", "features",
        "Add the candidate video's content type with train-vocabulary UNK handling.", "low",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "time_decay_item_popularity": OperatorSpec(
        "time_decay_item_popularity", (False, True), "features", "features",
        "Add strictly past-only 7-day half-life candidate item popularity.", "medium",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "time_decay_author_popularity": OperatorSpec(
        "time_decay_author_popularity", (False, True), "features", "features",
        "Add strictly past-only 7-day half-life candidate author popularity.", "medium",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "time_decay_tag_popularity": OperatorSpec(
        "time_decay_tag_popularity", (False, True), "features", "features",
        "Add strictly past-only 7-day half-life candidate tag popularity.", "medium",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "recent_history_similarity": OperatorSpec(
        "recent_history_similarity", (False, True), "features", "features",
        "Match each candidate's tag and author to the user's last 20 positive train events.",
        "medium", models=ENGINEERED_FEATURE_MODELS,
    ),
    "user_tag_impression_count": OperatorSpec(
        "user_tag_impression_count", (False, True), "features", "features",
        "Add a past-only bucketed user-tag interaction count.", "medium",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "user_tag_long_view_rate": OperatorSpec(
        "user_tag_long_view_rate", (False, True), "features", "features",
        "Add a past-only smoothed user-tag long-view rate with tag/global backoff.",
        "medium",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "author_impression_count": OperatorSpec(
        "author_impression_count", (False, True), "features", "features",
        "Add a leave-one-out train-only author impression-count bucket.", "medium",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "author_long_view_count": OperatorSpec(
        "author_long_view_count", (False, True), "features", "features",
        "Add a leave-one-out train-only author long-view-count bucket.", "medium",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "author_long_view_rate": OperatorSpec(
        "author_long_view_rate", (False, True), "features", "features",
        "Add a leave-one-out smoothed train-only author long-view rate.", "medium",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "user_author_impression_count": OperatorSpec(
        "user_author_impression_count", (False, True), "features", "features",
        "Add a strictly past-only user-author interaction-count bucket.", "high",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "user_author_long_view_count": OperatorSpec(
        "user_author_long_view_count", (False, True), "features", "features",
        "Add a strictly past-only user-author long-view-count bucket.", "high",
        models=ENGINEERED_FEATURE_MODELS,
    ),
    "user_author_long_view_rate": OperatorSpec(
        "user_author_long_view_rate", (False, True), "features", "features",
        "Add a strictly past-only user-author affinity with author/global backoff.", "high",
        models=ENGINEERED_FEATURE_MODELS,
    ),
}

ROOT_FIELDS = tuple(key for key, spec in OPERATORS.items() if spec.target == "root")
HYPERPARAMETER_FIELDS = tuple(
    key for key, spec in OPERATORS.items() if spec.target == "hyperparameters"
)
FEATURE_FIELDS = tuple(key for key, spec in OPERATORS.items() if spec.target == "features")
ALLOWED_VALUES = {key: OPERATORS[key].values for key in HYPERPARAMETER_FIELDS}
MODELS = OPERATORS["model"].values
OBJECTIVES = OPERATORS["training_objective"].values


def planner_registry() -> dict[str, dict[str, Any]]:
    return {key: spec.prompt_view() for key, spec in OPERATORS.items()}


def planner_model_registry() -> dict[str, dict[str, Any]]:
    return {key: spec.prompt_view() for key, spec in MODEL_SPECS.items()}


def branch_for_changes(changes: dict[str, Any]) -> str:
    if not changes:
        return "baseline"
    selected_model = changes.get("model")
    if selected_model in MODEL_SPECS and MODEL_SPECS[selected_model].supports_sequence:
        return "sequential"
    branches = {OPERATORS[key].branch for key in changes if key in OPERATORS}
    for preferred in ("features", "model", "ranking_objective", "replication", "optimization"):
        if preferred in branches:
            return preferred
    return "optimization"
