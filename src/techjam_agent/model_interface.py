from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .operator_registry import MODEL_SPECS, ModelSpec


class RecommenderOperator(Protocol):
    """Common execution contract used by every registered model family."""

    spec: ModelSpec

    def fit_validate(
        self, runner: Any, config: dict[str, Any], checkpoint: Path
    ) -> dict[str, Any]: ...

    def finalize(
        self,
        runner: Any,
        config: dict[str, Any],
        checkpoint: Path,
        output: Path,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class MethodRecommenderOperator:
    """Adapter from the common operator API to an ExperimentRunner method pair.

    Existing model implementations remain unchanged behind this adapter. New
    families register one fit method and one finalization method instead of
    adding another model-specific branch to ``ExperimentRunner.run``.
    """

    spec: ModelSpec
    fit_method: str
    finalize_method: str

    def fit_validate(
        self, runner: Any, config: dict[str, Any], checkpoint: Path
    ) -> dict[str, Any]:
        return getattr(runner, self.fit_method)(config, checkpoint)

    def finalize(
        self,
        runner: Any,
        config: dict[str, Any],
        checkpoint: Path,
        output: Path,
    ) -> dict[str, Any]:
        return getattr(runner, self.finalize_method)(config, checkpoint, output)


_OPERATORS: dict[str, MethodRecommenderOperator] = {
    **{
        model_id: MethodRecommenderOperator(
            MODEL_SPECS[model_id], "_run_latent", "_finalize_latent"
        )
        for model_id in ("linear", "fm", "ffm", "fm_ensemble")
    },
    "lightgbm": MethodRecommenderOperator(
        MODEL_SPECS["lightgbm"], "_run_lightgbm", "_finalize_lightgbm"
    ),
    "seq_fm": MethodRecommenderOperator(
        MODEL_SPECS["seq_fm"], "_run_seq_fm", "_finalize_seq_fm"
    ),
    "fpmc": MethodRecommenderOperator(
        MODEL_SPECS["fpmc"], "_run_fpmc", "_finalize_fpmc"
    ),
    "deepfm": MethodRecommenderOperator(
        MODEL_SPECS["deepfm"],
        "_run_deepfm",
        "_finalize_deepfm",
    ),

    "dcnv2": MethodRecommenderOperator(
        MODEL_SPECS["dcnv2"],
        "_run_dcnv2",
        "_finalize_dcnv2",
    ),
    "dcnv2_dense": MethodRecommenderOperator(
        MODEL_SPECS["dcnv2_dense"],
        "_run_dcnv2_dense",
        "_finalize_dcnv2_dense",
    ),
    "two_tower": MethodRecommenderOperator(
        MODEL_SPECS["two_tower"],
        "_run_two_tower",
        "_finalize_two_tower",
    ),
    "hybrid_blend": MethodRecommenderOperator(
        MODEL_SPECS["hybrid_blend"],
        "_run_hybrid_blend",
        "_finalize_hybrid_blend",
    ),

    "sasrec": MethodRecommenderOperator(
        MODEL_SPECS["sasrec"],
        "_run_sasrec",
        "_finalize_sasrec",
    ),
    "din": MethodRecommenderOperator(
        MODEL_SPECS["din"], "_run_din", "_finalize_din"
    ),
    "sasrec_meta": MethodRecommenderOperator(
        MODEL_SPECS["sasrec_meta"], "_run_sasrec_meta", "_finalize_sasrec_meta"
    ),
    "lightgcn": MethodRecommenderOperator(
        MODEL_SPECS["lightgcn"], "_run_lightgcn", "_finalize_lightgcn"
    ),
    "lightgcn_hybrid": MethodRecommenderOperator(
        MODEL_SPECS["lightgcn_hybrid"], "_run_lightgcn_hybrid", "_finalize_lightgcn_hybrid"
    ),

    "multitask": MethodRecommenderOperator(
        MODEL_SPECS["multitask"],
        "_run_multitask",
        "_finalize_multitask",
    ),
    "custom": MethodRecommenderOperator(
        MODEL_SPECS["custom"], "_run_custom", "_finalize_custom"
    ),
}


def recommender_operator(model_id: str) -> RecommenderOperator:
    try:
        return _OPERATORS[model_id]
    except KeyError as exc:
        raise ValueError(f"no recommender operator is registered for {model_id!r}") from exc


def registered_model_ids() -> tuple[str, ...]:
    return tuple(_OPERATORS)
