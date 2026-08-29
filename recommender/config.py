"""Pure configuration transformations used by safe experiment operators."""

from __future__ import annotations

from experiment.schemas import ExperimentSpec, ModelConfig, Operation


class UnsupportedOperation(ValueError):
    """Raised when a specification cannot be applied as a safe config change."""


def apply_experiment(parent: ModelConfig, spec: ExperimentSpec) -> ModelConfig:
    """Return a candidate config without mutating the parent configuration."""
    features = list(parent.features)
    hyperparameters = dict(parent.hyperparameters)
    model = parent.model

    if spec.operation is Operation.CHANGE_HYPERPARAMETER:
        name = str(spec.parameters["name"])
        hyperparameters[name] = spec.parameters["value"]
    elif spec.operation is Operation.ADD_FEATURE:
        feature = str(spec.parameters["feature"])
        if feature not in features:
            features.append(feature)
    elif spec.operation is Operation.REMOVE_FEATURE:
        feature = str(spec.parameters["feature"])
        if feature not in features:
            raise UnsupportedOperation(f"Cannot remove inactive feature: {feature}")
        features.remove(feature)
    elif spec.operation is Operation.CHANGE_MODEL:
        model = str(spec.parameters["model"])
    elif spec.operation is Operation.CHANGE_LOSS_WEIGHT:
        task = str(spec.parameters["task"])
        hyperparameters[f"loss_weight.{task}"] = spec.parameters["value"]
    else:
        raise UnsupportedOperation(
            f"{spec.operation.value} requires a specialized, separately validated builder"
        )

    return ModelConfig(
        model=model,
        features=tuple(features),
        hyperparameters=hyperparameters,
        seed=parent.seed,
    )

