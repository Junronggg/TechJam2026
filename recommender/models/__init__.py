"""Model registry. Model adapters will be added behind the common training contract."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    implemented: bool
    description: str


MODEL_REGISTRY: dict[str, ModelDefinition] = {
    "fm": ModelDefinition("fm", False, "Official NumPy FM; adapter not wired yet"),
    "lightgbm": ModelDefinition("lightgbm", False, "Planned fast tree-model alternative"),
    "deepfm": ModelDefinition("deepfm", False, "Optional neural extension"),
}

