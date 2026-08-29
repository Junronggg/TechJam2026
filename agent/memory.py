"""Experiment tree plus compact empirical lessons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from experiment.schemas import CriticResult, ExperimentResult, ExperimentSpec, ModelConfig


@dataclass
class ExperimentNode:
    node_id: str
    parent_id: str | None
    branch: str
    config: ModelConfig
    result: ExperimentResult
    spec: ExperimentSpec | None = None
    critic: CriticResult | None = None
    children: list[str] = field(default_factory=list)
    visits: int = 0
    is_best: bool = False

    @property
    def primary(self) -> float:
        if self.result.metrics is None:
            return float("-inf")
        return self.result.metrics.primary

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "branch": self.branch,
            "config": self.config.to_dict(),
            "spec": self.spec.to_dict() if self.spec else None,
            "result": self.result.to_dict(),
            "critic": self.critic.to_dict() if self.critic else None,
            "children": list(self.children),
            "visits": self.visits,
            "is_best": self.is_best,
        }


class ResearchMemory:
    def __init__(self) -> None:
        self.nodes: dict[str, ExperimentNode] = {}
        self._config_signatures: set[str] = set()

    def add_root(self, config: ModelConfig, result: ExperimentResult) -> ExperimentNode:
        if self.nodes:
            raise ValueError("Root already exists")
        node = ExperimentNode(
            node_id="baseline",
            parent_id=None,
            branch="baseline",
            config=config,
            result=result,
            is_best=True,
        )
        self.nodes[node.node_id] = node
        self._config_signatures.add(config.signature())
        return node

    def add_child(
        self,
        parent_id: str,
        config: ModelConfig,
        spec: ExperimentSpec,
        result: ExperimentResult,
        critic: CriticResult,
    ) -> ExperimentNode:
        if spec.experiment_id in self.nodes:
            raise ValueError(f"Duplicate node id: {spec.experiment_id}")
        parent = self.nodes[parent_id]
        node = ExperimentNode(
            node_id=spec.experiment_id,
            parent_id=parent_id,
            branch=spec.branch,
            config=config,
            spec=spec,
            result=result,
            critic=critic,
        )
        self.nodes[node.node_id] = node
        parent.children.append(node.node_id)
        self._config_signatures.add(config.signature())
        self._refresh_best()
        return node

    def contains_config(self, config: ModelConfig) -> bool:
        return config.signature() in self._config_signatures

    def successful_nodes(self) -> Iterable[ExperimentNode]:
        return (node for node in self.nodes.values() if node.result.metrics is not None)

    def best_node(self) -> ExperimentNode:
        candidates = list(self.successful_nodes())
        if not candidates:
            raise RuntimeError("No successful experiment nodes")
        return max(candidates, key=lambda node: node.primary)

    def lessons(self, limit: int = 8) -> tuple[str, ...]:
        findings = [
            f"{node.node_id}: {node.critic.observation} {node.critic.interpretation}"
            for node in self.nodes.values()
            if node.critic is not None
        ]
        return tuple(findings[-limit:])

    def to_dict(self) -> dict[str, object]:
        return {
            "best_node_id": self.best_node().node_id,
            "nodes": [node.to_dict() for node in self.nodes.values()],
        }

    def _refresh_best(self) -> None:
        if not any(True for _ in self.successful_nodes()):
            return
        best_id = self.best_node().node_id
        for node in self.nodes.values():
            node.is_best = node.node_id == best_id
