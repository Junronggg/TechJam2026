"""Failure-repair boundary; provider-specific repair is deferred."""

from __future__ import annotations

from typing import Protocol

from experiment.schemas import ExperimentResult, ExperimentSpec


class RepairAdvisor(Protocol):
    def repair(self, spec: ExperimentSpec, failed_result: ExperimentResult) -> ExperimentSpec:
        """Return one constrained repaired spec using the captured failure evidence."""

