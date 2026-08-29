"""Provider-neutral LLM boundary with explicit usage accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class StructuredLLMResponse:
    data: Mapping[str, Any]
    usage: LLMUsage
    model: str


class LLMClient(Protocol):
    def complete_structured(
        self,
        system_prompt: str,
        user_payload: Mapping[str, Any],
        schema_name: str,
    ) -> StructuredLLMResponse:
        """Return validated structured data and token usage without logging secrets."""


class LLMNotConnected(RuntimeError):
    pass


class MissingLLMClient:
    def complete_structured(
        self,
        system_prompt: str,
        user_payload: Mapping[str, Any],
        schema_name: str,
    ) -> StructuredLLMResponse:
        del system_prompt, user_payload, schema_name
        raise LLMNotConnected(
            "No LLM provider is configured. The deterministic dry-run planner remains available."
        )

