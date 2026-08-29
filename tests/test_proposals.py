"""Tests for planner prompt construction and LLM proposal retries. No real API."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.controller import Controller
from techjam_agent.proposals import (
    CHANGE_RULE,
    CONVERGENCE_EPSILON,
    HTTP_TIMEOUT_SECONDS,
    MAX_CHANGE_FIELDS,
    MAX_LLM_ATTEMPTS,
    OFFICIAL_BASELINE_PRIMARY,
    OpenAICompatibleResearcher,
    Proposal,
    build_planner_prompt,
    DeterministicResearcher,
    extract_token_usage,
    validation_metrics_only,
)


def load_config() -> dict:
    return json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))


def load_project() -> dict:
    return json.loads((ROOT / "configs" / "project.json").read_text(encoding="utf-8"))


def chat_payload(content: str | dict, usage: dict | None = None) -> dict:
    if not isinstance(content, str):
        content = json.dumps(content)
    payload = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


def legal_changes() -> dict:
    return {
        "hypothesis": "Try pairwise BPR instead of pointwise BCE.",
        "reason": "Ranking metrics should match a ranking loss.",
        "changes": {"training_objective": "bpr"},
    }


class FakeHTTPResponse:
    def __init__(self, payload: dict | str):
        if isinstance(payload, dict):
            raw = json.dumps(payload).encode()
        else:
            raw = payload.encode() if isinstance(payload, str) else payload
        self._raw = raw

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class ScriptedUrlOpen:
    def __init__(self, payloads: list):
        self.payloads = list(payloads)
        self.calls = []

    def __call__(self, request, timeout=None):
        self.calls.append({"timeout": timeout, "url": getattr(request, "full_url", None)})
        if not self.payloads:
            raise AssertionError("unexpected extra urlopen call")
        item = self.payloads.pop(0)
        return FakeHTTPResponse(item)


class FakeRunner:
    def __init__(self):
        self.calls = 0

    def run(self, config, checkpoint):
        self.calls += 1
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return {"GAUC": 0.60, "nDCG@5": 0.60, "primary": 0.60,
                "best_epoch": 1, "runtime_seconds": 0.01}

    def finalize(self, config, checkpoint, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("row_id,user_id,video_id,score\n", encoding="utf-8")
        return {"GAUC": 0.62, "nDCG@5": 0.58, "primary": 0.60}


class PromptTests(unittest.TestCase):
    def test_prompt_includes_validation_critique_epsilon_and_allowed_ops(self) -> None:
        history = [{
            "iteration": 0,
            "hypothesis": "Reproduce the official FM baseline.",
            "changes": {},
            "decision": "KEEP",
            "metrics": {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015,
                        "test_GAUC": 0.9999, "test": {"primary": 0.5953}},
            "critique": {
                "observation": "Validation Primary=0.601500",
                "interpretation": "This establishes the validation baseline.",
                "confidence": "high",
                "next_test": "Repeat promising results across seeds.",
            },
        }]
        prompt = build_planner_prompt(load_config(), history)
        blob = json.dumps(prompt)
        self.assertIn("validation Primary", prompt["objective"])
        self.assertEqual(prompt["official_baseline_primary"], OFFICIAL_BASELINE_PRIMARY)
        self.assertEqual(prompt["epsilon"], CONVERGENCE_EPSILON)
        self.assertEqual(prompt["epsilon"], 0.002)
        self.assertIn("training_objective", prompt["allowed_values"])
        self.assertIn("model", prompt["allowed_values"])
        self.assertEqual(prompt["allowed_values"]["max_change_fields"], MAX_CHANGE_FIELDS)
        self.assertIn("1 and 3", CHANGE_RULE)
        self.assertIn("LightGBM", prompt["change_rule"])
        self.assertEqual(prompt["global_best"]["validation_metrics"]["primary"], 0.6015)
        self.assertEqual(prompt["history"][0]["hypothesis"], "Reproduce the official FM baseline.")
        self.assertEqual(prompt["history"][0]["critique"]["observation"], "Validation Primary=0.601500")
        self.assertEqual(prompt["history"][0]["decision"], "KEEP")
        self.assertIn("remaining", prompt)
        self.assertIn("bpr", prompt["remaining"]["training_objectives"])

    def test_prompt_excludes_test_metrics(self) -> None:
        history = [{
            "iteration": 0,
            "decision": "KEEP",
            "metrics": {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015,
                        "test_GAUC": 0.6621, "test": {"primary": 0.5953}},
            "final_test_metrics": {"primary": 0.5953},
        }]
        prompt = build_planner_prompt(load_config(), history)
        blob = json.dumps(prompt)
        self.assertNotIn("0.5953", blob)
        self.assertNotIn("0.6621", blob)
        self.assertNotIn("test_GAUC", blob)
        self.assertNotIn("final_test_metrics", blob)
        self.assertEqual(prompt["history"][0]["metrics"],
                         {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015})

    def test_validation_metrics_only_drops_foreign_keys(self) -> None:
        self.assertEqual(
            validation_metrics_only({"GAUC": 1, "nDCG@5": 0, "primary": 0.5, "test": 9}),
            {"GAUC": 1, "nDCG@5": 0, "primary": 0.5},
        )


class ParseContractTests(unittest.TestCase):
    def test_parse_rejects_unknown_top_level_fields(self) -> None:
        with self.assertRaises(ValueError):
            Proposal.parse({**legal_changes(), "temperature": 0.2}, "llm")

    def test_parse_allows_model_and_objective_together(self) -> None:
        proposal = Proposal.parse({
            "hypothesis": "Try LightGBM on the base fields.",
            "reason": "Isolate model class after BPR.",
            "changes": {"model": "lightgbm", "training_objective": "bce"},
        }, "llm")
        self.assertEqual(len(proposal.changes), 2)


class TokenUsageTests(unittest.TestCase):
    def test_missing_usage_is_zeros(self) -> None:
        self.assertEqual(
            extract_token_usage({"choices": []}),
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        self.assertEqual(
            extract_token_usage(None),
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    def test_deterministic_records_zero_tokens(self) -> None:
        proposal = DeterministicResearcher().propose(load_config(), [])
        self.assertEqual(proposal.as_dict()["token_usage"],
                         {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        self.assertEqual(proposal.source, "deterministic")


class LlmResearcherTests(unittest.TestCase):
    def _researcher(self, payloads: list) -> tuple[OpenAICompatibleResearcher, ScriptedUrlOpen]:
        opener = ScriptedUrlOpen(payloads)
        researcher = OpenAICompatibleResearcher(
            "gpt-test", api_key="sk-test-not-real", urlopen=opener,
        )
        return researcher, opener

    def test_valid_json_succeeds_on_first_attempt(self) -> None:
        researcher, opener = self._researcher([
            chat_payload(legal_changes(), {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}),
        ])
        proposal = researcher.propose(load_config(), [])
        self.assertEqual(len(opener.calls), 1)
        self.assertEqual(opener.calls[0]["timeout"], HTTP_TIMEOUT_SECONDS)
        self.assertEqual(proposal.changes, {"training_objective": "bpr"})
        self.assertEqual(proposal.token_usage["total_tokens"], 15)
        self.assertEqual(proposal.source, "llm")

    def test_invalid_then_valid_succeeds_after_retry(self) -> None:
        researcher, opener = self._researcher([
            chat_payload("this is not json", {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}),
            chat_payload(legal_changes(), {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}),
        ])
        proposal = researcher.propose(load_config(), [])
        self.assertEqual(len(opener.calls), 2)
        self.assertEqual(proposal.changes["training_objective"], "bpr")
        self.assertEqual(proposal.token_usage["prompt_tokens"], 12)
        self.assertEqual(proposal.token_usage["completion_tokens"], 3)
        self.assertEqual(proposal.token_usage["total_tokens"], 15)

    def test_three_invalid_responses_raise_clear_error(self) -> None:
        payloads = [
            chat_payload("nope", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
            chat_payload("{", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
            chat_payload({"hypothesis": "", "reason": "x", "changes": {"training_objective": "bpr"}},
                         {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
        ]
        researcher, opener = self._researcher(payloads)
        with self.assertRaises(RuntimeError) as raised:
            researcher.propose(load_config(), [])
        self.assertIn("after 3 attempts", str(raised.exception))
        self.assertNotIn("sk-test-not-real", str(raised.exception))
        self.assertEqual(len(opener.calls), MAX_LLM_ATTEMPTS)
        self.assertEqual(researcher.last_token_usage["total_tokens"], 6)

    def test_token_counts_accumulate_across_retries(self) -> None:
        researcher, _ = self._researcher([
            chat_payload("bad", {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}),
            chat_payload(legal_changes(), {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}),
        ])
        proposal = researcher.propose(load_config(), [])
        self.assertEqual(proposal.token_usage, {
            "prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38,
        })

    def test_missing_usage_does_not_crash(self) -> None:
        researcher, _ = self._researcher([chat_payload(legal_changes())])
        proposal = researcher.propose(load_config(), [])
        self.assertEqual(proposal.token_usage["total_tokens"], 0)

    def test_repair_instruction_sent_on_retry_not_first_call(self) -> None:
        opener = ScriptedUrlOpen([
            chat_payload("bad", {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}),
            chat_payload(legal_changes(), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
        ])

        def capturing_urlopen(request, timeout=None):
            body = json.loads(request.data.decode())
            opener.bodies = getattr(opener, "bodies", [])
            opener.bodies.append(body)
            return opener(request, timeout=timeout)

        researcher = OpenAICompatibleResearcher(
            "gpt-test", api_key="sk-test-not-real", urlopen=capturing_urlopen,
        )
        researcher.propose(load_config(), [])
        self.assertEqual(len(opener.bodies[0]["messages"]), 2)
        self.assertEqual(len(opener.bodies[1]["messages"]), 3)
        self.assertIn("exactly these keys", opener.bodies[1]["messages"][-1]["content"])
        first_user = json.loads(opener.bodies[0]["messages"][1]["content"])
        self.assertNotIn("sk-test-not-real", json.dumps(first_user))
        self.assertNotIn("test_GAUC", json.dumps(first_user))


class FallbackTests(unittest.TestCase):
    def test_controller_falls_back_to_deterministic_after_llm_error(self) -> None:
        class FailingLlm:
            def propose(self, best, history):
                raise RuntimeError("LLM proposal failed after 3 attempts: JSONDecodeError")

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            controller = Controller(
                FakeRunner(), FailingLlm(), load_config(), load_project(),
                base / "logs", base / "artifacts", base / "submissions",
            )
            with patch("sys.stdout", new=io.StringIO()):
                controller.run(max_iterations=2)
            records = sorted((base / "logs").glob("iteration_*.json"))
            self.assertEqual(len(records), 2)
            second = json.loads(records[1].read_text(encoding="utf-8"))
            self.assertEqual(second["source"], "deterministic")
            self.assertEqual(second["changes"], {"training_objective": "bpr"})
            self.assertEqual(second["token_usage"]["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
