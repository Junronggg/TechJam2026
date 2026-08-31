"""Tests for planner prompt construction and LLM proposal retries. No real API."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.controller import Controller
from techjam_agent.config import apply_changes
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
    candidate_id_for,
    legal_candidate_catalog,
    standardize_proposal,
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
    candidate = apply_changes(load_config(), {"training_objective": "bpr"})
    return {
        "proposal_type": "config",
        "candidate_id": candidate_id_for(candidate),
        "observation": "The benchmark uses ranking metrics while the baseline uses BCE.",
        "diagnosis": "The pointwise objective may be misaligned with within-user ranking.",
        "hypothesis": "Try pairwise BPR instead of pointwise BCE.",
        "evidence_ids": ["benchmark_reference"],
        "reason": "Ranking metrics should match a ranking loss.",
        "changes": [{"field": "training_objective", "value": "bpr"}],
        "expected_effect": {"GAUC": "increase", "nDCG@5": "increase", "primary": "increase"},
        "risk": "A single-seed improvement may stay inside epsilon.",
        "estimated_cost": "medium",
        "success_condition": "Validation Primary exceeds the selected parent.",
    }


class FakeHTTPResponse:
    def __init__(self, payload: dict | str):
        if isinstance(payload, dict):
            raw = json.dumps(payload).encode()
        else:
            raw = payload.encode() if isinstance(payload, str) else payload
        self._raw = raw
        self.status = 200
        self.headers = {"x-request-id": "req_test"}

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
        if isinstance(item, Exception):
            raise item
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
        self.assertIn("budget", prompt)
        self.assertIn("remaining_seconds", prompt["budget"])
        self.assertIn("1 and 4", CHANGE_RULE)
        self.assertIn("lightgbm", prompt["remaining"]["models"])
        self.assertIn("ranked_candidates", prompt["research_search"])
        self.assertLessEqual(len(prompt["legal_candidates"]), 5)
        self.assertIn("benchmark_reference", prompt["valid_evidence_ids"])
        self.assertTrue(prompt["evidence_catalog"])
        self.assertTrue(all("candidate_id" in item for item in prompt["legal_candidates"]))
        self.assertEqual(prompt["global_best"]["validation_metrics"]["primary"], 0.6015)
        evidence = next(
            item for item in prompt["evidence_catalog"]
            if item["evidence_id"] == "iteration_000"
        )
        self.assertEqual(evidence["primary"], 0.6015)
        self.assertIn("remaining", prompt)
        self.assertIn("bpr", prompt["remaining"]["training_objectives"])

    def test_prompt_excludes_test_metrics(self) -> None:
        history = [{
            "iteration": 0,
            "decision": "KEEP",
            "metrics": {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015,
                        "test_GAUC": 0.6621, "test": {"primary": 0.5953}},
            "final_test_metrics": {"primary": 0.5953},
            "critique": {
                "verdict": "noise",
                "test_primary": 0.4321,
                "metric_deltas": {"primary": 0.0, "test": 0.4321},
            },
        }]
        prompt = build_planner_prompt(load_config(), history)
        blob = json.dumps(prompt)
        self.assertNotIn("0.5953", blob)
        self.assertNotIn("0.6621", blob)
        self.assertNotIn("test_GAUC", blob)
        self.assertNotIn("final_test_metrics", blob)
        self.assertNotIn("0.4321", blob)
        evidence = next(
            item for item in prompt["evidence_catalog"]
            if item["evidence_id"] == "iteration_000"
        )
        self.assertEqual(evidence["primary"], 0.6015)

    def test_validation_metrics_only_drops_foreign_keys(self) -> None:
        self.assertEqual(
            validation_metrics_only({"GAUC": 1, "nDCG@5": 0, "primary": 0.5, "test": 9}),
            {"GAUC": 1, "nDCG@5": 0, "primary": 0.5},
        )

    def test_candidate_catalog_is_legal_and_excludes_full_history(self) -> None:
        base = load_config()
        bpr = apply_changes(base, {"training_objective": "bpr"})
        catalog = legal_candidate_catalog(base, [{"config": bpr}])
        self.assertTrue(catalog)
        self.assertNotIn(
            candidate_id_for(bpr),
            {candidate["candidate_id"] for candidate in catalog},
        )
        for item in catalog:
            resolved = apply_changes(base, item["changes"])
            self.assertEqual(item["candidate_id"], candidate_id_for(resolved))

    def test_autonomous_catalog_adds_composite_candidates(self) -> None:
        base = load_config()
        regular = legal_candidate_catalog(base, [])
        autonomous = legal_candidate_catalog(base, [], autonomous=True)
        regular_ids = {item["candidate_id"] for item in regular}
        composites = [
            item for item in autonomous
            if item.get("candidate_kind") == "autonomous_composite"
        ]
        self.assertTrue(composites)
        self.assertGreater(len(autonomous), len(regular))
        for item in composites:
            self.assertNotIn(item["candidate_id"], regular_ids)
            self.assertGreaterEqual(len(item["changes"]), 2)
            self.assertLessEqual(len(item["changes"]), MAX_CHANGE_FIELDS)
            self.assertEqual(
                item["candidate_id"],
                candidate_id_for(apply_changes(base, item["changes"])),
            )

    def test_autonomous_prompt_advertises_self_expanding_search(self) -> None:
        prompt = build_planner_prompt(load_config(), [], autonomous=True)
        self.assertTrue(prompt["autonomy"]["autonomous_mode"])
        self.assertGreater(prompt["remaining"]["autonomous_composite_candidates"], 0)
        self.assertTrue(
            any(item.get("candidate_kind") == "autonomous_composite"
                for item in legal_candidate_catalog(load_config(), [], autonomous=True))
        )

    def test_catalog_skips_epoch_cap_after_observed_early_stop(self) -> None:
        base = load_config()
        history = [{
            "config": base,
            "status": "success",
            "metrics": {"primary": 0.6, "best_epoch": 5},
        }]
        catalog = legal_candidate_catalog(base, history)
        epoch_values = {
            item["changes"]["epochs"]
            for item in catalog if set(item["changes"]) == {"epochs"}
        }
        self.assertNotIn(10, epoch_values)
        self.assertNotIn(20, epoch_values)
        self.assertNotIn(30, epoch_values)

    def test_global_best_config_and_metrics_come_from_same_record(self) -> None:
        base = load_config()
        tagged = apply_changes(base, {"tag": True})
        history = [{
            "iteration": 0, "status": "success", "decision": "KEEP",
            "config": base,
            "metrics": {"GAUC": 0.61, "nDCG@5": 0.59, "primary": 0.60},
            "changes": {},
        }, {
            "iteration": 1, "status": "success", "decision": "KEEP",
            "config": tagged,
            "metrics": {"GAUC": 0.62, "nDCG@5": 0.60, "primary": 0.61},
            "changes": {"tag": True},
        }]
        prompt = build_planner_prompt(base, history)
        self.assertTrue(prompt["global_best"]["config"]["features"]["tag"])
        self.assertEqual(prompt["global_best"]["validation_metrics"]["primary"], 0.61)
        self.assertEqual(prompt["global_best"]["evidence_id"], "iteration_001")


class ParseContractTests(unittest.TestCase):
    def test_parse_rejects_unknown_top_level_fields(self) -> None:
        with self.assertRaises(ValueError):
            Proposal.parse({**legal_changes(), "temperature": 0.2}, "llm")

    def test_parse_allows_model_and_objective_together(self) -> None:
        payload = legal_changes()
        payload.update({
            "hypothesis": "Try LightGBM on the base fields.",
            "reason": "Isolate model class after BPR.",
            "changes": [
                {"field": "model", "value": "lightgbm"},
                {"field": "training_objective", "value": "bce"},
            ],
            "candidate_id": "candidate_parse_contract",
        })
        proposal = Proposal.parse(payload, "llm")
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

    def test_deterministic_uses_same_candidate_and_evidence_contract(self) -> None:
        config = load_config()
        raw = DeterministicResearcher().propose(config, [])
        proposal = standardize_proposal(raw, config, [])
        self.assertEqual(
            proposal.candidate_id,
            candidate_id_for(apply_changes(config, proposal.changes)),
        )
        self.assertIn("benchmark_reference", proposal.evidence_ids)
        self.assertTrue(any(item.startswith("strategy_") for item in proposal.evidence_ids))
        self.assertIsNotNone(proposal.observation)
        self.assertIsNotNone(proposal.diagnosis)
        self.assertEqual(set(proposal.expected_effect), {"GAUC", "nDCG@5", "primary"})


class LlmResearcherTests(unittest.TestCase):
    def _researcher(self, payloads: list) -> tuple[OpenAICompatibleResearcher, ScriptedUrlOpen]:
        opener = ScriptedUrlOpen(payloads)
        researcher = OpenAICompatibleResearcher(
            "gpt-test", api_key="sk-test-not-real", urlopen=opener,
            retry_backoff_seconds=0,
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
        self.assertEqual(proposal.llm_attempts, 1)
        self.assertEqual(proposal.observation, legal_changes()["observation"])
        self.assertEqual(proposal.evidence_ids, ("benchmark_reference",))
        self.assertEqual(len(researcher.last_call_records), 1)
        audit = researcher.last_call_records[0]
        self.assertEqual(audit["result"], "success")
        self.assertEqual(audit["http_status"], 200)
        self.assertEqual(audit["provider_request_id"], "req_test")
        self.assertEqual(audit["usage"]["total_tokens"], 15)
        self.assertNotIn("sk-test-not-real", json.dumps(audit))

    def test_data_profile_findings_are_valid_citable_evidence(self) -> None:
        payload = legal_changes()
        payload["evidence_ids"] = ["profile_affinity_coverage"]
        opener = ScriptedUrlOpen([chat_payload(payload)])
        researcher = OpenAICompatibleResearcher(
            "gpt-test", api_key="sk-test-not-real", urlopen=opener,
            retry_backoff_seconds=0,
            data_profile={
                "evidence_id": "data_profile_summary",
                "key_findings": [{"evidence_id": "profile_affinity_coverage"}],
            },
        )
        proposal = researcher.propose(load_config(), [])
        self.assertEqual(proposal.evidence_ids, ("profile_affinity_coverage",))
        prompt = researcher.last_call_records[0]["prompt"]
        self.assertEqual(prompt["data_profile"]["evidence_id"], "data_profile_summary")

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
        self.assertEqual(proposal.llm_attempts, 2)

    def test_timeout_then_valid_response_retries(self) -> None:
        researcher, opener = self._researcher([
            TimeoutError("mock timeout"),
            chat_payload(legal_changes(), {"total_tokens": 7}),
        ])
        proposal = researcher.propose(load_config(), [])
        self.assertEqual(len(opener.calls), 2)
        self.assertEqual(proposal.llm_attempts, 2)
        self.assertEqual(proposal.token_usage["total_tokens"], 7)

    def test_network_errors_exhaust_all_attempts(self) -> None:
        researcher, opener = self._researcher([
            urllib.error.URLError("offline") for _ in range(MAX_LLM_ATTEMPTS)
        ])
        with self.assertRaises(RuntimeError) as raised:
            researcher.propose(load_config(), [])
        self.assertIn("after 3 attempts", str(raised.exception))
        self.assertNotIn("sk-test-not-real", str(raised.exception))
        self.assertEqual(len(opener.calls), MAX_LLM_ATTEMPTS)
        self.assertEqual(
            [record["error"]["category"] for record in researcher.last_call_records],
            ["network"] * MAX_LLM_ATTEMPTS,
        )

    def test_authentication_error_does_not_retry(self) -> None:
        unauthorized = urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions", 401, "Unauthorized", {}, None
        )
        researcher, opener = self._researcher([unauthorized])
        with self.assertRaises(RuntimeError) as raised:
            researcher.propose(load_config(), [])
        self.assertEqual(len(opener.calls), 1)
        self.assertIn("authentication", str(raised.exception))
        self.assertEqual(researcher.last_call_records[0]["http_status"], 401)
        self.assertFalse(researcher.last_call_records[0]["error"]["retryable"])

    def test_illegal_change_then_valid_response_retries(self) -> None:
        researcher, opener = self._researcher([
            chat_payload({**legal_changes(), "changes": [{"field": "dropout", "value": 0.5}]}),
            chat_payload(legal_changes()),
        ])
        proposal = researcher.propose(load_config(), [])
        self.assertEqual(len(opener.calls), 2)
        self.assertEqual(proposal.changes, {"training_objective": "bpr"})

    def test_candidate_id_and_changes_must_match(self) -> None:
        wrong = legal_changes()
        wrong["changes"] = [{"field": "learning_rate", "value": 0.002}]
        researcher, opener = self._researcher([
            chat_payload(wrong),
            chat_payload(legal_changes()),
        ])
        proposal = researcher.propose(load_config(), [])
        self.assertEqual(len(opener.calls), 2)
        self.assertEqual(proposal.changes, {"training_objective": "bpr"})
        self.assertEqual(
            researcher.last_call_records[0]["error"]["category"],
            "candidate_validation",
        )

    def test_duplicate_change_then_new_change_retries(self) -> None:
        bpr = json.loads(json.dumps(load_config()))
        bpr["training_objective"] = "bpr"
        prompt = build_planner_prompt(load_config(), [{"config": bpr}])
        selected = prompt["legal_candidates"][0]
        repaired = {
            **legal_changes(),
            "hypothesis": "Use the highest-ranked available repair candidate.",
            "reason": "The original BPR candidate is already present in memory.",
            "candidate_id": selected["candidate_id"],
            "changes": [
                {"field": field, "value": value}
                for field, value in selected["changes"].items()
            ],
        }
        researcher, opener = self._researcher([
            chat_payload(legal_changes()),
            chat_payload(repaired),
        ])
        proposal = researcher.propose(load_config(), [{"config": bpr}])
        self.assertEqual(len(opener.calls), 2)
        self.assertEqual(proposal.changes, selected["changes"])

    def test_ten_mocked_calls_return_legal_structured_proposals(self) -> None:
        for _ in range(10):
            researcher, opener = self._researcher([chat_payload(legal_changes())])
            proposal = researcher.propose(load_config(), [])
            self.assertEqual(len(opener.calls), 1)
            self.assertEqual(proposal.changes, {"training_objective": "bpr"})
            self.assertTrue({
                "proposal_type", "observation", "diagnosis", "hypothesis", "evidence_ids",
                "candidate_id",
                "reason", "changes", "expected_effect", "risk", "success_condition",
                "estimated_cost",
                "source", "token_usage", "llm_attempts", "llm_call_ids", "fallback",
            }.issubset(proposal.as_dict()))

    def test_three_invalid_responses_raise_clear_error(self) -> None:
        payloads = [
            chat_payload("nope", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
            chat_payload("{", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
            chat_payload({**legal_changes(), "hypothesis": ""},
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
            retry_backoff_seconds=0,
        )
        researcher.propose(load_config(), [])
        self.assertEqual(len(opener.bodies[0]["messages"]), 2)
        self.assertEqual(len(opener.bodies[1]["messages"]), 3)
        self.assertIn("deterministic validation", opener.bodies[1]["messages"][-1]["content"])
        response_format = opener.bodies[0]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        schema = response_format["json_schema"]["schema"]
        evidence_enum = schema["properties"]["evidence_ids"]["items"]["enum"]
        self.assertIn("benchmark_reference", evidence_enum)
        self.assertTrue(any(item.startswith("strategy_") for item in evidence_enum))
        self.assertIn(
            legal_changes()["candidate_id"],
            schema["properties"]["candidate_id"]["enum"],
        )
        first_user = json.loads(opener.bodies[0]["messages"][1]["content"])
        self.assertNotIn("sk-test-not-real", json.dumps(first_user))
        self.assertNotIn("test_GAUC", json.dumps(first_user))


class FallbackTests(unittest.TestCase):
    def test_successful_llm_call_is_persisted_and_drives_iteration(self) -> None:
        researcher, _ = LlmResearcherTests()._researcher([
            chat_payload(
                legal_changes(),
                {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        ])
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            controller = Controller(
                FakeRunner(), researcher, load_config(), load_project(),
                base / "logs", base / "artifacts", base / "submissions",
            )
            with patch("sys.stdout", new=io.StringIO()):
                summary = controller.run(max_iterations=2)
            iteration = json.loads(
                (base / "logs" / "iteration_001.json").read_text(encoding="utf-8")
            )
            calls = [
                json.loads(line) for line in
                (base / "logs" / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            trajectory = json.loads(
                (base / "logs" / "research_trajectory.json").read_text(encoding="utf-8")
            )
        self.assertEqual(iteration["source"], "llm")
        self.assertEqual(iteration["observation"], legal_changes()["observation"])
        self.assertEqual(iteration["llm_call_ids"], [calls[0]["call_id"]])
        self.assertEqual(calls[0]["result"], "success")
        self.assertEqual(summary["llm_http_requests"], 1)
        self.assertEqual(summary["llm_http_failures"], 0)
        self.assertEqual(summary["llm_fallbacks"], 0)
        self.assertEqual(trajectory[1]["source"], "llm")
        self.assertEqual(trajectory[1]["evidence_ids"], ["benchmark_reference"])

    def test_controller_reports_llm_budget_and_token_accounting(self) -> None:
        class AccountedLlm:
            def __init__(self):
                self.last_attempts = 2
                self.last_token_usage = {
                    "prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15,
                }
                self.context = None

            def set_run_context(self, context):
                self.context = context

            def propose(self, best, history):
                return Proposal(
                    legal_changes()["hypothesis"], legal_changes()["reason"],
                    {"training_objective": "bpr"}, "llm", self.last_token_usage, 2,
                )

        researcher = AccountedLlm()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            controller = Controller(
                FakeRunner(), researcher, load_config(), load_project(),
                base / "logs", base / "artifacts", base / "submissions",
            )
            with patch("sys.stdout", new=io.StringIO()):
                summary = controller.run(max_iterations=2)
        self.assertEqual(summary["llm_requests"], 2)
        self.assertEqual(summary["llm_http_requests"], 2)
        self.assertEqual(summary["llm_tokens"]["total_tokens"], 15)
        self.assertEqual(summary["llm_failures"], 0)
        self.assertEqual(summary["llm_proposal_failures"], 0)
        self.assertEqual(researcher.context["remaining_iterations"], 1)
        self.assertGreater(researcher.context["remaining_seconds"], 0)
        self.assertEqual(researcher.context["estimated_next_experiment_seconds"], 900.0)

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
            self.assertEqual(second["source"], "deterministic_fallback")
            self.assertEqual(second["changes"], {"training_objective": "bpr"})
            self.assertEqual(second["token_usage"]["total_tokens"], 0)
            self.assertTrue(second["fallback"]["used"])
            self.assertEqual(second["fallback"]["reason_code"], "search_exhausted")

    def test_controller_persists_each_failed_call_and_links_fallback(self) -> None:
        researcher, _ = LlmResearcherTests()._researcher([
            urllib.error.URLError("offline") for _ in range(MAX_LLM_ATTEMPTS)
        ])
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            controller = Controller(
                FakeRunner(), researcher, load_config(), load_project(),
                base / "logs", base / "artifacts", base / "submissions",
            )
            with patch("sys.stdout", new=io.StringIO()):
                summary = controller.run(max_iterations=2)
            calls = [
                json.loads(line) for line in
                (base / "logs" / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            iteration = json.loads(
                (base / "logs" / "iteration_001.json").read_text(encoding="utf-8")
            )
        self.assertEqual(len(calls), MAX_LLM_ATTEMPTS)
        self.assertEqual(summary["llm_http_requests"], MAX_LLM_ATTEMPTS)
        self.assertEqual(summary["llm_http_failures"], MAX_LLM_ATTEMPTS)
        self.assertEqual(summary["llm_proposal_failures"], 1)
        self.assertEqual(summary["llm_fallbacks"], 1)
        self.assertEqual(summary["llm_error_categories"], {"network": MAX_LLM_ATTEMPTS})
        self.assertEqual(iteration["source"], "deterministic_fallback")
        self.assertEqual(iteration["llm_call_ids"], [row["call_id"] for row in calls])
        self.assertTrue(all(row["error"]["category"] == "network" for row in calls))

    def test_semantic_rejections_are_not_reported_as_http_failures(self) -> None:
        wrong = legal_changes()
        wrong["changes"] = [{"field": "learning_rate", "value": 0.002}]
        researcher, _ = LlmResearcherTests()._researcher([
            chat_payload(wrong) for _ in range(MAX_LLM_ATTEMPTS)
        ])
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            controller = Controller(
                FakeRunner(), researcher, load_config(), load_project(),
                base / "logs", base / "artifacts", base / "submissions",
            )
            with patch("sys.stdout", new=io.StringIO()):
                summary = controller.run(max_iterations=2)
        self.assertEqual(summary["llm_http_requests"], MAX_LLM_ATTEMPTS)
        self.assertEqual(summary["llm_http_failures"], 0)
        self.assertEqual(summary["llm_proposal_failures"], 1)
        self.assertEqual(summary["llm_fallbacks"], 1)


if __name__ == "__main__":
    unittest.main()
