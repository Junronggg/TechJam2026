from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.controller import Controller
from techjam_agent.evidence import build_generated_family_policies, merge_generated_policies
from techjam_agent.experiment_planner import MEMORY_MODES
from techjam_agent.local_env import load_local_env
from techjam_agent.proposals import DeterministicResearcher, OpenAICompatibleResearcher
from techjam_agent.isolated import IsolatedExperimentRunner


load_local_env(ROOT / ".env")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the autonomous FM research loop")
    parser.add_argument("--researcher", choices=("deterministic", "llm"), default="deterministic")
    parser.add_argument(
        "--memory-mode",
        choices=MEMORY_MODES,
        default="distilled_patterns",
        help=("Deterministic-planner ablation: ignore outcomes, read raw family "
              "history, or use distilled research patterns."),
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument(
        "--evidence-file",
        default=os.getenv("TECHJAM_RESEARCH_EVIDENCE", "configs/research_evidence.json"),
        help="Persistent validation-only evidence supplied to the selected Researcher.",
    )
    parser.add_argument(
        "--evidence-manifest",
        default=os.getenv(
            "TECHJAM_EVIDENCE_MANIFEST", "configs/evidence_manifest.json"
        ),
        help=("Manifest of validation artifacts used to regenerate scoped family "
              "policies before every run."),
    )
    parser.add_argument(
        "--research-context",
        default=os.getenv("TECHJAM_RESEARCH_CONTEXT", "configs/research_context.json"),
        help="Training-only dataset facts and method conditions supplied to the LLM planner.",
    )
    parser.add_argument("--max-iterations", type=int,
                        help="Maximum total executed experiments including the iteration-0 "
                             "baseline. Clamped to the official maximum in configs/project.json.")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--finalize-test",
        action="store_true",
        help="Evaluate test once after research. Omit during development runs.",
    )
    parser.add_argument(
        "--research-after-convergence",
        action="store_true",
        help=("Record the official convergence point but continue exploring legal "
              "branches until the experiment/time cap or search exhaustion."),
    )
    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        help=("Automatically spend action/time budget on rolling validation and "
              "paired-seed confirmation for promising new discoveries."),
    )
    parser.add_argument(
        "--intervention",
        action="append",
        default=[],
        metavar="REASON::ACTION::AVOIDABLE",
        help=("Audit a human intervention made during this run. AVOIDABLE is "
              "true/false. May be supplied more than once."),
    )
    args = parser.parse_args()

    try:
        from techjam_agent.runner import ExperimentRunner
    except ModuleNotFoundError as exc:
        if exc.name == "numpy":
            parser.error("NumPy is not installed. Create the project virtual environment and run "
                         "'python -m pip install -r requirements.txt'.")
        raise

    project = json.loads((ROOT / "configs" / "project.json").read_text(encoding="utf-8"))
    initial = json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))
    configured = args.data_dir or os.getenv("TECHJAM_DATA_DIR", project["data_dir"])
    data_dir = Path(configured)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    evidence_path = Path(args.evidence_file)
    if not evidence_path.is_absolute():
        evidence_path = ROOT / evidence_path
    try:
        prior_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot load research evidence from {evidence_path}: {exc}")
    if not isinstance(prior_evidence, dict):
        parser.error("research evidence must be a JSON object")
    manifest_path = Path(args.evidence_manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        generated_policies = build_generated_family_policies(ROOT, manifest)
        prior_evidence = merge_generated_policies(prior_evidence, generated_policies)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        parser.error(f"cannot build family policies from {manifest_path}: {exc}")
    context_path = Path(args.research_context)
    if not context_path.is_absolute():
        context_path = ROOT / context_path
    try:
        research_context = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot load research context from {context_path}: {exc}")
    if not isinstance(research_context, dict):
        parser.error("research context must be a JSON object")
    prior_evidence = {
        **prior_evidence,
        "dataset_facts": research_context.get("dataset_facts", {}),
        "method_reference": research_context.get("method_reference", {}),
    }
    if args.researcher == "llm":
        if args.memory_mode != "distilled_patterns":
            parser.error("--memory-mode ablation currently supports --researcher deterministic")
        researcher = OpenAICompatibleResearcher(
            args.model, args.base_url, prior_evidence=prior_evidence
        )
    else:
        researcher = DeterministicResearcher(
            project.get("autonomy", {}).get("candidate_scoring"),
            memory_mode=args.memory_mode,
            prior_evidence=prior_evidence,
        )
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    local_runner = ExperimentRunner(ROOT, data_dir, ROOT / project["starter_dir"],
                                    project.get("official_evaluator_sha256"))
    runner = IsolatedExperimentRunner(local_runner, project["experiment_timeout_seconds"])
    controller = Controller(runner, researcher, initial, project, ROOT / "logs" / run_id,
                            ROOT / "artifacts", ROOT / "submissions")
    for raw in args.intervention:
        parts = raw.split("::")
        if len(parts) != 3 or parts[2].strip().lower() not in {"true", "false"}:
            parser.error("--intervention must be REASON::ACTION::true|false")
        controller.record_intervention(
            parts[0], parts[1], parts[2].strip().lower() == "true"
        )
    summary = controller.run(
        args.max_iterations,
        finalize_test=args.finalize_test,
        research_after_convergence=args.research_after_convergence,
        auto_confirm=args.auto_confirm,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["best_primary"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
