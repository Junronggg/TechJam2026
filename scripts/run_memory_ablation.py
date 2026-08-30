from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODES = ("no_memory", "raw_history", "distilled_patterns")
RUN_LOG_PATTERN = re.compile(r"^Run log:\s*(.+)$")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _load_history(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _run_mode(mode: str, max_iterations: int, data_dir: str | None) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_agent.py"),
        "--researcher",
        "deterministic",
        "--memory-mode",
        mode,
        "--max-iterations",
        str(max_iterations),
    ]
    if data_dir:
        command.extend(("--data-dir", data_dir))
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    run_dir: Path | None = None
    assert process.stdout is not None
    for raw_line in process.stdout:
        print(f"[{mode}] {raw_line}", end="", flush=True)
        match = RUN_LOG_PATTERN.match(raw_line.strip())
        if match:
            run_dir = Path(match.group(1)).resolve()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{mode} run failed with exit code {return_code}")
    if run_dir is None:
        raise RuntimeError(f"{mode} run did not report its log directory")

    summary = _load_json(run_dir / "summary.json")
    history = _load_history(run_dir / "experiment_history.jsonl")
    candidates = [row for row in history if row.get("iteration") != 0]
    unproductive = sum(
        row.get("research_decision") in {
            "REJECT", "STOP_DIRECTION", "REINTERPRET"
        }
        for row in candidates
    )
    selected_families = []
    for row in candidates:
        selection = row.get("candidate_selection")
        selected_families.append(
            selection.get("selected_family") if isinstance(selection, dict) else None
        )
    return {
        "mode": mode,
        "run_dir": str(run_dir),
        "best_primary": summary.get("best_primary"),
        "best_iteration": summary.get("best_iteration"),
        "total_experiments": summary.get("total_experiments"),
        "candidate_experiments": summary.get("candidate_experiments"),
        "unproductive_experiments": unproductive,
        "automatic_controls": sum(row.get("decision") == "CONTROL" for row in candidates),
        "manual_interventions": summary.get("manual_interventions"),
        "memory_influenced_selections": summary.get("memory_influenced_selections"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "stop_reason": summary.get("stop_reason"),
        "selected_families": selected_families,
        "test_metrics_used": summary.get("final_test_metrics") is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a fair validation-only ablation of planner memory modes."
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Equal total experiment cap for every mode, including the baseline.",
    )
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    if args.max_iterations < 2:
        parser.error("--max-iterations must be at least 2")

    results = [
        _run_mode(mode, args.max_iterations, args.data_dir) for mode in MODES
    ]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "researcher": "deterministic",
        "validation_only": True,
        "same_static_candidate_priors": True,
        "max_iterations_per_mode": args.max_iterations,
        "results": results,
    }
    output = ROOT / "artifacts" / "memory_ablation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Ablation report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
