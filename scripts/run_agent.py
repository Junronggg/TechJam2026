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
from techjam_agent.proposals import DeterministicResearcher, OpenAICompatibleResearcher


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the autonomous FM research loop")
    parser.add_argument("--researcher", choices=("deterministic", "llm"), default="deterministic")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--data-dir")
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
    researcher = (OpenAICompatibleResearcher(args.model, args.base_url)
                  if args.researcher == "llm" else DeterministicResearcher())
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    runner = ExperimentRunner(ROOT, data_dir, ROOT / project["starter_dir"])
    controller = Controller(runner, researcher, initial, project, ROOT / "logs" / run_id,
                            ROOT / "artifacts", ROOT / "submissions")
    summary = controller.run(args.max_iterations)
    print(json.dumps(summary, indent=2))
    return 0 if summary["best_primary"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
