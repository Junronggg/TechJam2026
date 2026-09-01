from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.runner import ExperimentRunner
from techjam_agent.config import normalize_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explicitly evaluate the saved validation-best model on test once"
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--output", type=Path, default=ROOT / "submissions" / "final.csv")
    args = parser.parse_args()
    project = json.loads((ROOT / "configs" / "project.json").read_text(encoding="utf-8"))
    config = normalize_config(json.loads(
        (ROOT / "artifacts" / "best_config.json").read_text(encoding="utf-8")
    ))
    data_dir = Path(args.data_dir or project["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    runner = ExperimentRunner(
        ROOT, data_dir, ROOT / project["starter_dir"], project["official_evaluator_sha256"]
    )
    metrics = runner.finalize(config, ROOT / "artifacts" / "best_model.npz", args.output)
    metrics_path = ROOT / "artifacts" / "final_test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Submission: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
