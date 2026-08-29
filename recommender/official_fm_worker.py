"""Subprocess entry point for one isolated official-FM validation run."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from experiment.schemas import ExperimentResult, ExperimentStatus, ModelConfig, write_json
from recommender.official_fm import run_validation_fm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    request_path = parse_args().request.resolve()
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    run_dir = Path(payload["run_dir"]).resolve()
    output_path = run_dir / "backend_result.json"
    raw_config = payload["config"]
    config = ModelConfig(
        model=raw_config["model"],
        features=tuple(raw_config["features"]),
        hyperparameters=raw_config["hyperparameters"],
        seed=int(raw_config["seed"]),
    )
    try:
        result = run_validation_fm(
            experiment_id=payload["experiment_id"],
            config=config,
            starter_dir=Path(payload["starter_dir"]).resolve(),
            data_dir=Path(payload["data_dir"]).resolve(),
            run_dir=run_dir,
            evaluator_sha256=payload["evaluator_sha256"],
        )
        return_code = 0
    except Exception:
        result = ExperimentResult(
            experiment_id=payload["experiment_id"],
            status=ExperimentStatus.FAILED,
            error=traceback.format_exc(),
        )
        return_code = 1
    write_json(output_path, result.to_dict())
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

