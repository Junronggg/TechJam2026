from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.research_diagnostics import categorical_placebos, placebo_verdict
from techjam_agent.runner import ExperimentRunner
from techjam_agent.sequence_features import (
    SEQUENCE_FEATURE_DIMS,
    align_event_times,
    strict_sequence_categories,
)


def config_for(initial: dict, feature: str | None) -> dict:
    config = copy.deepcopy(initial)
    config["model"] = "fm"
    config["training_objective"] = "bpr"
    config["hyperparameters"]["learning_rate"] = 0.0003
    if feature is not None:
        config["features"][feature] = True
    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run real/constant/shuffled/random controls before attributing an FM "
            "gain to a categorical sequence feature"
        )
    )
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="runs/sequence_placebo")
    parser.add_argument(
        "--feature",
        default="prior_video_positive",
        choices=tuple(SEQUENCE_FEATURE_DIMS),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    initial = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    runner = ExperimentRunner(ROOT, data_dir, ROOT / "kuairand-starter-kit")
    runner.prepare()
    event_times = align_event_times(data_dir, runner._splits)
    real_categories = strict_sequence_categories(runner._splits, event_times)
    cardinality = SEQUENCE_FEATURE_DIMS[args.feature]
    variants_by_split = {
        split: categorical_placebos(
            categories[args.feature], cardinality, seed=args.seed + split_index
        )
        for split_index, (split, categories) in enumerate(real_categories.items())
    }

    results = {
        "baseline": runner.run(
            config_for(initial, None), output_dir / f"baseline_{args.feature}.npz"
        )
    }
    feature_config = config_for(initial, args.feature)
    for variant in ("real", "constant", "shuffled", "random_same_cardinality"):
        runner._sequence_categories = {
            split: {args.feature: controls[variant]}
            for split, controls in variants_by_split.items()
        }
        print(f"\n=== {variant} ===", flush=True)
        results[variant] = runner.run(
            feature_config, output_dir / f"{variant}_{args.feature}.npz"
        )
        results[variant]["delta_vs_baseline"] = (
            results[variant]["primary"] - results["baseline"]["primary"]
        )

    decision = placebo_verdict(
        results["real"]["primary"],
        {
            name: results[name]["primary"]
            for name in ("constant", "shuffled", "random_same_cardinality")
        },
    )
    payload = {
        "test_labels_used": False,
        "feature": args.feature,
        "cardinality": cardinality,
        "controls": (
            "constant; independently shuffled within each split; uniform random "
            "categorical with the same declared cardinality"
        ),
        "results": results,
        "decision": decision,
    }
    summary = output_dir / f"summary_{args.feature}.json"
    summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n=== automatic verdict ===")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
