from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.data_profile import build_profile, write_profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a validation-safe KuaiRand-Pure EDA/data profile."
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", default="artifacts/data-profile")
    parser.add_argument("--top-k", type=int, default=15,
                        help="Maximum categories retained for high-cardinality segment tables.")
    args = parser.parse_args()

    project = json.loads((ROOT / "configs" / "project.json").read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir or project["data_dir"])
    output_dir = Path(args.output_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    print(f"Profiling train and validation data from {data_dir} ...", flush=True)
    profile, planner = build_profile(data_dir, top_k=args.top_k)
    write_profile(output_dir, profile, planner)
    train, valid = profile["split_summary"]["train"], profile["split_summary"]["validation"]
    print(
        f"Profile complete: train={train['rows']:,}, validation={valid['rows']:,}, "
        f"validation positive rate={valid['long_view_rate']:.4f}", flush=True,
    )
    print(f"Readable report: {output_dir / 'profile.md'}", flush=True)
    print(f"Planner context: {output_dir / 'planner_context.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
