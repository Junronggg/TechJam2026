from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.evidence import build_generated_family_policies  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate scoped family policies from validation-only artifacts."
    )
    parser.add_argument("--manifest", default="configs/evidence_manifest.json")
    parser.add_argument(
        "--check-against",
        help="Fail if generated JSON differs from this existing file.",
    )
    args = parser.parse_args()
    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generated = build_generated_family_policies(ROOT, manifest)
    rendered = json.dumps(generated, indent=2, ensure_ascii=False) + "\n"
    if args.check_against:
        expected = (ROOT / args.check_against).read_text(encoding="utf-8")
        if rendered != expected:
            print("generated family policies are stale", file=sys.stderr)
            return 1
        print("generated family policies are current")
        return 0
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
