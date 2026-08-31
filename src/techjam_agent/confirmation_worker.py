from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from .confirmation import run_confirmation
from .evidence_escalator import action_from_dict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = Path(request["result"])
    try:
        payload = {
            "status": "success",
            "result": run_confirmation(
                Path(request["root"]),
                Path(request["data_dir"]),
                Path(request["starter_dir"]),
                request.get("evaluator_sha256"),
                action_from_dict(request["action"]),
                Path(request["output_dir"]),
            ),
        }
        code = 0
    except Exception:
        payload = {"status": "error", "error": traceback.format_exc()[-8000:]}
        code = 1
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(payload), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
