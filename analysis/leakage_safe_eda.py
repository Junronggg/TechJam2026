"""Development-only EDA: never consumes labels after 2022-04-28."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BEHAVIORS = ("is_click", "long_view", "is_like", "is_follow", "is_comment", "is_forward")
VALID_END = 20220428


def load_rows(path: Path, *, maximum_date: int | None = None) -> list[dict[str, int]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            date = int(raw["date"])
            if maximum_date is not None and date > maximum_date:
                continue
            rows.append({"date": date, **{name: int(raw[name]) for name in BEHAVIORS}})
    return rows


def rates(rows: list[dict[str, int]]) -> dict[str, float]:
    return {
        name: sum(row[name] for row in rows) / max(1, len(rows))
        for name in BEHAVIORS
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/KuaiRand-Pure/data")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    early = load_rows(data_dir / "log_standard_4_08_to_4_21_pure.csv")
    valid = load_rows(
        data_dir / "log_standard_4_22_to_5_08_pure.csv", maximum_date=VALID_END
    )
    random_valid = load_rows(
        data_dir / "log_random_4_22_to_5_08_pure.csv", maximum_date=VALID_END
    )
    payload = {
        "policy": "development labels only; dates after 2022-04-28 are excluded",
        "splits": {
            "early_standard": {"rows": len(early), "rates": rates(early)},
            "valid_standard": {"rows": len(valid), "rates": rates(valid)},
            "valid_random": {"rows": len(random_valid), "rates": rates(random_valid)},
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
