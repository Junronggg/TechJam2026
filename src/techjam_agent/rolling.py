from __future__ import annotations


ROLLING_FOLDS = (
    ("fold_1", 20220408, 20220414, 20220415, 20220417),
    ("fold_2", 20220408, 20220417, 20220418, 20220420),
    ("fold_3", 20220408, 20220420, 20220421, 20220423),
)


def build_rolling_splits(rows: list[tuple]) -> dict[str, dict[str, list[tuple]]]:
    """Build expanding train windows followed by disjoint future validation windows."""
    folds = {}
    for name, train_start, train_end, valid_start, valid_end in ROLLING_FOLDS:
        train = [row for row in rows if train_start <= int(row[0]) <= train_end]
        valid = [row for row in rows if valid_start <= int(row[0]) <= valid_end]
        if not train or not valid:
            raise ValueError(f"{name} is empty: train={len(train)}, valid={len(valid)}")
        if max(int(row[0]) for row in train) >= min(int(row[0]) for row in valid):
            raise ValueError(f"{name} does not preserve temporal order")
        folds[name] = {"train": train, "valid": valid}
    return folds
