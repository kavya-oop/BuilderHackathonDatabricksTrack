#!/usr/bin/env python3
"""Exact order-insensitive comparison of output folders or expected CSVs."""

from pathlib import Path
import sys

from run_fixtures import SCHEMAS, load_actual, load_expected


def load(folder: Path, table: str):
    csv = folder / f"{table}.csv"
    return load_expected(csv, SCHEMAS[table]) if csv.is_file() else load_actual(folder / table, SCHEMAS[table])


def main() -> int:
    left, right = map(Path, sys.argv[1:])
    matched = True
    for table, schema in SCHEMAS.items():
        a, b = (load(folder, table) for folder in (left, right))
        keys = [column for column in ("user_id", "campaign_id", "day") if column in schema]
        duplicate_a = a.height - a.n_unique(subset=keys)
        duplicate_b = b.height - b.n_unique(subset=keys)
        only_a = a.join(b, on=list(schema), how="anti", nulls_equal=True)
        only_b = b.join(a, on=list(schema), how="anti", nulls_equal=True)
        print(
            f"{table}: rows {a.height} vs {b.height}; only in A {only_a.height}; "
            f"only in B {only_b.height}; duplicate keys {duplicate_a} vs {duplicate_b}"
        )
        if duplicate_a or duplicate_b or only_a.height or only_b.height:
            matched = False
            print(only_a.sort(keys).head(5))
            print(only_b.sort(keys).head(5))
    print("MATCH" if matched else "DIFFER")
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
