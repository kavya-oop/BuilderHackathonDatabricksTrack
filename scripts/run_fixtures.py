#!/usr/bin/env python3
"""Run an implementation on every supplied fixture and compare exact rows."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import polars as pl


SCHEMAS = {
    "user_daily": {
        "user_id": pl.String,
        "day": pl.Date,
        "sessions": pl.Int128,
        "events": pl.Int128,
        "revenue_cents": pl.Int128,
        "top_category": pl.String,
    },
    "campaign_daily": {
        "campaign_id": pl.String,
        "channel": pl.String,
        "day": pl.Date,
        "sessions": pl.Int128,
        "purchasing_sessions": pl.Int128,
        "revenue_cents": pl.Int128,
        "assisted_sessions": pl.Int128,
    },
}


def load_expected(path: str | Path, schema: dict) -> pl.DataFrame:
    frame = pl.read_csv(
        path,
        schema_overrides={key: (pl.String if value == pl.Date else value) for key, value in schema.items()},
    )
    if frame.height == 0:
        return pl.DataFrame(schema=schema)
    return frame.with_columns(pl.col("day").str.to_date())


def load_actual(folder: str | Path, schema: dict) -> pl.DataFrame:
    folder = Path(folder)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError(f"expected an output directory: {folder}")
    frames = []
    for path in sorted(folder.glob("*.parquet")):
        if path.name.startswith((".", "_")) or path.is_symlink() or not path.is_file():
            continue
        frame = pl.read_parquet(path)
        if set(frame.columns) != set(schema):
            raise ValueError(f"{path}: expected {list(schema)}, got {frame.columns}")
        for column, expected in schema.items():
            actual = frame.schema[column]
            if expected == pl.Int128 and actual.is_integer():
                continue
            if expected == pl.String and actual.base_type() in (pl.String, pl.Categorical, pl.Enum):
                continue
            if actual != expected:
                raise ValueError(f"{path}: {column} has type {actual}, expected {expected}")
        frames.append(frame.select(schema.keys()).cast(schema))
    if not frames:
        raise ValueError(f"no parquet in {folder}")
    return pl.concat(frames)


def rows(frame: pl.DataFrame) -> list[tuple]:
    return sorted(frame.rows(), key=lambda row: tuple("" if value is None else str(value) for value in row))


def main() -> int:
    separator = sys.argv.index("--")
    fixtures, command = sys.argv[1], sys.argv[separator + 1 :]
    cases = sorted(name for name in os.listdir(fixtures) if os.path.isdir(os.path.join(fixtures, name)))
    failed = []
    for case in cases:
        data = os.path.join(fixtures, case, "input")
        expected = os.path.join(fixtures, case, "expected")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as output:
            environment = dict(os.environ, DATA_DIR=data, OUTPUT_DIR=output)
            process = subprocess.run(
                command + [data, output], env=environment, capture_output=True, text=True
            )
            if process.returncode != 0:
                failed.append(case)
                print(f"FAIL {case}: exit {process.returncode}\n{process.stderr[-2000:]}")
                continue
            good = True
            for table, schema in SCHEMAS.items():
                wanted = rows(load_expected(os.path.join(expected, f"{table}.csv"), schema))
                try:
                    actual = rows(load_actual(os.path.join(output, table), schema))
                except (ValueError, OSError, pl.exceptions.PolarsError) as error:
                    print(f"FAIL {case} {table}: {error}")
                    good = False
                    continue
                if wanted != actual:
                    print(f"FAIL {case} {table}\n  expected {wanted}\n  actual   {actual}")
                    good = False
            print(f"{'ok  ' if good else 'FAIL'} {case}")
            if not good:
                failed.append(case)
    print(f"\n{len(cases) - len(failed)}/{len(cases)} fixtures passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
