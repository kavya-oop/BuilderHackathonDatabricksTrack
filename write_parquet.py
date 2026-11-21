#!/usr/bin/env python3
"""Convert the compact Rust core's aggregate CSVs to the required Parquet tables."""

import os
from pathlib import Path

import polars as pl


SCRATCH = Path(os.environ.get("SCRATCH_DIR", "/scratch"))
OUT = Path(os.environ.get("OUTPUT_DIR", "/output"))

USER_SCHEMA = {
    "user_id": pl.String,
    "day": pl.Date,
    "sessions": pl.Int64,
    "events": pl.Int64,
    "revenue_cents": pl.Int64,
    "top_category": pl.String,
}
CAMPAIGN_SCHEMA = {
    "campaign_id": pl.String,
    "channel": pl.String,
    "day": pl.Date,
    "sessions": pl.Int64,
    "purchasing_sessions": pl.Int64,
    "revenue_cents": pl.Int64,
    "assisted_sessions": pl.Int64,
}


def convert(name: str, schema: dict[str, pl.DataType]) -> None:
    source = SCRATCH / f"{name}.csv"
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    frame = pl.read_csv(
        source,
        schema=schema,
        null_values={"top_category": ""} if name == "user_daily" else None,
        low_memory=False,
    )
    frame.write_parquet(
        folder / "part-0.parquet",
        compression="uncompressed",
        statistics=False,
        row_group_size=1_000_000,
    )


convert("user_daily", USER_SCHEMA)
convert("campaign_daily", CAMPAIGN_SCHEMA)
