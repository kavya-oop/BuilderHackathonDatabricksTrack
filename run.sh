#!/bin/sh
set -eu

: "${DATA_DIR:=/data}" "${OUTPUT_DIR:=/output}" "${SCRATCH_DIR:=/scratch}"
export DATA_DIR OUTPUT_DIR SCRATCH_DIR
export HOME="$SCRATCH_DIR" TMPDIR="$SCRATCH_DIR" POLARS_TEMP_DIR="$SCRATCH_DIR"

/app/shopping-sessions
exec python3 /app/write_parquet.py
