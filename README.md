# Shopping Sessions — native high-performance pipeline

This repository implements the Databricks Shopping Sessions track. It strictly
parses and cleans the CSV inputs, resolves linked identities, builds sessions,
attributes and assists campaigns, and writes the two required Parquet datasets.

The runtime uses a native Rust core for the large parsing, sorting and timeline
sweeps. A small Polars step converts the already-aggregated results to Parquet.

## Validation

- 36/36 supplied edge-case fixtures pass.
- The supplied one-million-event development answer matches exactly: no
  missing, extra, differing, or duplicate-key rows in either table.
- A locally generated 20-million-event seed completes the native processing
  core in about 12 seconds under an 8-vCPU, 7 GB Docker limit. Local timings
  are informational; the contest scorer is authoritative.

## Container contract

The image reads `DATA_DIR` (default `/data`), uses `SCRATCH_DIR` (default
`/scratch`), and writes:

- `OUTPUT_DIR/user_daily/part-0.parquet`
- `OUTPUT_DIR/campaign_daily/part-0.parquet`

Build for the scorer with:

```sh
docker buildx build --platform linux/amd64 --load -t shopping-sessions:latest .
```

The full behavior is defined in [`rules.md`](rules.md).
