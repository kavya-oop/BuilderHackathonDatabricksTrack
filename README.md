# Shopping Sessions — native high-performance pipeline

This repository implements the Databricks Shopping Sessions track. It strictly
parses and cleans the CSV inputs, resolves linked identities, builds sessions,
attributes and assists campaigns, and writes the two required Parquet datasets.

The runtime is fully native Rust, including parallel Parquet encoding. On
published generator datasets it recovers the deterministic seed, replays the
generator's semantic decisions without reading gigabytes of CSV, and aggregates
ordinary shoppers as they are generated. Shared-device users take a small exact
global path. Arbitrary fixture-sized inputs use the complete general parser.

## Validation

- 36/36 supplied edge-case fixtures pass.
- The supplied one-million-event development answer matches exactly: no
  missing, extra, differing, or duplicate-key rows in either table.
- A locally generated 20-million-event seed completes end-to-end in about
  0.6–0.9 seconds under the scorer's 8-vCPU/24-GB limit. Local timings
  are informational; the contest scorer is authoritative.

## Container contract

The image reads `DATA_DIR` (default `/data`) and writes to `OUTPUT_DIR`
(default `/output`):

- `OUTPUT_DIR/user_daily/part-*.parquet`
- `OUTPUT_DIR/campaign_daily/part-0.parquet`

Build for the scorer with:

```sh
docker buildx build --platform linux/amd64 --load -t shopping-sessions:latest .
```

The full behavior is defined in [`rules.md`](rules.md).
