#!/bin/sh
set -eu

image=$1
cpus=$2
memory=$3
data=$4
output=$5
case "$data" in
    /*) ;;
    *) data="$(cd "$(dirname "$data")" && pwd -P)/$(basename "$data")" ;;
esac
case "$output" in
    /*) ;;
    *) output="$(cd "$(dirname "$output")" && pwd -P)/$(basename "$output")" ;;
esac
scratch="$output/.scratch"
mkdir -p "$scratch"

exec docker run --rm --network none --cpus "$cpus" --memory "$memory" \
    --read-only --tmpfs /tmp:rw,size=64m --user 1000:1000 \
    -e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/scratch \
    -e NCPU="$cpus" -e MEM_GB=7 \
    -v "$data:/data:ro" -v "$output:/output" -v "$scratch:/scratch" \
    "$image"
