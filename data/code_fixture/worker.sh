#!/bin/sh
# SPDX-License-Identifier: MIT
set -eu

ROOT=${1:-.}
OUT=${2:-build/ring}
mkdir -p "$OUT"

for shard in "$ROOT"/*.shard; do
    [ -f "$shard" ] || continue
    name=$(basename "$shard" .shard)
    cp "$shard" "$OUT/$name.shard"
    echo "leased $name"
done

if [ -f "$OUT/manifest.json" ]; then
    echo "manifest present"
else
    echo "manifest missing" >&2
    exit 1
fi
