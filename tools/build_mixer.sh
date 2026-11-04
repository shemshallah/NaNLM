#!/bin/sh
# Multi-order integer mixer smoke gate. The static mixer is a real measurement,
# not a quality claim: it currently loses BPC to the 3-slot memory on Alice.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT="$ROOT/build/mixer"
mkdir -p "$OUT"
make -C "$ROOT" build/nanlm_mixer >/dev/null
"$ROOT/build/nanlm_mixer" --corpus "$ROOT/data/alice_excerpt.txt" \
    --table-bits 14 --doc 1 --online 0 \
    --log "$OUT/stream.tsv" > "$OUT/run.log"
cat "$OUT/run.log"
grep -q "mixer: bpc=" "$OUT/run.log"
grep -q "eval=" "$OUT/run.log"
echo "mixer: PASS (full-stream log written)"
