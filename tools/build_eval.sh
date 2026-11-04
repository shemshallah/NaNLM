#!/bin/sh
# Full held-out-stream gate: every byte, paired bootstrap, model-size accounting.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT="$ROOT/build/eval"
mkdir -p "$OUT"
make -C "$ROOT" build/nanlm_slot_train >/dev/null

"$ROOT/build/nanlm_slot_train" --corpus "$ROOT/data/alice_excerpt.txt" \
    --eval-log "$OUT/slot.tsv" \
    --out "$OUT/slot.nas" --expected "$OUT/slot.expected" \
    --manifest "$OUT/slot.manifest" > "$OUT/slot.log"
cat "$OUT/slot.log"
python3 "$ROOT/experiments/eval_stream.py" \
    --run "slot3=$OUT/slot.tsv=$OUT/slot.nas" --boot 5000
