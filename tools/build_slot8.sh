#!/bin/sh
# 8-byte / 16-slot student gate.
#
# Teacher records are exact 8-byte contexts. The student uses eight byte slots
# plus eight ctype slots (16 x 256 bits = 4096-bit keys). The fixture teacher is
# intentionally weak; this gate proves the longer-context path and the VM
# differential, not quality.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CORPUS=${CORPUS:-"$ROOT/data/alice_excerpt.txt"}
OUT="$ROOT/build/slot8"
MAX_KEYS=${MAX_KEYS:-2048}
mkdir -p "$OUT" "$ROOT/models"

make -C "$ROOT" build/nanlm_teacher build/nanlm_slot8_train \
    build/nanlm_asm build/nanlm_vm >/dev/null

SIZE=$(wc -c < "$CORPUS")
TRAIN_END=$((SIZE * 85 / 100))
"$ROOT/build/nanlm_teacher" build-ngram "$CORPUS" "$OUT/teacher.ntr" \
    0 "$TRAIN_END" 4 > "$OUT/teacher.log"
cat "$OUT/teacher.log"
"$ROOT/build/nanlm_slot8_train" \
    --teacher "$OUT/teacher.ntr" --corpus "$CORPUS" --max-keys "$MAX_KEYS" \
    --out "$OUT/model.nas" --expected "$OUT/model.expected" \
    --manifest "$OUT/model.manifest" > "$OUT/model.log"
cat "$OUT/model.log"

"$ROOT/build/nanlm_asm" "$OUT/model.nas" "$OUT/model.nan"
"$ROOT/build/nanlm_vm" --img "$OUT/model.nan" --steps 600000000 \
    > "$OUT/model.out" 2> "$OUT/model.err" || true
if ! grep -q "rc=3, .*trap=''" "$OUT/model.err"; then
    echo "slot8: run did not halt cleanly" >&2
    cat "$OUT/model.err" >&2
    exit 1
fi
diff -u "$OUT/model.expected" "$OUT/model.out"
echo "slot8: PASS ($(wc -l < "$OUT/model.out") queries, $(grep -o '[0-9]* steps' "$OUT/model.err" | tail -1))"
