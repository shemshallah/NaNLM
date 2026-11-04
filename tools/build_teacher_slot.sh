#!/bin/sh
# Teacher -> typed slot memory -> NaN16. Uses the fixture n-gram teacher to
# validate the pipeline; a Qwen/byte-level teacher can replace the .ntr file.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TMP=${TMPDIR:-"$ROOT/build/teacher_slot_tmp"}
mkdir -p "$TMP" "$ROOT/build"
make -C "$ROOT" build/nanlm_teacher build/nanlm_teacher_slot build/nanlm_asm build/nanlm_vm >/dev/null

"$ROOT/build/nanlm_teacher" build-ngram "$ROOT/data/alice_excerpt.txt" \
    "$TMP/teacher.ntr" 0 0 4 >/dev/null
"$ROOT/build/nanlm_teacher_slot" \
    --teacher "$TMP/teacher.ntr" \
    --corpus "$ROOT/data/alice_excerpt.txt" \
    --out "$TMP/teacher_slot.nas" \
    --expected "$TMP/teacher_slot.expected" \
    --manifest "$TMP/teacher_slot.manifest" > "$TMP/teacher_slot.log" 2>&1
cat "$TMP/teacher_slot.log"

"$ROOT/build/nanlm_asm" "$TMP/teacher_slot.nas" "$TMP/teacher_slot.nan"
"$ROOT/build/nanlm_vm" --img "$TMP/teacher_slot.nan" --steps 500000000 \
    > "$TMP/teacher_slot.out" 2> "$TMP/teacher_slot.err" || true
if ! grep -q "rc=3, .*trap=''" "$TMP/teacher_slot.err"; then
    echo "teacher_slot: run did not halt cleanly" >&2
    cat "$TMP/teacher_slot.err" >&2
    exit 1
fi
if ! diff -u "$TMP/teacher_slot.expected" "$TMP/teacher_slot.out"; then
    echo "teacher_slot: prediction mismatch" >&2
    exit 1
fi
echo "teacher_slot: PASS ($(wc -l < "$TMP/teacher_slot.out") queries)"
