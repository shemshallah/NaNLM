#!/bin/sh
# Teacher shard gate: build two teacher shards, merge both orders, require
# byte-identical output. Teacher records are additive, like corpus shards.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CORPUS=${CORPUS:-"$ROOT/data/alice_excerpt.txt"}
TMP=${TMPDIR:-"$ROOT/build/teacher_tmp"}
mkdir -p "$TMP" "$ROOT/build"
make -C "$ROOT" build/nanlm_teacher >/dev/null

LEN=$(wc -c < "$CORPUS")
MID=$((LEN / 2))
A="$TMP/teacher_a.ntr"
B="$TMP/teacher_b.ntr"
AB="$TMP/teacher_ab.ntr"
BA="$TMP/teacher_ba.ntr"

"$ROOT/build/nanlm_teacher" build-ngram "$CORPUS" "$A" 0 "$MID" 4 >/dev/null
"$ROOT/build/nanlm_teacher" build-ngram "$CORPUS" "$B" "$MID" "$LEN" 4 >/dev/null
"$ROOT/build/nanlm_teacher" merge "$AB" "$A" "$B" >/dev/null
"$ROOT/build/nanlm_teacher" merge "$BA" "$B" "$A" >/dev/null

if ! cmp -s "$AB" "$BA"; then
    echo "teacher_gate: merge order changed output" >&2
    exit 1
fi
echo "teacher_gate: PASS (A+B == B+A, byte-identical)"

"$ROOT/build/nanlm_teacher" stats "$A"
"$ROOT/build/nanlm_teacher" stats "$AB"
