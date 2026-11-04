#!/bin/sh
# Signed shard group gate: commutativity, exact unlearning, and inverse.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CORPUS=${CORPUS:-"$ROOT/data/alice_excerpt.txt"}
OUT="$ROOT/build/shard2"
mkdir -p "$OUT"

make -C "$ROOT" build/nanlm_shard2 >/dev/null

SIZE=$(wc -c < "$CORPUS")
MID=$((SIZE / 2))
A="$OUT/a.shard2"
B="$OUT/b.shard2"
"$ROOT/build/nanlm_shard2" build "$CORPUS" "$A" 0 "$MID" >/dev/null
"$ROOT/build/nanlm_shard2" build "$CORPUS" "$B" "$MID" "$SIZE" >/dev/null

"$ROOT/build/nanlm_shard2" merge "$OUT/ab.shard2" "$A" "$B" >/dev/null
"$ROOT/build/nanlm_shard2" merge "$OUT/ba.shard2" "$B" "$A" >/dev/null
cmp "$OUT/ab.shard2" "$OUT/ba.shard2"
echo "shard2: A+B == B+A PASS"

"$ROOT/build/nanlm_shard2" negate "$OUT/neg_b.shard2" "$B"
"$ROOT/build/nanlm_shard2" merge "$OUT/zero.shard2" "$B" "$OUT/neg_b.shard2" >/dev/null
"$ROOT/build/nanlm_shard2" stats "$OUT/zero.shard2"
RECORDS=$("$ROOT/build/nanlm_shard2" stats "$OUT/zero.shard2" | sed 's/.*records=\([0-9]*\).*/\1/')
if [ "$RECORDS" != "0" ]; then
    echo "shard2: B + (-B) did not cancel" >&2
    exit 1
fi
echo "shard2: B + (-B) == empty PASS"

"$ROOT/build/nanlm_shard2" merge "$OUT/a_minus_b.shard2" "$OUT/ab.shard2" \
    "$OUT/neg_b.shard2" >/dev/null
cmp "$A" "$OUT/a_minus_b.shard2"
echo "shard2: (A+B)+(-B) == A PASS (exact unlearning)"
"$ROOT/build/nanlm_shard2" stats "$A"
