#!/bin/sh
# Shard gate: build two corpus slices, merge them in both orders, and require
# byte-identical output. This is the CRDT property that makes volunteer
# training safe: merge is commutative and deterministic.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CORPUS=${CORPUS:-"$ROOT/data/alice_excerpt.txt"}
TMP=${TMPDIR:-/data/data/com.termux/files/usr/tmp/opencode}
mkdir -p "$TMP" "$ROOT/build"
make -C "$ROOT" build/nanlm_shard >/dev/null

LEN=$(wc -c < "$CORPUS")
MID=$((LEN / 2))
A="$TMP/shard_a.bin"
B="$TMP/shard_b.bin"
AB="$TMP/shard_ab.bin"
BA="$TMP/shard_ba.bin"

"$ROOT/build/nanlm_shard" build "$CORPUS" "$A" 0 "$MID" >/dev/null
"$ROOT/build/nanlm_shard" build "$CORPUS" "$B" "$MID" "$LEN" >/dev/null
"$ROOT/build/nanlm_shard" merge "$AB" "$A" "$B" >/dev/null
"$ROOT/build/nanlm_shard" merge "$BA" "$B" "$A" >/dev/null

if ! cmp -s "$AB" "$BA"; then
    echo "shard_gate: merge order changed output" >&2
    exit 1
fi
echo "shard_gate: PASS (A+B == B+A, byte-identical)"

# A single shard split in two must have the same total count as the concatenation
# of the two totals, unless a count saturates at 255. Alice has no such key.
TA=$("$ROOT/build/nanlm_shard" stats "$A" | sed 's/.*total=//')
TB=$("$ROOT/build/nanlm_shard" stats "$B" | sed 's/.*total=//')
TAB=$("$ROOT/build/nanlm_shard" stats "$AB" | sed 's/.*total=//')
if [ "$TAB" -ne $((TA + TB)) ]; then
    echo "shard_gate: merged total $TAB != $TA + $TB" >&2
    exit 1
fi
echo "shard_gate: PASS (total $TA + $TB = $TAB)"
