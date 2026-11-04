#!/bin/sh
# Prove the continuous-training bridge: raw counts in a shard compile to the
# same typed slot memory that the corpus trainer produces, and run on the VM.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CORPUS=${CORPUS:-"$ROOT/data/alice_excerpt.txt"}
OUT="$ROOT/build/shard_slot"
mkdir -p "$OUT" "$ROOT/models"

make -C "$ROOT" build/nanlm_shard build/nanlm_shard_slot build/nanlm_slot_train \
    build/nanlm_asm build/nanlm_vm >/dev/null

SIZE=$(wc -c < "$CORPUS")
TRAIN_END=$((SIZE * 85 / 100))
"$ROOT/build/nanlm_shard" build "$CORPUS" "$OUT/counts.shard" 0 "$TRAIN_END" >/dev/null
"$ROOT/build/nanlm_shard_slot" \
    --shard "$OUT/counts.shard" --corpus "$CORPUS" \
    --out "$OUT/from_shard.nas" --expected "$OUT/from_shard.expected" \
    --manifest "$OUT/from_shard.manifest" > "$OUT/from_shard.log"
cat "$OUT/from_shard.log"

# The direct corpus trainer must produce exactly the same prototype memory.
"$ROOT/build/nanlm_slot_train" --corpus "$CORPUS" --queries 32 \
    --out "$OUT/from_corpus.nas" --expected "$OUT/from_corpus.expected" \
    --manifest "$OUT/from_corpus.manifest" > "$OUT/from_corpus.log"
cmp "$OUT/from_shard.nas" "$OUT/from_corpus.nas"
cmp "$OUT/from_shard.expected" "$OUT/from_corpus.expected"
echo "shard_slot: corpus == shard compile PASS"

"$ROOT/build/nanlm_asm" "$OUT/from_shard.nas" "$OUT/from_shard.nan"
"$ROOT/build/nanlm_vm" --img "$OUT/from_shard.nan" --steps 300000000 \
    > "$OUT/from_shard.out" 2> "$OUT/from_shard.err" || true
grep -q "rc=3, .*trap=''" "$OUT/from_shard.err"
diff -u "$OUT/from_shard.expected" "$OUT/from_shard.out"
echo "shard_slot: VM PASS ($(wc -l < "$OUT/from_shard.out") queries)"
