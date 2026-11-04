#!/bin/sh
# Build the simplex slot memory (one prototype per observed 3-gram), assemble
# it, run it on nanlm_vm, and compare predicted bytes + query hashes.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ASM="$ROOT/build/nanlm_asm"
VM="$ROOT/build/nanlm_vm"
CORPUS=${CORPUS:-"$ROOT/data/alice_excerpt.txt"}
QUERIES=${QUERIES:-32}

mkdir -p "$ROOT/models" "$ROOT/build"
make -C "$ROOT" build/nanlm_slot_train build/nanlm_asm build/nanlm_vm >/dev/null

"$ROOT/build/nanlm_slot_train" \
    --corpus "$CORPUS" --queries "$QUERIES" \
    --out "$ROOT/models/nanlm_slot.nas" \
    --expected "$ROOT/models/nanlm_slot.expected" \
    --manifest "$ROOT/models/nanlm_slot.manifest" \
    > "$ROOT/build/slot.log" 2>&1

cat "$ROOT/build/slot.log"
sh "$ROOT/tools/check_labels.sh" "$ROOT/models/nanlm_slot.nas"
"$ASM" "$ROOT/models/nanlm_slot.nas" "$ROOT/models/nanlm_slot.nan"

"$VM" --img "$ROOT/models/nanlm_slot.nan" --steps 300000000 \
    > "$ROOT/build/slot.out" 2> "$ROOT/build/slot.err" || true
if ! grep -q "rc=3, .*trap=''" "$ROOT/build/slot.err"; then
    echo "build_slot: run did not halt cleanly" >&2
    cat "$ROOT/build/slot.err" >&2
    exit 1
fi
if ! diff -u "$ROOT/models/nanlm_slot.expected" "$ROOT/build/slot.out"; then
    echo "build_slot: prediction mismatch" >&2
    exit 1
fi
echo "build_slot: PASS ($(wc -l < "$ROOT/build/slot.out") queries)"

# Step starvation must not pass.
"$VM" --img "$ROOT/models/nanlm_slot.nan" --steps 1000 \
    > "$ROOT/build/slot_starved.out" 2> "$ROOT/build/slot_starved.err" || true
if grep -q "rc=3" "$ROOT/build/slot_starved.err"; then
    echo "build_slot: starved run halted" >&2
    exit 1
fi
if diff -q "$ROOT/models/nanlm_slot.expected" "$ROOT/build/slot_starved.out" >/dev/null 2>&1; then
    echo "build_slot: step-starved run passed" >&2
    exit 1
fi
echo "build_slot: NEGATIVE PASS (step starvation rejected)"
