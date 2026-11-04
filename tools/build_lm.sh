#!/bin/sh
# Build the byte-level NaNLM language-model base, assemble it, run it on
# NaNLM's own VM, and compare the C predictions + query hashes.
#
#   sh tools/build_lm.sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ASM="$ROOT/build/nanlm_asm"
VM="$ROOT/build/nanlm_vm"
CORPUS=${CORPUS:-"$ROOT/data/alice_excerpt.txt"}
DIM=${DIM:-2048}
QUERIES=${QUERIES:-32}

mkdir -p "$ROOT/models" "$ROOT/build"
make -C "$ROOT" build/nanlm_lm_train build/nanlm_asm build/nanlm_vm >/dev/null

run_model() {
    name=$1
    shift
    "$ROOT/build/nanlm_lm_train" \
        --corpus "$CORPUS" --dim "$DIM" --queries "$QUERIES" \
        --out "$ROOT/build/$name.nas" \
        --expected "$ROOT/build/$name.expected" \
        --manifest "$ROOT/build/$name.manifest" "$@" \
        > "$ROOT/build/$name.log" 2>&1
    sh "$ROOT/tools/check_labels.sh" "$ROOT/build/$name.nas"
    "$ASM" "$ROOT/build/$name.nas" "$ROOT/build/$name.nan"
    "$VM" --img "$ROOT/build/$name.nan" --steps 150000000 \
        > "$ROOT/build/$name.out" 2> "$ROOT/build/$name.err" || true
    if ! grep -q "rc=3, .*trap=''" "$ROOT/build/$name.err"; then
        echo "build_lm: $name did not halt cleanly" >&2
        cat "$ROOT/build/$name.err" >&2
        exit 1
    fi
}

run_model positive
if ! diff -u "$ROOT/build/positive.expected" "$ROOT/build/positive.out"; then
    echo "build_lm: positive mismatch" >&2
    exit 1
fi
cat "$ROOT/build/positive.log"
echo "build_lm: PASS ($(wc -l < "$ROOT/build/positive.out") queries)"

cp "$ROOT/build/positive.nas" "$ROOT/models/nanlm_lm.nas"
cp "$ROOT/build/positive.nan" "$ROOT/models/nanlm_lm.nan"
cp "$ROOT/build/positive.expected" "$ROOT/models/nanlm_lm.expected"
cp "$ROOT/build/positive.manifest" "$ROOT/models/nanlm_lm.manifest"

run_model zero --zero-positions
if diff -q "$ROOT/build/zero.expected" "$ROOT/build/zero.out" >/dev/null 2>&1; then
    echo "build_lm: zero-position gate did not fail" >&2
    exit 1
fi
echo "build_lm: NEGATIVE PASS (zero positions rejected)"

run_model swap --swap-predicted
if diff -q "$ROOT/build/swap.expected" "$ROOT/build/swap.out" >/dev/null 2>&1; then
    echo "build_lm: predicted-prototype swap did not fail" >&2
    exit 1
fi
echo "build_lm: NEGATIVE PASS (predicted prototypes rejected)"

# Step starvation: a truncated run must not be accepted as a pass.
"$VM" --img "$ROOT/build/positive.nan" --steps 1000 \
    > "$ROOT/build/starved.out" 2> "$ROOT/build/starved.err" || true
if grep -q "rc=3" "$ROOT/build/starved.err"; then
    echo "build_lm: starved run halted" >&2
    exit 1
fi
if diff -q "$ROOT/build/positive.expected" "$ROOT/build/starved.out" \
    >/dev/null 2>&1; then
    echo "build_lm: step-starved run passed" >&2
    exit 1
fi
echo "build_lm: NEGATIVE PASS (step starvation rejected)"
