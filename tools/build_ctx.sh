#!/bin/sh
# Build the M2a context model, assemble it with NaNLM's own assembler, run it
# on NaNLM's own VM, and compare class labels and encoded-vector hashes.
#
#   sh tools/build_ctx.sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ASM="$ROOT/build/nanlm_asm"
VM="$ROOT/build/nanlm_vm"
DIM=${DIM:-2048}
CLASSES=${CLASSES:-16}
QUERIES=${QUERIES:-16}

mkdir -p "$ROOT/models" "$ROOT/build"
make -C "$ROOT" build/nanlm_ctx_train build/nanlm_asm build/nanlm_vm >/dev/null

run_model() {
    name=$1
    shift
    "$ROOT/build/nanlm_ctx_train" \
        --dim "$DIM" --classes "$CLASSES" --queries "$QUERIES" \
        --out "$ROOT/build/$name.nas" \
        --expected "$ROOT/build/$name.expected" \
        --manifest "$ROOT/build/$name.manifest" "$@" >/dev/null
    sh "$ROOT/tools/check_labels.sh" "$ROOT/build/$name.nas"
    "$ASM" "$ROOT/build/$name.nas" "$ROOT/build/$name.nan"
    "$VM" --img "$ROOT/build/$name.nan" --steps 5000000 \
        > "$ROOT/build/$name.out" 2> "$ROOT/build/$name.err" || true
    if ! grep -q "rc=3, .*trap=''" "$ROOT/build/$name.err"; then
        echo "build_ctx: $name did not halt cleanly" >&2
        cat "$ROOT/build/$name.err" >&2
        exit 1
    fi
}

run_model positive
if ! diff -u "$ROOT/build/positive.expected" "$ROOT/build/positive.out"; then
    echo "build_ctx: positive mismatch" >&2
    exit 1
fi
echo "build_ctx: PASS ($(wc -l < "$ROOT/build/positive.out") queries)"

cp "$ROOT/build/positive.nas" "$ROOT/models/nanlm_ctx.nas"
cp "$ROOT/build/positive.nan" "$ROOT/models/nanlm_ctx.nan"
cp "$ROOT/build/positive.expected" "$ROOT/models/nanlm_ctx.expected"
cp "$ROOT/build/positive.manifest" "$ROOT/models/nanlm_ctx.manifest"

run_model zero --zero-positions
if diff -q "$ROOT/build/zero.expected" "$ROOT/build/zero.out" >/dev/null 2>&1; then
    echo "build_ctx: zero-position gate did not fail" >&2
    exit 1
fi
echo "build_ctx: NEGATIVE PASS (zero positions rejected)"

run_model swap --swap01
if diff -q "$ROOT/build/swap.expected" "$ROOT/build/swap.out" >/dev/null 2>&1; then
    echo "build_ctx: swap gate did not fail" >&2
    exit 1
fi
echo "build_ctx: NEGATIVE PASS (swapped prototypes rejected)"

# Step starvation: a truncated run must not be accepted as a pass.
"$VM" --img "$ROOT/build/positive.nan" --steps 1000 \
    > "$ROOT/build/starved.out" 2> "$ROOT/build/starved.err" || true
if grep -q "rc=3" "$ROOT/build/starved.err"; then
    echo "build_ctx: starved run halted" >&2
    exit 1
fi
if diff -q "$ROOT/build/positive.expected" "$ROOT/build/starved.out" \
    >/dev/null 2>&1; then
    echo "build_ctx: step-starved run passed" >&2
    exit 1
fi
echo "build_ctx: NEGATIVE PASS (step starvation rejected)"
