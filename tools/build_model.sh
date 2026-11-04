#!/bin/sh
# Build the first NaNLM model, assemble it with NaNLM's own assembler, run it
# on NaNLM's own VM, and compare.
#
#   sh tools/build_model.sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ASM="$ROOT/build/nanlm_asm"
VM="$ROOT/build/nanlm_vm"

mkdir -p "$ROOT/models" "$ROOT/build"
make -C "$ROOT" build/nanlm_train build/nanlm_asm build/nanlm_vm >/dev/null

"$ROOT/build/nanlm_train" \
    --out "$ROOT/models/nanlm0.nas" \
    --expected "$ROOT/models/nanlm0.expected" \
    --manifest "$ROOT/models/nanlm0.manifest"

sh "$ROOT/tools/check_labels.sh" "$ROOT/models/nanlm0.nas"
"$ASM" "$ROOT/models/nanlm0.nas" "$ROOT/models/nanlm0.nan"

# A pass must be a clean rc=3 HALT, not just a run that printed the right bytes
# before exhausting its budget.
check_halt() {
    err=$1
    if ! grep -q "rc=3, .*trap=''" "$err"; then
        echo "build_model: run did not halt cleanly" >&2
        cat "$err" >&2
        exit 1
    fi
}

"$VM" --img "$ROOT/models/nanlm0.nan" --steps 20000000 \
    > "$ROOT/build/nanlm0.out" 2> "$ROOT/build/nanlm0.err" || true
check_halt "$ROOT/build/nanlm0.err"

if ! diff -u "$ROOT/models/nanlm0.expected" "$ROOT/build/nanlm0.out"; then
    echo "build_model: prediction mismatch" >&2
    exit 1
fi

echo "build_model: PASS ($(wc -l < "$ROOT/build/nanlm0.out") queries)"

# Negative gate: swap prototypes 0 and 1 after training. The expected labels
# are unchanged, so a correct runtime must disagree with them.
"$ROOT/build/nanlm_train" \
    --out "$ROOT/build/nanlm0_bad.nas" \
    --expected "$ROOT/build/nanlm0_bad.expected" \
    --manifest "$ROOT/build/nanlm0_bad.manifest" \
    --swap01 >/dev/null

sh "$ROOT/tools/check_labels.sh" "$ROOT/build/nanlm0_bad.nas"
"$ASM" "$ROOT/build/nanlm0_bad.nas" "$ROOT/build/nanlm0_bad.nan"

"$VM" --img "$ROOT/build/nanlm0_bad.nan" --steps 20000000 \
    > "$ROOT/build/nanlm0_bad.out" 2> "$ROOT/build/nanlm0_bad.err" || true
check_halt "$ROOT/build/nanlm0_bad.err"

if diff -q "$ROOT/build/nanlm0_bad.expected" "$ROOT/build/nanlm0_bad.out" \
    >/dev/null 2>&1; then
    echo "build_model: negative gate did not fail" >&2
    exit 1
fi

echo "build_model: NEGATIVE PASS (swapped prototypes rejected)"

# Step starvation: a truncated run must not be accepted as a pass.
"$VM" --img "$ROOT/models/nanlm0.nan" --steps 1000 \
    > "$ROOT/build/nanlm0_starved.out" 2> "$ROOT/build/nanlm0_starved.err" || true
if grep -q "rc=3" "$ROOT/build/nanlm0_starved.err"; then
    echo "build_model: starved run halted" >&2
    exit 1
fi
if diff -q "$ROOT/models/nanlm0.expected" "$ROOT/build/nanlm0_starved.out" \
    >/dev/null 2>&1; then
    echo "build_model: step-starved run passed" >&2
    exit 1
fi
echo "build_model: NEGATIVE PASS (step starvation rejected)"
