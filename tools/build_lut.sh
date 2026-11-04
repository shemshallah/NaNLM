#!/bin/sh
# Compile a learned routed LUT tree to NaN16, run it on nanlm_vm, and compare
# all 256 outputs.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TMP=${TMPDIR:-"$ROOT/build/lut_tmp"}
mkdir -p "$TMP" "$ROOT/build"
make -C "$ROOT" build/nanlm_asm build/nanlm_vm >/dev/null

python3 "$ROOT/experiments/export_lut.py" \
    --routed --seed 2 \
    --out "$TMP/lut_regime.nas" \
    --expected "$TMP/lut_regime.expected" >/dev/null

"$ROOT/build/nanlm_asm" "$TMP/lut_regime.nas" "$TMP/lut_regime.nan"
"$ROOT/build/nanlm_vm" --img "$TMP/lut_regime.nan" --steps 1000000 \
    > "$TMP/lut_regime.out" 2> "$TMP/lut_regime.err" || true
if ! grep -q "rc=3, .*trap=''" "$TMP/lut_regime.err"; then
    echo "lut_gate: run did not halt cleanly" >&2
    cat "$TMP/lut_regime.err" >&2
    exit 1
fi
if ! diff -u "$TMP/lut_regime.expected" "$TMP/lut_regime.out"; then
    echo "lut_gate: output mismatch" >&2
    exit 1
fi
echo "lut_gate: PASS (256 routed LUT outputs, $(wc -l < "$TMP/lut_regime.out") lines)"
