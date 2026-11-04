#!/bin/sh
# In-VM Boolean plasticity gate:
#   1. 32-lane bit-sliced counters (c0, c1, c2) updated in-VM by BOOL.8/BOOL.6
#      half-adders and thresholded in-VM to recover 0xCAFEBABE from noisy data.
#   2. Self-modifying STW truth-table search over n=0..15 inside nanlm_vm.
#   3. Negative controls: breaking the carry rule or freezing STW must fail.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT="$ROOT/build/plasticity"
mkdir -p "$OUT"
make -C "$ROOT" build/nanlm_asm build/nanlm_vm >/dev/null

"$ROOT/build/nanlm_asm" "$ROOT/tests/plasticity.nas" "$OUT/plasticity.nan"
"$ROOT/build/nanlm_vm" --img "$OUT/plasticity.nan" --steps 100000 \
    > "$OUT/plasticity.out" 2> "$OUT/plasticity.err"
grep -q "rc=3, .*trap=''" "$OUT/plasticity.err"
grep -qx PASS "$OUT/plasticity.out"
echo "plasticity: PASS (bit-sliced counter + STW regime evolution in 325 steps)"

# Negative control 1: zero out the half-adder carry (BOOL.8 -> BOOL.0)
sed 's/BOOL\.8 r5,r1,r4/BOOL.0 r5,r1,r4/' "$ROOT/tests/plasticity.nas" \
    > "$OUT/bad_carry.nas"
"$ROOT/build/nanlm_asm" "$OUT/bad_carry.nas" "$OUT/bad_carry.nan"
"$ROOT/build/nanlm_vm" --img "$OUT/bad_carry.nan" --steps 100000 \
    > "$OUT/bad_carry.out" 2> "$OUT/bad_carry.err"
if grep -qx PASS "$OUT/bad_carry.out"; then
    echo "plasticity: broken carry control unexpectedly passed" >&2
    exit 1
fi
echo "plasticity: NEGATIVE PASS (broken bit-sliced carry rejected)"

# Negative control 2: disable STW truth-table mutation (NOP out STW)
sed 's/STW r5,r6,r7/NOP/' "$ROOT/tests/plasticity.nas" \
    > "$OUT/no_stw.nas"
"$ROOT/build/nanlm_asm" "$OUT/no_stw.nas" "$OUT/no_stw.nan"
"$ROOT/build/nanlm_vm" --img "$OUT/no_stw.nan" --steps 100000 \
    > "$OUT/no_stw.out" 2> "$OUT/no_stw.err"
if grep -qx PASS "$OUT/no_stw.out"; then
    echo "plasticity: frozen STW control unexpectedly passed" >&2
    exit 1
fi
echo "plasticity: NEGATIVE PASS (frozen STW regime rejected)"
