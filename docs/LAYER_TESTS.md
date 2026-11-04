# Layer-by-layer test plan

One word, many algebras. Each layer gets a different use of NaN, a different
test, and a stop condition.

## Test matrix

| layer | best use of NaN | test | metric | status |
|---|---|---|---|---|
| **B** bit-sliced | `n` as truth table, 32 lanes | all 16 functions; regime vs fixed | exact lanes; accuracy | all-16 static + dynamic done |
| **N** nibble | 8 × 4-bit SWAR | nibble add/max with carry guard | exact lanes; instructions | planned |
| **Y** byte | 4 × 8-bit SWAR | byte saturating add / compare | exact lanes; instructions | planned |
| **H** halfword | 2 × 16-bit counters | halfword add/carry, 16-bit scale | exact; steps | planned |
| **S** scalar | 1 × 32-bit accumulator | dot/readout, hash | exact; BPC | in use |
| **V** hypervector | bind/bundle/Hamming | D sweep; BSWAPM binding | top-1/BPC; steps | slot memory done |
| **M** memory | labels + Hamming | top-k vote; teacher distillation | top-1/BPC; size | k=1 done |
| **R** regime | `n` as writable code | all 16 self-modified; regime vs fixed | accuracy; determinism | all-16 done |
| **X** mirror | `rot2(n)` / BSWAPM | mirror ensemble; seal | accuracy or integrity | law tested; ensemble planned |
| **stack** | heterogeneous | B→V→M→S vs homogeneous V | top-1/BPC; steps | planned |

## B — bit-sliced Boolean

### Best use

- `BOOL.n` is 32 independent Boolean lanes.
- `n` is both the truth table and the predicate.
- Use `n=15` for control, `n` for function.
- Self-modifying `n` is the regime primitive.

### Tests

1. **All 16 truth tables, static** — `make test` (word/VM) and `tests/lane.nas`.
2. **All 16 at runtime** — `tests/regime_all.nas`: rewrite the slot through
   `0..15`, check `n * 0x11111111`. **PASS, 248 steps.**
3. **Regime vs fixed** — the decisive model test:
   - task: small binary classification;
   - fixed circuit: one `BOOL.n` per layer;
   - dynamic: router selects `n` per input cluster;
   - compare accuracy at equal gate count.
4. **Majority/parity circuit** — 3-input majority from three `BOOL` gates;
   measure exactness and steps.

### Stop condition

If dynamic `n` never beats the best fixed `n` on a held-out task at equal
size, the Regime Machine is not worth the complexity.

## N — nibble

### Best use

- 8 lanes of 4 bits; small alphabets and quantized activations.
- `BOOL` is still two-input; a 4-bit LUT is built from several Boolean gates
  and masks, not one instruction.
- Use `n=6` for XOR, `n=8` for AND, `n=14` for OR inside SWAR routines.

### Tests

1. **Nibble-wise add with carry guard**:
   - split `0x0F0F0F0F`-style masks;
   - add per nibble;
   - suppress carries with masks;
   - compare to a scalar reference on random words.
2. **Nibble-wise max/min**:
   - compare per nibble via XOR/AND/OR;
   - select lanes with masks;
   - exactness gate.
3. **Nibble LUT composition**:
   - implement a 2-input 4-bit function with a small Boolean circuit;
   - compare to a table;
   - count gates.

### Stop condition

If a nibble routine needs more gates than a byte/scalar equivalent and is not
used by a layer that actually helps quality, drop it.

## Y — byte

### Best use

- 4 × 8-bit lanes for byte-level text, masks, ctype, and quantized features.
- `ADD` carries across bytes; use `BOOL` masks to detect/suppress carry.
- `BOOL.6` + popcount gives per-byte Hamming/parity.

### Tests

1. **Byte saturating add**:
   - add two vectors;
   - detect carry into the next byte;
   - clamp;
   - compare to reference on random data.
2. **Byte compare**:
   - per-byte equality via XOR + OR-reduce;
   - mask the result.
3. **Byte popcount**:
   - SWAR popcount per byte;
   - compare to scalar.

### Stop condition

If byte SWAR is slower than splitting into scalar operations for the actual
model shape, keep it only for feature layers where it improves accuracy.

## H — halfword

### Best use

- 2 × 16-bit counters, indices, scaled probabilities.
- `ADD` carry detection with masks.
- 16-bit weights for teacher distributions.

### Tests

1. **Halfword add/carry** — exact lanes, carry flag.
2. **16-bit scale** — multiply a halfword by a constant with 32-bit `MUL` and
   mask.
3. **Saturating counter** — clamp at 65535.

### Stop condition

If no model layer needs 16-bit lanes, skip.

## S — scalar

### Best use

- 32-bit accumulation, addresses, hashes, exact readout.
- `MUL`, `DIV`, `CMP` are already native.
- The scalar layer is the bridge between Boolean/HDC and the outside.

### Tests

1. **Integer readout** — ridge/perceptron scores computed in C, emitted as
   scalar ops; compare to C.
2. **Hash** — FNV over a vector; already used in model images.
3. **Dot product** — 4-byte dot via `MUL`/`ADD`, compare.

### Stop condition

None; this layer is already load-bearing.

## V — hypervector

### Best use

- `BOOL.6` bind, majority bundle, popcount similarity.
- Data words neutral `n=10`.
- `BSWAPM` is a free permutation; `rot2` is the mirror law.
- The slot memory is the current V→M stack.

### Tests

1. **D sweep** — 1,024 / 2,048 / 4,096 / 8,192 at fixed keys.
2. **Binding ablation** — XOR vs `BSWAPM` permutation vs slot concat.
3. **Bundling ablation** — majority-of-3 vs counter threshold.
4. **Capacity** — keys vs top-1/BPC.

### Stop condition

If the slot memory cannot beat the n-gram on top-1 after teacher distillation
and 8-slot context, V is not the right memory shape.

## M — memory

### Best use

- Hamming nearest neighbour over prototypes; labels; counts.
- `LDN` gives a class check; the gate catches corrupted words.
- `PROTOSHIFT` maps prototype index to class.

### Tests

1. **k=1 vs k=2/4 vote** — C first; implement in VM only if it wins.
2. **Teacher distillation** — top-k byte weights; student vs teacher.
3. **Prototype pruning** — top-K by count; accuracy vs size.
4. **Seal/capability** — reject corrupted prototypes.

### Stop condition

If top-k and distillation do not beat the n-gram, the memory is not the
bottleneck; the encoder is.

## R — regime / router

### Best use

- Compute `n` from hash/popcount/small circuit.
- `STW n` into the next `BOOL` slot; `JMPR`.
- The router is the policy; the truth table is the expert.

### Tests

1. **All 16 runtime regimes** — done (`regime_all.nas`, PASS).
2. **Regime vs fixed** — the decisive test.
3. **Router stability** — same input -> same `n`; deterministic hash.
4. **Self-modifying loop safety** — writable slot, no gate trap, bounded.

### Stop condition

If dynamic regimes do not beat fixed ones on real data, keep self-modification
as a demo, not a model.

## X — mirror

### Best use

- `BSWAPM` swaps faces and payload bytes.
- `rot2(n)` gives a complementary Boolean function.
- Neutral words survive the swap.

### Tests

1. **Mirror law** — done in `test_word` and the lane gate.
2. **Mirror ensemble** — byte-swapped prototype table; combine votes.
3. **Seal** — dual-face payload hash.
4. **Position binding** — BSWAPM as a free permutation vs XOR position codes.

### Stop condition

If the mirror ensemble does not improve accuracy, use X only for integrity.

## The heterogeneous stack

The first real stack test:

```
B (32-lane mask) -> V (bind/bundle) -> M (nearest memory) -> S (readout)
```

Compare against a homogeneous V stack of the same total word count.

- If the heterogeneous stack wins on a small task, the layer idea is real.
- If it ties, use the simpler stack.
- If it loses, the extra dispatch cost is not paying for itself.

## Priority

1. **Regime vs fixed** — highest value; decides the alien architecture.
2. **Heterogeneous B→V→M→S** — decides whether layers are more than taxonomy.
3. **Teacher distillation + 8-slot V/M** — decides whether the memory beats
   the n-gram.
4. **Nibble/byte SWAR** — only if a model layer uses it.
5. **Mirror ensemble** — cheap; decide integrity vs accuracy.
6. **Capability loader** — trust layer for volunteer shards.

## Evidence already in the repo

- `make regime-check` — all 16 self-modified truth tables, PASS, 248 steps.
- `make layers-check` — one payload as scalar, Boolean lanes, byte masks; PASS.
- `make model-slot-check` — slot memory 52.3% top-1 / 2.67 BPC vs n-gram
  55.5% / 3.13; exact runtime diff.
- `make shard-check`, `make teacher-check` — merge is `A+B == B+A`.
- Boolean reservoir: 11.15% top-1 / 7.91 BPC — failed, kept as a research note.
