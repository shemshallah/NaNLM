# Heterogeneous NaN layers

A NaN16 word is 32 payload bits. The same 32 bits can be read as:

- 32 Boolean lanes (`BOOL.n`);
- 8 nibbles;
- 4 bytes;
- 2 halfwords;
- 1 32-bit scalar (`ADD`, `MUL`, `CMP`);
- one slice of a D-bit hypervector;
- one memory key;
- one address/tag.

A model does not have to use the same interpretation in every layer. That is
the architecture idea: **a stack of heterogeneous layers over one word
format.**

## The layer descriptor

A layer is described by an 8-bit kind plus the usual regime fields:

```
kind:   B  bit-sliced Boolean (32 lanes)
        N  nibble (8 x 4-bit)
        Y  byte (4 x 8-bit)
        H  halfword (2 x 16-bit)
        S  scalar (1 x 32-bit)
        V  hypervector slice (D/32 words)
        M  memory / Hamming
        R  router / regime
        X  mirror / byte-swap
n:      4-bit truth table or predicate
route:  8-bit expert/region id
mask:   32-bit lane mask
flags:  signed, saturating, reduce, ...
```

The descriptor is control data. The model data stays neutral (`n=10`).

## Layer kinds

### B — bit-sliced Boolean

- `BOOL.n` over 32 lanes.
- Best for masks, routing, parity, majority, popcount support.
- The function is the class nibble.
- Can be self-modified at runtime (`STW` into the next slot).

### N — nibble

- 8 lanes of 4 bits.
- Implement small LUTs by composing `BOOL` functions and masks, or use a
  nibble-wise truth table built from 16-bit tables.
- Good for quantized activations, small alphabets, and lookup layers.

### Y — byte

- 4 lanes of 8 bits.
- SWAR add/compare with carry guards.
- Natural for byte-level text, token classes, and byte masks.

### H — halfword

- 2 lanes of 16 bits.
- Counters, indices, probabilities scaled to 16 bits.

### S — scalar

- 1 × 32-bit.
- `ADD`, `SUB`, `MUL`, `DIV`, `CMP`.
- Accumulators, addresses, and exact integer scores.

### V — hypervector

- A D-bit state is D/32 words.
- Bind with `BOOL.6`, bundle with majority, compare with popcount.
- This is the slot memory / HDC layer.

### M — memory

- Nearest-neighbour Hamming over prototypes.
- Labels, distributions, and top-k.
- The layer is a table, not a matrix.

### R — router / regime

- Computes a 4-bit `n` from a hash, popcount, small circuit, or learned key.
- Writes `n` into the next `BOOL` slot with `STW`.
- This is the self-modifying layer.

### X — mirror

- `BSWAPM` and `rot2(n)`.
- A free byte permutation and a complementary Boolean view.
- Use for integrity, position binding, or a dual-face ensemble.

## Why heterogeneous layers are interesting

A homogeneous transformer uses the same operation everywhere. A heterogeneous
NaN stack can use the right algebra per layer:

- **routing** is bit-sliced Boolean;
- **features** are nibble/byte quantized;
- **accumulation** is scalar;
- **memory** is Hamming;
- **selection** is a 4-bit regime;
- **integrity** is a dual-face seal.

The word format is the common carrier. The layer descriptor is the type.

This is closer to a **heterogeneous machine** than to a neural network with a
single matmul primitive.

## The self-modifying layer

The regime layer can write the next layer's truth table:

```
router -> n
STW n into BOOL slot
JMPR slot
```

`tests/regime.nas` proves this on `nanlm_vm` in 31 steps: the slot starts as
`BOOL.0`, is rewritten to `BOOL.6`, and computes XOR.

A stack can therefore change its Boolean function per token, per layer, or per
expert without branching.

## Training a heterogeneous stack

Each layer type needs a different learner:

| layer | learner |
|---|---|
| V, M | counting / HDC bundling / distillation |
| B, N | evolution, STE, or exhaustive search over 16 functions |
| Y, H, S | integer regression, counting, or a small readout |
| R | hash/popcount baseline, then a learned key table |
| X | fixed permutation or integrity check |

Do not try to backprop through the VM. Train the layer's parameters in C or a
host adapter, then emit the layer as NaN16 words and gate the compiled model.

## Test matrix

1. **Interpretation gate** — one payload, four interpretations. Done:
   `tests/layers.nas` prints PASS.
2. **Regime gate** — self-modifying truth table. Done: `tests/regime.nas`.
3. **Regime vs fixed** — same circuit size, dynamic `n` vs static `n`.
4. **Nibble layer** — 8×4-bit LUT vs 32-lane bit mask on a small task.
5. **Byte layer** — 4×8-bit SWAR add vs scalar add, with carry guards.
6. **Mirror layer** — byte-swapped inference vs normal; does the second view
   add accuracy or only integrity?
7. **Heterogeneous stack** — B → V → M → S on a byte-level task, compared to
   a homogeneous V stack.

## What not to claim

- A layer kind is not free compute; it is a different interpretation of the
  same 32 bits.
- The nibble/byte layers still need carry/mask discipline.
- The router still has to be learned or specified.
- Heterogeneity does not guarantee quality; it guarantees expressiveness.

## Current status

- `make layers-check` passes: scalar add, Boolean complement, byte mask, and
  byte OR on the same payload.
- `make regime-check` passes: self-modifying truth table.
- The slot memory is 52.3% top-1 / 2.67 BPC vs the n-gram 55.5% / 3.13.

Next: a **Regime vs fixed** classifier and a **heterogeneous B → V → M** stack.
Those two tests decide whether the layer idea is a real architecture or only a
nice taxonomy.
