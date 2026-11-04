# M2 plan: from pipeline proof to a language-shaped model

M1 proved the boundary:

```
C trainer -> .nas -> nanlm_asm -> .nan -> nanlm_vm -> diff
```

It did **not** prove the model is interesting. M1 is a synthetic prototype
classifier. The runtime does not yet bind or bundle; it only does XOR +
popcount inside Hamming. M2 fixes that in two stages.

**Status:** M2a is implemented and passing. The NaN16 runtime now performs
`v_i = C[b_i] XOR P[i]`, bundles three vectors with majority, hashes the
encoded query, and prints `CC HHHHHHHH`. The C adapter and the `.nan` runtime
agree on all 16 class/hash pairs. Zeroed positions, swapped prototypes, and
step starvation all fail the differential gate.

## M2a — force the runtime to use binding and bundling

Keep the synthetic task, but generate each class as a bundle of bound parts:

```
base[c][i] = random vector
centroid[c] = majority(
    bind(base[c][0], P[0]),
    bind(base[c][1], P[1]),
    bind(base[c][2], P[2])
)
```

Training samples are noisy copies of `centroid[c]`; queries are held-out noisy
copies. The C adapter knows the decomposition. The NaN16 runtime receives only
the codebook, the position vectors, and the query bytes, and must reconstruct
`centroid` with:

```
v_i = C[b_i] XOR P[i]        // BOOL.6
q   = majority(v_0,v_1,v_2)  // (a&b)|(a&c)|(b&c)
```

This forces the runtime to exercise the actual HDC operations. The
differential test is still exact: C encoder and NaN16 encoder must produce the
same query vector, and both classifiers must return the same class.

### M2a gates

- [x] C and NaN16 encoders produce identical query hashes for every query.
- [x] C and NaN16 classifiers agree on 100% of queries (16/16).
- [x] Zeroing the position vectors changes the query hash and fails the diff.
- [x] A runtime that skips binding cannot pass the hash gate.
- [x] Step starvation fails the diff.

## M2b — byte-level corpus

Switch the vocabulary to 256 bytes and use a real text corpus.

| parameter | M2b value |
|---|---:|
| D | 8,192 bits |
| words per vector | 256 |
| vocabulary | 256 |
| context | 3 bytes |
| codebook | 256 × 256 = 65,536 words |
| positions | 3 × 256 = 768 words |
| prototypes | 256 × 256 = 65,536 words |
| query set | 64 × 256 = 16,384 words |
| total | ~148k words = 1.18 MB |

### Training

1. Split corpus into train/validation by byte offset, not by random shuffle.
2. For each position `t` with at least 3 bytes of context:
   - encode `(x[t-2], x[t-1], x[t])` into a query vector;
   - increment bit-sliced counters for class `x[t+1]`.
3. Prototype bit = majority of counters.
4. Evaluate on held-out bytes.

This is an order-3 HDC associative memory. It is not backprop. It may lose to a
plain order-3 n-gram; that is a valid, publishable result.

### Baselines

- Uniform byte: 8 bits/byte.
- Order-1 byte frequency.
- Order-3 byte n-gram with add-k smoothing.
- HDC order-3.
- HDC order-2 and order-4 as ablations.

### Metrics

- bits/byte (primary);
- top-1 accuracy;
- model words and bytes;
- training time in the C adapter;
- inference steps per token on `nanlm_vm`;
- memory high-water mark;
- variance across at least 3 corpus splits.

### Runtime shape

```
model_encode:
  for i in 0..2:
      a = codebook + context[i] * VW
      b = positions + i * VW
      temp[i] = a XOR b
  query = majority(temp[0], temp[1], temp[2])

model_predict:
  best = D+1
  for c in 0..255:
      d = hamming(query, prototype[c])
      if d < best: best = d; best_c = c
```

The encoder needs three temp vectors of 256 words each (3 KB), plus the query
(1 KB). The hot loop is `256 × 256 = 65,536` word comparisons; at ~20
instructions each that is ~1.3M instructions per token. The C core can do it;
measure before optimizing.

### Optimization order

1. Exact scalar runtime.
2. Unroll the Hamming loop by 4 words.
3. Use `BCOPY` for vector copies.
4. Precompute codebook addresses.
5. Consider a 16-bit popcount approximation only after the exact baseline.

## M2 model image and seal

The model image needs a manifest and a seal:

- `magic NANLM0`
- `version 2`
- `dim`, `words`, `vocab`, `context`
- offsets for codebook, positions, prototypes, queries, code
- seal A: FNV-1a over near payloads
- seal B: FNV-1a over byte-swapped payloads
- seal C: rotate/XOR over class nibbles
- corpus hash and split description

The C adapter verifies the seal before emitting. The loader/runtime verifies
it before inference. A single flipped payload byte must fail the seal.

## M2 double-endian experiment

Add an ablation where position binding uses `BSWAPM` on neutral data words
instead of XOR with a position vector:

```
v_i = BSWAPM(C[b_i]) XOR mask_i
```

Because `n=10` is neutral, `BSWAPM` preserves the nibble and reverses the
payload bytes. This is a free fixed permutation. Compare:

- XOR with learned/random position vectors;
- `BSWAPM` byte reversal;
- XOR + `BSWAPM`;
- rotate by a per-position amount.

The result may be worse. Publish it either way.

## M2 negative gates

A gate that cannot fail is not a check. M2 must include:

1. **Shuffled labels:** query classes are not emitted in class order, so a
   runtime that returns `query_index % classes` fails.
2. **Swapped prototypes:** swap prototypes 0 and 1 after training; the
   differential test must fail.
3. **Broken encoder:** an encoder that ignores positions must produce a
   different query and fail the exact-vector gate.
4. **Seal corruption:** flip one payload byte; the loader must reject.
5. **Step starvation:** run with too few steps; the harness must report a
   failure, not a pass.

## What M2 does not do

- No backprop in the NaN machine.
- No transformer parity claim.
- No dynamic expert routing yet (M3).
- No self-modifying `BOOL` instructions yet (M3).
- No C-to-NaN compiler.
- No new word format.

## Exit criteria

M2 is done when:

- the NaN16 runtime actually performs binding and bundling;
- it produces the same query vectors and predictions as the C adapter;
- a byte-level model runs on a held-out corpus;
- the n-gram baseline is reported alongside it;
- at least one negative gate fails as expected;
- the model image is sealed and a corrupted byte is rejected.
