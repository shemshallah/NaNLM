# NaNLM-0: a concrete first model

**Working name:** `nanlm0`
**Status:** specification, not yet implemented.
**Goal:** prove that a high-dimensional associative model can be trained in C,
exported as NaN16 words, and run as a `.nan` image with no C model code.

## 1. Task

Byte-level next-symbol prediction on a small UTF-8/ASCII corpus.

Given bytes `x[t-2], x[t-1], x[t]`, predict `x[t+1]`.

This is a language-model-shaped task. It is not a claim to be an LLM. It is a
task with a clear baseline (order-3 byte n-gram) and a clear failure mode.

## 2. Dimensions

| parameter | value | rationale |
|---|---:|---|
| `D` | 8,192 bits | 256 words; high-dimensional but fits a phone |
| vocabulary | 256 | byte-level, no tokenizer |
| context | 3 | one bind per position, one majority bundle |
| prototypes | 256 | one per next byte |
| codebook | 256 × 256 words | 65,536 words = 512 KB |
| position vectors | 3 × 256 words | 768 words = 6 KB |
| prototype memory | 256 × 256 words | 65,536 words = 512 KB |
| total weights | ~132k words | ~1.05 MB |

A 1 MB model is deliberately small. It should fit in the C core's 4M-word
space with room for code, stack, and buffers. If the model needs more capacity,
increase `D` or add experts; do not hide the cost.

## 3. Word conventions

- **All model data words are neutral (`n=10`, both heads `+q`).** This is
  enforced by the emitter and checked by the loader.
- **All `BOOL` instructions use `n` as the truth table.** The model's logic is
  in the program, not in the data words.
- **All control instructions use `n=15` (ALWAYS)** unless they are deliberately
  predicated.
- **No floating-point ops.** Only `BOOL`, `ADD`, `SUB`, `SHL`, `SHR`, `CMP`,
  `LD`, `ST`, `JMP`, `JMPR`, `CALL`, `RET`, `HALT`, `BCOPY`, `BFILL`, `BSWAPM`.

## 4. Encoding

Let `C[b]` be the 256-word codevector for byte `b`.
Let `P[i]` be the 256-word position vector for position `i` in `{0,1,2}`.

For a context `(a,b,c)`:

```
v0 = C[a] XOR P[0]        // BOOL.6
v1 = C[b] XOR P[1]        // BOOL.6
v2 = C[c] XOR P[2]        // BOOL.6
q  = majority(v0, v1, v2) // (v0&v1)|(v0&v2)|(v1&v2)
```

`q` is the query hypervector. It is neutral data.

### 4.1 Why XOR for binding

XOR is self-inverse and preserves Hamming structure:

```
Hamming(a XOR b, a XOR c) = Hamming(b, c)
```

That makes unbinding and similarity analysis clean. The other 15 functions are
available for experiments, but they are not the default.

### 4.2 Why majority for bundling

Majority-of-three is exact, cheap, and order-independent for three inputs. For
more than three, use a tree of majorities or a bit-sliced counter. Do not use
XOR for bundling; XOR cancels repeated symbols.

## 5. Memory and prediction

For each class `c` (next byte), store a prototype `M[c]` of `D` bits.

Prediction:

```
best = 0
best_dist = D + 1
for c in 0..255:
    d = popcount(q XOR M[c])
    if d < best_dist:
        best_dist = d
        best = c
return best
```

Ties go to the lowest class index, deterministically.

### 5.1 Training in the C adapter

For each training position `t`:

1. Build `q` as above.
2. For each bit `j` of `q`, increment `ones[c][j]` if `q[j] == 1`, and
   `total[c]`.
3. After the pass, set `M[c][j] = 1` iff `ones[c][j] * 2 > total[c]`.

This is an integer-only prototype learner. It is order-independent and
requires no gradients. It can be run for multiple passes; counts accumulate.

For class imbalance, cap `total[c]` or normalize the threshold. The first
version should keep it simple and report the imbalance.

### 5.2 Alternative: exemplar memory

Instead of one prototype per class, store `K` context vectors per class and
take the minimum distance over all exemplars. This is more accurate but larger
and slower. It is a later ablation, not v0.

## 6. The NaN16 runtime

### 6.1 Memory map (suggested)

| range | use |
|---|---|
| `0x00000..0x00FFF` | code and constants |
| `0x01000..0x10FFF` | codebook (65,536 words) |
| `0x11000..0x11FFF` | position vectors (768 words) |
| `0x12000..0x21FFF` | prototypes (65,536 words) |
| `0x22000..0x22FFF` | query, temporaries, stack |
| `0x23000..` | free |

The exact addresses are linker/assembler choices. The model manifest records
them so the runtime and the verifier agree.

### 6.2 Registers and ABI

Use the NaNLM register ABI:

- `r0 = 0`
- `r13 = 1`
- `r14 = SP` (software)
- `r15 = link`
- `r1..r12` temporaries
- arguments in `r1..r5`, results in `r1/r2`

Functions:

```
vec_xor(dst, a, b)          // 256 words, BOOL.6
vec_majority(dst, a, b, c)  // 256 words
vec_popcount_xor(a, b)      // returns 32-bit count for one word
vec_hamming(a, b)           // sum over 256 words
vec_copy(dst, src)          // BCOPY or loop
model_encode(q, a, b, c)    // codebook lookup + bind + bundle
model_predict(q)            // loop over 256 prototypes
```

### 6.3 Inner loop

For each prototype word `w`:

```
BOOL.6 t, q, w          ; t = q XOR w
; popcount t (SWAR)
```

The SWAR popcount uses `SHR`, `BOOL.8`, `SUB`, `ADD`, and constants. It is
about 10 instructions per word. A 256-word vector × 256 prototypes is 65,536
word comparisons; at ~10 instructions each, ~655k instructions per prediction.
That is acceptable for a demo but not fast. Measure it.

### 6.4 Optimization order

1. Correct scalar loop.
2. Unroll by 4 words.
3. Use `BCOPY`/`BFILL` for copies and zeroing.
4. Precompute the query once per token.
5. Use early termination only if it preserves exact tie-breaking.
6. Consider a 16-bit or 8-bit approximation only after exact baseline.

Do not optimize before the exact model matches the C adapter.

## 7. Export format

The C adapter emits:

- `models/nanlm0.nas` — source of record, human-readable, `nanlm_asm`-compatible.
- `models/nanlm0.nan` — assembled words, `NNNN_PPPPPPPP_FFFF` lines.
- `models/nanlm0.manifest` — JSON or key/value:
  - `magic = NANLM0`
  - `version = 1`
  - `D = 8192`
  - `vocab = 256`
  - `context = 3`
  - offsets for codebook, positions, prototypes, code
  - seal A, B, C
  - corpus hash and training parameters

The `.nas` file should contain:

- a header comment with the manifest;
- `.org` and labels;
- `.word` data for codebook/positions/prototypes (or a compact generator if
  the data is too large);
- the runtime code;
- a `.str` prompt/usage string for the CLI demo.

The `.nan` file is generated by `nanlm_asm`, not by a custom binary writer. The adapter may emit `.nan` directly for speed, but the `.nas` must
assemble to the same words.

## 8. Verification gates

Every gate must be able to fail.

1. **Word gate:** `nan16_make(payload,n)` roundtrips payload and `n`; all
   heads legal.
2. **Lane gate:** for all `n`, `BOOL.n(0xCCCCCCCC,0xAAAAAAAA) ==
   n*0x11111111`.
3. **Mirror gate:** for all `n`, `BOOL.rot2(n)(a,b) == BOOL.n(~a,b)`.
4. **Data gate:** every model word has `n=10`; every code word has a legal
   head.
5. **Vector gate:** adapter `vec_xor`/`vec_majority`/`hamming` match scalar
   reference on random vectors.
6. **Model gate:** adapter prediction == `nanlm_vm` prediction for a held-out
   set, byte-for-byte.
7. **Seal gate:** flipping one payload byte changes the seal and the loader
   rejects the model.
8. **Baseline gate:** report accuracy/BPC against an order-3 n-gram, including
   the case where the baseline wins.

## 9. Baselines and metrics

- Uniform byte baseline: 8 bits/byte.
- Order-3 byte n-gram with add-k smoothing.
- NaNLM-0: bits/byte or top-1 accuracy on held-out text.
- Model size in words and bytes.
- Inference steps per token on `nanlm_vm` (and native if available).
- Training time in the C adapter.
- Memory high-water mark.

Do not report only the best number. Report the baseline, the variance across
seeds, and the corpus hash.

## 10. What would make this fail

- The HDC prototype is too weak: the n-gram wins by a large margin. That is a
  valid result; publish it and try exemplar memory or a larger `D`.
- The NaN16 runtime is too slow: 655k instructions/token may be seconds on the
  C core. If so, reduce `D`, reduce prototypes, or use a smaller demo corpus.
- The model is just a lookup table: if the prototype collapses to a class
  histogram, say so. The interesting case is when binding/bundling adds
  structure beyond the marginal distribution.
- The C adapter and the `.nan` runtime diverge: this is a bug, not a research
  result. Fix it before benchmarking.

## 11. First implementation slice

The first commit that counts:

1. `nan16_word.h` + word tests (done in this foundation).
2. `nanlm_vec` + vector tests.
3. A C trainer for a **2,048-bit, 16-class** toy problem (e.g., 4×4 glyphs or
   synthetic categories).
4. Emit `.nan`, run on `nanlm_vm`, compare.
5. Only then scale to 8,192 bits and 256 byte classes.

Starting at 8,192 bits and 256 classes is a good target but a bad first debug
loop. Scale after the pipeline is exact.

## 12. Open decisions

- Should the codebook be random or learned? Start random; learned codebooks are
  an ablation.
- Should position be XOR with a vector or `BSWAPM`? Start XOR; test `BSWAPM` as
  a free permutation later.
- Should prototypes be one per class or exemplars? Start one per class.
- Should the router be a branch table or self-modifying `BOOL`? Start branch
  table.
- Should the model image include the training corpus? No; include its hash.
- Should the runtime be a standalone `.nan` or an optional cartridge? Standalone
  first; cartridge after it works.
