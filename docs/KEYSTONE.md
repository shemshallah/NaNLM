# Keystone and supports

NaNLM is organized around a small **keystone** that must stay exact, and a set
of **supports** that are allowed to change.

## The keystone

The keystone is the part that defines the machine:

1. **NaN16 word** — two legal NaN heads around a 32-bit payload; class nibble
   `n`; both-face gate; `BSWAPM` mirror law.
2. **`nanlm_asm`** — assembles the generated `.nas` subset into `.nan` words.
3. **`nanlm_vm`** — executes the subset, gates every fetched word, traps on a
   bad head or opcode, and returns `rc=3` only on a clean HALT.
4. **Model image contract** — `.nas` source, `.nan` image, manifest, expected
   output, query hash, and a build script that must fail on mismatch.
5. **Gates** — lane identity, mirror law, word packing, vector reference,
   clean-HALT requirement, negative gates, step starvation.

If a change touches the keystone, it needs a new gate and a version bump. The
keystone is deliberately boring; the model is where the experiments live.

```
                 +--------------------------------------+
                 | keystone                             |
                 |  word -> asm -> vm -> image -> gate  |
                 +------------------+-------------------+
                                    |
        +---------------------------+---------------------------+
        |               supports (replaceable)                  |
        |  corpus  binder  bundler  memory  router  decoder     |
        |  trainer  verifier  tokenizer  experts  web demo      |
        +-------------------------------------------------------+
```

## Support contracts

Each support has a narrow contract. A support may be replaced without changing
the keystone.

### Corpus / tokenizer

Input: bytes or symbols.

Contract:

- deterministic split by offset;
- no hidden cleaning;
- a provenance file;
- a held-out validation slice.

Current: raw bytes from `data/alice_excerpt.txt`, 85/15 split.

Next: byte-level is the baseline; BPE/word tokens are a support experiment, not
a keystone change.

### Encoder / binder

Input: one symbol vector and one position vector.

Contract:

- `nanlm_vec_bind` (XOR, `BOOL.6`);
- output is neutral `n=10` data;
- C encoder and NaN16 encoder must produce the same query hash.

Next supports:

- `BSWAPM` byte-reversal binding;
- learned `BOOL.n` binding (test invertibility);
- random projection from a continuous embedding.

### Bundler

Input: three bound vectors.

Contract:

- majority `(a&b)|(a&c)|(b&c)`;
- neutral output;
- C and NaN16 encoders agree.

Next: weighted bundle, bit-sliced counter, tree of majorities.

### Memory

Input: a query vector.

Contract:

- a fixed number of neutral prototypes;
- deterministic tie-breaking by lowest prototype index;
- `PROTOSHIFT` maps prototype index to class.

Current: `vocab * clusters` majority prototypes, 8 per byte. An optional
exemplar mode (`--exemplar`, reservoir sampling) is a negative result in the
research log.

Next supports:

- exemplar memory (store contexts, not class centroids);
- k-means clusters instead of hash buckets;
- a sparse expert memory with a branch-table router.

### Router

Current: a hash of the query selects one of 8 prototype buckets per byte. The
hash is training-only; inference searches all prototypes.

Next supports:

- `BSWAPM`/popcount signature;
- a small learned key table;
- a branch-table expert router;
- a self-modifying `BOOL.n` regime (later).

### Decoder

Current: nearest Hamming prototype, class = `prototype_index >> 3`.

Next supports:

- top-k prototype vote;
- calibrated softmax for bits/byte;
- a codebook lookup decoder;
- a conventional readout trained in the C adapter and emitted as a circuit.

### Trainer

Current: integer bit-sliced counters, no gradients.

Next supports:

- Tsetlin clauses;
- evolutionary circuit search;
- straight-through/Gumbel training in the adapter, then harden to `BOOL.n`;
- distillation from a conventional model.

### Verifier

Current: C expected predictions + query-vector hash; negative gates for zeroed
positions, swapped prototypes, and step starvation.

Next supports:

- FNV A/B/C seal over near payloads, byte-swapped payloads, and class nibbles;
- a `.nan` loader that rejects a corrupted model before execution;
- a model card with corpus hash, split, metrics, and known failures.

## What is keystone, what is support

| item | keystone? | reason |
|---|---|---|
| word layout / gate | yes | every model depends on it |
| `BOOL.n` truth table | yes | the primitive |
| assembler op subset | yes | `.nan` must run |
| VM status codes | yes | gates depend on `rc=3` |
| XOR binder | no | replaceable support |
| majority bundler | no | replaceable support |
| 8 clusters/byte | no | experiment |
| Alice corpus | no | fixture |
| nearest-Hamming decoder | no | experiment |
| n-gram baseline | no | comparison |
| seal format | planned | support that will become keystone when shipped |

## Current support mapping

| support | file |
|---|---|
| word | `include/nan16_word.h` |
| vector ops | `include/nanlm_vec.h`, `src/nanlm_vec.c` |
| assembler | `src/nanlm_asm.c`, `tools/nanlm_asm.c` |
| VM | `src/nanlm_vm.c`, `tools/nanlm_vm.c` |
| M1 trainer | `src/nanlm_model.c`, `tools/nanlm_train.c` |
| M2a context | `src/nanlm_ctx.c`, `tools/nanlm_ctx_train.c` |
| M2b LM | `src/nanlm_lm.c`, `tools/nanlm_lm_train.c` |
| runtime template | `src/nanlm_ctx.c` (`nanlm_runtime_source`) |
| corpus | `data/alice_excerpt.txt` |
| gates | `tools/build_model.sh`, `build_ctx.sh`, `build_lm.sh` |

## Design rules for new supports

1. **No new word format.** A support may not change the 64-bit word.
2. **No floats in the artifact.** Host diagnostics may use doubles; the model
   image is integer/Boolean.
3. **No hidden runtime.** A support that only works in C is a diagnostic, not
   a release.
4. **Hash the encoded vector.** Labels alone can be produced by a decoder that
   skips the encoder.
5. **Add a negative gate.** If the support cannot fail, it is not tested.
6. **Measure against the n-gram.** A new support must beat 50% top-1 or explain
   why it is worth its cost anyway.

## Next three supports

1. **Top-k decoding.** Nearest-1 leaves information on the table. A k-nearest
   class vote is the next decoder experiment; it can keep the same prototypes.
2. **Context-aware memory.** Hash buckets beat k-means and reservoir exemplars.
   A context-aware hash or a learned partition is the next memory experiment.
3. **Seal/loader.** FNV A/B/C over the model words, a `.nan` loader that
   rejects corruption, and a model card that names the failure modes.

Calibrated HDC decoding is implemented: class-level log-sum-exp with a
temperature selected on a disjoint 128-context slice. The model still loses to
the n-gram, and the card says so.

Only after those should the project try a router, learned `BOOL.n` circuits, or
a larger D.
