# NaNLM foundations

**Status:** design v0.1, after studying NaN/0S.

## 0. The thesis in one paragraph

NaNLM is a high-dimensional Boolean model whose atoms are NaN16 words. A word
is not a floating-point number. It is a 64-bit object with two legal NaN heads
and a 32-bit payload. The two heads produce a four-bit class nibble `n`; for
`BOOL`, `n` is a truth table applied independently to 32 bit lanes. A model is
a sparse population of short Boolean circuits over a very high-dimensional
binary state, with binding and bundling for structure, Hamming similarity for
memory, and integer-only local learning. It is not a smaller transformer. It
is a different substrate, chosen because it is inspectable, gated, portable,
and cheap at the edge.

## 1. Why the NaN framing is load-bearing

A critic can say: "You are just using 64-bit integers with two tag words. Why
call it NaN?" There are four honest answers.

1. **The tag is the computation.** In NaN16 the class nibble is not metadata
   sitting beside the payload. For `BOOL` it *is* the truth table. The
   instruction's tag selects the function. The same nibble is the predicate
   for every other op. A normal tagged word cannot do that without a decode
   step.
2. **The gate is a safety property.** Both heads must be legal before a word
   executes. A corrupted head traps `!G`; it does not silently become another
   instruction. The double-ended seal extends this to payload integrity. That
   is useful for model weights, which are otherwise a large uninterpreted
   binary blob.
3. **The format is inert to accidental arithmetic.** If a NaN16 word is
   accidentally fed to a floating-point operation, it stays NaN. The bug is
   loud instead of becoming a silent numeric drift. This is a real property
   for a model whose whole point is to avoid FP.
4. **The double endianness is a second view.** `BSWAPM` swaps the faces and
   transforms `n` by `rot2`. Neutral words (`n=0,5,10,15`) read the same from
   both ends. That gives a free fixed permutation of the payload, a mirror law,
   and an integrity channel. It is not a second ALU, but it is a real
   representation-level capability.

If a future design uses NaN16 only as a storage container and does all work in
C, the framing has failed. The test is simple: **can the model run as a `.nan`
image with no C model code?** If not, it is a C project wearing a NaN costume.

## 2. The four planes

NaNLM is easier to reason about as four planes with hard boundaries.

```
+-------------------------------------------------------------+
| 4. Model plane       codebook, hypervectors, experts, memory |
|                      .nas source, .nan image, manifest       |
+-------------------------------------------------------------+
| 3. Model compiler    C adapter: train, quantize, pack, emit  |
|                      S6 test bench, never shipped runtime    |
+-------------------------------------------------------------+
| 2. NaN16 machine     BOOL, ADD, LD/ST, BSWAPM, JMP, ...      |
|                      nanlm_vm / future native cores             |
+-------------------------------------------------------------+
| 1. Word plane        two NaN heads + 32-bit payload          |
|                      gate, class nibble, double endianness   |
+-------------------------------------------------------------+
```

The word plane is studied from NaN/0S. NaNLM owns the machine plane
(`nanlm_asm`, `nanlm_vm`), the model plane, and the model compiler. The
compiler is allowed to be C because it is S6. The model plane is not.

## 3. What a "high-dimensional" state is

A vector of dimension `D` bits is stored as `D/32` NaN16 words. Each word is a
neutral data word (`n=10`) whose 32-bit payload is one slice.

| D | words | bytes |
|---:|---:|---:|
| 1,024 | 32 | 256 |
| 2,048 | 64 | 512 |
| 4,096 | 128 | 1,024 |
| 8,192 | 256 | 2,048 |
| 16,384 | 512 | 4,096 |
| 65,536 | 2,048 | 16,384 |

This is compact. A 65,536-bit hypervector is 16 KB. A model with 1,000 such
prototypes is 16 MB, which fits in the 32 MB C core and comfortably in the
4M-word (32 MB) default. The constraint is not memory; it is **throughput**:
every operation is word-by-word, and every similarity is a popcount over the
whole vector.

The word is 32 lanes, not 64. That is the price of the 32-bit payload. A
`BOOL` instruction still does 32 useful bit operations per word, which is the
right unit for a bit-sliced model.

## 4. Data words vs regime words

The most important design decision is to keep three roles separate even though
the hardware can fuse them.

### 4.1 Data

Model data is always `n=10` (both heads `+q`). This is the neutral nibble. It
means:

- the word is not a predicate;
- `BSWAPM` preserves `n=10`;
- the word reads the same from both ends at the head level;
- the payload is the only varying part.

All codebook vectors, prototypes, masks, and intermediate vectors are `n=10`.
This is not a convention we can skip: if data words carry arbitrary `n`, then
`BSWAPM`, `LDN`, and any code that inspects the class will see noise.

### 4.2 Function

A `BOOL` instruction carries the truth table in its own `n`. The operands can
be any valid words; `BOOL` only reads their payloads. So the function is a
property of the **operation**, not the data.

This is cleaner than the original "n is stored in every weight word" idea. It
also means a model's learned truth tables are encoded in its **program**, not
in its weights. A `BOOL.6` is XOR; a `BOOL.8` is AND; a `BOOL.14` is OR. The
program is the model's logic.

### 4.3 Route / predicate

Control instructions use `n=15` (ALWAYS) by default. To predicate, use
`n` in `{1,2,3,4,5,6}` after a `CMP`. The predicate is a four-bit table over
`(Z,L)`, not a general condition. This is enough for loops and branches, but it
is not a rich control-flow mechanism.

A router can be implemented in two clean ways:

1. **Branch table**: compute a 4-bit route, use it to index a table of `JMPR`
   targets. Each target is a short circuit with its own `BOOL.n` sequence.
2. **Self-modifying `BOOL`**: write the chosen `n` into the `n` field of a
   `BOOL` instruction word with `STW`, then execute it. This is powerful but
   requires writable code and careful sealing. Use it only after the branch
   table works.

Do not conflate "the router's n" with "the data's n". The router is a control
decision. The data is neutral. The function is an instruction property.

## 5. Binding, bundling, similarity

The model's primitive operations map directly to NaN16:

| operation | NaN16 | notes |
|---|---|---|
| bind | `BOOL.6` (XOR) | self-inverse, preserves Hamming structure |
| unbind | `BOOL.6` | same instruction |
| bundle 3 | majority | `(a&b)|(a&c)|(b&c)` via `BOOL.8`/`BOOL.14` |
| similarity | XOR + popcount | Hamming distance |
| permutation | `BSWAPM`, `SHL`, `SHR` | byte reversal and bit rotation |
| mask | `BOOL.8` | AND with a constant vector |
| select | `BOOL` with `n` 12/10 etc. | projections |

Binding with XOR is standard hyperdimensional computing (HDC). It is
invertible, it distributes over bundling in the usual approximate sense, and it
is one instruction. The other 15 functions are available for research, but
they are not all invertible. For example, AND loses information; OR saturates.
A "learned algebra" that uses arbitrary `n` must handle non-invertibility
explicitly. Start with XOR.

Bundling by majority-of-three is exact and cheap. For even counts, use a
bit-sliced counter or duplicate one vector to break ties. A binary tree of
majority-of-three is not the same as a true majority of all inputs, but it is
a good approximation and it composes.

Similarity is Hamming distance. On NaN16, popcount is a short SWAR routine:

```
x = x - ((x >> 1) & 0x55555555)
x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
x = (x + (x >> 4)) & 0x0F0F0F0F
x = (x * 0x01010101) >> 24
```

That is five shifts/adds/masks per 32-bit word. A 256-word vector costs about
1,280 simple ops per distance. For 256 prototypes, ~328k ops per query. The C
core can do that; the native core can do it faster. This is the real cost
model, and it should be measured, not hand-waved.

## 6. Double endianness in the model

There are three legitimate uses, and one illegitimate one.

**Legitimate: neutral data.** `n=10` words are invariant under `BSWAPM` at the
head level. The payload bytes reverse. That gives a free permutation
`pi(x) = byteswap32(x)` on each 32-bit slice. It is a fixed, invertible,
hardware-supported permutation. Use it as a position binding or as a second
view for a seal.

**Legitimate: integrity.** A model image can store a dual-face seal: FNV-1a
over near payloads, FNV-1a over byte-swapped payloads, and a class-channel
hash. The loader verifies all three before using the weights. A single
corrupted byte changes the seal. This follows the dual-face seal idea,
applied to a model.

**Legitimate: mirror law.** If a circuit is written with `BOOL.n`, its
byte-swapped form uses `BOOL.rot2(n)` and computes the mirror function. This
can implement a "reverse pass" or a complement without a second table.

**Illegitimate: pretending the two faces double the arithmetic.** They do not.
There is one payload. The second face is a head, not a second data plane. The
value is integrity, ordering, and a free permutation—not 2x FLOPs.

## 7. The model: NaNLM-0

The first model should be small enough to finish and rich enough to exercise
every primitive.

**Task:** byte-level next-symbol prediction on a small corpus. This is a
language-model-shaped task without pretending to be an LLM.

**Representation:**

- `D = 8192` bits (256 words).
- Codebook: 256 random neutral hypervectors, one per byte.
- Position vectors: 3 random neutral hypervectors (or a deterministic hash).
- Context: `majority(bind(token[t-2], pos0), bind(token[t-1], pos1),
  bind(token[t], pos2))`.
- Memory: one prototype per next-byte class, `P[c]` of dimension `D`.
- Similarity: Hamming distance between context and each `P[c]`.
- Output: `argmin_c Hamming(context, P[c])`.

**Training (C adapter):**

For each training position, compute the context, then update a bit-sliced
counter for the observed next byte. After the pass, set each prototype bit to
the majority of its counter. This is an integer-only, order-independent
estimate. It is not backprop; it is a one-pass prototype learner. The adapter
can also do multiple passes, class balancing, and held-out calibration.

**Inference (NaN16):**

1. Load codebook, position vectors, and prototypes from the model image.
2. For each of the three context bytes, `BOOL.6` the codebook vector with the
   position vector.
3. Bundle the three with majority.
4. For each class, XOR the context with the prototype and popcount.
5. Track the minimum; output the class.

This is a few thousand NaN16 instructions per token. It can run on `nanlm_vm`
and, with a display, as an optional cartridge.

**Why this first:**

- It uses `BOOL`, `ADD`, `SHL`, `SHR`, `LD`, `ST`, `JMP`, and the gate.
- It has a clear baseline: order-3 byte n-gram.
- It is small enough to inspect and fast enough to demo.
- It is honest: a high-dimensional associative memory, not a transformer.

**What success looks like:**

- The model runs as a `.nan` image with no C model code.
- Its output matches the C adapter's reference exactly.
- It beats a uniform baseline and is competitive with a small n-gram on a
  held-out slice, or it loses in a way we can explain.
- The model is sealed, and a corrupted byte is rejected.
- The whole thing fits in a few hundred KB.

## 8. Learning beyond the first model

There are four learning stories that fit the substrate. They should be tried
in this order.

### 8.1 One-pass bundling (v0)

No gradients. Compute prototypes from counts. Fast, interpretable, and enough
to prove the pipeline. The model is a memory, not a neural network.

### 8.2 Tsetlin / clause learning (v1)

A Tsetlin machine learns conjunctions of Boolean literals with local rewards.
It maps naturally to 32-bit lanes and to the 16-function alphabet. Each clause
is a mask; a team of clauses votes. The update only needs the clause output,
the label, and a reward bit. This is a much better fit than backprop for a
machine with no FP and no dense matrix unit.

### 8.3 Evolutionary circuit search (v2)

A genome is a short sequence of `(BOOL.n, operand mask, destination)` plus a
router. Mutation changes `n`, flips a mask bit, inserts/deletes a gate. Fitness
is task accuracy minus a size/step penalty. The C adapter evolves; the best
genome is emitted as `.nas`. This is the "learned digital circuits" idea with a
real, testable artifact.

### 8.4 Straight-through / Gumbel in the adapter (research)

If we want to connect to modern ML, treat each gate's `n` as a categorical
variable and train a relaxed distribution in C/PyTorch, then harden to a 4-bit
choice and emit NaN16. This is a bridge, not the core. It must be labeled as
such: the shipped model has no gradients.

## 9. The C adapter

The adapter is the "C translator" in the only form that makes sense here.

It does three things:

1. **Train / quantize.** Build codebooks, prototypes, masks, and circuit
   genomes in C. Use integers, popcount, and deterministic PRNGs. No FP in the
   model path (host convenience code may use it, but it must not leak into the
   artifact).
2. **Emit.** Write `.nan` text (the `NNNN_PPPPPPPP_FFFF` format) and a small
   manifest. The emitter uses only forms `nanlm_asm` can assemble. The `.nas`
   source is checked in; the `.nan` is generated.
3. **Verify.** Load the `.nan` back, reconstruct the vectors, and compare
   against the C reference. Run the same model through `nanlm_vm` and compare
   outputs bit-for-bit.

What it does **not** do:

- It does not replace the NaN runtime.
- It does not add a new word format.
- It does not write an x86 JIT. `nanlm_vm` is the execution path; a native
  core is a later optimization.
- It does not put floats in the model.

### 9.1 The "back to x86" path

The correct path is:

```
.nas source
   -> nanlm_asm
   -> .nan words
   -> nanlm_vm (portable C core)        <- test bench, x86/ARM
   -> future native core               <- performance path
   -> future A64/RISC-V cores
```

The wrong path is a custom C parser plus an x86 backend. That is a multi-month
detour with no model payoff. If the community wants to write models in C, give
them a header-only C adapter (`nan16_word.h`, `nanlm_vec.h`) that compiles to
native code for experiments and emits `.nan` for the real artifact. That is
more useful than a C-to-NaN compiler.

### 9.2 The model image

A model image is a `.nan` stream plus a manifest:

- magic/version;
- `D`, vocabulary size, context length;
- codebook offset, position-vector offset, prototype offset;
- code entry point;
- seal A/B/C;
- ISA version and word model version;
- optional training metadata (corpus hash, hyperparameters).

The manifest can be a small `.nas` data block or a separate JSON file used only
by the adapter. The shipped image must be self-contained and gated.

## 10. Toolchain and repo shape

Start as one repo. Split only when a boundary is real.

```
NaNLM/
  README.md
  LICENSE
  Makefile
  include/
    nan16_word.h        exact word model, header-only
    nanlm_vec.h         vector ops
  src/
    nanlm_vec.c
    nanlm_adapter.c     train/pack/emit/verify (S6)
    nanlm_cli.c         demo and inspector
  models/
    nanlm0.nas          source of record
    nanlm0.nan          assembled words
    nanlm0.manifest
  tests/
    test_word.c
    test_vec.c
    test_model.c
  docs/
    NAN16_STUDY.md
    FOUNDATIONS.md
    MODEL_0.md
    ADAPTER.md
    ROADMAP.md
  .github/workflows/ci.yml
```

Later, if needed:

- `nanlm-models`: model zoo and leaderboard.
- `nanlm-web`: browser playground (WASM build of the C core).
- `nanlm-bench`: benchmark harness and baselines.
- `nanlm-cart`: optional cartridge exhibit.

Do not create five repos on day one. Deeprepo's ring exists because the model
did not fit on one runner. NaNLM-0 fits in one repo and one phone.

## 11. Community strategy

The project needs a reason to exist that is clear in one sentence:

> A language-model-shaped system where every weight is a gated NaN word and
> every operation is a 32-lane Boolean function.

That is interesting to three audiences:

- **NaN/0S people**: a new use for the word model, with seals and gates.
- **HDC / neurosymbolic people**: a concrete Boolean substrate for binding and
  memory.
- **systems / esolang people**: a model that is a `.nan` image, not a Python
  stack.

What to publish first:

1. A 60-second demo: train a tiny model, export `.nan`, run it on `nanlm_vm`,
   show the prediction and the Hamming distances.
2. A word inspector: paste `7FF8_..._F87F`, see heads, `n`, truth table,
   payload, mirror. This is shareable and teaches the format.
3. A benchmark table with a real baseline and honest numbers.
4. A model zoo with tiny, inspectable models.

Community hooks that fit the constraints:

- **NaN golf**: smallest `.nan` model that solves a task.
- **Regime zoo**: submit a short `BOOL.n` circuit and its truth table.
- **Seal challenge**: corrupt a model byte; the loader must reject it.
- **Bring your own hypervector**: a manifest + `.nan` file; the harness runs it.
- **Double-endian mirror**: submit a circuit and its `BSWAPM` mirror.

What not to say:

- "This beats transformers." It does not.
- "AGI." No.
- "NaN is faster." It is not inherently faster; it is a different substrate.
- "C to NaN compiler" before the VM and the model work. Sequence matters.
- "High-dimensional" without a dimension and a benchmark. Always give numbers.

Governance:

- Apache-2.0.
- A short CONTRIBUTING with three contribution types: models, adapters,
  benchmarks.
- An RFC folder for word/model/format changes.
- A rule that every claim has a gate that can fail.
- A rule that the C adapter is S6 and cannot become the runtime.

## 12. Roadmap

### M0 — foundation (1 week)

- `nan16_word.h` with exact word model.
- Tests: lane identity, mirror law, gate, bswap, neutral words.
- `nanlm_vec` bind/bundle/Hamming against scalar references.
- This document and the study.

### M1 — first model (2 weeks)

- C adapter trains a 2,048-bit HDC classifier on a small dataset.
- Emit `.nan`; run inference on `nanlm_vm`; compare bit-for-bit.
- Benchmark memory, steps, accuracy vs a simple baseline.

### M2 — language-shaped model (2–3 weeks)

- 8,192-bit byte-level associative predictor.
- `.nas` runtime; model image; seal.
- Compare to order-3 n-gram; publish the result, win or lose.
- 60-second demo.

### M3 — experts and learning (4–6 weeks)

- 4–16 Boolean experts with a branch-table router.
- Evolutionary search over `n`/masks; export the best circuit.
- Tsetlin clause learner as a second model family.
- Model zoo.

### M4 — community surface (6–8 weeks)

- Browser playground (WASM `nanlm_vm` + canvas).
- Optional cartridge exhibit, if it belongs there.
- Benchmark leaderboard.
- RFC process and first external contributions.

### M5 — scale and hardware (later)

- Larger D, more prototypes, streaming memory.
- Native A64/RISC-V cores.
- FPGA/ASIC exploration of the 32-lane LUT.
- Distillation from a conventional model into a Boolean circuit.

## 13. Risks and how to kill them early

| risk | early test |
|---|---|
| NaN is only a container; the real model is C | run the model with `nanlm_vm` and no adapter |
| HDC model is too weak to be interesting | baseline it against n-gram on a small corpus; publish both |
| C core too slow for inference | measure steps/token; optimize popcount/bundling; use native core |
| 32-bit payload too small | measure D vs accuracy; D=8192 is 2 KB per vector |
| per-word `n` confused with data | enforce `n=10` for data in the adapter and the loader |
| `BSWAPM` corrupts code | only bswap data; test mirror law and seal |
| model format drifts from NaN16 | roundtrip `.nan` through `nanlm_vm` in CI |
| community expects an LLM | say "associative predictor" in the README, not "LLM" |
| C adapter becomes the product | make the `.nan` model the release artifact |

## 14. Open research questions

1. What is the capacity of a D-bit HDC memory when binding is a 2-input
   Boolean function chosen from 16, rather than a fixed XOR?
2. Which of the 16 functions are useful binding operators? XOR/XNOR are
   invertible; AND/OR are not. Is there a useful non-invertible algebra?
3. Can a router learn to select a Boolean regime (a short circuit) from the
   input, and does that beat a single circuit of the same size?
4. Can evolutionary search find small circuits that beat random masks on a
   held-out task?
5. Does per-slice `BSWAPM` permutation add capacity over XOR position codes?
6. Can dual-face seals make model weights tamper-evident without a large
   integrity overhead?
7. What is the scaling law for D, number of prototypes, and context length on
   a fixed compute budget?
8. Can a conventional model be distilled into a sparse Boolean circuit that
   runs on NaN16, and what is lost?

These are real questions with experiments, not manifestos. The project should
publish the negative results too.
