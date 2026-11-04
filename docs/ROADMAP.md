# Roadmap

The order is chosen to make the next failure cheap.

## M0 — Foundation

**Goal:** the word model and the vector primitives are exact, tested, and
documented.

- [x] Study the NaN16 word format: gate, class nibble, double endianness, ISA.
- [x] `include/nan16_word.h`: exact word helpers.
- [x] `docs/NAN16_STUDY.md`, `docs/FOUNDATIONS.md`, `docs/MODEL_0.md`,
      `docs/ADAPTER.md`.
- [x] `include/nanlm_vec.h`, `src/nanlm_vec.c`: bind, bundle, Hamming.
- [x] Tests: lane identity, mirror law, gate, bswap, vector reference.
- [ ] CI: `make test` on x86-64 and aarch64.

**Exit gate:** `make test` passes; every test can fail when the implementation
is corrupted.

## M1 — First model, first export

**Goal:** a tiny model trains in C, exports to `.nan`, and runs on `nanlm_vm`
with identical predictions.

- [x] 2,048-bit, 16-class synthetic dataset.
- [x] C trainer with integer counters.
- [x] `.nas` emitter and `.nan` emitter.
- [x] `nanlm_vm` inference program.
- [x] Differential test: adapter == `.nan` (32/32).
- [x] Benchmark: accuracy, model size, steps/prediction.

**Exit gate:** a model image that runs without the C adapter, with a trace or
prediction match.

**Result:** `sh tools/build_model.sh` assembles 11,264 words and runs 32
queries in ~1,098,403 steps on the C core, with 32/32 predictions matching.
The labels are shuffled, and a negative gate swaps prototypes 0/1 and asserts
that the differential test fails. It does. This is a pipeline proof, not a
language model.

## M2 — NaNLM-0

**Goal:** the byte-level associative predictor in `docs/MODEL_0.md`.

- [x] M2a: context binding/bundling runtime (`v_i = C[b_i] XOR P[i]`,
      `q = majority(v_0,v_1,v_2)`).
- [x] M2a: query-vector FNV hash gate; class + hash must match.
- [x] M2a: negative gates for zeroed positions, swapped prototypes, and step
      starvation.
- [x] M2b base: 2,048-bit, 256-byte vocabulary, 8 prototypes per byte.
- [x] M2b base: codebook, position vectors, prototypes, held-out Alice excerpt.
- [x] M2b base: order-3 n-gram baseline.
- [x] M2b base: held-out top-1 accuracy (26.6% HDC / 5.50 BPC vs 50.0% / 3.85 n-gram).
- [x] M2b base: `tools/build_lm.sh` end-to-end gate.
- [ ] M2b next: 8,192-bit scale.
- [ ] M2b next: seal A/B/C and loader verification.
- [x] M2b: HDC probability calibration and bits/byte (5.50 vs n-gram 3.85).
- [ ] M2b next: 60-second demo script.

**Exit gate:** a published result, including the case where the n-gram wins.

**Result:** the n-gram wins on quality, but the byte-level `.nan` model runs on
`nanlm_vm` and matches the C adapter on every predicted byte and hash.

## M3 — Experts and learning

**Goal:** move beyond one circuit and one prototype.

- [ ] 4–16 Boolean experts with a branch-table router.
- [ ] Evolutionary search over `BOOL.n` sequences and masks.
- [ ] Tsetlin clause learner as a second model family.
- [ ] Model zoo with tiny, inspectable models.
- [ ] Ablations: D, context, binding function, bundling, experts.

**Exit gate:** at least one task where a learned circuit beats a random-mask
baseline, or a clear negative result.

## M4 — Community surface

**Goal:** make it easy for someone else to run, inspect, and contribute.

- [ ] Browser playground (WASM `nanlm_vm` + canvas).
- [ ] Word inspector: heads, `n`, truth table, mirror.
- [ ] Benchmark leaderboard.
- [ ] RFC process.
- [ ] Optional NaN/0S cartridge exhibit.
- [ ] `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates.

**Exit gate:** an external contributor can add a model or a benchmark without
touching the core.

## M5 — Scale and hardware

**Goal:** find the real limits.

- [ ] Larger D and streaming prototype memory.
- [ ] Native A64/RISC-V cores for the model runtime.
- [ ] FPGA/ASIC exploration of the 32-lane LUT.
- [ ] Distillation from a conventional model into a Boolean circuit.
- [ ] Formal properties: gate, seal, and exact inference.

## What not to do

- Do not build a general C-to-NaN compiler in M0–M2.
- Do not add a second word format.
- Do not put floats in the model.
- Do not claim transformer parity.
- Do not create a multi-repo ring before the single-repo model works.
- Do not let the C adapter become the runtime.
