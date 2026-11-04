# NaNLM

[![ci](https://github.com/shemshallah/NaNLM/actions/workflows/ci.yml/badge.svg)](https://github.com/shemshallah/NaNLM/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![word](https://img.shields.io/badge/word-NaN16-7ff8.svg)](docs/VM.md)

**A high-dimensional Boolean model that lives in NaN16 words.**

NaNLM is a standalone project. It uses the NaN16 word format studied from
NaN/0S as its foundation, but it does **not** depend on the NaN/0S repository:
NaNLM has its own assembler (`nanlm_asm`) and runtime (`nanlm_vm`).

```
 byte     b0 b1   |  b2 b3 b4 b5   |  b6 b7
          FAR head |  32-bit payload | NEAR head
          big-endian                 little-endian
```

The two heads produce a four-bit class nibble `n`. For `BOOL`, `n` is a truth
table: 32 independent Boolean lanes run in one instruction. For every other
op, `n` is a predicate over the machine flags. The same four bits do both jobs.

NaNLM asks a different question from a dense transformer:

> Can a useful model be built from a sparse population of these words, without
> floating-point arithmetic, matrix multiplication, or backpropagation in the
> machine?

The first target is deliberately small and testable: an 8,192-bit
hyperdimensional associative memory that predicts the next character in a
short context. It is **not** a transformer competitor. It is a substrate
experiment with a measurable baseline.

**Status:** M1, M2a, M2b, and the M3 simplex slot memory pass. On the
**full 4,497-byte held-out stream** the slot memory reaches **49.05% top-1 /
2.959 BPC**; the interpolated n-gram reaches **48.14% / 3.420 BPC**. The paired
block-bootstrap BPC difference is decisive
(**−0.460 [−0.521, −0.399]**), while the top-1 difference is a statistical tie
(**+0.91 [−0.13, +2.02]** points). The old 52.3% vs 55.5% comparison used 128
sampled contexts and was too small to separate the models.

The continuous-training path is now closed end-to-end: pinned repository →
license/secret-filtered chunks → additive shards → typed slot memory → `.nas` →
`nanlm_vm`. Signed 32-bit shards make unlearning a re-merge; the 8-slot student
and multi-order mixer exist, but their first results are negative.

A Qwen2.5-7B teacher now passes its token-boundary card (**88.8% / 0.66 BPC**)
but does **not** improve the byte-level student. The ngram-only 8-slot control
matches the n-gram BPC at 4,096 keys; the raw 3-slot memory still wins BPC.
The next model is token-level. See [QWEN7B.md](docs/QWEN7B.md).

## What this is

- A **NaN16-native model format**: weights, codebook, and program are 64-bit
  double-ended NaN words.
- A **32-lane Boolean vector machine**: `BOOL.n` applies one of 16 two-input
  functions to 32 bit positions at once.
- A **sparse high-dimensional associative memory**: binding, bundling, Hamming
  similarity, prototype memory.
- A **C adapter** for training, packing, emitting `.nan` images, and verifying
  the NaN runtime. The adapter is S6: a test bench, not the shipped machine.

## What this is not

- Not a floating-point neural network with NaNs as a storage trick.
- Not a claim that NaN16 beats tensor cores.
- Not a full C-to-NaN compiler in v0.
- Not an LLM. The first model is a small associative predictor with honest
  baselines.

## The byte-level M2b base

M2a exercises binding and bundling on a synthetic context model. M2b switches
the vocabulary to bytes and adds a real corpus plus an n-gram baseline. The
table below describes the current M2b base.

`NaNLM-0` (working name):

| property | value |
|---|---|
| vector dimension | 2,048 bits = 64 NaN words (8,192 is the next scale) |
| vocabulary | byte-level, 256 symbols |
| context | 3 bound bytes, majority-bundled |
| memory | 8 prototypes per next-byte class |
| similarity | Hamming distance / popcount |
| training | integer bit-sliced counters in the C adapter |
| inference | NaN16 `BOOL`, `ADD`, `SHL`, `SHR`, `LD`, `ST`, `JMP` |
| arithmetic | integer only; no FP in the model |
| model size | ~1.2 MB of NaN words including codebook |
| quality | 26.6% top-1, 5.50 BPC / n-gram 50%, 3.85 BPC (128 contexts) |

The point is not the score. The point is that every weight is a gated NaN
word, every operation is a Boolean function, and the whole model can be
inspected bit for bit.

## Quick start

```sh
make test
```

To build, assemble, run, and diff the first model against the C adapter:

```sh
sh tools/build_model.sh
```

To run the M2a context model, which exercises binding/bundling and a
query-vector hash gate:

```sh
sh tools/build_ctx.sh
```

To run the byte-level M2b class-memory base on `data/alice_excerpt.txt`:

```sh
sh tools/build_lm.sh
```

To run the M3 simplex slot memory (one prototype per observed 3-gram):

```sh
sh tools/build_slot.sh
```

To prove additive shard merge (`A+B == B+A` byte-for-byte):

```sh
sh tools/build_shard.sh
```

To compile additive count shards into typed slot memory and prove it matches
the direct corpus trainer byte-for-byte:

```sh
make shard-slot-check
```

To run the full continuous code pipeline — GitHub-ready ingestion, license and
secret gates, shards, exact merge, typed slot memory, VM diff, and a
runner-ring failover plan:

```sh
make continuous-check
```

To score every held-out byte with paired bootstrap intervals and model-size
accounting:

```sh
./build/nanlm_slot_train --corpus data/alice_excerpt.txt   --eval-log build/slot_stream.tsv   --out build/slot_stream.nas --expected build/slot_stream.expected   --manifest build/slot_stream.manifest
python3 experiments/eval_stream.py   --run slot3=build/slot_stream.tsv=build/slot_stream.nas
```

To build and run the 8-byte / 16-slot student and the multi-order mixer:

```sh
make slot8-check
make mixer-check
```

M2a output is `CC HHHHHHHH`: two hex class digits, a space, and the FNV-1a hash
of the encoded query vector. The negative gates zero the position vectors,
swap prototypes 0/1, and starve the step budget; all three must fail.

Everything runs locally: the model is assembled by `build/nanlm_asm` and
executed by `build/nanlm_vm`. No external tree or native runtime is required.

The first model is a 2,048-bit, 16-class synthetic prototype classifier. It is
a pipeline proof. M2a adds the actual HDC operations:
`v_i = C[b_i] XOR P[i]`, `q = majority(v_0,v_1,v_2)`. M2b replaces the synthetic
classes with 256 byte classes, eight prototypes per byte, and a held-out Alice
excerpt. The n-gram baseline wins today; the model card says so.

The tests verify the parts that must not drift:

- the lane identity `BOOL.n(0xCCCCCCCC, 0xAAAAAAAA) == n * 0x11111111`;
- the mirror law `BOOL.rot2(n)(a,b) == BOOL.n(~a,b)`;
- both-face gate behaviour;
- vector bind/bundle/Hamming against scalar references;
- neutral data words and `BSWAPM` involution.

## Documentation

- [NaN16 study](docs/NAN16_STUDY.md) — the word format and its constraints, with provenance.
- [Foundations](docs/FOUNDATIONS.md) — the full design argument.
- [Keystone and supports](docs/KEYSTONE.md) — the stable core and the replaceable model parts.
- [Distributed training](docs/DISTRIBUTED.md) — reporam/repovram shards, volunteers, and the S3 path.
- [Continuous training from GitHub](docs/CONTINUOUS.md) — code ingestion, runner ring, memory generations, and routing.
- [Repo-anchored brains](docs/REPO_BRAIN.md) — identity vs residence, manifests, and mirrors.
- [Smart base / teacher distillation](docs/TEACHER.md) — how to get a smart model immediately.
- [Open-source lifecycle](docs/OPEN_SOURCE.md) — why mergeable counting is a world effort, stage by stage.
- [Alien architecture](docs/ALIEN.md) — self-modifying regimes, mirror law, and what failed.
- [Heterogeneous layers](docs/LAYERS.md) — one word, different interpretations per layer.
- [Layer test plan](docs/LAYER_TESTS.md) — per-layer NaN use, tests, and stop conditions.
- [NaN LUT / Regime Network](docs/LUT_REGIME.md) — the special architecture candidate and its decisive tests.
- [Regime results](docs/REGIME_RESULTS.md) — fixed vs learned-router LUT trees, with controls.
- [Feature and teacher probes](docs/FEATURE_RESULTS.md) — typed ctype gain, learned-feature and small-teacher negatives.
- [Full-stream evaluation](docs/EVAL.md) — why 128 contexts were not enough, bootstrap CIs, and Hutter-style model accounting.
- [Multi-order context mixing](docs/MIXER.md) — PAQ/lpaq-style template bank, integer mixer, and the first negative result.
- [Packing and fast kernels](docs/PACKING.md) — bit-plane pages, VM conformance, additive HDC counters, and the T1/T2/T3 split.
- [Qwen teacher runner](docs/QWEN.md) — the first real Qwen run, its teacher card, and why 0.5B/256 samples is not enough.
- [Qwen 7B teacher result](docs/QWEN7B.md) — the T4 teacher card, the scale grid, and why the byte student cannot consume it.
- [Token-level teacher/student](docs/TOKEN.md) — exact token-ID shards, merge/scale tools, token backoff evaluation, and the token-student design.
- [In-VM plasticity](docs/PLASTICITY.md) — bit-sliced counters and STW truth-table search inside `nanlm_vm`.
- [Token HDC prototype](docs/TOKEN_HDC.md) — first token-level HDC student, corrected teacher backoff result, and the model-size reality check.
- [Sparse high-dimensional sweep](docs/SPARSE.md) — dimensions, active fraction, and noise fragility.
- [Free-GPU strategy](docs/GPU.md) — Modal batch teacher inference, free-tier options, and what not to spend GPU hours on.
- [Kaggle runbook](docs/KAGGLE.md) — one script for a free T4/GPU session, with teacher, hybrid backoff, and student artifacts.
- `kaggle_hybrid.py` — rebuild hybrid students from an existing 7B teacher shard without reloading the model.
- `kaggle_grid.py` — CPU-only teacher-scale and 8-slot-size grid from existing shards.
- `kaggle_token.py` — one-line Kaggle token teacher run and NTT packaging.
- [Modal GPU runbook](docs/MODAL.md) — install, authenticate, run one Qwen 7B teacher job, fetch the shard, and compile the student.
- [AMD Radeon / ROCm runbook](docs/AMD_GPU.md) — Qwen 7B teacher commands for the AMD cloud box.
- [AMD teacher notebook](notebooks/qwen7b_teacher.ipynb) — open in JupyterLab and run the whole Qwen 7B teacher/student gate.
- [Model 0](docs/MODEL_0.md) — a concrete, buildable first model.
- [M2 plan](docs/M2_PLAN.md) — context binding, a real corpus, and negative gates.
- [C adapter](docs/ADAPTER.md) — the S6 boundary and the `.nan` compiler.
- [VM and assembler](docs/VM.md) — NaNLM's own runtime, subset, and status codes.
- [Roadmap](docs/ROADMAP.md) — what ships first, and what does not.
- [Community plan](docs/COMMUNITY.md) — launch sequence, hooks, and anti-patterns.
- [Evidence](docs/EVIDENCE.md) — the gates that have actually run.
- [Research log](docs/RESEARCH_LOG.md) — experiment matrix, including the negative results.
- [Model card](docs/MODEL_CARD.md) — intended use, metrics, caveats, and reproduction.
- [M1 retrospective](docs/M1_RETROSPECTIVE.md) — what the first pipeline proved, and did not.

## Relationship to NaN16

NaNLM uses the NaN16 word model: two legal NaN heads around a 32-bit payload,
with the class nibble as truth table or predicate. The format is studied and
credited in [docs/NAN16_STUDY.md](docs/NAN16_STUDY.md), but NaNLM owns its
assembler, runtime, model format, training adapter, and gates.

The final artifact is a `.nan` image. The C adapter may train and pack it, but
the model must run on `nanlm_vm` without the adapter. That is the boundary that
keeps this from becoming a C project with a NaN file extension.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version: use NaN16 exactly,
keep data neutral, make the model run as `.nan`, and add a gate that can fail.

## License

Apache-2.0. See [LICENSE](LICENSE).
