# Evidence

The foundation is only useful if it matches the real NaN16 core. These are the
checks run so far, on aarch64 Termux with clang 21.

## 1. Host tests

```sh
make test
```

Output:

```
test_word: PASS
test_vec: PASS
test_model: PASS
test_ctx: PASS
```

`test_word` checks:

- word roundtrip for all 16 nibbles;
- both-face gate acceptance and rejection;
- the lane identity for all 16 functions;
- the mirror law for all 16 functions over several operand pairs;
- `BSWAPM` involution and the `n -> rot2(n)` transform;
- neutral data nibbles;
- popcount reference values.

`test_vec` checks:

- `BOOL.n` vector operations against scalar references for all 16 functions;
- XOR binding;
- majority-of-three bundling;
- Hamming distance and XOR cancellation;
- `BSWAPM` payload reversal and involution;
- vector popcount;
- a 128-bit, 4-class nearest-prototype classifier with two flipped bits per
  sample.

`test_model` checks:

- a 256-bit, 4-class trained model is deterministic across two runs;
- every prototype and query is neutral data;
- the C classifier matches the ground-truth labels;
- the emitted `.nas` contains the expected `.equ`, labels, and `.org`.

`test_ctx` checks:

- codebook, positions, prototypes, and queries are neutral data;
- manual bind+bundle equals `nanlm_ctx_encode`;
- query hashes match the encoded vectors;
- the C classifier matches the labels;
- zeroed positions change the hash;
- swapped prototypes change predictions;
- the emitted `.nas` contains the context runtime data labels.

## 2. Lane identity on the NaNLM VM

A small NaN16 program is assembled by `nanlm_asm` and run on `nanlm_vm`:

```asm
MOVI r1,0xCCCC
MOVHI r1,0xCCCC
MOVI r2,0xAAAA
MOVHI r2,0xAAAA
BOOL.6 r3,r1,r2
MOVI r4,0x6666
MOVHI r4,0x6666
CMP r3,r4
JMP[NE] @fail
; print PASS
```

```sh
build/nanlm_asm tests/lane.nas build/lane.nan
build/nanlm_vm --img build/lane.nan --steps 100000
```

Output:

```
PASS
```

This confirms that `BOOL.6` produces `0x66666666` for the lane constants, as
the word model predicts, using NaNLM's own tools.

## 3. Word packing

`test_word` roundtrips `nan16_make(payload, n)` for all 16 nibbles, checks the
both-face gate, the lane identity, the mirror law, and `BSWAPM` involution.
`nanlm_asm` emits `.word` data as neutral `n=10` words, and `nanlm_vm` gates
every fetched word. The model gates below confirm that the emitted data is
neutral and that the runtime reaches it correctly.

## 4. M1 end-to-end model

Build, assemble, run, and diff:

```sh
sh tools/build_model.sh
```

Output:

```
nanlm0: dim=2048 classes=16 queries=32 samples=64 noise=64
prototype_words=1024 query_words=2048 c_accuracy=32/32
build_model: PASS (32 queries)
build_model: NEGATIVE PASS (swapped prototypes rejected)
build_model: NEGATIVE PASS (step starvation rejected)
```

The NaNLM VM trace for the positive model:

```
[nanlm-vm] nan-text: 1098403 steps, rc=3, pc=24, trap=''
```

The query labels are shuffled, so a runtime that returns
`query_index % classes` cannot pass. The negative gate then swaps prototypes
0 and 1 after training and asserts that the differential test **fails**. It
does, with a clean mismatch and no trap.

This is the important boundary: the C adapter trained and emitted the model,
but the predictions came from the `.nan` image running on `nanlm_vm`. The
image is 11,264 words (90,112 bytes) and the runtime prints one two-digit hex
class per query.

## 5. M2a context model

Build, assemble, run, and diff:

```sh
sh tools/build_ctx.sh
```

Output:

```
build_ctx: PASS (16 queries)
build_ctx: NEGATIVE PASS (zero positions rejected)
build_ctx: NEGATIVE PASS (swapped prototypes rejected)
build_ctx: NEGATIVE PASS (step starvation rejected)
```

The NaNLM VM trace:

```
[nanlm-vm] nan-text: 612743 steps, rc=3, pc=39, trap=''
```

Each line is `CC HHHHHHHH`: two hex class digits, a space, and the FNV-1a hash
of the 2,048-bit encoded query vector. The C adapter and the `.nan` runtime
agree on all 16 lines, which means the runtime really performed:

```
v_i = C[b_i] XOR P[i]
q   = majority(v_0, v_1, v_2)
```

and did not take a label-only shortcut. The negative gates prove the hash gate
and classifier are not vacuous:

- zeroing the position vectors changes the query hash and fails;
- swapping prototypes 0/1 changes the class labels and fails;
- a 1,000-step starvation run truncates the output and fails.

## 6. M2b byte-level language-model base

Build, assemble, run, and diff:

```sh
sh tools/build_lm.sh
```

Output:

```
nanlm-lm: dim=2048 vocab=256 clusters=8 queries=32 context=3
train_bytes=25500 val_bytes=4500 train_frac=0.850
eval_n=128 calib_n=128 temperature=128.0
hdc_top1=26.56% ngram_top1=50.00%
hdc_bpc=5.4982 ngram_bpc=3.8505
build_lm: PASS (32 queries)
build_lm: NEGATIVE PASS (zero positions rejected)
build_lm: NEGATIVE PASS (predicted prototypes rejected)
build_lm: NEGATIVE PASS (step starvation rejected)
```

The NaNLM VM trace:

```
[nanlm-vm] nan-text: 140442458 steps, rc=3, pc=40, trap=''
```

This is a real byte-level model: 2,048-bit codebook, 256 byte classes, eight
prototypes per class, three-byte context, held-out Alice excerpt. The runtime
matches the C adapter on all 32 emitted predictions and hashes. The honest
result on 128 calibrated held-out contexts is that the order-3 n-gram wins:
26.6% top-1 / 5.50 bits-per-byte vs 50.0% / 3.85.

The model is a structural base, not a quality claim.

## 7. M3 simplex slot memory

```sh
sh tools/build_slot.sh
```

Output:

```
nanlm-slot: keys=3832 bits=1536 queries=32
eval_n=128 hdc_top1=52.34% hdc_bpc=2.6703
ngram_top1=55.47% ngram_bpc=3.1304
build_slot: PASS (32 queries)
build_slot: NEGATIVE PASS (step starvation rejected)
```

The NaNLM VM trace:

```
[nanlm-vm] nan-text: 197829777 steps, rc=3, pc=39, trap=''
```

This is one prototype per observed 3-gram, with 255-bit simplex codewords in
three slots. The runtime matches the C predictions and query hashes exactly.
It wins bits-per-byte (2.67 vs 3.13) and loses top-1 (52.3% vs 55.5%) to the
in-repo interpolated n-gram. Honest, not hidden.

## 8. Additive shard merge

```sh
sh tools/build_shard.sh
```

Output:

```
shard_gate: PASS (A+B == B+A, byte-identical)
shard_gate: PASS (total 14997 + 14997 = 29994)
```

Training is `(3-gram, next byte, count)`; merging is integer addition. Two
corpus slices merged in either order produce byte-identical shards. This is
the CRDT property that lets volunteers train without coordination.

## 9. Teacher -> typed slot pipeline

```sh
make teacher-slot-check
```

Output:

```
teacher-slot: teacher_records=83504 contexts=15312 keys=3828
eval_n=128 hdc_top1=52.34% hdc_bpc=3.6219
stream_n=4497 stream_top1=49.05% stream_bpc=3.9316
ngram_top1=55.47% ngram_bpc=3.1307
teacher_slot: PASS (32 queries)
```

The pipeline is exact: teacher shard → 3-byte aggregation → typed slot model →
`.nas` → `nanlm_vm` → diff. The quality is poor because the fixture n-gram
teacher is weak; this is a plumbing gate, not a model result.

## 10. Continuous code pipeline and shard -> slot compile

```sh
make shard-slot-check
make continuous-check
```

`shard-slot-check` proves that count shards compile to exactly the same
prototype memory as the direct corpus trainer:

```
shard-slot: shard_records=8317 keys=3832
eval_n=128 hdc_top1=52.34% hdc_bpc=2.6608
stream_n=4497 stream_top1=49.05% stream_bpc=2.9593
ngram_top1=55.47% ngram_bpc=3.1304
shard_slot: corpus == shard compile PASS
shard_slot: VM PASS (32 queries)
```

The two `.nas` files are byte-identical. This is the continuous-training
invariant: a merged shard set is a model, not just a checkpoint.

`continuous-check` ingests a deliberately adversarial local code fixture,
rejects secret/generated/minified files, validates the MIT license, builds and
merges shards, compiles typed slot memory, runs the VM, and plans a four-runner
ring with one failed runner:

```
ingest: files=8 chunks=8 bytes=5766 skipped=4
shard: keys=1912 records=2579 total=4901
continuous: A+B == B+A PASS
shard-slot: eval_n=128 hdc_top1=27.34% hdc_bpc=6.2286
ngram_top1=26.56% ngram_bpc=5.5575
continuous: shard -> typed slot -> VM PASS (32 queries)
ring: runners=4 jobs=4 chunks=8 bytes=5766
ring: simulated done=4 failed=[1]
continuous: runner ring plan/failover/materialize/shard PASS
```

The code fixture is tiny and high-entropy; its quality numbers are a plumbing
check, not a code-model claim. The important result is deterministic
provenance-to-runtime closure.

## 11. Full held-out stream, not 128 contexts

```sh
make eval-check
```

The 128-context split was too small: the binomial standard error near 50% is
about ±4.4 points, so 52.3% vs 55.5% was a statistical tie. Every held-out byte
is now scored, with paired moving-block bootstrap intervals and model-size
accounting.

```
slot3: 4497 bytes  top1=49.05%  bpc=2.9593
ngram:             top1=48.14%  bpc=3.4196
paired dBPC 95% CI: [-0.5214, -0.3989]
paired dTop1 95% CI: [-0.13, +2.02] points
```

The BPC win is real; the top-1 difference is a tie. The result is recorded in
[docs/EVAL.md](EVAL.md) with the Hutter-style model-size penalty. On a 4.5 KB
stream the 3-slot `.nas` (65.9 KB gzip) dominates total BPC; model size is now
part of the evidence, not hidden.

## 12. Signed 32-bit shards and exact unlearning

```sh
make shard2-check
```

Output:

```
shard2: A+B == B+A PASS
shard2: B + (-B) == empty PASS
shard2: (A+B)+(-B) == A PASS (exact unlearning)
```

Counts are signed 32-bit integers added with two's-complement modular
arithmetic. A takedown is a negated shard; a generation can be rebuilt by
re-merging without the offending shard. The older uint16 shard remains for
compatibility; the signed format is the unlearning path.

## 13. 8-slot and mixer first results

`make slot8-check` runs an 8-byte / 16-slot student (8 byte + 8 ctype slots)
with recency-graded codewords and suffix-covering prototype selection. With the
fixture n-gram teacher it scores 44.79% / 3.765 BPC on the full stream, worse
than the 3-slot memory and the n-gram. This is a fixture-teacher result, not a
verdict on 8-byte context with a real Qwen teacher.

`make mixer-check` runs the C99 multi-order mixer. Its static equal-weight
variant scores 45.97% / 3.725 BPC at table 14; the learned mixer update was
unstable and is disabled by default. The first mixer is a negative result, not
a PAQ claim. The design and next steps are in [docs/MIXER.md](MIXER.md).

## 14. Qwen 7B teacher and the byte-student ceiling

A Kaggle T4 run of Qwen2.5-7B-Instruct 4-bit produced:

```
teacher_top1  88.78%
teacher_bpc   0.6606
ngram_bpc     2.9757 (same token-boundary positions)
```

The teacher card passes. The byte-level students do not improve. On the full
4,492-byte held-out stream:

| teacher scale | 8-slot top-1 | 8-slot BPC |
|---:|---:|---:|
| 0 (ngram only) | **49.51%** | **3.4201** |
| 8 | 46.75% | 3.6406 |
| 16 | 47.60% | 3.5225 |
| 32 | 47.86% | 3.4420 |

The ngram-only 8-slot matches the n-gram BPC and slightly beats top-1; every
non-zero teacher scale makes BPC worse. The raw 3-slot direct model remains
**49.05% / 2.9593 BPC**.

The conclusion is representational: a token-level teacher cannot be fully
consumed by a byte-level 3/8-slot student through first-byte marginalization.
The next model is token-level with 3–6 token context. See
[QWEN7B.md](QWEN7B.md).

## 15. Additional checks

- ASan + UBSan builds of `test_word`, `test_vec`, `test_model`, and `test_ctx` pass.
- The same pipeline was run with `--dim 512 --classes 8 --queries 16` and
  `--dim 512 --classes 32 --queries 64`; both matched `nanlm_vm` byte-for-byte.
  The 32-class run exercises the two-digit hex output path.
- `clang --analyze` reports no findings on the C adapter and trainer.

## What is not proven yet

- The class memory (M2b) loses to the n-gram: 26.6% / 5.50 BPC vs 50.0% / 3.85.
- The slot memory (M3) wins BPC on the full stream (2.959 vs 3.420; paired CI
  excludes zero) but top-1 is a statistical tie (49.05% vs 48.14%).
- The teacher → typed slot pipeline exists and passes plumbing, but the fixture
  teacher is not smart enough to beat the n-gram.
- The continuous code fixture is a plumbing result; it is not a code-quality
  result and does not beat the n-gram on BPC.
- The 8-slot fixture-teacher student is worse than 3-slot; real Qwen teacher
  artifacts are still pending.
- The first mixer loses BPC and its learned weight update was unstable.
- Model size dominates Hutter-style totals on a 4.5 KB stream; packed
  bit-plane pages and native kernels are designed but not implemented.
- reporam/repovram page mapping and generation manifests are designed but not
  implemented as a live service.
- The seal/loader path does not exist yet.
- CI runs the full gates on x86-64; local gates run on aarch64.

The next gates are: Qwen teacher card on a held-out code stream, an 8-slot
student with a real teacher, a stable per-bit mixer, and a packed page + native
kernel whose hashes match `nanlm_vm`. The continuous supply and shard algebra
are already in place.
