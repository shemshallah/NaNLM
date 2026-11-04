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
nanlm-slot: keys=3832 bits=765 queries=32
eval_n=128 hdc_top1=51.56% hdc_bpc=2.7196
ngram_top1=55.47% ngram_bpc=3.1304
build_slot: PASS (32 queries)
build_slot: NEGATIVE PASS (step starvation rejected)
```

The NaNLM VM trace:

```
[nanlm-vm] nan-text: 100695708 steps, rc=3, pc=39, trap=''
```

This is one prototype per observed 3-gram, with 255-bit simplex codewords in
three slots. The runtime matches the C predictions and query hashes exactly.
It wins bits-per-byte (2.72 vs 3.13) and loses top-1 (51.6% vs 55.5%) to the
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

## 9. Additional checks

- ASan + UBSan builds of `test_word`, `test_vec`, `test_model`, and `test_ctx` pass.
- The same pipeline was run with `--dim 512 --classes 8 --queries 16` and
  `--dim 512 --classes 32 --queries 64`; both matched `nanlm_vm` byte-for-byte.
  The 32-class run exercises the two-digit hex output path.
- `clang --analyze` reports no findings on the C adapter and trainer.

## What is not proven yet

- The class memory (M2b) loses to the n-gram: 26.6% / 5.50 BPC vs 50.0% / 3.85.
- The slot memory (M3) wins bits-per-byte (2.72 vs 3.13) but loses top-1
  (51.6% vs 55.5%) on the 128-context Alice split.
- The slot model is built from a corpus, not yet compiled directly from merged
  shards.
- reporam/repovram page mapping and generation manifests are designed but not
  implemented.
- Teacher distillation (storing a small teacher's distribution) is not
  implemented.
- The seal/loader path does not exist yet.
- The 8,192-bit scale has not been retested.
- CI runs the full gates on x86-64; local gates run on aarch64.

The next gate is a merged shard → slot model → `.nan` build, then teacher
distillation and a reporam/repovram generation.
