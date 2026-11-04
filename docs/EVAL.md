# Full-stream evaluation and the end of the 128-context claim

The previous 128-context comparison was too small to separate the models. At
n=128, the binomial standard error on a top-1 rate near 50% is about 4.4
points. A 52.3% vs 55.5% difference is therefore a statistical tie, not a win
or a loss. The 128-context table is retained only as a regression smoke test.

## Protocol

1. Train on the first 85% of the corpus.
2. Score **every** next byte in the last 15%, not a sample of contexts.
3. Freeze global model counts during scoring.
4. Record per-byte model NLL and the interpolated 3/2/1-gram NLL.
5. Bootstrap moving blocks (64 bytes) for 95% intervals, paired by byte.
6. Count model bytes once: `model_bits = 8 * gzip(model file)`.
7. Report `total_bpc = (stream_bits + model_bits) / stream_bytes`.

The TSV emitted by the C evaluator is:

```
pos actual pred slot_bits ngram_bits slot_correct ngram_correct
```

Reproduce:

```sh
./build/nanlm_slot_train --corpus data/alice_excerpt.txt \
  --eval-log build/slot_stream.tsv \
  --out build/slot_stream.nas --expected build/slot_stream.expected \
  --manifest build/slot_stream.manifest

python3 experiments/eval_stream.py \
  --run slot3=build/slot_stream.tsv=build/slot_stream.nas
```

## Results: Alice, full held-out stream

4,497 held-out bytes (the 8-slot run has 4,492 because its context starts later).

| model | top-1 | BPC | paired ΔBPC (95% CI) | paired Δtop1 (95% CI) |
|---|---:|---:|---|---|
| 3-slot typed memory | **49.05%** | **2.9593** | **−0.4603 [−0.521, −0.399]** | +0.91 [−0.13, +2.02] |
| 3-gram baseline | 48.14% | 3.4196 | — | — |
| 8-slot / 16-slot student (fixture teacher) | 44.79% | 3.7648 | +0.3442 [+0.266, +0.425] | −3.25 [−4.28, −2.25] |
| static multi-order mixer, table 14 | 45.97% | 3.7249 | +0.3046 [+0.272, +0.335] | −2.16 [−3.01, −1.28] |

The 3-slot memory has a **real BPC advantage** on the full stream: the paired
CI excludes zero. Its top-1 difference is still a tie: the CI includes zero.
That is the honest reading.

The 8-slot and mixer rows are **not** a verdict on the 8-byte idea or on
PAQ-style mixing. The 8-slot row uses a deliberately weak fixture teacher; the
mixer row is an equal-weight fixed mixer, not a trained PAQ mixer. Both are
plumbing/negative results.

## Model size and Hutter accounting

On a 4.5 KB held-out stream, model size dominates the total:

| model | `.nas` bytes | gzip bytes | gzip BPC equivalent | total BPC (gzip) |
|---|---:|---:|---:|---:|
| 3-slot | 3,384,343 | 65,862 | 117.17 | 120.13 |
| 8-slot | 5,046,578 | 106,626 | 189.89 | 193.66 |
| mixer table 14 | 6,258,784 | — | — | 11,150.27 |

This is not a reason to hide model size. It is the reason to:

- pack bodies as bit-planes, not per-word NaN text;
- use fixed-size hashed tables instead of 96 bytes/key;
- evaluate on a stream large enough to amortize the model;
- publish a Pareto table with model size and BPC, not a single top-1 number.

A 1 MB model on a 4.5 KB stream is ~1,800 BPC equivalent. On a 1 GB stream it is
~0.008 BPC. The community leaderboard must state the stream size.

## What this changes

- **Retract** "slot memory loses top-1 52.3 vs 55.5" as a model conclusion.
  Full-stream top-1 is 49.05 vs 48.14, which is a tie.
- **Keep** the BPC win: full-stream 2.9593 vs 3.4196, paired CI excludes zero.
- **Do not** claim an 8-slot win yet. Its teacher was a fixture.
- **Do not** claim a mixer win yet. The static mixer loses BPC; the learned
  mixer needs per-bit weight sets and stable integer updates.

## Next evaluation jobs

1. PG-19 / Gutenberg full-stream BPC with a held-out book, not a 30 KB excerpt.
2. Repository-holdout and time-holdout code streams.
3. Qwen teacher card on the same held-out bytes; teacher top-1 must exceed the
   3-gram before distillation is meaningful.
4. Pareto table with raw, gzip, and packed-page model sizes.
5. Paired block bootstrap for every table; never report a bare 128-context
   top-1 again.
