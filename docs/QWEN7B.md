# Qwen 7B teacher: first real GPU result

The Kaggle T4 run completed a Qwen2.5-7B-Instruct 4-bit teacher pass over the
first 2,048 tokens of Alice. The teacher card passed, but the byte-level
student did not improve. This is the most useful negative result so far.

## Teacher card

| field | value |
|---|---:|
| model | Qwen2.5-7B-Instruct, 4-bit |
| tokens | 2,048 |
| eval positions | 303 token boundaries |
| teacher top-1 | **88.78%** |
| teacher BPC | **0.6606** |
| n-gram BPC at the same positions | 2.9757 |
| teacher contexts | 1,521 |
| teacher records | 5,734 |

The teacher is genuinely strong at **token boundaries**. It is not a byte
teacher.

## Pure teacher student

| model | full-stream top-1 | full-stream BPC |
|---|---:|---:|
| 8-slot, 2,048 keys, pure Qwen | 16.32% | 9.0443 |
| 8-slot, 2,048 keys, Qwen + ngram, teacher weight 1/16 | 40.78% | 4.4930 |
| 8-slot, 4,096 keys, Qwen + ngram, teacher weight 1/16 | 47.60% | 3.5225 |

Pure teacher distillation is coverage-starved: the teacher emits only at token
boundaries, while the student is scored on every byte.

## Scale grid

The grid uses the existing 7B shard and a raw 3-gram teacher over the same
training range. `scale=0` is the n-gram-only control: no Qwen weight at all.

| teacher scale | 8-slot top-1 | 8-slot BPC | 3-slot teacher-slot top-1 | 3-slot teacher-slot BPC |
|---:|---:|---:|---:|---:|
| **0 (ngram only)** | **49.51%** | **3.4201** | 49.05% | 3.9316 |
| 8 | 46.75% | 3.6406 | 47.10% | 4.1110 |
| 16 | 47.60% | 3.5225 | 47.65% | 3.9976 |
| 32 | 47.86% | 3.4420 | 48.08% | 3.9179 |

Baselines on the same full held-out stream:

| model | top-1 | BPC |
|---|---:|---:|
| n-gram | 48.13% | 3.4207 |
| **raw 3-slot direct** | **49.05%** | **2.9593** |
| 8-slot ngram-only, 4,096 keys | 49.51% | 3.4201 |

The ngram-only 8-slot has a small top-1 win over the n-gram
(+1.38 points, 95% CI [0.45, 2.43]), but BPC is a tie
(−0.0005, 95% CI [−0.0645, 0.0668]). It is also a 9.5 MB model
(198 KB gzip), so it is not Pareto against the raw 3-slot.

At every non-zero teacher scale, the Qwen teacher makes the byte student worse
than the ngram-only control. Scale 32 is closest, but still ~0.02 BPC worse and
not better than scale 0.

## What this means

1. **The 7B teacher is not useless; the byte student is the wrong consumer.**
   The teacher predicts tokens. The student predicts bytes. First-byte
   marginalization plus token-boundary sampling loses most of the signal.
2. **Weight blending matters.** The raw teacher had 102.9M weight units vs
   6.5M for the n-gram. Without scaling it dominated and the hybrid was worse.
   At 1/32 it approaches neutral, but never beats the ngram-only control.
3. **The 3-slot teacher-slot is the wrong place for long-context teacher data.**
   Compressing an 8-byte teacher into 3 bytes destroys it. The 8-slot is better,
   but still not enough.
4. **Raw 3-slot memory remains the best BPC model** on the full stream:
   49.05% / 2.9593.
5. **Do not spend more GPU hours on byte-level distillation.** The next
   experiment is a token-level student.

## Why the ngram-only 8-slot is interesting

The 8-slot ngram-only control proves that the recency-graded codewords and
suffix-covering prototype selection work: it matches the n-gram BPC at 4,096
keys and slightly beats its top-1. That is an 8-byte memory result, not a
teacher result. It still has a large model and does not beat the raw 3-slot
BPC.

## Next model: token-level student

The clean match to the teacher:

- 4–16k token vocabulary, byte fallback;
- 3–6 token context, roughly 12–24 bytes;
- token IDs and top-k token probabilities stored directly;
- no BPE first-byte marginalization;
- token-level slot memory and `.nan` runtime later.

The byte-level result is a useful control. It shows that the pipeline can
produce a strong teacher card and exact `nanlm_vm` execution, but it cannot
turn a token teacher into a byte model by counting alone.

## Artifacts

The Kaggle run packaged:

- `qwen7b_alice.ntr` and card;
- `qwen7b_ngram.ntr`;
- `qwen7b_teacher_scaled.ntr`;
- `qwen7b_hybrid.ntr`;
- 8-slot and 3-slot students, streams, manifests.

The raw 7B weights are not part of the repo. Teacher shards are additive and
can be mirrored; the card and manifests are the durable evidence.
