# Feature and teacher probes

Two honest probes: a learned feature layer and a small neural teacher.

## 1. Typed ctype feature layer — positive

The typed slot memory is implemented in C and NaN16:

```
3 byte slots + 3 ctype slots = 1,536 bits
```

On the Alice split:

| model | top-1 | BPC |
|---|---:|---:|
| raw 3-slot memory | 51.56% | 2.7196 |
| **typed 6-slot memory** | **52.34%** | **2.6703** |
| in-repo n-gram | 55.47% | 3.1304 |

Runtime: 197,829,777 steps for 32 queries, exact output, no trap.

The ctype layer is real: it improves top-1 and BPC without changing the
memory or the decoder.

## 2. Learned embedding features — negative

A 3-byte MLP teacher was trained for 5 epochs and its 16-dimensional byte
embedding was quantized into 8-bit codes. Those codes replaced the ctype slots.

Results:

| feature slots | top-1 | BPC |
|---|---:|---:|
| hand ctype | **52.34%** | **2.6703** |
| learned embedding bits [0..7] | 51.56% | 2.723 |
| learned embedding bits [8..15] | 50.78% | 2.752 |
| learned embedding, interleaved | 50.78% | 2.755 |

The learned embedding did not beat the hand-written ctype feature on this
corpus. This is a useful negative: **learned features are not automatically
better than a small, well-chosen typed feature layer.**

## 3. Small neural teacher — insufficient for top-1

A 3-byte embedding MLP (48 → 128 → 256, 5 epochs) on the full Alice text:

| model | top-1 | BPC |
|---|---:|---:|
| MLP teacher | 49.86% | **2.579** |
| in-repo n-gram | **55.47%** | 3.130 |
| typed slot memory | 52.34% | 2.670 |

The small teacher beats the n-gram on bits-per-byte but not on top-1. A student
distilled from it cannot win top-1 on this split.

Conclusion: the teacher needs **more context and more capacity**, not a better
decoder. The next teacher should be Qwen2.5-0.5B/7B (Apache-2.0) or a byte-level
transformer trained on a larger public corpus.

## 4. Teacher -> typed slot plumbing

The full compile path now exists:

```
teacher records -> last-3-byte aggregation -> typed slot model -> .nas -> nanlm_vm
```

`make teacher-slot-check` passes, but the fixture n-gram teacher is weak:

```
teacher_records=97124 contexts=16376 keys=4094
hdc_top1=33.59% hdc_bpc=3.5187
ngram_top1=55.47% ngram_bpc=3.1324
```

The plumbing is exact; the quality is not. A real Qwen/byte-level teacher
replaces the fixture without changing the path.

## 5. What this rules out

- More context alone (4–6 bytes) made the raw memory worse.
- Top-k decoding improved BPC but not top-1.
- A small learned embedding did not beat ctype.
- A small 3-byte neural teacher did not beat the n-gram on top-1.

## 6. What remains

1. **Qwen teacher distillation** — byte-marginalized top-k distributions,
   additive shards, typed slot memory.
2. **8-slot student / longer teacher context** — a teacher conditioned on more
   than 3 bytes, evaluated at 8-byte student context.
3. **True learned LUT features** — not an embedding quantized to bits; a LUT
   network trained end-to-end with STE.
4. **8×8 digit task** — parameter-matched sparse vs dense on real data.
5. **Mirror ensemble and capability loader.**

The feature layer is real. The teacher is the missing quality ceiling.


## 7. Full-stream correction and 8-slot / mixer status

The 128-context table was too small: at n=128 the top-1 standard error is about
±4.4 points. On the full 4,497-byte held-out stream:

| model | top-1 | BPC |
|---|---:|---:|
| 3-slot typed memory | **49.05%** | **2.9593** |
| 3-gram | 48.14% | 3.4196 |
| 8-slot (fixture teacher) | 44.79% | 3.7648 |
| static mixer (table 14) | 45.97% | 3.7249 |

The 3-slot BPC win is real (paired CI excludes zero); the top-1 difference is a
tie. The 8-slot fixture-teacher and static-mixer rows are negatives, not
verdicts on the 8-byte or PAQ directions. See `docs/EVAL.md`, `docs/QWEN.md`,
and `docs/MIXER.md`.


## 8. Qwen 7B teacher result

The 7B teacher card is **88.78% top-1 / 0.6606 BPC** on 303 held-out token
boundaries, but the byte-level student does not improve. A scale grid with the
existing shard shows the ngram-only 8-slot control at **49.51% / 3.4201 BPC**;
teacher scales 8, 16, and 32 are worse. The raw 3-slot model remains
**49.05% / 2.9593 BPC**. This is a representation mismatch, not a teacher
failure: token-level teachers need a token-level student. See
`docs/QWEN7B.md`.
