# Research log

Short, dated entries. Negative results stay in.

## 2026-11-03 — M2b byte-level base

Corpus: first 30,000 bytes of Project Gutenberg #11 (*Alice*), 85/15 split by
byte offset. D=2,048, vocab=256, context=3, 32 held-out contexts.

| model | top-1 | n-gram top-1 | n-gram BPC | runtime |
|---|---:|---:|---:|---:|
| HDC 1 prototype/byte | 21.9% | 50.0% | 3.78 | not measured |
| HDC 2 prototypes/byte | 25.0% | 50.0% | 3.78 | not measured |
| HDC 4 prototypes/byte | 25.0% | 50.0% | 3.78 | 70.3M steps |
| HDC 8 prototypes/byte | **34.4%** | 50.0% | 3.78 | 140.4M steps |
| HDC 16 prototypes/byte | 28.1% | 50.0% | 3.78 | 280.8M steps |

All numbers use only 32 held-out contexts. The standard error is large: one
query is 3.1 percentage points. Do not treat the cluster ranking as settled.

The n-gram wins on quality. The HDC model is real, runs as a `.nan` image, and
matches the C adapter on every predicted byte and hash, but it is not
competitive.

### What this says

- Four prototypes per byte is not enough; eight is better.
- Sixteen is worse at this D and corpus; more capacity is not monotonic.
- The decoder is the bottleneck, not the encoder: the query hash matches
  exactly, so the runtime is doing the binding/bundling correctly.
- HDC probability calibration is not implemented. Do not report HDC bits/byte
  until a temperature or readout is selected on held-out data.

### Dimension sweep at 8 prototypes/byte

| D | top-1 |
|---:|---:|
| 1,024 | 25.0% |
| 1,536 | 21.9% |
| 2,048 | **34.4%** |
| 2,560 | 31.3% |
| 4,096 | 31.3% |
| 8,192 | 28.1% |

Bigger D did not help. That points at the memory/decoder, not raw capacity.

### Exemplar memory, first-K (negative)

First K training contexts per class, nearest exemplar:

| K | top-1 |
|---:|---:|
| 4 | 3.1% |
| 8 | 9.4% |
| 16 | 21.9% |

This is worse than the majority prototypes. First-K sampling is biased to the
start of the corpus. A reservoir or k-means selection is the next version, not
this one.

### Hypotheses

1. **Capacity:** D=2,048 is too small for 256 byte classes with order-3
   contexts. The next scale is D=8,192.
2. **Memory shape:** class centroids discard context diversity. Exemplar
   memory or k-means clusters should beat hash buckets.
3. **Decoder:** nearest Hamming with 8 buckets is crude. A top-k vote or a
   small learned readout could help without changing the keystone.
4. **Tokenizer:** byte classes are hard. A 32–64 symbol tokenizer might make
   the same substrate look much better, but it changes the task.
5. **Binding:** XOR is not necessarily the best binding for this corpus.
   `BSWAPM` and learned `BOOL.n` binding are worth a controlled ablation.

### Next experiments, in order

1. D=8,192, 8 clusters, same corpus, same queries. Measure top-1 and steps.
2. Exemplar memory: 256–1,024 held-out training contexts, nearest exemplar.
3. Calibrated decoding: temperature sweep on a disjoint validation slice;
   report BPC only if it beats uniform.
4. k-means buckets instead of hash buckets; same number of prototypes.
5. `BSWAPM` position binding vs XOR position vectors.
6. Context length 2/3/4.
7. A 64-symbol tokenizer, clearly labeled as a different task.

### What not to do

- Do not claim the n-gram result is a failure of the substrate; it is a
  failure of this memory/decoder at this scale.
- Do not hide the 50% baseline.
- Do not report HDC BPC from an uncalibrated softmax.
- Do not add a router or experts before the memory/decoder baseline is honest.
