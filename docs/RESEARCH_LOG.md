# Research log

Short, dated entries. Negative results stay in.

## 2026-11-03 — simplex slot memory

One prototype per observed 3-gram; 255-bit simplex codewords in three slots;
nearest-key retrieval; majority next byte.

| model | top-1 | BPC |
|---|---:|---:|
| slot memory (3832 keys) | 51.6% | **2.72** |
| in-repo interpolated n-gram | **55.5%** | 3.13 |

The slot memory wins bits-per-byte and loses top-1 on 25 KB of Alice. The
important architectural change is that training is now **additive counting**:
a shard is `(3-gram, next byte, count)`, and merge is integer addition.
`A+B == B+A` byte-for-byte (`make shard-check`). This is the volunteer/S3
training unit.

## 2026-11-03 — M2b calibrated evaluation

128 held-out contexts, temperature calibrated on a disjoint 128-context slice.
D=2,048, hash-bucket memory.

| clusters/byte | top-1 | BPC |
|---:|---:|---:|
| 1 | 20.3% | 5.74 |
| 2 | 24.2% | 5.57 |
| 4 | 22.7% | 5.51 |
| 8 | **26.6%** | 5.50 |
| 16 | 25.0% | **5.38** |
| order-3 n-gram | 50.0% | 3.85 |

The class-level log-sum-exp over prototypes fixed the earlier uncalibrated
BPC. The model still loses to the n-gram, but the number is now meaningful.
Eight clusters is the best top-1; sixteen is the best BPC. The default stays at
eight because the runtime is half the size.

## 2026-11-03 — 32-query exploration (superseded)

These numbers use only 32 contexts and a single softmax over prototypes; treat
them as exploratory, not headline results. The 128-context table above
supersedes them.

Corpus: first 30,000 bytes of Project Gutenberg #11 (*Alice*), 85/15 split by
byte offset. D=2,048, vocab=256, context=3, 32 held-out contexts.

| model | top-1 | n-gram top-1 | n-gram BPC | runtime |
|---|---:|---:|---:|---:|
| HDC 1 prototype/byte | 21.9% | 50.0% | 3.78 | not measured |
| HDC 2 prototypes/byte | 25.0% | 50.0% | 3.78 | not measured |
| HDC 4 prototypes/byte | 25.0% | 50.0% | 3.78 | 70.3M steps |
| HDC 8 prototypes/byte | 34.4%* | 50.0% | 3.78 | 140.4M steps |
| HDC 16 prototypes/byte | 28.1% | 50.0% | 3.78 | 280.8M steps |

`*` 32-query subset. The 128-context table is the headline; one query is 3.1
points, so the old ranking was noisy.

The n-gram wins on quality. The HDC model is real, runs as a `.nan` image, and
matches the C adapter on every predicted byte and hash, but it is not
competitive.

### What this says

The 32-query table is too noisy to rank cluster counts. It was useful for
finding the pipeline and the first negative results, not for a model card.
The 128-context calibrated table is the headline.

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

### K-means memory (negative)

K-means per byte class (4 iterations) on the 128-context eval:

| K | top-1 | BPC |
|---:|---:|---:|
| 4 | 21.1% | 5.60 |
| 8 | 23.4% | 5.56 |
| 16 | 21.1% | 5.66 |

Below the hash-bucket memory (26.6% / 5.50). The centroids collapse toward
common contexts. Keep k-means as an ablation, not the default.

### Exemplar memory (negative)

Reservoir exemplars on the 128-context eval:

| K | top-1 | BPC |
|---:|---:|---:|
| 4 | 14.1% | 6.56 |
| 8 | 18.0% | 6.25 |
| 16 | 21.9% | 5.90 |

First-K exemplars were worse (3–22% on 32 contexts). Nearest-exemplar is not
automatically better than a class centroid; reservoir sampling helps but does
not win.

### Hypotheses after the 128-context run

1. **Capacity is not the bottleneck.** Bigger D did not help in the 32-query
   sweep; retest D=4,096/8,192 at 128 contexts before scaling.
2. **Memory shape mattered, but not as expected.** Hash buckets beat k-means
   and reservoir exemplars. The next memory experiment should be a
   context-aware hash or a learned partition, not vanilla k-means.
3. **Decoder calibration mattered.** Class-level log-sum-exp fixed the
   uncalibrated BPC. A top-k vote or a small readout is the next decoder
   experiment.
4. **Byte classes are hard.** A 32–64 symbol tokenizer is a different task but
   may reveal whether the substrate is the problem or the class count.
5. **Binding is still open.** XOR is a baseline; `BSWAPM` and learned
   `BOOL.n` binding are controlled ablations.

### Next experiments, in order

1. Retest D=4,096 and D=8,192 at 128 calibrated contexts; the earlier sweep
   used 32.
2. Context-aware hash or learned partition; vanilla k-means lost.
3. Top-k prototype vote; nearest-1 is leaving information on the table.
4. A 64-symbol tokenizer, clearly labeled as a different task.
5. `BSWAPM` vs XOR position binding.
6. Seal/loader, so the model image is tamper-evident.

### What not to do

- Do not claim the n-gram result is a failure of the substrate; it is a
  failure of this memory/decoder at this scale.
- Do not hide the 50% baseline.
- Do not report HDC BPC from an uncalibrated softmax.
- Do not add a router or experts before the memory/decoder baseline is honest.
