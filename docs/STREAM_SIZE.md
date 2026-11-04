# Stream-size results

## Full Alice held-out stream

`data/alice_full.txt` (151,191 bytes), 85/15 split:

```sh
./build/nanlm_slot_train --corpus data/alice_full.txt \
  --eval-log build/slot_full.tsv \
  --out build/slot_full.nas \
  --expected build/slot_full.expected \
  --manifest build/slot_full.manifest
```

Result:

```text
keys=6417
stream_n=22676
stream_top1=55.91%
stream_bpc=2.3291
stream_ngram_top1=55.00%
stream_ngram_bpc=2.6704
```

Paired BPC difference: **−0.341 [−0.360, −0.322]**. Top-1 +0.91 points.

Model size:

```text
5,522,138 bytes raw
103,458 bytes gzip
```

On 22,676 held-out bytes that is still **38.83 total BPC** including model size.
The stream got larger; the model grew with it. This is the same scaling problem.

## 1.2 MB Gutenberg attempt

I downloaded Project Gutenberg #2701 (Moby Dick, 1,234,646 bytes) and tried the
same slot trainer. The naive slot model did not complete: the prototype table
grows with every observed 3-gram, and the full-stream Hamming evaluation grows
with both keys and stream length. This is exactly why fixed-size hashed tables
and a capped prototype/scorer are required.

## Conclusion

- Larger stream: **55.91% / 2.3291 BPC** on 22,676 bytes, a real improvement
  over the 4.5 KB run.
- Model size still dominates: 103 KB gzip is 36.5 BPC equivalent on 22 KB.
- The naive slot model is not the path to 1 MB+ evaluation.
- The next large-stream evaluator must use a **bounded** model: hashed count
  tables, capped prototypes, or token HDC with a hard model-size budget.

## 1.2 MB mixer test

Project Gutenberg #2701, 1,234,646 bytes, 85/15 split (185,189 held-out bytes):

| model | BPC | model bytes | total BPC |
|---|---:|---:|---:|
| mixer, table 14 | 3.6024 | 6,258,784 | 273.98 |
| mixer, table 12 | 3.9865 | 2,719,840 | 121.48 |
| n-gram baseline | **2.6417** | — | — |

Top-1 was disabled (`--top1-limit 0`) because exact byte argmax over 185k
positions is too slow for this evaluator; BPC is unaffected.

The fixed-size mixer loses badly to the n-gram baseline and its tables still
dominate the model-size budget. Larger table sizes buy BPC but not a Pareto
point. The mixer needs a much smaller/quantized count representation before it
is a 1 MB-stream candidate.

## What is actually bounded

- The naive slot model grows with every 3-gram; it failed on 1.2 MB.
- The mixer is bounded but too large; table 12 = 2.72 MB, table 14 = 6.26 MB.
- The 512/1024-prototype token HDC is 36–67 KB sparse, but its n-gram fallback
  is 141 KB and its BPC is ~2.47 on a tiny stream.
- The teacher lookup + document cache is 2.15 BPC but has no compiled model
  size yet.

The next credible 1 MB candidate is the **token HDC + doc cache with the
n-gram fallback removed or bounded**, not another larger mixer table.
