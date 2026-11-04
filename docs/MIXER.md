# Multi-order context mixing

The mixer is the next quality path. The target is the PAQ/lpaq recipe:

```
context bank -> hashed count models -> integer logistic mixer -> arithmetic coder
```

The slot memory is a high-dimensional associative memory. A mixer is a
**compression ensemble**: many weak next-byte predictions are combined by a
small model selector. It is complementary, not a replacement.

## Current C implementation

`src/nanlm_mixer.c` / `tools/nanlm_mixer.c`:

- 12 global templates:
  - orders 0, 1, 2, 3, 4, 6, 8;
  - skip-gram `(-2,-4,-6)`;
  - ctype of the last five bytes;
  - word-0 and word-1 hashes;
  - code state: indentation, bracket depth, last byte, line start.
- 3 document-local templates: orders 2, 3, 4.
- a rolling match model over the last four bytes.
- 512 mixer regimes = 64 context regimes × 8 bit positions.
- 16-bit fixed-point logistic weights, Q8.
- fixed-size 2-way hashed count tables with 16-bit check tags and signed
  32-bit counters.
- a per-document cache that can adapt while global counts stay frozen.
- exact count merge by integer addition (`nanlm_mixer_merge_counts`).

The current runner defaults to **static equal-weight mixing** because the
integer weight update was unstable in the first C pass. That is recorded as a
negative result, not hidden.

## Full-stream result

Alice, 4,492 held-out bytes, table bits 14:

| model | top-1 | BPC |
|---|---:|---:|
| static mixer | 45.97% | 3.7249 |
| static mixer + online document cache | 46.68% | 3.7003 |
| 3-gram baseline | 48.13% | 3.4203 |
| 3-slot typed memory | **49.05%** | **2.9593** |

The static mixer loses. Its model is 6.26 MB, so it is not Pareto either.

## Why the first mixer lost

1. **Equal weights are not a mixer.** The templates are not equally useful at
   every bit position or context; a static average dilutes strong predictors.
2. **Integer learning was unstable.** The first update rule saturated weights
   and flipped their sign; this is why the default is static.
3. **No state maps / APM / SSE.** PAQ gets much of its gain from secondary
   symbol estimation and calibrated probability maps, not just counts.
4. **Hashed collisions.** 12 templates at 2^14-2^16 buckets with 2-way probing
   is small for code; collisions matter.
5. **No match-model confidence calibration.** A raw match prediction is too
   strong or too weak depending on match length.
6. **The document-local cache helps only slightly.** Adapting the document
   table online improves 3.7249 → 3.7003 BPC, still behind the n-gram. The
   global model remains frozen.
7. **The stream is tiny.** Model-size accounting dominates; this is a
   plumbing result, not a compression result.

## The right next mixer

### Per-bit, per-regime weight sets

The mixer should select a weight vector by:

```
regime = H(last 2-4 bytes, ctype, indent, bracket depth, word shape)
weight = W[regime][bit_position][template]
```

The current 512 regimes already include bit position; the next step is to make
the learning update stable:

- weights Q8 or Q16, not int16 accumulators;
- fractional update accumulator per weight;
- clamp only at the probability boundary, not the weight;
- train on a warm-up split, then freeze for the eval split;
- report both fixed and adapted mixer rows.

### Two-layer mixer

A single mixer cannot model “which mixer is good here.” Use:

```
layer 1: 8-16 template groups -> 4-8 group predictions
layer 2: group predictions -> final byte probability
```

The regime router is the **selector**, exactly as the LUT/regime work proposed:
it picks a weight set, not a whole model.

### Count models

Keep counts signed and additive. Use:

- 3-4 hash rows per template (count-min);
- 16-bit check tags for diagnostics, not destructive replacement;
- merge by integer addition;
- per-document cache as a separate table with a document-generation id;
- exact unlearning by re-merging without a shard or by adding a negated shard.

### Match model

The match model should output:

- match length;
- candidate byte;
- confidence by length bucket;
- a separate “match broke” state.

Feed all four as mixer inputs, not a single fixed probability.

### Calibration

Add a small SSE/APM stage after the mixer:

```
p1 = mixer(...)
p2 = apm[quantized_p1][small_context]
```

The APM table is tiny, integer-only, and mergeable by counting.

## Evaluation contract

Every mixer run must produce:

- full held-out stream BPC, not 128 sampled contexts;
- paired block-bootstrap CI against the n-gram;
- top-1 with CI;
- raw/gzip/packed model bytes;
- Hutter-style total BPC;
- a fixed training/eval split and a frozen eval model;
- exact model hash.

`make mixer-check` is only a smoke gate. The result is not promoted until it
beats the 3-slot memory on the declared stream at a declared model size.

## Relation to the other layers

- **3-slot memory**: the current quality baseline.
- **8-slot memory**: longer context; needs a real teacher and suffix-covering
  prototypes. The first 8-slot run with the fixture teacher is not a verdict.
- **Mixer**: the compression path. It can use the slot memory as one input
  template later.
- **LUT/regime network**: the mixer selector is its natural inference form.
- **Datastore LM**: the mixer can interpolate a global memory, a document
  cache, and a transformer next-token distribution.
