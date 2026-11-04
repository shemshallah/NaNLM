# Distributed training on reporam / repovram

NaNLM is designed to be trained by volunteers in **additive shards**. There is
no gradient sync, no floating-point optimizer state, and no need for a single
machine to hold the model.

## Why counting

The model is a **simplex slot memory**: one prototype per observed context,
with a next-byte count table. Training is:

```
for each 3-gram context:
    counts[context][next_byte] += 1
```

That operation is:

- **commutative** — shard order does not matter;
- **associative** — merging is a fold;
- **integer-only** — no floats, no rounding drift;
- **mergeable** — `sum(shards)` is exact;
- **slow-training friendly** — a volunteer can process 1 MB or 1 GB; more
  words only add counts.

This is the "special training method": **train by counting, merge by adding.**
It turns volunteer training into a CRDT, not a distributed optimizer.

The repository already has the primitive:

```sh
build/nanlm_shard build CORPUS OUT [START END]
build/nanlm_shard merge OUT A.shard B.shard ...
build/nanlm_shard stats FILE
```

`tools/build_shard.sh` proves `A+B == B+A` byte-for-byte.

## Storage tiers (no Cloudflare yet)

Until R2/S3 is available, use the deeprepo tiers that already exist:

| tier | medium | contents | mutability |
|---|---|---|---|
| **repovram** | GitHub release bins, 4 MiB pages, HTTP Range | compiled model generation: codebook, prototypes, labels, `.nan` image | immutable per generation |
| **reporam** | GitHub release bins + journal deltas | training shards, merge manifests, volunteer leases, metrics | append-only deltas + compaction |
| L0 | runner RAM / SSD | page cache for the current shard and hot prototypes | ephemeral |
| GitHub issues | control plane | volunteer claims, shard attachments, model cards | append-only |

Geometry follows deeprepo: 16 bins × 510 pages × 4 MiB ≈ 31.9 GiB virtual per
tier, sparse (unwritten pages read as zeros). A shard is small: 30 KB of Alice
produces ~72 KB of merged shard records. A 1 GB corpus produces ~2.4 GB of
records; S3/R2 is the eventual home, reporam is the bridge.

### repovram layout (proposed generation)

| page | content |
|---|---|
| 0 | `NANLMSB1` superblock: generation, key count, matrix words, codebook offset, labels offset, shard-set hash |
| 1.. | prototype matrix (each key = 24 words = 96 bytes), codebook (256 × 8 words), labels |
| tail | optional `.nan` hot image for `nanlm_vm` |

A generation is write-once. Readers page it over HTTP Range; the hot cache is
on the runner.

### reporam layout (proposed)

| region | content |
|---|---|
| superblock | `NANLMRB1` |
| `shards/` | one record file per volunteer contribution, content-addressed by hash |
| `merge/` | merge manifest: input shard hashes, output hash, totals, timestamp |
| `leases/` | volunteer/job lease: corpus range, shard id, expiry |
| `metrics/` | eval card, n-gram baseline, top-1/BPC, step counts |
| `generations/` | pointer to the repovram generation and its parent |

Writes are append-only `d-<seq>.dlt` page deltas, compacted into trimmed bins,
exactly like deeprepo.

## Training loop

```
volunteer / runner:
  1. lease a corpus slice from reporam
  2. build a shard:  (3-gram key, next byte, count)
  3. hash the shard; upload it as a reporam delta
  4. release the lease

merge job:
  1. read the shard set (or a prefix)
  2. merge: sum counts, saturating at 65535
  3. compile: one simplex-coded prototype per context
  4. eval: top-1 / BPC against an interpolated n-gram on a held-out slice
  5. publish a new repovram generation + model card
```

The compile step is deterministic: same shard set → same prototype matrix →
same `.nan` image → same query hashes.

## Why this can win at scale

Current small-corpus result (Alice, 128 held-out contexts):

| model | top-1 | bits/byte |
|---|---:|---:|
| simplex slot memory | 51.6% | **2.72** |
| in-repo interpolated n-gram | **55.5%** | 3.13 |

The slot memory already wins bits-per-byte. It loses top-1 on 25 KB of text.
That is the honest starting point, not the finish line.

The scaling path:

1. **More data.** Counting has no convergence requirement; every shard helps.
2. **Longer contexts.** A slot memory can use 4–8 slots without changing the
   training rule; the table grows, reporam absorbs it.
3. **Teacher distillation.** Volunteers run a small open teacher on a shard and
   emit `(context, next-byte, weight)` counts. The slot memory then stores the
   teacher's distribution, not just raw text. This is how the memory can beat a
   plain n-gram on top-1.
4. **Hot/cold split.** Repovram holds the top-K prototypes; reporam holds the
   long tail. Inference streams the hot set; a shard server can query cold
   prototypes later.
5. **WASM volunteers.** `nanlm_shard` and `nanlm_vm` are C99; both can compile
   to WASM. A browser volunteer can count a text shard or run the teacher on a
   WebGPU backend, then upload a shard.

## Volunteer phases

### Phase 0 — GitHub Actions (now)

- A volunteer opens an issue with a corpus slice or a text file.
- A workflow dispatches a runner, builds the shard, uploads a reporam delta.
- A merge workflow folds shards and publishes a new generation.
- No inbound connections; no R2.

### Phase 1 — local CLI (now)

- `nanlm_shard build` runs on a phone or laptop.
- The shard is attached to an issue/PR or uploaded as a release asset.
- A maintainer merges with `nanlm_shard merge`.

### Phase 2 — WASM volunteers (long term)

- Browser page loads a text slice and `nanlm_shard.wasm`.
- The volunteer contributes CPU (counts) or GPU (teacher inference).
- Shards upload to R2/S3 when available; the same format works unchanged.

## Gates

- `make shard-check` — `A+B == B+A`, exact totals.
- `make model-slot-check` — shard → model → `.nan` → `nanlm_vm` → diff.
- `make model-lm-check` — class-memory baseline and n-gram comparison.
- Every generation records the shard-set hash and the eval card.
- A corrupted shard must fail the content hash; a corrupted model must fail the
  query hash or the `.nan` gate.

## What not to do

- Do not put gradients or float optimizer state in reporam. Counts only.
- Do not require a single writer during training. Merge is commutative.
- Do not claim a top-1 win on Alice; report the table.
- Do not put the whole corpus in repovram. Shards go to reporam; only compiled
  prototypes go to repovram.
- Do not wait for R2. The shard format is storage-neutral.
