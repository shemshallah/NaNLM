# Repo-anchored brains

The model must stay **open, reproducible, and GitHub-anchored** even when its
memory lives on a RAM farm or GPU farm. The way to do that is to separate
**identity** from **residence**.

## Identity vs residence

| layer | lives where | must be |
|---|---|---|
| code | git repo | open source, Apache-2.0 |
| generation manifest | git repo | small, diffable, hashed |
| shard index | git repo | hashes + counts, not bytes |
| eval card | git repo | reproducible metrics |
| training protocol | git repo | exact merge rule |
| model bytes | any content-addressed store | verifiable by hash |
| hot `.nan` image | repo, release, or mirror | runnable offline |
| RAM/GPU farm | optional backend | a mirror, not a dependency |

The repo answers **what the brain is**. The store answers **where it sleeps**.
No brain is allowed to exist only in a backend.

## Generation manifest

Every published model generation gets a small manifest committed to git:

```
magic NANLMGEN
version 1
model nanlm-slot
generation 3
format NANLMSB1
context 3
keys 3832
matrix_words 24
codebook_sha256 <hex>
prototypes_sha256 <hex>
labels_sha256 <hex>
hot_image_sha256 <hex>
shards <n>
shard_sha256 <hex> ...
eval_top1 <pct>
eval_bpc <bpc>
ngram_top1 <pct>
ngram_bpc <bpc>
corpus_sha256 <hex>
train_frac 0.85
code_commit <git sha>
storage <backend>:<locator>
mirror <backend>:<locator>
```

`nanlm_vm` never reads the manifest; it runs the `.nan` file. The manifest is
for humans, CI, mirrors, and recovery.

## Storage contract

A backend only has to provide:

```
read(hash, offset, length) -> bytes
exists(hash) -> bool
```

Optional write path for training:

```
put_shard(hash, bytes)
list_shards()
```

The current backends:

- **repovram** — immutable compiled generations, GitHub release bins, 4 MiB
  pages, HTTP Range.
- **reporam** — mutable shard/lease/merge register, append-only deltas.

Future backends use the same contract:

- S3/R2 bucket (when available);
- a volunteer RAM farm serving range requests;
- a GPU farm that runs the teacher and emits shards;
- IPFS/Arweave for archival mirrors.

Nothing in `nanlm_vm`, `nanlm_asm`, or the model format knows which backend is
in use.

## Why GitHub still matters

GitHub is the **coordination and provenance layer**, not the execution layer:

- the code and protocol are public;
- the manifest is reviewable in a PR;
- CI verifies the hashes and the eval card;
- issues/PRs carry volunteer claims and shard attachments;
- releases carry the hot image and, when small, the shards.

If a backend disappears, the manifest still names the hashes. Any surviving
mirror can serve them. If the hot image is committed, the model still runs
offline.

## Failure modes

| failure | recovery |
|---|---|
| GitHub release asset deleted | fetch the same hash from a mirror; re-upload |
| R2/S3 unavailable | repovram or a volunteer mirror serves the pages |
| one shard corrupted | reject by hash; merge the remaining shards |
| all shards lost, hot image survives | the model still runs; training is not reproducible |
| all bytes lost | the manifest is not a model; mark the generation dead |
| repo deleted | Apache-2.0 code can be forked; mirrors keep the manifests |

A generation is only "published" when it has:

1. a committed manifest;
2. a committed or mirrored hot `.nan` image;
3. at least one independent mirror for bytes that are too large for git;
4. a passing eval card and a hash check in CI.

## Anti-patterns

- Do not make the runtime call GitHub at inference time.
- Do not store a brain only in a release asset.
- Do not call a model "open" if its manifest is private.
- Do not merge shards without content hashes.
- Do not trust a backend's word for a page; verify the hash.
- Do not let a GPU farm hold a model that cannot be rebuilt from the manifest.

## The contract

> **The repo defines the brain. The store holds its memory. The hash is the
> identity. `nanlm_vm` is the execution.**

That holds whether the memory is reporam today, an S3 bucket tomorrow, or a
volunteer RAM farm later.
