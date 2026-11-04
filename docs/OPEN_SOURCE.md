# Why this can be an open-source world effort

NaN is the aesthetic. The architecture is what makes it a world effort.

## 1. The properties that matter beyond NaN

### 1.1 Training is mergeable

The training unit is a shard:

```
(context, next byte, count/weight)
```

Merge is integer addition. It is commutative, associative, and order-free.
That means:

- a volunteer can train for five minutes or five days;
- shards can arrive out of order;
- no global optimizer state, no lockstep, no gradient sync;
- a fork can merge a different subset and publish a different generation;
- a failed shard is rejected by hash, not by a central scheduler.

This is the single most important property. It turns training from a
coordinated job into a **CRDT**.

### 1.2 Identity is content-addressed

The repo holds the manifest, the hashes, the eval card, and the code. The
bytes can live anywhere. A generation is a hash, not a server.

That means:

- a fork is a real model, not a copy of a service;
- a mirror can replace a backend without changing the model;
- a RAM farm or GPU farm is a residence, not an owner;
- GitHub can disappear and the model can still be rebuilt from shards and
  manifests.

### 1.3 Teacher and student are separate

The teacher is a pretrained Apache model. The student is an open slot memory
distilled from it. The teacher supplies quality; the student supplies
openness, portability, and mergeable training.

That means:

- the project does not need to train a frontier model from scratch;
- the student can be re-distilled when a better teacher appears;
- the teacher can be swapped without changing the student format;
- the artifact that ships is small enough for a phone.

### 1.4 Memory is storage-agnostic

The model only needs:

```
read(hash, offset, length)
exists(hash)
```

reporam and repovram are the current backends. S3/R2, a RAM farm, a GPU farm,
or a volunteer mirror can serve the same pages later.

### 1.5 The substrate is small and portable

- `nanlm_asm` and `nanlm_vm` are C99.
- A model runs offline from one `.nan` file.
- The hot image is megabytes, not gigabytes.
- WASM is a realistic volunteer target.
- The whole thing is inspectable: 32-bit payloads, truth tables, labels.

### 1.6 Every claim has a gate

- lane identity and mirror law;
- word packing;
- clean-HALT requirement;
- shard merge is byte-identical;
- teacher vs n-gram;
- student vs teacher;
- C adapter vs `.nan`;
- negative gates: zeroed positions, swapped labels, step starvation.

Open-source trust is not a license file. It is evidence that can fail.

## 2. Why open source is the right shape

A closed version of this would be a service that owns the shards and the
memory. An open version can be:

- **forkable** — anyone can take the manifest and shards and publish a
  generation;
- **auditable** — the counts, labels, teacher card, and eval card are public;
- **contributable in small units** — a shard, a teacher run, a mirror, a
  benchmark, a port;
- **hardware-neutral** — CPU, phone, WASM, GPU farm, RAM farm;
- **resilient** — no single backend is the brain.

The community contribution ladder is unusually smooth:

1. run a benchmark;
2. submit a shard;
3. review a shard;
4. run a teacher on a corpus slice;
5. mirror a generation;
6. port the VM;
7. propose an RFC;
8. maintain a generation.

No step requires owning a cluster.

## 3. The initial lifespan

### Stage 0 — Local proof (now)

- Alice corpus, simplex slot memory, 3 slots.
- `nanlm_asm` / `nanlm_vm`, shard merge, model card.
- Metrics: 51.6% top-1 / 2.72 BPC vs n-gram 55.5% / 3.13.
- Gate: exact shard merge, exact runtime diff, CI green.

What it proves: the format, the merge, and the runtime are real.

What it does not prove: a smart model, a teacher, or a public training ring.

### Stage 1 — Teacher bootstrap

- Teacher: Qwen2.5-0.5B-Instruct (Apache-2.0), byte-marginalized at token
  boundaries.
- Corpus: 10 MB public-domain text.
- Student: 3-slot slot memory, top-4 teacher bytes per context.
- Artifacts: teacher card, teacher shards, student `.nan`, model card.
- Gate: teacher must beat the n-gram before distillation; student must match
  the C adapter; student quality must be reported honestly.

What it proves: a pretrained teacher can make the student smart immediately.

Risk: BPE-to-byte marginalization is approximate. The card must say so.

### Stage 2 — GitHub shard ring

- Issue/PR-driven volunteer shards.
- reporam holds shards, leases, merge manifests, eval cards.
- A merge workflow folds shards, compiles, evaluates, and publishes a
  generation.
- A simple leaderboard: top-1/BPC vs n-gram, shard count, model size.

What it proves: volunteers can contribute without coordination.

Risks: GitHub API/storage budgets, shard spam, license drift, review load.

### Stage 3 — 8-slot student and 7B teacher

- Student context: 8 bytes, 64 words per prototype.
- Teacher: Qwen2.5-7B-Instruct (Apache-2.0), Q4, paged from repovram.
- Top-K contexts in repovram; cold contexts in reporam.
- Gate: teacher vs student vs n-gram on the same 128-context split.

What it proves: longer context and a stronger teacher can beat the n-gram
top-1, not just BPC.

Risk: prototype table size and teacher inference cost.

### Stage 4 — Public model zoo and benchmark

- Multiple corpora: public domain, code, multilingual.
- Multiple teachers: 0.5B, 1.5B, 7B.
- Multiple student sizes: 3-slot, 8-slot.
- A benchmark harness that anyone can run.
- Model cards with teacher hash, corpus hash, shard set, eval numbers.

What it proves: the architecture generalizes beyond one corpus.

Risk: benchmark overfitting, leaderboard gaming.

### Stage 5 — WASM and GPU volunteers

- `nanlm_shard.wasm` in the browser: count a text slice, upload a shard.
- WebGPU teacher inference: run the teacher on a volunteer GPU.
- R2/S3 storage when available; the shard format does not change.
- Federation: any store implementing `read(hash)` can serve a generation.

What it proves: the world effort can run without a central cluster.

Risks: browser memory limits, teacher quality differences, shard
verification, abuse.

### Stage 6 — Foundation and long-term stewardship

- Maintainers, release managers, mirror operators.
- Apache-2.0 code and weights; public manifests.
- Archival: at least two independent mirrors per generation.
- Deprecation policy: generations are frozen, not silently rewritten.
- Funding: grants, sponsors, compute donations.

What it proves: the model is a public artifact, not a startup asset.

Risks: maintainer burnout, quality plateau, license drift, storage cost,
centralization pressure.

## 4. Stage matrix

| stage | compute | storage | people | artifact | gate |
|---|---|---|---|---|---|
| 0 local | phone/CPU | git | 1 | `.nan` + shards | merge + diff |
| 1 teacher | runner | git/reporam | 1–3 | teacher card + student | teacher beats n-gram |
| 2 ring | runners | reporam | 5–20 | generation | shard hash + eval |
| 3 scale | runner/GPU | reporam + repovram | 20–100 | 8-slot student | student beats n-gram |
| 4 zoo | mixed | reporam/repovram | 100+ | model zoo | reproducible cards |
| 5 WASM | volunteers | R2/S3/mirrors | 1k+ | federated generations | store contract |
| 6 foundation | donated | multi-mirror | org | public utility | archive + governance |

## 5. The open-source contract

1. **Code is Apache-2.0.** No hidden runtime.
2. **Weights are Apache-2.0 or public domain.** No teacher license cloud.
3. **Manifests are in git.** A generation is reviewable in a PR.
4. **Every large byte range has a hash and a mirror.** No single-store brain.
5. **Every generation has a teacher card and a student card.**
6. **Every claim has a gate that can fail.**
7. **Forks can publish.** The format is the permission.

## 6. What could kill it

| failure | mitigation |
|---|---|
| Teacher license contaminates the artifact | Apache-only teacher matrix; teacher card |
| Student never beats the n-gram on top-1 | teacher distillation + 8-slot context; publish the negative if it fails |
| GitHub storage/API limits | shards on reporam now, R2/S3 later; mirrors |
| GitHub becomes the only home | content hashes + two mirrors + offline `.nan` |
| Shard spam / low-quality data | hash, size, eval regression, reviewer sign-off |
| Benchmark overfitting | held-out corpora, teacher card, negative gates |
| Maintainer burnout | small RFC process, release managers, funded mirrors |
| “Open” in name only | manifest + hashes + reproducible build in CI |

## 7. First 90 days

**Weeks 1–2**

- Teacher shard format and merge gate (done in code).
- Qwen0.5B byte-marginalized teacher card.
- 3-slot student distilled from the teacher.
- Teacher vs n-gram gate.

**Weeks 3–4**

- 8-slot student and 7B teacher.
- Generation manifest and mirror policy.
- Public benchmark harness.

**Month 2**

- GitHub shard ring.
- First external shard.
- Model zoo page and leaderboard.

**Month 3**

- WASM shard builder.
- Second mirror.
- First RFC: shard format v2 and teacher card.

## 8. The sentence to hold onto

> **NaN gives the machine its face. Mergeable counting gives the project its
> world. Content-addressed manifests give it a life after GitHub.**

That is the architecture.
