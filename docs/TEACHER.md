# Teacher selection

A public model needs to be smart immediately. The fastest honest route is
**teacher distillation into the additive slot memory**, not training a
transformer inside NaN16.

This document picks the teacher by role, not by hype.

## The candidate matrix

| candidate | params | license | byte-aligned? | causal? | role |
|---|---:|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 494M | Apache-2.0 | no (BPE) | yes | bootstrap teacher |
| Qwen2.5-1.5B-Instruct | 1.54B | Apache-2.0 | no (BPE) | yes | stronger bootstrap |
| Qwen2.5-7B-Instruct | 7.6B | Apache-2.0 | no (BPE) | yes | primary teacher |
| Qwen2.5-14B-Instruct | 14.8B | Apache-2.0 | no (BPE) | yes | high-quality, expensive |
| SmolLM2-360M-Instruct | 362M | Apache-2.0 | no (BPE) | yes | tiny bootstrap |
| SmolLM2-1.7B-Instruct | 1.71B | Apache-2.0 | no (BPE) | yes | small primary |
| TinyLlama-1.1B-Chat | 1.1B | Apache-2.0 | no (BPE) | yes | fallback |
| ByT5-small | ~300M | Apache-2.0 | yes | no (encoder-decoder) | not a streaming teacher |
| CANINE-s | ~120M | Apache-2.0 | yes | no (encoder) | not a streaming teacher |
| Llama-3.2-1B | 1.2B | Llama license | no | yes | avoid for an Apache artifact |
| Qwen2.5-3B | 3.1B | “other” | no | yes | avoid; license ambiguity |
| char GRU (ours) | 2–10M | Apache-2.0 | yes | yes | CI/dev teacher |
| byte-level transformer (ours) | 10–40M | Apache-2.0 | yes | yes | long-term teacher |

The two non-negotiables:

1. **Apache-2.0 or public domain.** The student is an open artifact; its
   teacher should not add a license cloud.
2. **Causal next-byte signal.** Encoder-only or span-denoising models
   (ByT5, CANINE) are byte-aligned but not streaming; scoring every next byte
   requires a decoder pass per position.

## Bootstrap teacher: Qwen2.5-0.5B-Instruct

Why:

- Apache-2.0.
- 494M params; Q4_K_M is roughly 300–400 MB.
- Runs on a standard GitHub runner without paging.
- Strong enough to produce a smart byte distribution on natural text.
- A 1.5B variant is a drop-in upgrade when a runner has room.

Its weakness: it is a BPE model, so the next-token distribution must be
marginalized to the next byte. That is acceptable for the bootstrap.

## Primary teacher: Qwen2.5-7B-Instruct

Why:

- Apache-2.0.
- 7.6B params; Q4_K_M is roughly 4.3–4.7 GB.
- Fits a 7 GB runner with paging and a bounded cache.
- Fits repovram as a paged teacher: the same tier that holds a 30B deeprepo
  model can hold a 7B teacher comfortably.
- Better long-range context and semantics than 0.5B/1.5B.

This is the "smart base" for the first serious generation. The teacher runs on
a runner or GPU farm, never on the phone. The student is what ships.

## Long-term teacher: our own byte-level causal transformer

The BPE marginalization is a bridge, not the destination. The durable teacher
should be:

- **byte-level** — every target is exactly `P(next byte | prefix)`;
- **causal** — one forward pass gives all next-byte logits;
- **10–40M params** — trainable on a single GPU, cheap enough for volunteers;
- **trained on public-domain text** — Project Gutenberg / PG-19, not a
  share-alike corpus, so the artifact is clean;
- **Apache-2.0 code and weights** — no third-party model license.

A 12-layer, 384-dim byte transformer with 512-byte context is a good first
target. A 2-layer GRU with 512 hidden is the CPU/CI fallback.

This teacher can be improved by volunteers with GPU time. The student is
re-distilled after each teacher generation.

## BPE-to-byte marginalization

If the teacher is BPE, distill only at **token boundaries**:

1. Tokenize the prefix.
2. Run the teacher; get `P(next token = t | prefix)`.
3. Keep the top-k tokens.
4. For each token, take its first byte `b = spell(t)[0]`.
5. `P(next byte = b) = sum_{t: spell(t)[0]=b} P(t)`.

This is exact for the first byte of the next token. It ignores the case where
the context ends inside a token. That is acceptable for a bootstrap; the
byte-level teacher removes the limitation.

Do not pretend a BPE distribution is a full byte distribution. Label the
student card accordingly.

## Why not the obvious alternatives

- **ByT5 / CANINE**: byte-aligned but non-causal. Scoring a corpus requires a
  decoder pass per position, which is far more expensive than one causal
  forward pass.
- **Llama 3.2**: strong, but the license is not Apache. Keep the artifact
  clean.
- **Qwen2.5-3B**: license marked “other”; avoid when 1.5B and 7B are Apache.
- **30B+ teachers**: better quality, but the distillation throughput and
  repovram pressure are not worth it for the first generation. Scale after the
  student pipeline works.
- **A random/n-gram teacher**: not smart; it cannot beat the n-gram baseline
  that the student is supposed to beat.

## Distillation shard

The teacher shard is additive, like the corpus shard:

```
(context key, next byte, weight)
```

- `context key`: 3 bytes for the current runtime, 8 bytes for the next scale.
- `weight`: quantized teacher probability, 16-bit.
- top-k bytes per context, k = 4–8.
- merge is integer addition; shard order does not matter.

The student compiles the shard into one simplex-coded prototype per context.
Top-1 is the teacher argmax for seen contexts; unseen contexts use the nearest
simplex key.

## Cost model

For a 10 MB corpus, one teacher forward pass per byte is expensive if done
naively. Use teacher forcing over chunks:

- batch the corpus into windows of 512–2048 bytes;
- one forward pass yields logits for every position;
- marginalize to bytes;
- emit top-k records.

On a 0.5B teacher, a 10 MB corpus is a few GPU-minutes or tens of CPU-minutes.
On a 7B teacher, use a GPU farm or a long runner shift. The work is shardable
by corpus range, so volunteers can contribute.

## Teacher evaluation gate

Before distilling, measure the teacher on the same 128-context split:

- top-1 next-byte accuracy after marginalization;
- bits/byte after marginalization;
- comparison to the in-repo interpolated n-gram.

If the teacher does not beat the n-gram, distillation cannot win. Publish the
teacher card, not just the student.

## Recommended first release

1. Teacher: Qwen2.5-0.5B-Instruct Q4.
2. Corpus: Project Gutenberg public-domain text, 10 MB.
3. Context: 3 bytes for the current slot runtime.
4. Distill top-4 bytes per context.
5. Student: slot memory, compile to `.nan`, run on `nanlm_vm`.
6. Report teacher and student top-1/BPC against the n-gram.

Then repeat with Qwen2.5-7B and an 8-byte student.

## What this buys

- **Smart immediately** — a pretrained Apache teacher supplies the signal.
- **Open student** — shards + manifest + `.nan` + Apache code.
- **Volunteer-trainable** — teacher inference is shard-local.
- **Sparse-VRAM friendly** — only the hot student table must be paged.
- **No backprop in NaN16** — the VM only runs inference.

The teacher is the quality ceiling. The student is the durable artifact. The
repo is the identity. The store is replaceable.
