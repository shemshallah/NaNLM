# Getting a smart base

A public model needs to be smart immediately. The fastest honest route is
**teacher distillation into the additive slot memory**, not training a
transformer inside NaN16.

## Why not train the NaN16 VM directly

Backprop through `nanlm_vm` would be slow, fragile, and would put floating
point back into the critical path. The substrate is good at inference and
counting, not at dense gradient descent.

## The two-model release

1. **Teacher** — a small byte-level neural model (GRU/LSTM/transformer) trained
   on a public corpus. It runs on a runner or a volunteer GPU, not on the
   phone. Its weights can live in repovram.
2. **Student** — a simplex slot memory distilled from the teacher. It runs on
   `nanlm_vm`, trains by additive shards, and is the open artifact.

The teacher is the quality ceiling. The student inherits its next-byte
distribution for every observed context and generalizes by Hamming distance.

## Byte alignment matters

A BPE LLM is not a byte-level teacher. Its next-token distribution does not map
cleanly to the next byte at token boundaries. Use a **byte-level teacher**
(char GRU/LSTM/transformer) so every target is a byte distribution.

If a BPE teacher is used, the distillation must explicitly marginalize token
probabilities over their byte spellings and handle boundaries. That is a later
optimization, not the first base.

## Distillation shard

A teacher shard is the same additive shape as a corpus shard, with integer
weights instead of raw counts:

```
(context key, next byte, weight)
```

For each context, keep the top-k bytes and quantize their probabilities to
16-bit weights. Merge is still addition. A context may have several records
with different next bytes.

The student compiles these records into prototypes exactly as it does corpus
counts. Top-1 is the teacher's argmax for seen contexts; unseen contexts use
the nearest simplex key.

## Context length

Start with 3 bytes to match the current `nanlm_slot` runtime. The first
teacher can be a small byte-level model conditioned on 3 bytes (an embedding
MLP) or a sequence model evaluated with a 3-byte prefix.

The next scale is 8 slots:

- key = 8 × 255-bit simplex codewords = 2040 bits = 64 words;
- prototype table = top-K contexts × 64 words;
- runtime scans K prototypes, exactly like the current slot runtime.

That is a change of `SW` from 3 to 8 and a 64-bit context key in the shard
format. It does not change the training rule or the merge rule.

## Sparse VRAM placement

| tier | contents |
|---|---|
| repovram | teacher weights (optional), student prototype matrix, codebook, labels, hot `.nan` |
| reporam | teacher distribution shards, merge manifests, leases, eval cards |
| runner/GPU | teacher inference, shard generation |
| phone | `nanlm_vm` + hot `.nan` |

The teacher can disappear after distillation; the student is the durable open
artifact. The teacher weights stay in repovram as provenance if there is room.

## Distillation pipeline

```
corpus shard
   -> teacher inference (runner/GPU)
   -> top-k byte distribution
   -> additive shard (context, byte, weight)
   -> reporam
   -> merge
   -> slot compile
   -> repovram generation
   -> nanlm_vm
```

## Gates

- Teacher top-1/BPC on a held-out split.
- Student top-1/BPC after distillation, same split.
- Student must match the C adapter on every emitted byte/hash.
- A shuffled-teacher negative gate: permute the teacher's byte labels; the
  student must lose quality, proving the distillation is real.
- A shard merge gate: `A+B == B+A`.

## What this buys

- **Smart immediately** — the teacher is a pretrained/trained neural model.
- **Open student** — the artifact is a manifest + `.nan` image + shards.
- **Volunteer-trainable** — teacher inference is parallel and shard-local.
- **Sparse-VRAM friendly** — only the hot prototype table must be paged.
- **No backprop in NaN16** — the VM only runs inference.

## Open questions

1. What is the smallest byte-level teacher that beats the n-gram by a useful
   margin on a 1–10 MB corpus?
2. How many top-k bytes per context are worth storing?
3. Does teacher distillation preserve its advantage after slot quantization?
4. How much does 8-slot context improve over 3-slot on the same teacher?
5. Can the teacher run in WASM/WebGPU for volunteers, or only on runners?
