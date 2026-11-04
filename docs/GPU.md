# Free-GPU strategy

The goal is not to train a 7B model from scratch. The goal is to spend a small
number of free GPU hours on the expensive part — **teacher inference over a
pinned corpus** — and keep the student as CPU counting.

## Rule zero: buy the smallest useful GPU job

Free hours are scarce and variable. Do not spend them on:

- full 7B fine-tuning;
- long-running inference servers;
- teaching the model general language from zero.

Do spend them on:

- one batched Qwen 7B teacher pass over a high-value held-out split;
- a small differentiable logic-gate network (T3), if that experiment is ready;
- a LoRA adapter for T2 only after the datastore interpolation is measured.

## Free / low-cost GPU options

| source | typical free allocation | best use |
|---|---|---|
| **Modal** | monthly starter credits; A10G/L4/A100 on demand | batch teacher inference, cached model, short jobs |
| **Kaggle** | ~30 GPU-hours/week, T4 ×2 / P100 | 4-bit 7B teacher on a notebook; dataset-backed |
| **Google Colab** | free T4, session-limited | small teacher / gate-network experiments |
| **Lightning AI** | limited monthly GPU hours | short PyTorch jobs |
| **Hugging Face ZeroGPU** | shared short jobs | demos and small teacher slices |
| **Paperspace** | limited free GPU | backup batch job |
| **Google Cloud / Azure** | trial credits | only when the job is ready and cost is bounded |
| **Oracle Always Free** | no GPU | object storage / CPU mirrors, not teacher |
| **GitHub Actions** | CPU only (standard runners) | 0.5B llama.cpp teacher; not 7B |

Verify current quotas before relying on them. Free tiers change.

## The Modal app

`modal_teacher.py` runs the batch Hugging Face teacher on one GPU function:

```sh
pip install modal
modal token new

modal run modal_teacher.py \
  --corpus data/alice_excerpt.txt \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-tokens 4096 \
  --top-k 16
```

It:

1. loads the model in 4-bit NF4 on an A10G;
2. caches the model in the `nanlm-hf-cache` Volume;
3. runs a single forward pass in 256-token chunks with overlap;
4. emits an additive NTR1 teacher shard;
5. writes a teacher card with top-1 and BPC on held-out token boundaries;
6. stores artifacts in the `nanlm-teacher-shards` Volume.

Fetch:

```sh
modal volume get nanlm-teacher-shards qwen_teacher.ntr ./build/qwen_teacher.ntr
modal volume get nanlm-teacher-shards qwen_teacher.ntr.card.json ./build/qwen_teacher.ntr.card.json
```

The app is a batch job, not an inference service. It does not require Modal at
model runtime.

## Batch teacher design

`tools/hf_teacher.py`:

- tokenizes once with the teacher tokenizer;
- reconstructs exact byte spans from token pieces;
- runs chunks of 256 tokens with 16-token overlap;
- takes top-k next-token logits;
- marginalizes to the first byte;
- emits `(8-byte context, next byte, weight)` records;
- writes a teacher card on the held-out token boundaries.

This avoids the 0.5B llama-server bottleneck: one request per token boundary
was slow and generated only 256 samples in nine minutes on a CPU runner.

## Teacher gate

Do not distill a teacher just because it exists. Require:

1. teacher top-1 > the 3-gram baseline on a declared held-out stream;
2. teacher BPC < the 3-gram baseline;
3. the teacher card records model revision, tokenizer revision, top-k, context
   bytes, and corpus hash;
4. the student card is scored on the same held-out bytes;
5. the student beats the 3-slot baseline at the declared model size.

Qwen0.5B/256 samples failed the gate. Qwen7B/4k+ tokens is the next candidate.

## Free-hour budget sketch

- 7B 4-bit teacher, 4k tokens, A10G: one short job plus model download/cache.
- 7B teacher, 32k tokens: still a single job; cache and shard only once.
- T3 gate network: small enough for a T4/Colab session; export truth tables.
- T2 LoRA: only after a KV/datastore interpolation baseline; use a small
  Apache model, not 7B full fine-tuning.

The expensive failure mode is not the GPU bill. It is rerunning an uncached
teacher on the same corpus. Cache by:

```
(model_revision, tokenizer_revision, corpus_sha256, max_tokens,
 top_k, context_bytes, teacher_mode)
```

## What to run next

1. Qwen2.5-7B-Instruct 4-bit, 4k tokens, top-k 16 on Alice.
2. If the teacher card beats the 3-gram, compile the 8-slot student and score
   the full held-out stream.
3. Then run the same teacher on a PG-19 book and a pinned code repository.
4. Only then spend hours on T3 gate training.

The free-GPU path is:

> **one cached teacher pass → one additive shard → CPU counting → full-stream
> evaluation → publish or reject.**

Not:

> **rent a cluster and fine-tune a 7B model before the evaluation harness is
> honest.**
