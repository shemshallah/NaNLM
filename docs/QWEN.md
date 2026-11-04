# Qwen teacher runner: first real run

The `qwen-teacher` workflow now runs on GitHub Actions with an Apache-2.0
Qwen2.5-0.5B-Instruct Q4_K_M GGUF teacher and a llama.cpp server.

Run: [`41856101819`](https://github.com/shemshallah/NaNLM/actions/runs/41856101819)
(success, 9m20s), 2026-11-04.

## Teacher card

```
eval_n         128
teacher_top1   32.03%
teacher_bpc    5.7798
teacher_samples 256
top_k          8
```

The teacher was queried at 256 token boundaries in the training split, and
scored at the same full held-out positions used by the student. It is **worse
than the 3-gram baseline** (48.14% top-1 / 3.420 BPC) on this task. Do not
distill this checkpoint into a quality claim.

## Student results

| student | context | top-1 (full stream) | BPC (full stream) |
|---|---|---:|---:|
| pure Qwen, 8-slot | 8 bytes + ctype | 16.32% | 8.7736 |
| hybrid Qwen + corpus ngram, 8-slot | 8 bytes + ctype | 37.11% | 4.3108 |
| hybrid Qwen + corpus ngram, 3-slot | 3 bytes + ctype | 41.92% | 4.3650 |

The hybrid 3-slot number was produced locally from the downloaded
`qwen_hybrid.ntr`; the 8-slot number is from the workflow manifest.

None beats the 3-slot typed memory (49.05% / 2.959) or the n-gram BPC.

## What this says

- The **runner works**: pinned Qwen GGUF, llama.cpp server, tokenizer-exact
  prompts, NTR1 teacher records, additive teacher merge, 8-slot compilation,
  and VM differential.
- The **teacher is not smart enough yet**. Qwen0.5B-Instruct, raw completion,
  no chat template, 256 sampled boundaries, and top-k first-byte
  marginalization is a weak teacher on *Alice*.
- **More samples do not fix a weak teacher**, but 256 is also far too few. The
  next run must use thousands of boundaries or a batch logits path.
- The **hybrid** (Qwen + corpus ngram) is a useful safety net but still loses
  to the pure 3-slot memory. It shows that the teacher shard merge path works,
  not that it helps.

## Artifacts

Downloaded from the workflow artifact `qwen-teacher-slot8`:

| artifact | sha256 |
|---|---|
| `qwen_card.json` | `15d5ce0d1e7276c3e101219cd044bf7eca0b643d522ff85b6b8293564f4a88d8` |
| `qwen_teacher.ntr` | `c99f3a0eb1a97b3059e6d71da61241afb81c62f75099b85a176cd4b7614ac1af` |
| `qwen_hybrid.ntr` | `9cc8f838e64d1e139c4d0536581147473db528d7dcb1996de370bad431924523` |
| `qwen_slot8.nas` | `4dc5f30f828d8edebcb0b2b017ad7578d31b5be6dd43dc224c82e228034286bb` |
| `qwen_hybrid_slot8.manifest` | `60f52762aa118a2d346ae90d0d4d67b9c544cffb7c96e3d61261085e0aedd6c9` |
| `qwen_hybrid_slot3.manifest` | `5e1c0d1518448191a0deab06004fb01a31d92487e6a2982d05cda7a0049b20b5` |

The `.ntr` files are additively mergeable. The `.nas` files are reproducible
from the manifests and teacher records; these binaries are artifacts, not the
repo source of truth.

## Next teacher runs

1. **Qwen2.5-7B-Instruct** (Apache-2.0) on a GPU runner. 0.5B is a plumbing
   teacher, not a quality teacher.
2. **Batch logits**, not one HTTP request per boundary: use
   `llama-perplexity --save-all-logits` or a native llama.cpp logits emitter.
3. **Byte-exact token spans** from the tokenizer, and a proper test that the
   teacher's next-token first-byte marginalization matches the corpus byte at
   each boundary.
4. **Teacher card on a held-out book/repository**, not the first 30 KB of
   Alice.
5. **Distillation gate**: teacher top-1 must exceed the n-gram before student
   promotion.

The correct claim today is narrow:

> The Qwen teacher runner is operational and reproducible, but Qwen0.5B with
> 256 samples is not a quality teacher for this corpus.
