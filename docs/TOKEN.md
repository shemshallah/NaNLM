# Token-level teacher and student

The byte-level 7B result proved the mismatch: a token-level teacher cannot be
efficiently consumed by a byte-level 3/8-slot student through first-byte
marginalization. The next model keeps the teacher’s native token IDs.

## Why token-level

- Qwen predicts the next **token**, not the next byte.
- A token can be several bytes; first-byte marginalization throws away the rest.
- A token-boundary card of 88.78% / 0.6606 BPC became a byte student near 47%
  top-1 / 3.52 BPC.
- Token context 3–6 tokens is roughly 12–24 bytes, enough to carry the signal.

## NTT1 teacher shard

`tools/hf_token_teacher.py` emits a token-level additive shard:

```
header: magic NTT1, version 1, nrec, context_len
record: uint64 context, uint32 next_token_id, uint32 weight
```

The current format packs 3 token IDs at 18 bits each (Qwen vocabulary is
<2^18):

```
context = (t0 << 36) | (t1 << 18) | t2
```

The record is the exact teacher distribution over token IDs at a token
boundary. No BPE marginalization. Weights are additive; merge and scale tools
are in `tools/token_shard.py`.

## One-line Kaggle run

The script handles the pull, token teacher, card, stats, and tar:

```python
!python /kaggle/working/NaNLM/kaggle_token.py
```

This avoids multi-line Jupyter formatting. If the repo is not cloned yet:

```python
!git clone --depth 1 https://github.com/shemshallah/NaNLM.git /kaggle/working/NaNLM
!python /kaggle/working/NaNLM/kaggle_token.py
```

Outputs:

```text
/kaggle/working/qwen7b_tokens.ntt
/kaggle/working/qwen7b_tokens.card.json
/kaggle/working/token_teacher.log
/kaggle/working/qwen7b_tokens.tgz
```

## Manual run

```python
!git -C /kaggle/working/NaNLM pull --ff-only
!python /kaggle/working/NaNLM/tools/hf_token_teacher.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --corpus /kaggle/working/NaNLM/data/alice_excerpt.txt \
  --out /kaggle/working/qwen7b_tokens.ntt \
  --card /kaggle/working/qwen7b_tokens.card.json \
  --max-tokens 2048 \
  --top-k 16 \
  --load-in-4bit \
  --device cuda \
  --dtype float16 \
  --chunk 64
```

This is one more 7B pass, but the model is cached. If the byte teacher is
already loaded in the same session, a separate token pass is still the clean
measurement; do not mix the two shard formats.

Inspect:

```python
!python /kaggle/working/NaNLM/tools/token_shard.py stats \
  /kaggle/working/qwen7b_tokens.ntt
```

The token card reports:

- `teacher_top1` and `teacher_bpc` on held-out token boundaries;
- `ngram_top1` / `ngram_bpc` for a token 3-gram at the same positions;
- records, contexts, vocabulary, context length.

**Gate:** teacher BPC must beat token-ngram BPC. If it does not, do not build a
student.

## Merge and scale

```sh
python3 tools/token_shard.py stats qwen7b_tokens.ntt
python3 tools/token_shard.py merge qwen7b_tokens_hybrid.ntt \
  qwen7b_tokens.ntt qwen7b_tokens_ngram.ntt
python3 tools/token_shard.py scale qwen7b_tokens_scaled.ntt \
  qwen7b_tokens.ntt 1 16
```

Unlike the byte teacher, the token teacher has no coverage gap at token
boundaries: every next-token position is covered. The raw token n-gram is still
useful as a low-order fallback.

## Token student design

The next C model:

- token vocabulary: Qwen IDs, with a compact 4–16k student vocabulary later;
- context: 3–6 token slots;
- token codebook: 18-bit IDs encoded into simplex or hashed 256-bit codes;
- counts: next-token sparse top-k, signed and additive;
- prediction: nearest Hamming prototype + calibrated token distribution;
- fallback: byte model for unknown/OOV tokens;
- output: `.nas` runtime with token IDs and byte fallback.

The 8-byte recency-graded slot result gives us the core trick: recent slots get
larger Hamming budgets, so flat nearest-prototype search implements
longest-suffix backoff. The same idea applies to token slots.

## What not to do

- Do not feed token shards into `nanlm_slot8_train`; that expects NTR byte
  records.
- Do not aggregate a token teacher down to 3 bytes.
- Do not spend GPU hours on a token teacher until the byte grid has been read;
  it has: the byte path is closed.
- Do not promise a token student win before the token card and full-stream
  evaluation.

## Status

- Token teacher emitter: implemented.
- Token shard stats/merge/scale: implemented.
- Token student: not implemented yet.
- Token runtime: not implemented yet.
- One 7B token pass is the next GPU job, and only if it is cached/cheap.
