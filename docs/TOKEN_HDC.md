# Token-level HDC prototype

`experiments/token_hdc.py` is the first token student simulation:

- word / punctuation / whitespace tokenizer;
- 1,024-token vocabulary on Alice;
- 3-token context;
- sparse random token codes;
- `BOOL.n` binding with position codes;
- majority bundle of three bound slot vectors;
- one prototype per observed context;
- nearest-Hamming prediction with the prototype's next-token counts;
- full held-out token top-1 and byte-normalized BPC.

## Results

Alice, 12,497 tokens, 5,571 prototypes, 1,874 held-out tokens / 4,550 bytes:

| mode | dim | top-1 | BPT | BPC |
|---|---:|---:|---:|---:|
| fixed XOR | 2,048 | 39.49% | 7.7698 | 3.2001 |
| regime-selected `n` | 2,048 | 39.81% | 7.7474 | 3.1909 |
| regime-selected `n` | 8,192 | 39.86% | 7.7181 | 3.1788 |

Regime specialization gives a small positive shift: **+0.32 points / −0.009 BPC**
at 2,048 bits. That is not enough to claim the dual-use `n` is paying for
itself yet.

The token HDC prototype does **not** beat the byte 3-slot model
(**2.9593 BPC**). It is a research prototype, not a replacement.

## Model size and sparsity

At 2,048 bits, 5,571 prototypes means 11.4 million model bits. At 8,192 bits it
is 45.6 million bits. The active fraction of prototypes is ~50% because binding
a sparse token code to a dense position code densifies the representation.

This is the key next problem: **sparse token codes are becoming dense after
binding/bundling**. The 8k-bit model does not help unless the encoded vectors
stay cold.

## What this prototype says

1. HDC token binding works mechanically.
2. A token HDC student already reaches ~3.19 BPC on word tokens, close to the
   byte 3-slot model.
3. Regime-selected `n` is measurable but tiny.
4. The current prototype is too large and too dense.
5. It does not use the Qwen token teacher shard yet.

## Next token HDC steps

- use Qwen token IDs and the 3-token teacher shard as the count source;
- make token codes sparse and keep them sparse after binding (active masks,
  regime-gated lanes);
- recency-graded token slots (the byte 8-slot trick);
- top-k counts rather than full next-token tables;
- token-level full-stream BPC against token n-gram and byte 3-slot;
- model-size/Hutter accounting;
- then emit a `.nas` runtime.

Do not call this a token-level win yet.
