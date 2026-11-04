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

| mode | dim | top-1 | BPT | BPC | active % |
|---|---:|---:|---:|---:|---:|
| fixed XOR, dense position codes | 2,048 | 39.49% | 7.7698 | 3.2001 | 49.5 |
| regime-selected `n`, dense position codes | 2,048 | 39.81% | 7.7474 | 3.1909 | 49.6 |
| fixed XOR, sparse position codes | 2,048 | 37.46% | 7.8342 | 3.2267 | 2.88 |
| regime-selected `n`, sparse position codes | 2,048 | 38.69% | 7.7301 | 3.1838 | 9.85 |
| regime-selected `n`, sparse position codes | 8,192 | 39.27% | 7.7008 | 3.1717 | 9.91 |

Regime specialization gives a small positive shift at fixed dimension:
**+1.23 points / −0.043 BPC** with sparse position codes, but it raises active
fraction from 2.88% to 9.85%. That is not enough to claim the dual-use `n` is
paying for itself yet.

The token HDC prototype does **not** beat the byte 3-slot model
(**2.9593 BPC**). It is a research prototype, not a replacement.

## Model size and sparsity

At 2,048 bits, 5,571 prototypes means 11.4 million model bits. At 8,192 bits it
is 45.6 million bits. Sparse position codes reduce active prototype bits from
~50% to 3–10%, but they also cost accuracy. Higher dimension gives only a tiny
BPC improvement at 4x the model bits.

This is the key next problem: **sparse representation is not yet paying for
itself**. The next step is not more bits; it is teacher-guided token counts,
learned regime selection, and recency-graded slots.

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

## Teacher-integrated token HDC

`experiments/token_hdc_ntt.py` replaces the word-token corpus with:

- Qwen token IDs;
- the NTT1 token teacher shard;
- sparse token/position HDC codes;
- `BOOL.n` binding by token class;
- one prototype per teacher context;
- exact teacher lookup when the context is known;
- nearest HDC prototype when it is not;
- hierarchical token n-gram backoff;
- full held-out top-1, BPT, byte BPC, coverage, and model bits.

Run on Kaggle (no GPU):

```python
!cd /kaggle/working/NaNLM && python3 experiments/token_hdc_ntt.py --ntt /kaggle/working/qwen7b_tokens.ntt --corpus /kaggle/working/NaNLM/data/alice_excerpt.txt --model Qwen/Qwen2.5-7B-Instruct --dim 2048 --mode regime
```

This is the first test that can directly compare the compact token HDC memory
against the 2.4161 BPC teacher/backoff lookup. Do not promote it until it pays
for its model bits and beats the token n-gram on the same held-out tokens.

## Corrected teacher-integrated result

After sparse truth tables and 2/1-token teacher backoff:

| mode | active | exact3 | any teacher | teacher backoff BPC | token ngram BPC | hybrid BPC | hybrid top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| regime | 0.69% | 7.31% | 81.45% | 3.1960 | 2.7324 | 2.4177 | 10.17% |
| fixed | 2.25% | 7.31% | 81.45% | 3.1217 | 2.7324 | **2.4069** | 10.70% |

The sparse truth-table fix dropped regime active fraction from 83% to 0.69%.
The 2/1 teacher backoff raised teacher coverage from 7.3% to 81.45%. A causal
document-local token cache then reached **2.1509 BPC** in the lookup model,
which is now the target for the HDC compression.

This matches the **2.4161 BPC** token lookup target. But it is not yet a compact
student:

- the 2.40 BPC hybrid is dominated by the exact/backoff teacher lookup;
- HDC nearest is used for only 208 of 1,121 eval tokens;
- prototypes are 11,857,920 dense bits (1.48 MB) at 2,048 dimensions;
- the teacher lookup table is still present in the evaluator.

Dense model accounting on the 4,476-byte held-out stream is ~2,650 BPC before
teacher counts. This is not Pareto. The next compression target is the teacher
table and prototype storage, not more dimensions.

Fixed binding beats regime binding on BPC (2.4069 vs 2.4177). Regime
specialization is still not earning its metadata cost.

## Model-size accounting update

`experiments/token_hdc_ntt.py` now:

- compiles each teacher context to the top `--top-counts` next tokens (default 8);
- estimates sparse prototype storage from active coordinates;
- counts prototype keys, top-k counts, token-ID table, and the token n-gram
  fallback bytes;
- prints Hutter totals for dense, sparse, sparse+ngram, and sparse+ngram+teacher
  shard.

This is the gate the 2.40 BPC result must pass. A run that reports 2.40 BPC but
hundreds of BPC of model overhead is not a Pareto win.

## Top-k and model-size result

With top-8 counts per prototype:

| mode | active | stream BPC | sparse model | sparse + ngram total |
|---|---:|---:|---:|---:|
| regime | 0.69% | 2.5182 | 362,227 B | 901.60 BPC |
| fixed | 2.25% | 2.5080 | 640,536 B | 1399.02 BPC |

Top-8 truncation cost about 0.10 BPC versus the full teacher table. The sparse
model is much smaller than dense (1.48 MB) but still dominates a 4.5 KB stream.
The token n-gram fallback alone is 140,811 B.

This is not Pareto. The next experiment is **prototype pruning**:

```python
!python experiments/token_hdc_ntt.py \
  --ntt /kaggle/working/qwen7b_tokens.ntt \
  --corpus /kaggle/working/NaNLM/data/alice_excerpt.txt \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dim 2048 --mode regime --top-counts 8 --max-prototypes 512
```

The question is whether 512/1024/2048 prototypes retain the 2.5 BPC signal. If
not, the teacher table is too specific for a compact model at this corpus size;
the answer is more data, not a larger prototype matrix.
