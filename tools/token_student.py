#!/usr/bin/env python3
"""Evaluate a token-level teacher shard with token n-gram backoff.

This is the bridge from NTT1 teacher shards to a token student. It does not
load Qwen; it only tokenizes the corpus, reads the teacher shard, and scores:

  * teacher exact 3-token context;
  * token n-gram 3/2/1 + unigram;
  * hybrid: teacher when present, n-gram backoff otherwise.

Run on Kaggle (no GPU):
    python3 tools/token_student.py \
      --ntt /kaggle/working/qwen7b_tokens.ntt \
      --corpus data/alice_excerpt.txt \
      --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from token_shard import read as read_ntt  # noqa: E402

TOKEN_MAX = 1 << 18


def pack(ids: list[int], end: int, k: int) -> int | None:
    if end < k:
        return None
    v = 0
    for token_id in ids[end - k:end]:
        v = (v << 18) | token_id
    return v


def add_counts(ids: list[int], end: int, k: int, table: dict, unigram: dict):
    for i in range(k - 1, end - 1):
        if i + 1 >= end:
            break
        ctx = pack(ids, i + 1, k)
        if ctx is None:
            continue
        nxt = ids[i + 1]
        table[ctx][nxt] += 1
        unigram[nxt] += 1


def ngram_prob(ctxs: list[int], actual: int, tables: list[dict],
               unigram: dict, total: int, vocab: int) -> tuple[float, int]:
    probs = []
    totals = []
    for table, ctx in zip(tables, ctxs):
        bucket = table.get(ctx)
        if bucket:
            count = bucket.get(actual, 0)
            t = sum(bucket.values())
            probs.append((count + 0.1) / (t + 0.1 * vocab))
            totals.append(t)
        else:
            probs.append((0.1) / (0.1 * vocab))
            totals.append(0)
    pu = (unigram.get(actual, 0) + 0.1) / (total + 0.1 * vocab)
    p = 0.6 * probs[0] + 0.3 * probs[1] + 0.09 * probs[2] + 0.01 * pu
    if totals[0] == 0 and totals[1] == 0 and totals[2] == 0:
        p = pu
    elif totals[0] == 0:
        p = 0.75 * probs[1] + 0.2 * probs[2] + 0.05 * pu
    # top-1 from the union of the three buckets, else the unigram winner
    candidates = set()
    for table, ctx in zip(tables, ctxs):
        bucket = table.get(ctx)
        if bucket:
            candidates.update(bucket.keys())
    if candidates:
        best = max(candidates, key=lambda b: (
            0.6 * (((tables[0].get(ctxs[0], {}).get(b, 0) + 0.1) /
                    (sum(tables[0].get(ctxs[0], {}).values()) + 0.1 * vocab))
                   if tables[0].get(ctxs[0]) else 0.0)
            + 0.3 * (((tables[1].get(ctxs[1], {}).get(b, 0) + 0.1) /
                      (sum(tables[1].get(ctxs[1], {}).values()) + 0.1 * vocab))
                     if tables[1].get(ctxs[1]) else 0.0)
            + 0.09 * (((tables[2].get(ctxs[2], {}).get(b, 0) + 0.1) /
                       (sum(tables[2].get(ctxs[2], {}).values()) + 0.1 * vocab))
                      if tables[2].get(ctxs[2]) else 0.0)
        ))
    else:
        best = max(unigram.items(), key=lambda kv: kv[1])[0] if unigram else 0
    return max(1e-12, p), best


def teacher_bucket_prob(bucket: dict[int, int], actual: int,
                        vocab: int) -> tuple[float, int]:
    total = sum(bucket.values())
    if total <= 0:
        return 1.0 / vocab, 0
    p = bucket.get(actual, 0) / total
    # Residual mass is not represented in the top-k shard.
    p = p * 0.9 + 0.1 / vocab
    best = max(bucket.items(), key=lambda kv: kv[1])[0]
    return max(1e-12, p), best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntt", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train-frac", type=float, default=0.85)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        print(f"token-student: missing dependency: {exc}", file=sys.stderr)
        return 2

    text = Path(args.corpus).read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    vocab = int(tokenizer.vocab_size)
    if max(ids) >= TOKEN_MAX:
        print("token-student: vocabulary exceeds 18 bits", file=sys.stderr)
        return 1

    ctx_len, recs = read_ntt(Path(args.ntt))
    if ctx_len != 3:
        print("token-student: expected 3-token shard", file=sys.stderr)
        return 1
    teacher: dict[int, dict[int, int]] = defaultdict(dict)
    for ctx, nxt, weight in recs:
        old = teacher[ctx].get(nxt, 0)
        teacher[ctx][nxt] = old + weight

    train_tokens = int(len(ids) * args.train_frac)
    t1, t2, t3 = defaultdict(lambda: defaultdict(int)), \
                 defaultdict(lambda: defaultdict(int)), \
                 defaultdict(lambda: defaultdict(int))
    unigram: dict[int, int] = defaultdict(int)
    add_counts(ids, train_tokens, 1, t1, unigram)
    add_counts(ids, train_tokens, 2, t2, unigram)
    add_counts(ids, train_tokens, 3, t3, unigram)
    total = sum(unigram.values())

    eval_n = 0
    t_correct = t_bits = ng_correct = ng_bits = hy_correct = hy_bits = 0.0
    exact = 0
    exact_n = 0
    for i in range(train_tokens, len(ids) - 1):
        c3 = pack(ids, i + 1, 3)
        c2 = pack(ids, i + 1, 2)
        c1 = pack(ids, i + 1, 1)
        if c3 is None or c2 is None or c1 is None:
            continue
        actual = ids[i + 1]
        eval_n += 1
        bucket = teacher.get(c3)
        if bucket:
            exact += 1
            exact_n += 1
            p, best = teacher_bucket_prob(bucket, actual, vocab)
            t_bits += -math.log2(p)
            if best == actual:
                t_correct += 1

        p_ng, best_ng = ngram_prob([c3, c2, c1], actual,
                                   [t3, t2, t1], unigram, total, vocab)
        ng_bits += -math.log2(p_ng)
        if best_ng == actual:
            ng_correct += 1

        if bucket:
            p_hy, best_hy = teacher_bucket_prob(bucket, actual, vocab)
        else:
            p_hy, best_hy = p_ng, best_ng
        hy_bits += -math.log2(p_hy)
        if best_hy == actual:
            hy_correct += 1

    def show(name, bits, correct, n):
        n = n or 1
        print(f"{name}: top1={100.0 * correct / n:.2f}% "
              f"bpc={bits / n:.4f}")

    print(f"token-student: tokens={len(ids)} train={train_tokens} "
          f"eval={eval_n} teacher_exact={exact} "
          f"coverage={100.0 * exact / eval_n:.2f}%")
    show("teacher_exact", t_bits, t_correct, exact_n)
    show("token_ngram", ng_bits, ng_correct, eval_n)
    show("teacher_plus_backoff", hy_bits, hy_correct, eval_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
