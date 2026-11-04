#!/usr/bin/env python3
"""Evaluate a token-level teacher shard with hierarchical Kneser-Ney/Witten-Bell
token backoff and an optional document-local adaptive token cache.

Fixes the large-vocabulary Laplace denominator trap: in a 151k vocabulary,
adding 0.1 * V = 15,164 to the denominator of a 3-gram or 2-gram crushes
empirical counts into uniform noise. Instead, each order interpolates
recursively:

    p_0(w) = (unigram(w) + alpha) / (total + alpha * V)
    p_k(w) = lambda_k * (c_k(w) / t_k) + (1 - lambda_k) * p_{k-1}(w)

where lambda_k = t_k / (t_k + tau_k). Every order sums to 1.0 over V.
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


def unpack(ctx: int, k: int) -> list[int]:
    return [(ctx >> (18 * (k - 1 - i))) & (TOKEN_MAX - 1) for i in range(k)]


def pack(ids: list[int], end: int, k: int) -> int | None:
    if end < k:
        return None
    v = 0
    for token_id in ids[end - k:end]:
        v = (v << 18) | token_id
    return v


def add_counts(ids: list[int], end: int, k: int, table: dict,
               totals: dict, unigram: dict | None = None) -> None:
    for i in range(k - 1, end - 1):
        if i + 1 >= end:
            break
        ctx = pack(ids, i + 1, k)
        if ctx is None:
            continue
        nxt = ids[i + 1]
        table[ctx][nxt] += 1
        totals[ctx] += 1
        if unigram is not None:
            unigram[nxt] += 1


def hierarchical_prob(
    token: int,
    b3: dict[int, int] | None, t3: int,
    b2: dict[int, int] | None, t2: int,
    b1: dict[int, int] | None, t1: int,
    unigram: dict[int, int], utot: int,
    vocab: int,
    taus: tuple[float, float, float] = (1.5, 2.5, 4.0),
    alpha: float = 0.005,
) -> float:
    p = (unigram.get(token, 0) + alpha) / (utot + alpha * vocab)
    if b1 and t1 > 0:
        lam1 = t1 / (t1 + taus[2])
        p = lam1 * (b1.get(token, 0) / t1) + (1.0 - lam1) * p
    if b2 and t2 > 0:
        lam2 = t2 / (t2 + taus[1])
        p = lam2 * (b2.get(token, 0) / t2) + (1.0 - lam2) * p
    if b3 and t3 > 0:
        lam3 = t3 / (t3 + taus[0])
        p = lam3 * (b3.get(token, 0) / t3) + (1.0 - lam3) * p
    return max(1e-12, p)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntt", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train-frac", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="optional token cap matching teacher run")
    parser.add_argument("--teacher-weight", type=float, default=0.35,
                        help="interpolation weight on teacher hierarchy")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        print(f"token-student: missing dependency: {exc}", file=sys.stderr)
        return 2

    text = Path(args.corpus).read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if args.max_tokens and len(ids) > args.max_tokens:
        ids = ids[:args.max_tokens]
    vocab = int(tokenizer.vocab_size)
    if max(ids) >= TOKEN_MAX:
        print("token-student: vocabulary exceeds 18 bits", file=sys.stderr)
        return 1

    ctx_len, recs = read_ntt(Path(args.ntt))
    if ctx_len != 3:
        print("token-student: expected 3-token shard", file=sys.stderr)
        return 1

    teacher3: dict[int, dict[int, int]] = defaultdict(dict)
    t3_tot: dict[int, int] = defaultdict(int)
    teacher2: dict[int, dict[int, int]] = defaultdict(dict)
    t2_tot: dict[int, int] = defaultdict(int)
    teacher1: dict[int, dict[int, int]] = defaultdict(dict)
    t1_tot: dict[int, int] = defaultdict(int)
    teacher0: dict[int, int] = defaultdict(int)
    t0_tot = 0

    for ctx, nxt, weight in recs:
        teacher3[ctx][nxt] = teacher3[ctx].get(nxt, 0) + weight
        t3_tot[ctx] += weight
        t = unpack(ctx, 3)
        c2 = (t[1] << 18) | t[2]
        c1 = t[2]
        teacher2[c2][nxt] = teacher2[c2].get(nxt, 0) + weight
        t2_tot[c2] += weight
        teacher1[c1][nxt] = teacher1[c1].get(nxt, 0) + weight
        t1_tot[c1] += weight
        teacher0[nxt] += weight
        t0_tot += weight

    train_tokens = int(len(ids) * args.train_frac)
    ng1: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    ng2: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    ng3: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    ng1_tot: dict[int, int] = defaultdict(int)
    ng2_tot: dict[int, int] = defaultdict(int)
    ng3_tot: dict[int, int] = defaultdict(int)
    unigram: dict[int, int] = defaultdict(int)

    add_counts(ids, train_tokens, 1, ng1, ng1_tot, unigram)
    add_counts(ids, train_tokens, 2, ng2, ng2_tot, None)
    add_counts(ids, train_tokens, 3, ng3, ng3_tot, None)
    utot = sum(unigram.values())
    top_uni = sorted(unigram.items(), key=lambda kv: kv[1], reverse=True)[:16]
    top_uni_ids = [k for k, _ in top_uni]

    # Document-local adaptive token cache (starts empty at eval start, updates
    # strictly causally after each evaluated token).
    doc1: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    doc2: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    doc3: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    doc1_tot: dict[int, int] = defaultdict(int)
    doc2_tw: dict[int, int] = defaultdict(int)
    doc3_tw: dict[int, int] = defaultdict(int)
    doc_uni: dict[int, int] = defaultdict(int)
    doc_utot = 0

    eval_n = 0
    eval_tok_ids: list[int] = []
    t_correct = t_bits = 0.0
    ng_correct = ng_bits = 0.0
    hy_correct = hy_bits = 0.0
    dy_correct = dy_bits = 0.0
    exact3 = exact2 = exact1 = 0
    tw = args.teacher_weight

    for i in range(train_tokens, len(ids) - 1):
        c3 = pack(ids, i + 1, 3)
        c2 = pack(ids, i + 1, 2)
        c1 = pack(ids, i + 1, 1)
        if c3 is None or c2 is None or c1 is None:
            continue
        actual = ids[i + 1]
        eval_n += 1
        eval_tok_ids.append(actual)

        tb3, tt3 = teacher3.get(c3), t3_tot.get(c3, 0)
        tb2, tt2 = teacher2.get(c2), t2_tot.get(c2, 0)
        tb1, tt1 = teacher1.get(c1), t1_tot.get(c1, 0)
        nb3, nt3 = ng3.get(c3), ng3_tot.get(c3, 0)
        nb2, nt2 = ng2.get(c2), ng2_tot.get(c2, 0)
        nb1, nt1 = ng1.get(c1), ng1_tot.get(c1, 0)

        if tb3:
            exact3 += 1
        elif tb2:
            exact2 += 1
        elif tb1:
            exact1 += 1

        # Candidate pool for exact top-1 argmax
        cands = set(top_uni_ids)
        for b in (tb3, tb2, tb1, nb3, nb2, nb1, doc3.get(c3), doc2.get(c2), doc1.get(c1)):
            if b:
                if len(b) <= 32:
                    cands.update(b.keys())
                else:
                    cands.update(k for k, _ in sorted(b.items(), key=lambda kv: kv[1], reverse=True)[:16])

        # 1. Teacher hierarchy probability
        p_t = hierarchical_prob(
            actual, tb3, tt3, tb2, tt2, tb1, tt1,
            teacher0, max(1, t0_tot), vocab,
            taus=(65535.0, 131070.0, 262140.0), alpha=0.005,
        )
        if tb3:
            t_bits += -math.log2(p_t)
            best_t = max(tb3.items(), key=lambda kv: kv[1])[0]
            if best_t == actual:
                t_correct += 1

        # 2. Corpus token n-gram hierarchy probability
        p_ng = hierarchical_prob(
            actual, nb3, nt3, nb2, nt2, nb1, nt1,
            unigram, max(1, utot), vocab,
            taus=(1.5, 2.5, 4.0), alpha=0.005,
        )
        ng_bits += -math.log2(p_ng)
        best_ng = max(
            cands,
            key=lambda w: hierarchical_prob(
                w, nb3, nt3, nb2, nt2, nb1, nt1,
                unigram, max(1, utot), vocab,
                taus=(1.5, 2.5, 4.0), alpha=0.005,
            ),
        )
        if best_ng == actual:
            ng_correct += 1

        # 3. Hybrid static (Teacher hierarchy + Corpus token n-gram hierarchy)
        p_hy = tw * p_t + (1.0 - tw) * p_ng
        hy_bits += -math.log2(p_hy)
        best_hy = max(
            cands,
            key=lambda w: (
                tw * hierarchical_prob(
                    w, tb3, tt3, tb2, tt2, tb1, tt1,
                    teacher0, max(1, t0_tot), vocab,
                    taus=(65535.0, 131070.0, 262140.0), alpha=0.005,
                )
                + (1.0 - tw) * hierarchical_prob(
                    w, nb3, nt3, nb2, nt2, nb1, nt1,
                    unigram, max(1, utot), vocab,
                    taus=(1.5, 2.5, 4.0), alpha=0.005,
                )
            ),
        )
        if best_hy == actual:
            hy_correct += 1

        # 4. Hybrid + Document-local dynamic cache
        if doc_utot > 0:
            db3, dt3 = doc3.get(c3), doc3_tw.get(c3, 0)
            db2, dt2 = doc2.get(c2), doc2_tw.get(c2, 0)
            db1, dt1 = doc1.get(c1), doc1_tot.get(c1, 0)
            p_doc = hierarchical_prob(
                actual, db3, dt3, db2, dt2, db1, dt1,
                doc_uni, doc_utot, vocab,
                taus=(1.0, 2.0, 3.0), alpha=0.005,
            )
            p_dy = 0.25 * p_doc + 0.75 * p_hy
            best_dy = max(
                cands,
                key=lambda w: (
                    0.25 * hierarchical_prob(
                        w, db3, dt3, db2, dt2, db1, dt1,
                        doc_uni, doc_utot, vocab,
                        taus=(1.0, 2.0, 3.0), alpha=0.005,
                    )
                    + 0.75 * (
                        tw * hierarchical_prob(
                            w, tb3, tt3, tb2, tt2, tb1, tt1,
                            teacher0, max(1, t0_tot), vocab,
                            taus=(65535.0, 131070.0, 262140.0), alpha=0.005,
                        )
                        + (1.0 - tw) * hierarchical_prob(
                            w, nb3, nt3, nb2, nt2, nb1, nt1,
                            unigram, max(1, utot), vocab,
                            taus=(1.5, 2.5, 4.0), alpha=0.005,
                        )
                    )
                ),
            )
        else:
            p_dy, best_dy = p_hy, best_hy
        dy_bits += -math.log2(p_dy)
        if best_dy == actual:
            dy_correct += 1

        # Strictly causal update of document-local cache
        doc3[c3][actual] += 1
        doc3_tw[c3] += 1
        doc2[c2][actual] += 1
        doc2_tw[c2] += 1
        doc1[c1][actual] += 1
        doc1_tot[c1] += 1
        doc_uni[actual] += 1
        doc_utot += 1

    eval_bytes = len(tokenizer.decode(
        eval_tok_ids, skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    ).encode("utf-8", errors="replace"))
    if eval_bytes <= 0:
        eval_bytes = eval_n

    def show(name: str, bits: float, correct: float, n: int, n_bytes: int) -> None:
        n = n or 1
        n_bytes = n_bytes or 1
        print(f"{name}: top1={100.0 * correct / n:.2f}% "
              f"bpt={bits / n:.4f} bpc={bits / n_bytes:.4f}")

    cov_any = exact3 + exact2 + exact1
    print(f"token-student: tokens={len(ids)} train={train_tokens} "
          f"eval_tokens={eval_n} eval_bytes={eval_bytes} "
          f"teacher3={len(teacher3)} teacher2={len(teacher2)} "
          f"teacher1={len(teacher1)} "
          f"hit3={exact3} ({100.0 * exact3 / max(1, eval_n):.1f}%) "
          f"hit123={cov_any} ({100.0 * cov_any / max(1, eval_n):.1f}%)")
    show("teacher_exact3", t_bits, t_correct, exact3,
         int(round(eval_bytes * (exact3 / max(1, eval_n)))))
    show("token_ngram", ng_bits, ng_correct, eval_n, eval_bytes)
    show("teacher_plus_backoff", hy_bits, hy_correct, eval_n, eval_bytes)
    show("teacher_backoff_doc_cache", dy_bits, dy_correct, eval_n, eval_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
