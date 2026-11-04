#!/usr/bin/env python3
"""Token HDC student over Qwen token IDs + NTT teacher shard.

This is the teacher-integrated version of experiments/token_hdc.py.

It uses:
  * Qwen token IDs and the 3-token NTT teacher shard;
  * sparse random token codes and position codes;
  * BOOL.n binding selected by token class (fixed vs regime);
  * one HDC prototype per teacher context;
  * exact teacher lookup when present, HDC nearest-prototype otherwise;
  * hierarchical token n-gram backoff when the HDC match is poor;
  * full held-out top-1, BPT and byte-normalized BPC;
  * model-bit and active-fraction accounting.

Run on Kaggle (CPU, no GPU):
    python3 experiments/token_hdc_ntt.py \
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

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
from token_shard import read as read_ntt  # noqa: E402
from token_hdc import bool_n, bundle3, pack_matrix, token_class  # noqa: E402
from token_student import add_counts, hierarchical_prob, unpack  # noqa: E402

TOKEN_MAX = 1 << 18


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ntt", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--density", type=float, default=0.05)
    p.add_argument("--pos-density", type=float, default=0.05)
    p.add_argument("--train-frac", type=float, default=0.85)
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--mode", choices=("fixed", "regime"), default="regime")
    p.add_argument("--nearest-threshold", type=float, default=0.35,
                   help="max Hamming fraction to trust an HDC prototype")
    p.add_argument("--top-counts", type=int, default=8,
                   help="top next-token counts stored per prototype")
    p.add_argument("--max-prototypes", type=int, default=0,
                   help="keep the highest-mass N teacher contexts (0=all)")
    p.add_argument("--teacher-weight", type=float, default=0.55)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        print(f"token_hdc_ntt: missing dependency: {exc}", file=sys.stderr)
        return 2

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text = Path(args.corpus).read_text(encoding="utf-8")
    ids = tokenizer.encode(text, add_special_tokens=False)
    if args.max_tokens and len(ids) > args.max_tokens:
        ids = ids[:args.max_tokens]
    if max(ids) >= TOKEN_MAX:
        print("token_hdc_ntt: vocabulary exceeds 18 bits", file=sys.stderr)
        return 1
    vocab = int(tokenizer.vocab_size)
    train_tokens = int(len(ids) * args.train_frac)

    ctx_len, recs = read_ntt(Path(args.ntt))
    if ctx_len != 3:
        print("token_hdc_ntt: expected 3-token NTT1 shard", file=sys.stderr)
        return 1
    teacher: dict[tuple[int, int, int], dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for ctx, nxt, weight in recs:
        t = unpack(ctx, 3)
        teacher[(t[0], t[1], t[2])][nxt] += weight
    # Compile each teacher context to top-k next-token counts.
    for ctx, bucket in list(teacher.items()):
        teacher[ctx] = dict(sorted(bucket.items(), key=lambda kv: kv[1],
                                   reverse=True)[:args.top_counts])
    teacher2: dict[tuple[int, int], dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    teacher1: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for (t0, t1, t2), bucket in teacher.items():
        for nxt, weight in bucket.items():
            teacher2[(t1, t2)][nxt] += weight
            teacher1[t2][nxt] += weight

    # Token text/class cache for binding-operator selection.
    text_cache: dict[int, str] = {}

    def tok_text(tid: int) -> str:
        s = text_cache.get(tid)
        if s is None:
            s = tokenizer.decode([tid], skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False)
            text_cache[tid] = s
        return s

    unique_ids = sorted(set(ids) | {t for c in teacher for t in c})
    id_to_row = {tid: i for i, tid in enumerate(unique_ids)}
    rng = np.random.default_rng(args.seed)
    codebook = (rng.random((len(unique_ids), args.dim)) < args.density)
    pos = (rng.random((3, args.dim)) < args.pos_density)
    # Sparse-preserving gates only. n=9/15 contain dense complements.
    n_by_class = ({0: 6, 1: 6, 2: 12, 3: 10}
                  if args.mode == "regime" else {0: 6, 1: 6, 2: 6, 3: 6})

    def encode(ctx: tuple[int, int, int]) -> np.ndarray:
        parts = []
        for slot, tid in enumerate(ctx):
            n = n_by_class[token_class(tok_text(tid))]
            parts.append(bool_n(codebook[id_to_row[tid]], pos[slot], n))
        return bundle3(parts[0], parts[1], parts[2])

    keys = list(teacher.keys())
    if args.max_prototypes and len(keys) > args.max_prototypes:
        mass = {k: sum(teacher[k].values()) for k in keys}
        keys = sorted(keys, key=lambda k: -mass[k])[:args.max_prototypes]
        teacher = {k: teacher[k] for k in keys}
        teacher2 = defaultdict(lambda: defaultdict(int))
        teacher1 = defaultdict(lambda: defaultdict(int))
        for (t0, t1, t2), bucket in teacher.items():
            for nxt, weight in bucket.items():
                teacher2[(t1, t2)][nxt] += weight
                teacher1[t2][nxt] += weight
    prototypes = np.stack([encode(k) for k in keys])
    P = pack_matrix(prototypes)
    labels = {k: max(teacher[k].items(), key=lambda kv: kv[1])[0] for k in keys}
    dense_bits = int(P.size * 64)
    active_bits = int(np.bitwise_count(P).sum())
    print(f"token_hdc_ntt: mode={args.mode} dim={args.dim} "
          f"contexts={len(keys)} dense_model_bits={dense_bits} "
          f"active_bits={active_bits} "
          f"active={100.0 * active_bits / max(1, dense_bits):.2f}% "
          f"top_counts={args.top_counts}")

    # Token n-gram fallback.
    g1, g2, g3 = defaultdict(lambda: defaultdict(int)), \
                 defaultdict(lambda: defaultdict(int)), \
                 defaultdict(lambda: defaultdict(int))
    t1, t2, t3 = defaultdict(int), defaultdict(int), defaultdict(int)
    uni = defaultdict(int)
    add_counts(ids, train_tokens, 1, g1, t1, uni)
    add_counts(ids, train_tokens, 2, g2, t2, None)
    add_counts(ids, train_tokens, 3, g3, t3, None)
    utot = sum(uni.values())

    # Cache exact/nearest decisions.
    eval_n = 0
    eval_bytes = 0
    exact3 = exact2 = exact1 = nearest_n = 0
    teacher_n = 0
    teacher_bytes = 0
    correct_t = 0
    bits_t = 0.0
    correct_ng = 0
    bits_ng = 0.0
    correct_h = 0
    bits_h = 0.0
    correct_dy = 0
    bits_dy = 0.0
    doc1, doc2, doc3 = defaultdict(lambda: defaultdict(int)), \
                       defaultdict(lambda: defaultdict(int)), \
                       defaultdict(lambda: defaultdict(int))
    doc1_tot, doc2_tot, doc3_tot = defaultdict(int), defaultdict(int), defaultdict(int)
    doc_uni = defaultdict(int)
    doc_utot = 0

    def bucket_prob(bucket: dict[int, int], token: int) -> float:
        total = sum(bucket.values())
        return (bucket.get(token, 0) + 0.1) / (total + 0.1 * vocab)

    for i in range(train_tokens, len(ids) - 1):
        ctx = (ids[i - 2], ids[i - 1], ids[i])
        actual = ids[i + 1]
        b3 = teacher.get(ctx)
        b2 = teacher2.get((ctx[1], ctx[2]))
        b1 = teacher1.get(ctx[2])

        q = encode(ctx)
        qp = pack_matrix(q[None, :])[0]
        d = np.bitwise_count(P ^ qp).sum(axis=1)
        best_i = int(np.argmin(d))
        dist_frac = float(d[best_i]) / args.dim

        use_bucket = None
        if b3:
            exact3 += 1
            use_bucket = b3
        elif b2:
            exact2 += 1
            use_bucket = b2
        elif b1:
            exact1 += 1
            use_bucket = b1
        elif dist_frac <= args.nearest_threshold:
            nearest_n += 1
            use_bucket = teacher[keys[best_i]]

        if use_bucket:
            teacher_n += 1
            teacher_bytes += len(tok_text(actual).encode("utf-8", errors="replace"))
            p_t = bucket_prob(use_bucket, actual)
            best_t = max(use_bucket.items(), key=lambda kv: kv[1])[0]
            if best_t == actual:
                correct_t += 1
            bits_t += -math.log2(p_t)
        else:
            p_t = None
            best_t = None

        c1 = ctx[2]
        c2 = (ctx[1] << 18) | ctx[2]
        c3 = (ctx[0] << 36) | (ctx[1] << 18) | ctx[2]
        p_ng = hierarchical_prob(
            actual, g3.get(c3), t3.get(c3, 0), g2.get(c2), t2.get(c2, 0),
            g1.get(c1), t1.get(c1, 0), uni, utot, vocab,
        )
        bits_ng += -math.log2(p_ng)
        cands = set(uni.keys()) if len(uni) < 64 else {
            k for k, _ in sorted(uni.items(), key=lambda kv: kv[1], reverse=True)[:32]
        }
        for b in (g3.get(c3), g2.get(c2), g1.get(c1)):
            if b:
                cands.update(b.keys())
        best_ng = max(cands, key=lambda w: hierarchical_prob(
            w, g3.get(c3), t3.get(c3, 0), g2.get(c2), t2.get(c2, 0),
            g1.get(c1), t1.get(c1, 0), uni, utot, vocab,
        )) if cands else 0
        if best_ng == actual:
            correct_ng += 1

        if p_t is None:
            p_h, best_h = p_ng, best_ng
        else:
            p_h = args.teacher_weight * p_t + (1.0 - args.teacher_weight) * p_ng
            best_h = best_t
        bits_h += -math.log2(p_h)
        if best_h == actual:
            correct_h += 1

        # Document-local online cache. Updates only after the current token.
        if doc_utot:
            p_doc = hierarchical_prob(
                actual, doc3.get(c3), doc3_tot.get(c3, 0),
                doc2.get(c2), doc2_tot.get(c2, 0),
                doc1.get(c1), doc1_tot.get(c1, 0),
                doc_uni, doc_utot, vocab, taus=(1.0, 2.0, 3.0),
            )
            p_dy = 0.25 * p_doc + 0.75 * p_h
        else:
            p_dy, best_dy = p_h, best_h
        bits_dy += -math.log2(p_dy)
        if best_h == actual:
            correct_dy += 1
        doc3[c3][actual] += 1; doc3_tot[c3] += 1
        doc2[c2][actual] += 1; doc2_tot[c2] += 1
        doc1[c1][actual] += 1; doc1_tot[c1] += 1
        doc_uni[actual] += 1; doc_utot += 1
        eval_n += 1
        eval_bytes += len(tok_text(actual).encode("utf-8", errors="replace"))

    def show(name, c, b, n, nbytes):
        n = n or 1
        nbytes = nbytes or 1
        print(f"{name}: top1={100.0*c/n:.2f}% bpt={b/n:.4f} "
              f"bpc={b/nbytes:.4f}")

    cov = exact3 + exact2 + exact1
    print(f"token_hdc_ntt: eval_tokens={eval_n} eval_bytes={eval_bytes} "
          f"exact3={exact3} exact2={exact2} exact1={exact1} nearest={nearest_n} "
          f"coverage3={100.0*exact3/max(1,eval_n):.2f}% "
          f"coverage123={100.0*cov/max(1,eval_n):.2f}%")
    # Rough model-size accounting. Dense prototypes, sparse coordinates,
    # top-8 next-token counts, and the teacher shard are all reported.
    dense_bytes = (dense_bits + 7) // 8
    coord_bits = active_bits * (math.ceil(math.log2(args.dim)) + 1)
    key_bits = len(keys) * 54
    count_bits = len(keys) * args.top_counts * (18 + 16)
    unique_bits = len(unique_ids) * 18
    sparse_bits = coord_bits + key_bits + count_bits + unique_bits + 32
    sparse_bytes = (sparse_bits + 7) // 8
    ngram_records = (sum(len(b) for b in g1.values()) +
                     sum(len(b) for b in g2.values()) +
                     sum(len(b) for b in g3.values()))
    ngram_bits = ngram_records * (18 + 18 + 32)
    ngram_bytes = (ngram_bits + 7) // 8
    teacher_bytes_all = len(recs) * 16
    show("teacher_backoff", correct_t, bits_t, teacher_n, teacher_bytes)
    show("token_ngram", correct_ng, bits_ng, eval_n, eval_bytes)
    show("hdc_teacher_plus_ngram", correct_h, bits_h, eval_n, eval_bytes)
    show("hdc_plus_doc_cache", correct_dy, bits_dy, eval_n, eval_bytes)
    print(f"model_size: dense={dense_bytes} B sparse={sparse_bytes} B "
          f"ngram_fallback={ngram_bytes} B teacher_shard={teacher_bytes_all} B")
    print(f"hutter_total_bpc: dense={(bits_h + 8.0 * dense_bytes) / max(1, eval_bytes):.4f} "
          f"sparse={(bits_h + 8.0 * sparse_bytes) / max(1, eval_bytes):.4f} "
          f"sparse_plus_ngram={(bits_h + 8.0 * (sparse_bytes + ngram_bytes)) / max(1, eval_bytes):.4f} "
          f"with_teacher_shard={(bits_h + 8.0 * (sparse_bytes + ngram_bytes + teacher_bytes_all)) / max(1, eval_bytes):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
