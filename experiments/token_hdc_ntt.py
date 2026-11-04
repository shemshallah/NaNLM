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
    n_by_class = ({0: 6, 1: 6, 2: 12, 3: 9}
                  if args.mode == "regime" else {0: 6, 1: 6, 2: 6, 3: 6})

    def encode(ctx: tuple[int, int, int]) -> np.ndarray:
        parts = []
        for slot, tid in enumerate(ctx):
            n = n_by_class[token_class(tok_text(tid))]
            parts.append(bool_n(codebook[id_to_row[tid]], pos[slot], n))
        return bundle3(parts[0], parts[1], parts[2])

    keys = list(teacher.keys())
    prototypes = np.stack([encode(k) for k in keys])
    P = pack_matrix(prototypes)
    labels = {k: max(teacher[k].items(), key=lambda kv: kv[1])[0] for k in keys}
    print(f"token_hdc_ntt: mode={args.mode} dim={args.dim} "
          f"contexts={len(keys)} model_bits={len(keys) * args.dim} "
          f"active={100.0 * float(np.bitwise_count(P).sum()) / max(1, P.size * 64):.2f}%")

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
    exact_n = nearest_n = 0
    teacher_n = 0
    teacher_bytes = 0
    correct = 0
    bits = 0.0
    correct_ng = 0
    bits_ng = 0.0
    correct_h = 0
    bits_h = 0.0

    for i in range(train_tokens, len(ids) - 1):
        ctx = (ids[i - 2], ids[i - 1], ids[i])
        actual = ids[i + 1]
        q = encode(ctx)
        qp = pack_matrix(q[None, :])[0]
        d = np.bitwise_count(P ^ qp).sum(axis=1)
        best_i = int(np.argmin(d))
        best_key = keys[best_i]
        dist_frac = float(d[best_i]) / args.dim
        bucket = teacher.get(ctx)
        if bucket:
            exact_n += 1
            pred_bucket = bucket
        elif dist_frac <= args.nearest_threshold:
            nearest_n += 1
            pred_bucket = teacher[best_key]
        else:
            pred_bucket = None

        # Teacher/hybrid probability
        if pred_bucket:
            teacher_n += 1
            teacher_bytes += len(tok_text(actual).encode("utf-8", errors="replace"))
            total = sum(pred_bucket.values())
            p_t = (pred_bucket.get(actual, 0) + 0.1) / (total + 0.1 * vocab)
            best_t = max(pred_bucket.items(), key=lambda kv: kv[1])[0]
            if best_t == actual:
                correct += 1
        else:
            p_t = None
            best_t = None

        # Token n-gram
        c1 = ids[i]
        c2 = (ids[i - 1] << 18) | ids[i]
        c3 = (ids[i - 2] << 36) | (ids[i - 1] << 18) | ids[i]
        p_ng = hierarchical_prob(
            actual, g3.get(c3), t3.get(c3, 0), g2.get(c2), t2.get(c2, 0),
            g1.get(c1), t1.get(c1, 0), uni, utot, vocab,
        )
        bits_ng += -math.log2(p_ng)
        # top-1 ngram
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
        bits_ng += 0.0

        if p_t is None:
            p_h, best_h = p_ng, best_ng
        else:
            p_h = args.teacher_weight * p_t + (1.0 - args.teacher_weight) * p_ng
            best_h = best_t
        bits += -math.log2(p_t) if p_t is not None else 0.0
        bits_h += -math.log2(p_h)
        if best_h == actual:
            correct_h += 1
        eval_n += 1
        eval_bytes += len(tok_text(actual).encode("utf-8", errors="replace"))

    def show(name, c, b, n, nbytes):
        n = n or 1
        nbytes = nbytes or 1
        print(f"{name}: top1={100.0*c/n:.2f}% bpt={b/n:.4f} "
              f"bpc={b/nbytes:.4f}")

    print(f"token_hdc_ntt: eval_tokens={eval_n} eval_bytes={eval_bytes} "
          f"exact={exact_n} nearest={nearest_n} "
          f"coverage={100.0*(exact_n+nearest_n)/max(1,eval_n):.2f}%")
    show("hdc_teacher", correct, bits, teacher_n, teacher_bytes)
    show("token_ngram", correct_ng, bits_ng, eval_n, eval_bytes)
    show("hdc_teacher_plus_ngram", correct_h, bits_h, eval_n, eval_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
