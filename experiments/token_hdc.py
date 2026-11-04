#!/usr/bin/env python3
"""Token-level HDC prototype with regime-specialized bind operators.

This is the first token-student simulation. It tokenizes text into word /
punct / space tokens, encodes a 3-token context with sparse random HDC codes,
binds each token to a position code using BOOL.n, bundles the three, and stores
one prototype per observed context.

Two encoders are compared:
  fixed: every slot binds with XOR (n=6)
  regime: n is selected by the token class (space/alpha/digit/punct)

The model is evaluated on held-out tokens with full-stream BPC and top-1.
This is a CPU research prototype, not the VM model.
"""
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

TOKEN_RE = re.compile(r"\w+|[^\w\s]|\s+")


def bool_n(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(a, dtype=bool)
    if n & 1: out |= (~(a | b))
    if n & 2: out |= ((~a) & b)
    if n & 4: out |= (a & (~b))
    if n & 8: out |= (a & b)
    return out


def bundle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (a & b) | (a & c) | (b & c)


def token_class(tok: str) -> int:
    if tok.isspace(): return 0
    if tok.isalpha() or ("_" in tok and tok.replace("_", "").isalnum()): return 1
    if tok.isdigit(): return 2
    return 3


def pack_matrix(bits: np.ndarray) -> np.ndarray:
    # bits [N,D] bool -> [N, D/64] uint64
    packed = np.packbits(bits, axis=1, bitorder="little")
    if packed.shape[1] % 8:
        pad = 8 - packed.shape[1] % 8
        packed = np.pad(packed, ((0, 0), (0, pad)))
    return packed.view(np.uint64)


def run(mode: str, dim: int, density: float, max_vocab: int, top_k: int,
        seed: int) -> None:
    rng = np.random.default_rng(seed)
    text = Path("data/alice_excerpt.txt").read_text(encoding="utf-8")
    raw = TOKEN_RE.findall(text)
    counts = defaultdict(int)
    for t in raw:
        counts[t] += 1
    vocab = [t for t, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:max_vocab]]
    to_id = {t: i for i, t in enumerate(vocab)}
    ids = [to_id.get(t, 0) for t in raw]
    ntok = len(ids)
    split = int(ntok * 0.85)
    print(f"token_hdc: mode={mode} tokens={ntok} vocab={len(vocab)} "
          f"dim={dim} density={density}")

    # Sparse random token codes and dense position codes.
    codes = (rng.random((len(vocab), dim)) < density)
    pos = (rng.random((3, dim)) < 0.5)
    # Repaired per-class truth tables. 6 = XOR; 9/12/15 add mixing.
    n_by_class = {0: 6, 1: 6, 2: 12, 3: 9} if mode == "regime" else {
        0: 6, 1: 6, 2: 6, 3: 6
    }

    def encode(ctx: list[int]) -> np.ndarray:
        parts = []
        for slot, tid in enumerate(ctx):
            # regime is based on the actual token's class
            token_text = vocab[tid] if tid < len(vocab) else ""
            n = n_by_class[token_class(token_text)] if mode == "regime" else 6
            parts.append(bool_n(codes[tid], pos[slot], n))
        return bundle3(parts[0], parts[1], parts[2])

    # Context prototypes and next-token counts.
    proto = {}
    labels = {}
    next_counts = {}
    for i in range(2, split - 1):
        key = (ids[i - 2], ids[i - 1], ids[i])
        if key not in proto:
            proto[key] = encode(list(key))
            next_counts[key] = defaultdict(int)
        next_counts[key][ids[i + 1]] += 1
    keys = list(proto.keys())
    if not keys:
        print("token_hdc: no contexts")
        return
    P = pack_matrix(np.stack([proto[k] for k in keys]))
    print(f"token_hdc: prototypes={len(keys)} "
          f"model_bits={len(keys) * dim}")

    correct = 0
    bits = 0.0
    eval_bytes = 0
    eval_n = 0
    for i in range(split, ntok - 1):
        key = (ids[i - 2], ids[i - 1], ids[i])
        q = pack_matrix(encode(list(key))[None, :])[0]
        d = np.bitwise_count(P ^ q).sum(axis=1)
        best = keys[int(np.argmin(d))]
        bucket = next_counts[best]
        total = sum(bucket.values())
        actual = ids[i + 1]
        best_tok = max(bucket.items(), key=lambda kv: kv[1])[0]
        if best_tok == actual:
            correct += 1
        p = (bucket.get(actual, 0) + 0.1) / (total + 0.1 * len(vocab))
        bits += -math.log2(max(1e-12, p))
        eval_bytes += len(raw[i + 1].encode("utf-8", errors="replace"))
        eval_n += 1
    print(f"token_hdc: eval_tokens={eval_n} eval_bytes={eval_bytes} "
          f"top1={100.0 * correct / eval_n:.2f}% "
          f"bpt={bits / eval_n:.4f} bpc={bits / max(1, eval_bytes):.4f} "
          f"active_proto={100.0 * float(np.bitwise_count(P).sum()) / max(1, P.size * 64):.2f}%")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("fixed", "regime"), default="regime")
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--density", type=float, default=0.05)
    p.add_argument("--max-vocab", type=int, default=4096)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    run(args.mode, args.dim, args.density, args.max_vocab, args.top_k,
        args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
