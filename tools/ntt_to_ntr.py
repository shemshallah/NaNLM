#!/usr/bin/env python3
"""Convert an NTT1 token-level teacher shard into a full-byte-coverage NTR1 shard.

Why the first byte-level teacher shard failed:
  A token averages ~4 bytes. Emitting only the first byte of each predicted
  token at token boundaries leaves ~75% of byte positions inside multi-byte
  tokens with zero teacher records.

How prefix-trie byte expansion fixes this without touching the GPU:
  Each NTT1 record is (3-token context [t0, t1, t2], predicted next_token, weight).
  Let prefix_bytes = bytes(t0) + bytes(t1) + bytes(t2), and
      target_bytes = bytes(next_token) = b_0 b_1 ... b_{m-1}.
  Then predicting `next_token` with `weight` implies the chain of byte
  transitions:
      last_8_bytes(prefix_bytes + b_0..b_{j-1}) -> b_j   (for j = 0 .. m-1)
  with weight `weight`.

Summing these across all NTT1 records yields an NTR1 byte shard that covers
both token boundaries (j=0) and intra-token continuations (j>0).
"""

from __future__ import annotations

import argparse
import struct
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from token_shard import read as read_ntt  # noqa: E402

NTR_MAGIC = 0x3152544E
NTR_HEADER = struct.Struct("<IIII")
NTR_RECORD = struct.Struct("<QBxH4x")
TOKEN_MAX = 1 << 18


def unpack3(ctx: int) -> tuple[int, int, int]:
    return (
        (ctx >> 36) & (TOKEN_MAX - 1),
        (ctx >> 18) & (TOKEN_MAX - 1),
        ctx & (TOKEN_MAX - 1),
    )


def pack8(buf: bytes) -> int:
    if len(buf) < 8:
        buf = (b" " * (8 - len(buf))) + buf
    else:
        buf = buf[-8:]
    key = 0
    for b in buf:
        key = (key << 8) | b
    return key


def write_ntr(path: Path, counts: dict[tuple[int, int], int]) -> None:
    items = sorted(counts.items(), key=lambda kv: kv[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(NTR_HEADER.pack(NTR_MAGIC, 1, len(items), 0))
        for (ctx8, nxt), w in items:
            f.write(NTR_RECORD.pack(ctx8, nxt, min(65535, max(1, w))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--scale-div", type=int, default=64,
                        help="divide NTT weights by this factor before accumulating")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        print(f"ntt-to-ntr: missing dependency: {exc}", file=sys.stderr)
        return 2

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ctx_len, recs = read_ntt(Path(args.ntt))
    if ctx_len != 3:
        print("ntt-to-ntr: expected 3-token NTT1 shard", file=sys.stderr)
        return 1

    piece_cache: dict[int, bytes] = {}

    def piece(tid: int) -> bytes:
        res = piece_cache.get(tid)
        if res is None:
            res = tokenizer.decode(
                [tid], skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ).encode("utf-8", errors="replace")
            piece_cache[tid] = res
        return res

    byte_counts: dict[tuple[int, int], int] = defaultdict(int)
    boundary_transitions = 0
    internal_transitions = 0
    div = max(1, args.scale_div)

    for ctx, nxt, weight in recs:
        t0, t1, t2 = unpack3(ctx)
        prefix = piece(t0) + piece(t1) + piece(t2)
        target = piece(nxt)
        if not prefix or not target:
            continue
        w = max(1, (weight + div // 2) // div)
        rolling = prefix
        for j, b in enumerate(target):
            key8 = pack8(rolling)
            byte_counts[(key8, b)] = min(65535, byte_counts[(key8, b)] + w)
            rolling = rolling + bytes([b])
            if j == 0:
                boundary_transitions += 1
            else:
                internal_transitions += 1

    out = Path(args.out)
    write_ntr(out, byte_counts)
    contexts = len({k for k, _ in byte_counts})
    print(f"ntt-to-ntr: ntt_records={len(recs)} "
          f"boundary={boundary_transitions} internal={internal_transitions} "
          f"byte_contexts={contexts} ntr_records={len(byte_counts)} out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
