#!/usr/bin/env python3
"""Inspect, merge, or scale NTT1 token-level teacher shards.

NTT1 is emitted by tools/hf_token_teacher.py:
  header: magic, version, nrec, context_len
  record: uint64 context, uint32 next token id, uint32 weight
"""
from __future__ import annotations

import argparse
import struct
from collections import defaultdict
from pathlib import Path

MAGIC = 0x3154544E
HEADER = struct.Struct("<IIII")
RECORD = struct.Struct("<QII")


def read(path: Path):
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise SystemExit(f"{path}: truncated")
    magic, version, n, ctx_len = HEADER.unpack_from(data, 0)
    if magic != MAGIC or version != 1:
        raise SystemExit(f"{path}: not NTT1 v1")
    if len(data) != HEADER.size + n * RECORD.size:
        raise SystemExit(f"{path}: bad record count")
    recs = []
    off = HEADER.size
    for _ in range(n):
        recs.append(RECORD.unpack_from(data, off))
        off += RECORD.size
    return ctx_len, recs


def write(path: Path, ctx_len: int, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(HEADER.pack(MAGIC, 1, len(recs), ctx_len))
        for rec in recs:
            f.write(RECORD.pack(*rec))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats")
    s.add_argument("file")
    m = sub.add_parser("merge")
    m.add_argument("out")
    m.add_argument("inputs", nargs="+")
    sc = sub.add_parser("scale")
    sc.add_argument("out")
    sc.add_argument("input")
    sc.add_argument("num", type=int)
    sc.add_argument("den", type=int)
    args = parser.parse_args()

    if args.cmd == "stats":
        ctx_len, recs = read(Path(args.file))
        contexts = len({r[0] for r in recs})
        total = sum(r[2] for r in recs)
        print(f"token_shard: {args.file} context={ctx_len} "
              f"contexts={contexts} records={len(recs)} weight={total}")
        return 0
    if args.cmd == "merge":
        merged = defaultdict(int)
        ctx_len = None
        for name in args.inputs:
            cl, recs = read(Path(name))
            if ctx_len is None:
                ctx_len = cl
            if cl != ctx_len:
                raise SystemExit("cannot merge different context lengths")
            for ctx, nxt, weight in recs:
                merged[(ctx, nxt)] += weight
        out = sorted(merged.items(), key=lambda kv: kv[0])
        write(Path(args.out), ctx_len or 3,
              [(ctx, nxt, min(0xFFFFFFFF, w)) for (ctx, nxt), w in out])
        print(f"token_shard: merged {len(out)} records -> {args.out}")
        return 0
    if args.cmd == "scale":
        if args.den == 0:
            raise SystemExit("denominator must be nonzero")
        ctx_len, recs = read(Path(args.input))
        out = []
        for ctx, nxt, weight in recs:
            w = (weight * args.num + args.den // 2) // args.den
            if w == 0 and weight:
                w = 1
            out.append((ctx, nxt, min(0xFFFFFFFF, w)))
        write(Path(args.out), ctx_len, out)
        print(f"token_shard: scaled {len(out)} records -> {args.out}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
