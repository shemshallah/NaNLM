#!/usr/bin/env python3
"""Score full held-out streams with paired block-bootstrap intervals.

Input is the per-byte TSV emitted by nanlm_slot_eval_stream:

    pos actual pred slot_bits ngram_bits slot_correct ngram_correct

Unlike a 128-context sample, this scores every held-out byte. Confidence
intervals use a moving-block bootstrap (default block 64) so serial
dependence is not treated as independent evidence.

Hutter-style total:
    model_bits = 8 * model_file_bytes
    compressed_stream_bits = sum(slot_bits)
    total_bpc = (model_bits + compressed_stream_bits) / stream_bytes

The model file is counted once; a bigger model must earn its bytes.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean


def read_rows(path: Path) -> list[tuple[float, float, int, int]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        header = f.readline().strip().split()
        required = {"slot_bits", "ngram_bits", "slot_correct", "ngram_correct"}
        if not required.issubset(header):
            raise SystemExit(f"{path}: missing columns {required - set(header)}")
        idx = {name: header.index(name) for name in header}
        for line in f:
            fields = line.split()
            if len(fields) < len(header):
                continue
            rows.append((
                float(fields[idx["slot_bits"]]),
                float(fields[idx["ngram_bits"]]),
                int(fields[idx["slot_correct"]]),
                int(fields[idx["ngram_correct"]]),
            ))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def block_sums(rows: list[tuple[float, float, int, int]], block: int):
    blocks = []
    for start in range(0, len(rows), block):
        chunk = rows[start:start + block]
        blocks.append((
            len(chunk),
            sum(r[0] for r in chunk),
            sum(r[1] for r in chunk),
            sum(r[2] for r in chunk),
            sum(r[3] for r in chunk),
        ))
    return blocks


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def bootstrap(rows, block: int, n_boot: int, seed: int):
    blocks = block_sums(rows, block)
    rng = random.Random(seed)
    n = len(rows)
    bpc = []
    top1 = []
    diff_bpc = []
    diff_top1 = []
    for _ in range(n_boot):
        total_n = 0
        slot_bits = 0.0
        ng_bits = 0.0
        slot_correct = 0
        ng_correct = 0
        for _ in range(len(blocks)):
            bn, sb, nb, sc, nc = blocks[rng.randrange(len(blocks))]
            total_n += bn
            slot_bits += sb
            ng_bits += nb
            slot_correct += sc
            ng_correct += nc
        bpc.append(slot_bits / total_n)
        top1.append(100.0 * slot_correct / total_n)
        diff_bpc.append((slot_bits - ng_bits) / total_n)
        diff_top1.append(100.0 * (slot_correct - ng_correct) / total_n)
    bpc.sort()
    top1.sort()
    diff_bpc.sort()
    diff_top1.sort()
    return {
        "bpc": (percentile(bpc, 0.025), percentile(bpc, 0.975)),
        "top1": (percentile(top1, 0.025), percentile(top1, 0.975)),
        "diff_bpc": (percentile(diff_bpc, 0.025), percentile(diff_bpc, 0.975)),
        "diff_top1": (percentile(diff_top1, 0.025), percentile(diff_top1, 0.975)),
    }


def evaluate(label: str, log_path: Path, model_path: Path | None,
             block: int, n_boot: int, seed: int) -> dict:
    rows = read_rows(log_path)
    n = len(rows)
    slot_bpc = sum(r[0] for r in rows) / n
    ng_bpc = sum(r[1] for r in rows) / n
    slot_top1 = 100.0 * sum(r[2] for r in rows) / n
    ng_top1 = 100.0 * sum(r[3] for r in rows) / n
    model_bytes = model_path.stat().st_size if model_path and model_path.exists() else 0
    total_bits = sum(r[0] for r in rows) + 8.0 * model_bytes
    total_bpc = total_bits / n
    cis = bootstrap(rows, block, n_boot, seed)
    return {
        "label": label,
        "log": str(log_path),
        "model": str(model_path) if model_path else "",
        "n": n,
        "slot_bpc": slot_bpc,
        "slot_top1": slot_top1,
        "ngram_bpc": ng_bpc,
        "ngram_top1": ng_top1,
        "diff_bpc": slot_bpc - ng_bpc,
        "diff_top1": slot_top1 - ng_top1,
        "bpc_ci": cis["bpc"],
        "top1_ci": cis["top1"],
        "diff_bpc_ci": cis["diff_bpc"],
        "diff_top1_ci": cis["diff_top1"],
        "model_bytes": model_bytes,
        "model_bpc": 8.0 * model_bytes / n,
        "total_bpc": total_bpc,
    }


def fmt(x: float, places: int = 4) -> str:
    return f"{x:.{places}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        metavar="LABEL=LOG[=MODEL]")
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--json", metavar="FILE")
    args = parser.parse_args()

    results = []
    for spec in args.run:
        parts = spec.split("=", 2)
        if len(parts) < 2:
            parser.error("--run needs LABEL=LOG[=MODEL]")
        label, log = parts[0], Path(parts[1])
        model = Path(parts[2]) if len(parts) > 2 and parts[2] else None
        results.append(evaluate(label, log, model, args.block, args.boot, args.seed))

    print("| run | bytes | top1 | BPC | diff top1 | diff BPC | model B | total BPC |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r['label']} | {r['n']} | {r['slot_top1']:.2f}% | "
              f"{r['slot_bpc']:.4f} | {r['diff_top1']:+.2f} | "
              f"{r['diff_bpc']:+.4f} | {r['model_bytes']} | "
              f"{r['total_bpc']:.4f} |")
    print()
    for r in results:
        print(f"{r['label']}:")
        print(f"  BPC 95% block-bootstrap CI: "
              f"[{fmt(r['bpc_ci'][0])}, {fmt(r['bpc_ci'][1])}]")
        print(f"  top1 95% CI: [{fmt(r['top1_ci'][0], 2)}, "
              f"{fmt(r['top1_ci'][1], 2)}]%")
        print(f"  paired diff BPC 95% CI: "
              f"[{fmt(r['diff_bpc_ci'][0])}, {fmt(r['diff_bpc_ci'][1])}]")
        print(f"  paired diff top1 95% CI: "
              f"[{fmt(r['diff_top1_ci'][0], 2)}, "
              f"{fmt(r['diff_top1_ci'][1], 2)}] points")
        print(f"  model bytes: {r['model_bytes']} "
              f"({r['model_bpc']:.4f} BPC equivalent)")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n",
                                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
