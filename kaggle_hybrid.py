#!/usr/bin/env python3
"""Kaggle recovery: build hybrid teacher + students from an existing 7B shard.

Use this after a pure Qwen teacher run has produced:
    /kaggle/working/NaNLM/build/qwen7b_alice.ntr
    /kaggle/working/NaNLM/build/qwen7b_alice.ntr.card.json

It does NOT load Qwen or use the GPU. It:

  1. builds a raw 3-gram teacher over the same training range;
  2. merges Qwen + ngram into qwen7b_hybrid.ntr;
  3. compiles the 8-slot and 3-slot students;
  4. scores full held-out streams;
  5. packages /kaggle/working/qwen7b_hybrid_run.tgz.

Run:
    !git -C /kaggle/working/NaNLM pull --ff-only
    !python /kaggle/working/NaNLM/kaggle_hybrid.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get("NANLM_WORK", "/kaggle/working"))
REPO = WORK / "NaNLM"


def sh(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="build/qwen7b_alice.ntr")
    parser.add_argument("--corpus", default="data/alice_excerpt.txt")
    parser.add_argument("--max-keys", type=int, default=4096)
    parser.add_argument("--teacher-scale", type=int, default=16)
    parser.add_argument("--out-prefix", default="qwen7b_hybrid")
    args = parser.parse_args()

    if not REPO.is_dir():
        print(f"missing repo: {REPO}", file=sys.stderr)
        return 1
    teacher = REPO / args.teacher
    corpus = REPO / args.corpus
    if not teacher.exists():
        print(f"missing teacher shard: {teacher}", file=sys.stderr)
        return 1
    if not corpus.exists():
        print(f"missing corpus: {corpus}", file=sys.stderr)
        return 1

    build = REPO / "build"
    build.mkdir(parents=True, exist_ok=True)
    ngram = build / "qwen7b_ngram.ntr"
    hybrid = build / f"{args.out_prefix}.ntr"
    scaled = build / "qwen7b_teacher_scaled.ntr"
    card = Path(str(teacher) + ".card.json")
    corpus_size = corpus.stat().st_size
    train_end = int(corpus_size * 0.85)

    print("=== tools ===", flush=True)
    sh(["make", "build/nanlm_teacher", "build/nanlm_slot8_train",
        "build/nanlm_teacher_slot", "build/nanlm_asm", "build/nanlm_vm"],
       cwd=REPO)

    print("=== hybrid backoff ===", flush=True)
    if args.teacher_scale < 1:
        raise SystemExit("--teacher-scale must be >= 1")
    sh(["./build/nanlm_teacher", "build-ngram", str(corpus), str(ngram),
        "0", str(train_end), "4"], cwd=REPO)
    sh(["./build/nanlm_teacher", "scale", str(scaled), str(teacher),
        "1", str(args.teacher_scale)], cwd=REPO)
    sh(["./build/nanlm_teacher", "merge", str(hybrid), str(ngram),
        str(scaled)], cwd=REPO)
    sh(["./build/nanlm_teacher", "stats", str(teacher)], cwd=REPO)
    sh(["./build/nanlm_teacher", "stats", str(ngram)], cwd=REPO)
    sh(["./build/nanlm_teacher", "stats", str(scaled)], cwd=REPO)
    sh(["./build/nanlm_teacher", "stats", str(hybrid)], cwd=REPO)

    slot8_tsv = build / f"{args.out_prefix}_slot8.tsv"
    slot8_nas = build / f"{args.out_prefix}_slot8.nas"
    slot8_exp = build / f"{args.out_prefix}_slot8.expected"
    slot8_man = build / f"{args.out_prefix}_slot8.manifest"
    slot3_tsv = build / f"{args.out_prefix}_slot3.tsv"
    slot3_nas = build / f"{args.out_prefix}_slot3.nas"
    slot3_exp = build / f"{args.out_prefix}_slot3.expected"
    slot3_man = build / f"{args.out_prefix}_slot3.manifest"

    print("=== 8-slot student ===", flush=True)
    sh(["./build/nanlm_slot8_train",
        "--teacher", str(hybrid), "--corpus", str(corpus),
        "--max-keys", str(args.max_keys),
        "--eval-log", str(slot8_tsv),
        "--out", str(slot8_nas), "--expected", str(slot8_exp),
        "--manifest", str(slot8_man)], cwd=REPO)
    sh(["./build/nanlm_asm", str(slot8_nas),
        str(build / f"{args.out_prefix}_slot8.nan")], cwd=REPO)
    sh(["./build/nanlm_vm", "--img", str(build / f"{args.out_prefix}_slot8.nan"),
        "--steps", "600000000"], cwd=REPO)

    print("=== 3-slot student ===", flush=True)
    sh(["./build/nanlm_teacher_slot",
        "--teacher", str(hybrid), "--corpus", str(corpus),
        "--eval-log", str(slot3_tsv),
        "--out", str(slot3_nas), "--expected", str(slot3_exp),
        "--manifest", str(slot3_man)], cwd=REPO)

    print("=== full-stream score ===", flush=True)
    sh([sys.executable, "experiments/eval_stream.py",
        "--run", f"slot8={slot8_tsv}={slot8_nas}",
        "--run", f"slot3={slot3_tsv}={slot3_nas}",
        "--boot", "5000"], cwd=REPO)

    print("=== package ===", flush=True)
    files = [teacher, card, ngram, scaled, hybrid, slot8_tsv, slot3_tsv, slot8_nas,
             slot8_exp, slot8_man, slot3_nas, slot3_exp, slot3_man]
    tar = WORK / "qwen7b_hybrid_run.tgz"
    sh(["tar", "czf", str(tar), "-C", str(REPO)] +
       [str(p.relative_to(REPO)) for p in files if p.exists()])
    for p in files:
        if p.exists():
            shutil.copy2(p, WORK / p.name)
    print(f"download: {tar}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
