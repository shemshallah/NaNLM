#!/usr/bin/env python3
"""Kaggle CPU-only grid: tune teacher scale / 8-slot size from existing shards.

No Qwen load, no GPU. It reuses:
    build/qwen7b_alice.ntr
and builds the n-gram once, then tests:
    teacher scale = 0 (ngram-only control), 4, 8, 16, 32, 64
    8-slot max keys = 2048 and 4096

Run:
    !python /kaggle/working/NaNLM/kaggle_grid.py
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get("NANLM_WORK", "/kaggle/working"))
REPO = WORK / "NaNLM"


def sh(cmd: list[str], cwd: Path = REPO) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    if proc.stdout:
        print(proc.stdout, end="", flush=True)


def eval_pair(slot8_tsv: Path, slot8_nas: Path,
              slot3_tsv: Path, slot3_nas: Path) -> str:
    proc = subprocess.run(
        [sys.executable, "experiments/eval_stream.py",
         "--run", f"slot8={slot8_tsv}={slot8_nas}",
         "--run", f"slot3={slot3_tsv}={slot3_nas}",
         "--boot", "500"],
        cwd=str(REPO), check=True, capture_output=True, text=True,
    )
    print(proc.stdout, flush=True)
    return proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="build/qwen7b_alice.ntr")
    parser.add_argument("--corpus", default="data/alice_excerpt.txt")
    parser.add_argument("--scales", default="0,4,8,16,32,64")
    parser.add_argument("--max-keys", default="2048,4096")
    args = parser.parse_args()

    teacher = REPO / args.teacher
    corpus = REPO / args.corpus
    if not teacher.exists():
        print(f"missing teacher: {teacher}", file=sys.stderr)
        return 1
    if not corpus.exists():
        print(f"missing corpus: {corpus}", file=sys.stderr)
        return 1

    build = REPO / "build"
    grid = build / "grid"
    grid.mkdir(parents=True, exist_ok=True)
    ngram = grid / "ngram.ntr"
    log_path = WORK / "kaggle_grid.log"
    log = log_path.open("w", encoding="utf-8")
    original_stdout = sys.stdout

    class Tee:
        def write(self, data: str) -> None:
            original_stdout.write(data)
            log.write(data)
        def flush(self) -> None:
            original_stdout.flush()
            log.flush()

    sys.stdout = Tee()  # type: ignore[assignment]
    try:
        sh(["make", "build/nanlm_teacher", "build/nanlm_slot8_train",
            "build/nanlm_teacher_slot", "build/nanlm_asm", "build/nanlm_vm"])
        size = corpus.stat().st_size
        train_end = int(size * 0.85)
        sh(["./build/nanlm_teacher", "build-ngram", str(corpus), str(ngram),
            "0", str(train_end), "4"])
        sh(["./build/nanlm_teacher", "stats", str(teacher)])
        sh(["./build/nanlm_teacher", "stats", str(ngram)])

        scales = [int(x) for x in args.scales.split(",") if x.strip()]
        key_sizes = [int(x) for x in args.max_keys.split(",") if x.strip()]
        summary: list[dict] = []
        for scale in scales:
            if scale == 0:
                merged = ngram
            else:
                scaled = grid / f"teacher_s{scale}.ntr"
                merged = grid / f"hybrid_s{scale}.ntr"
                sh(["./build/nanlm_teacher", "scale", str(scaled), str(teacher),
                    "1", str(scale)])
                sh(["./build/nanlm_teacher", "merge", str(merged), str(ngram),
                    str(scaled)])
            for keys in key_sizes:
                tag = f"s{scale}_k{keys}"
                s8 = grid / f"{tag}_slot8"
                s3 = grid / f"{tag}_slot3"
                sh(["./build/nanlm_slot8_train",
                    "--teacher", str(merged), "--corpus", str(corpus),
                    "--max-keys", str(keys),
                    "--eval-log", str(s8) + ".tsv",
                    "--out", str(s8) + ".nas",
                    "--expected", str(s8) + ".expected",
                    "--manifest", str(s8) + ".manifest"])
                sh(["./build/nanlm_teacher_slot",
                    "--teacher", str(merged), "--corpus", str(corpus),
                    "--eval-log", str(s3) + ".tsv",
                    "--out", str(s3) + ".nas",
                    "--expected", str(s3) + ".expected",
                    "--manifest", str(s3) + ".manifest"])
                text = eval_pair(Path(str(s8) + ".tsv"), Path(str(s8) + ".nas"),
                                 Path(str(s3) + ".tsv"), Path(str(s3) + ".nas"))
                row = {"scale": scale, "keys": keys, "eval": text}
                summary.append(row)
                if s8.with_suffix(".nas").exists():
                    s8.with_suffix(".nas").unlink()
                if s3.with_suffix(".nas").exists():
                    s3.with_suffix(".nas").unlink()
        print("\n=== SUMMARY ===", flush=True)
        for row in summary:
            print(f"scale={row['scale']} keys={row['keys']}", flush=True)
            for line in row["eval"].splitlines():
                if line.startswith("| slot8") or line.startswith("| slot3"):
                    print(line, flush=True)
        print(f"\nlog: {log_path}", flush=True)
    finally:
        sys.stdout = original_stdout
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
