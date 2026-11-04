#!/usr/bin/env python3
"""Kaggle one-command token teacher run.

Run in a Kaggle notebook:
    !python /kaggle/working/NaNLM/kaggle_token.py

It uses the cached Qwen 7B model, emits an NTT1 token shard, writes the token
teacher card, runs token-shard stats, and packages everything into
/kaggle/working/qwen7b_tokens.tgz.

No byte teacher, no 3/8-slot compile. This is the token-level next step.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get("NANLM_WORK", "/kaggle/working"))
REPO = WORK / "NaNLM"


def sh(cmd: list[str], cwd: Path = REPO, log=None) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    if log is None:
        subprocess.run(cmd, cwd=str(cwd), check=True)
    else:
        proc = subprocess.run(
            cmd, cwd=str(cwd), check=True,
            stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        if proc.stdout:
            log.write(proc.stdout)
            log.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--corpus", default="data/alice_excerpt.txt")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--context-tokens", type=int, default=3)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--skip-pull", action="store_true")
    args = parser.parse_args()

    try:
        WORK.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"cannot create {WORK}: {exc}", file=sys.stderr)
        return 1

    os.environ["HF_HOME"] = str(WORK / "hf")
    os.environ["TRANSFORMERS_CACHE"] = str(WORK / "hf")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(WORK / "hf")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not (REPO / "tools/hf_token_teacher.py").exists():
        print(f"missing repo/script: {REPO}", file=sys.stderr)
        print("clone: git clone --depth 1 "
              "https://github.com/shemshallah/NaNLM.git /kaggle/working/NaNLM",
              file=sys.stderr)
        return 1

    log_path = WORK / "token_teacher.log"
    log = log_path.open("w", encoding="utf-8")
    try:
        if not args.skip_pull:
            sh(["git", "-C", str(REPO), "pull", "--ff-only"], log=log)
        corpus = REPO / args.corpus
        if not corpus.exists():
            print(f"missing corpus: {corpus}", file=sys.stderr)
            return 1
        ntt = WORK / "qwen7b_tokens.ntt"
        card = WORK / "qwen7b_tokens.card.json"
        cmd = [
            sys.executable, str(REPO / "tools/hf_token_teacher.py"),
            "--model", args.model,
            "--corpus", str(corpus),
            "--out", str(ntt),
            "--card", str(card),
            "--max-tokens", str(args.max_tokens),
            "--top-k", str(args.top_k),
            "--chunk", str(args.chunk),
            "--context-tokens", str(args.context_tokens),
            "--device", "cuda",
            "--dtype", "float16",
        ]
        if not args.no_4bit:
            cmd.append("--load-in-4bit")
        sh(cmd, log=log)
        sh([sys.executable, str(REPO / "tools/token_shard.py"),
            "stats", str(ntt)], log=log)
        student_cmd = [
            sys.executable, str(REPO / "tools/token_student.py"),
            "--ntt", str(ntt),
            "--corpus", str(corpus),
            "--model", args.model,
            "--max-tokens", str(args.max_tokens),
        ]
        proc = subprocess.run(
            student_cmd, cwd=str(REPO), check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        print(proc.stdout, end="", flush=True)
        log.write(proc.stdout)
        log.flush()
        data = json.loads(card.read_text(encoding="utf-8"))
        print(json.dumps(data, indent=2, sort_keys=True), flush=True)
        print("TOKEN TEACHER GATE:",
              "PASS" if data.get("teacher_bpc", 99) < data.get("ngram_bpc", 0)
              else "FAIL", flush=True)
        log.flush()
        tar = WORK / "qwen7b_tokens.tgz"
        subprocess.run(
            ["tar", "czf", str(tar), "-C", str(WORK),
             ntt.name, card.name, log_path.name],
            check=True,
        )
        print(f"download: {tar}", flush=True)
        print(f"log: {log_path}", flush=True)
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"failed: {exc}", file=sys.stderr)
        print(f"log: {log_path}", file=sys.stderr)
        return 1
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
