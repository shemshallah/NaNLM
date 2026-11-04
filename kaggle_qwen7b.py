#!/usr/bin/env python3
"""Kaggle one-shot runner: Qwen 7B teacher -> NaNLM 8-slot student.

Run in a Kaggle notebook with:
    !git clone --depth 1 https://github.com/shemshallah/NaNLM.git
    !python NaNLM/kaggle_qwen7b.py

Enable Internet and a GPU (T4 x2 or P100). The script uses one GPU and 4-bit
Qwen; it never installs torch (Kaggle's CUDA torch is kept).

Outputs land in /kaggle/working:
    qwen7b_alice.ntr
    qwen7b_alice.card.json
    qwen7b_slot8.nas / .expected / .manifest / .tsv
    qwen7b_slot3.nas / .expected / .manifest / .tsv
    qwen7b_run.tgz
    kaggle_run.log
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
MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"
CORPUS_DEFAULT = "data/alice_excerpt.txt"


def sh(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None,
       dry: bool = False) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    if dry:
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--corpus", default=CORPUS_DEFAULT)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--out-name", default="qwen7b_alice.ntr")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        WORK.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"cannot create {WORK}: {exc}", file=sys.stderr)
        return 1
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("HF_HOME", str(WORK / "hf"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(WORK / "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(WORK / "hf"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env = dict(os.environ)

    print("=== disk ===", flush=True)
    usage = shutil.disk_usage(WORK)
    print(f"{WORK} free: {usage.free / 1e9:.1f} GB", flush=True)
    if usage.free < 20e9:
        print("WARNING: 7B 4-bit download may not fit; free space <20 GB",
              file=sys.stderr, flush=True)

    print("=== GPU ===", flush=True)
    sh([sys.executable, "-c",
        "import torch; print(torch.__version__, torch.version.cuda, "
        "torch.cuda.is_available(), "
        "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"],
       env=env, dry=args.dry_run)

    print("=== install teacher deps (not torch) ===", flush=True)
    sh([sys.executable, "-m", "pip", "install", "-q", "-U",
        "transformers>=4.46", "accelerate", "bitsandbytes",
        "sentencepiece", "huggingface_hub"], env=env, dry=args.dry_run)

    print("=== repo ===", flush=True)
    if not (REPO / ".git").exists():
        sh(["git", "clone", "--depth", "1",
            "https://github.com/shemshallah/NaNLM.git", str(REPO)],
           env=env, dry=args.dry_run)
    else:
        sh(["git", "-C", str(REPO), "pull", "--ff-only"], env=env,
           dry=args.dry_run)
    if args.dry_run:
        return 0

    corpus = REPO / args.corpus
    if not corpus.exists():
        print(f"missing corpus: {corpus}", file=sys.stderr)
        return 1

    ntr = REPO / "build" / args.out_name
    card = REPO / "build" / (args.out_name + ".card.json")
    ntr.parent.mkdir(parents=True, exist_ok=True)

    teacher = [
        sys.executable, "tools/hf_teacher.py",
        "--model", args.model,
        "--corpus", str(corpus),
        "--out", str(ntr),
        "--card", str(card),
        "--max-tokens", str(args.max_tokens),
        "--top-k", str(args.top_k),
        "--chunk", str(args.chunk),
        "--device", "cuda",
        "--dtype", "float16",
    ]
    if not args.no_4bit:
        teacher += ["--load-in-4bit"]
    print("=== teacher ===", flush=True)
    sh(teacher, cwd=REPO, env=env, dry=args.dry_run)

    if args.dry_run:
        return 0
    if not card.exists():
        print("teacher card missing", file=sys.stderr)
        return 1
    data = json.loads(card.read_text(encoding="utf-8"))
    print("teacher card:", json.dumps(data, indent=2, sort_keys=True), flush=True)
    teacher_ok = (
        data.get("teacher_bpc", 99) < data.get("ngram_bpc", 0)
        and data.get("eval_positions", 0) > 0
    )
    print(f"TEACHER GATE: {'PASS' if teacher_ok else 'FAIL'}", flush=True)

    if args.no_compile:
        shutil.copy2(ntr, WORK / ntr.name)
        shutil.copy2(card, WORK / card.name)
        print(f"wrote {WORK / ntr.name} and {WORK / card.name}")
        return 0

    print("=== compile students ===", flush=True)
    sh(["make", "build/nanlm_teacher", "build/nanlm_slot8_train",
        "build/nanlm_teacher_slot", "build/nanlm_asm", "build/nanlm_vm"],
       cwd=REPO, env=env)
    sh(["./build/nanlm_teacher", "stats", str(ntr)], cwd=REPO, env=env)

    slot8_tsv = REPO / "build/qwen7b_slot8.tsv"
    slot8_nas = REPO / "build/qwen7b_slot8.nas"
    slot8_exp = REPO / "build/qwen7b_slot8.expected"
    slot8_man = REPO / "build/qwen7b_slot8.manifest"
    slot3_tsv = REPO / "build/qwen7b_slot3.tsv"
    slot3_nas = REPO / "build/qwen7b_slot3.nas"
    slot3_exp = REPO / "build/qwen7b_slot3.expected"
    slot3_man = REPO / "build/qwen7b_slot3.manifest"

    sh(["./build/nanlm_slot8_train",
        "--teacher", str(ntr), "--corpus", str(corpus),
        "--max-keys", "2048", "--eval-log", str(slot8_tsv),
        "--out", str(slot8_nas), "--expected", str(slot8_exp),
        "--manifest", str(slot8_man)], cwd=REPO, env=env)
    sh(["./build/nanlm_asm", str(slot8_nas), str(REPO / "build/qwen7b_slot8.nan")],
       cwd=REPO, env=env)
    sh(["./build/nanlm_vm", "--img", str(REPO / "build/qwen7b_slot8.nan"),
        "--steps", "600000000"], cwd=REPO, env=env)
    sh(["./build/nanlm_teacher_slot",
        "--teacher", str(ntr), "--corpus", str(corpus),
        "--eval-log", str(slot3_tsv),
        "--out", str(slot3_nas), "--expected", str(slot3_exp),
        "--manifest", str(slot3_man)], cwd=REPO, env=env)

    sh([sys.executable, "experiments/eval_stream.py",
        "--run", f"slot8={slot8_tsv}={slot8_nas}",
        "--run", f"slot3={slot3_tsv}={slot3_nas}",
        "--boot", "5000"], cwd=REPO, env=env)

    print("=== package ===", flush=True)
    files = [ntr, card, slot8_tsv, slot3_tsv, slot8_nas, slot8_exp,
             slot8_man, slot3_nas, slot3_exp, slot3_man]
    tar = WORK / "qwen7b_run.tgz"
    sh(["tar", "czf", str(tar), "-C", str(REPO)] +
       [str(p.relative_to(REPO)) for p in files], env=env)
    for p in files:
        if p.exists():
            shutil.copy2(p, WORK / p.name)
    print(f"download: {tar}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
