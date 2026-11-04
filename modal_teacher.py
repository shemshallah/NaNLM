"""Modal runner for the batch HF teacher.

This is intentionally a batch inference job, not an always-on service. The
expensive part is a short GPU pass over a pinned corpus; the output is a small
additive teacher shard plus a teacher card.

Setup:
    pip install modal
    modal token new
    modal run modal_teacher.py --corpus data/alice_excerpt.txt

For a 7B teacher:
    modal run modal_teacher.py --corpus data/alice_excerpt.txt \
      --model Qwen/Qwen2.5-7B-Instruct --max-tokens 4096 --top-k 16

The model is cached in a Modal Volume. The output shard is written to the
`nanlm-teacher-shards` Volume; run `modal volume get nanlm-teacher-shards ...`
to fetch it. No Modal credentials are used by the runtime model.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("nanlm-teacher")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "bitsandbytes==0.44.1",
        "sentencepiece==0.2.0",
        "protobuf==5.28.3",
    )
    .add_local_file("tools/hf_teacher.py", "/root/hf_teacher.py")
)

hf_cache = modal.Volume.from_name("nanlm-hf-cache", create_if_missing=True)
shard_cache = modal.Volume.from_name("nanlm-teacher-shards", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={"/cache": hf_cache, "/out": shard_cache},
)
def teacher(
    corpus_text: str,
    model: str,
    revision: str,
    max_tokens: int,
    top_k: int,
    train_frac: float,
    out_name: str,
) -> dict:
    corpus = Path("/tmp/corpus.txt")
    corpus.write_text(corpus_text, encoding="utf-8")
    out = Path("/out") / out_name
    card = Path("/out") / (out_name + ".card.json")
    env = {
        **os.environ,
        "HF_HOME": "/cache/hf",
        "TRANSFORMERS_CACHE": "/cache/hf",
        "HUGGINGFACE_HUB_CACHE": "/cache/hf",
        "PYTHONUNBUFFERED": "1",
    }
    command = [
        sys.executable, "/root/hf_teacher.py",
        "--model", model,
        "--revision", revision,
        "--corpus", str(corpus),
        "--out", str(out),
        "--card", str(card),
        "--max-tokens", str(max_tokens),
        "--top-k", str(top_k),
        "--train-frac", str(train_frac),
        "--load-in-4bit",
    ]
    subprocess.run(command, check=True, env=env, cwd="/root")
    hf_cache.commit()
    shard_cache.commit()
    return {
        "card": json.loads(card.read_text(encoding="utf-8")),
        "shard": str(out),
        "card_path": str(card),
        "size": out.stat().st_size if out.exists() else 0,
    }


@app.local_entrypoint()
def main(
    corpus: str = "data/alice_excerpt.txt",
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    revision: str = "main",
    max_tokens: int = 4096,
    top_k: int = 16,
    train_frac: float = 0.85,
    out_name: str = "qwen_teacher.ntr",
):
    text = Path(corpus).read_text(encoding="utf-8")
    result = teacher.remote(
        text, model, revision, max_tokens, top_k, train_frac, out_name
    )
    print(json.dumps(result["card"], indent=2, sort_keys=True))
    print(f"shard: {result['shard']} ({result['size']} bytes)")
    print("fetch: modal volume get nanlm-teacher-shards "
          f"{out_name} ./build/{out_name}")
    print("fetch card: modal volume get nanlm-teacher-shards "
          f"{out_name}.card.json ./build/{out_name}.card.json")
