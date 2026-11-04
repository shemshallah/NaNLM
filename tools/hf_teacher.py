#!/usr/bin/env python3
"""Batch HF teacher runner: Qwen (or any causal LM) -> additive NTR shards.

This is designed for free/limited GPU hours. It does one pass over a corpus:

  tokenize -> forward in chunks -> top-k next-token logits -> first-byte
  marginalization -> (8-byte context, next byte, weight) records

It also emits a teacher card on the held-out token boundaries. The student
training remains CPU counting; no gradients are needed for distillation.

Examples:
  python3 tools/hf_teacher.py --model Qwen/Qwen2.5-7B-Instruct \
      --corpus data/alice_excerpt.txt --out build/qwen7b.ntr \
      --card build/qwen7b.card.json --load-in-4bit --max-tokens 4096

On Modal, see modal_teacher.py.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import defaultdict
from pathlib import Path

import sys

NTR_MAGIC = 0x3152544E
NTR_HEADER = struct.Struct("<IIII")
NTR_RECORD = struct.Struct("<QBxH4x")


def context_key(data: bytes, end: int, width: int = 8) -> int:
    if end < width:
        return -1
    key = 0
    for byte in data[end - width:end]:
        key = (key << 8) | byte
    return key


def write_ntr(path: Path, records: dict[tuple[int, int], int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(records.items(), key=lambda kv: kv[0])
    with path.open("wb") as f:
        f.write(NTR_HEADER.pack(NTR_MAGIC, 1, len(items), 0))
        for (ctx, next_byte), weight in items:
            f.write(NTR_RECORD.pack(ctx, next_byte, min(65535, weight)))


def token_pieces(tokenizer, ids: list[int]) -> list[bytes]:
    pieces = []
    for token_id in ids:
        text = tokenizer.decode(
            [token_id], skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        pieces.append(text.encode("utf-8", errors="replace"))
    return pieces


def byte_layout(pieces: list[bytes]) -> tuple[bytes, list[int]]:
    full = bytearray()
    cumulative = []
    for piece in pieces:
        cumulative.append(len(full))
        full.extend(piece)
    cumulative.append(len(full))
    return bytes(full), cumulative


def first_byte(tokenizer, token_id: int,
               cache: dict[int, bytes]) -> int | None:
    piece = cache.get(token_id)
    if piece is None:
        piece = tokenizer.decode(
            [token_id], skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).encode("utf-8", errors="replace")
        cache[token_id] = piece
    return piece[0] if piece else None


def run(args: argparse.Namespace) -> int:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        print(f"hf-teacher: missing dependency: {exc}", file=sys.stderr)
        print("install: pip install torch transformers accelerate bitsandbytes",
              file=sys.stderr)
        return 2

    path = Path(args.corpus)
    text = path.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < 16:
        print("hf-teacher: corpus too short", file=sys.stderr)
        return 1
    if args.max_tokens and len(ids) > args.max_tokens:
        ids = ids[:args.max_tokens]

    pieces = token_pieces(tokenizer, ids)
    # Map token ids to bytes for top-k vocabulary ids, not corpus positions.
    token_byte_cache: dict[int, bytes] = {}
    full_bytes, cumulative = byte_layout(pieces)
    original = text.encode("utf-8")
    if full_bytes != original[:len(full_bytes)]:
        print("hf-teacher: warning: tokenizer round-trip differs from raw bytes",
              file=sys.stderr)

    kwargs = {
        "revision": args.revision,
        "trust_remote_code": True,
        "device_map": args.device_map,
    }
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()

    records: dict[tuple[int, int], int] = defaultdict(int)
    eval_positions = 0
    eval_correct = 0
    eval_bits = 0.0
    train_tokens = int(len(ids) * args.train_frac)
    overlap = max(8, args.overlap)
    chunk = max(32, args.chunk)
    written = 0

    with torch.inference_mode():
        for start in range(0, len(ids), chunk):
            lo = max(0, start - overlap)
            hi = min(len(ids), start + chunk)
            if hi - lo < 2:
                continue
            input_ids = torch.tensor([ids[lo:hi]], device=args.device)
            logits = model(input_ids=input_ids).logits[0]
            for i in range(max(start, 1), hi - 1):
                local = i - lo
                if local < 0 or local >= logits.shape[0] - 1:
                    continue
                token_id = ids[i + 1]
                actual = pieces[i + 1][0] if pieces[i + 1] else None
                if actual is None:
                    continue
                ctx = context_key(full_bytes, cumulative[i + 1], 8)
                if ctx < 0:
                    continue
                top = torch.topk(logits[local], k=min(args.top_k, logits.shape[-1]))
                top_ids = top.indices.tolist()
                top_probs = torch.softmax(top.values.float(), dim=-1).tolist()
                by_byte: dict[int, float] = defaultdict(float)
                for tid, probability in zip(top_ids, top_probs):
                    b = first_byte(tokenizer, tid, token_byte_cache)
                    if b is not None:
                        by_byte[b] += probability
                mass = sum(by_byte.values())
                if mass <= 0:
                    continue
                if i < train_tokens:
                    for b, probability in by_byte.items():
                        w = int(round(65535.0 * probability / mass))
                        if w > 0:
                            records[(ctx, b)] += w
                            written += 1
                else:
                    eval_positions += 1
                    best = max(by_byte.items(), key=lambda kv: kv[1])[0]
                    if best == actual:
                        eval_correct += 1
                    p_actual = by_byte.get(actual, 0.0) / mass
                    kept = min(0.9, mass)
                    p_actual = p_actual * kept + (1.0 - kept) / 256.0
                    eval_bits += -math.log2(max(1e-12, p_actual))
            if args.log_every and (start // chunk) % args.log_every == 0:
                print(f"hf-teacher: tokens={min(start + chunk, len(ids))} "
                      f"records={len(records)}", file=sys.stderr)

    if not records:
        print("hf-teacher: no teacher records", file=sys.stderr)
        return 1
    out = Path(args.out)
    write_ntr(out, records)
    card = {
        "model": args.model,
        "revision": args.revision,
        "corpus": str(path),
        "tokens": len(ids),
        "train_tokens": train_tokens,
        "eval_positions": eval_positions,
        "teacher_top1": 100.0 * eval_correct / eval_positions if eval_positions else 0.0,
        "teacher_bpc": eval_bits / eval_positions if eval_positions else 0.0,
        "contexts": len({ctx for ctx, _ in records}),
        "records": len(records),
        "top_k": args.top_k,
        "load_in_4bit": args.load_in_4bit,
    }
    if args.card:
        card_path = Path(args.card)
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(f"hf-teacher: tokens={len(ids)} records={written} contexts={card['contexts']} "
          f"eval_n={eval_positions} top1={card['teacher_top1']:.2f}% "
          f"bpc={card['teacher_bpc']:.4f} out={out}")
    if args.card:
        print(f"hf-teacher: card={args.card}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--card")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--train-frac", type=float, default=0.85)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--log-every", type=int, default=4)
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > 256:
        parser.error("--top-k must be 1..256")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
