#!/usr/bin/env python3
"""Exact token-level teacher shard from Qwen (or any causal LM).

This avoids the byte-level first-byte marginalization that made the 7B teacher
useless to the 3/8-slot byte student.

Format NTT1 (little-endian):
    header: magic=0x3154544E, version=1, nrec, context_len
    record: uint64 packed context, uint32 next token id, uint32 weight

The context packs context_len token IDs at 18 bits each (Qwen vocabulary is
<2^18), oldest token in the high bits. For 3 tokens this is a 12-24 byte
context, exactly the teacher window the token student will consume.

Run:
    python3 tools/hf_token_teacher.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --corpus data/alice_excerpt.txt \
      --out build/qwen7b_tokens.ntt \
      --card build/qwen7b_tokens.card.json \
      --max-tokens 2048 --top-k 16 --load-in-4bit \
      --device cuda --chunk 64
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

NTT_MAGIC = 0x3154544E
NTT_HEADER = struct.Struct("<IIII")
NTT_RECORD = struct.Struct("<QII")
TOKEN_BITS = 18
TOKEN_MAX = 1 << TOKEN_BITS


def pack_context(ids: list[int], end: int, length: int) -> int | None:
    if end < length:
        return None
    ctx = 0
    for token_id in ids[end - length:end]:
        if token_id < 0 or token_id >= TOKEN_MAX:
            raise ValueError(f"token id {token_id} exceeds 18-bit vocabulary")
        ctx = (ctx << TOKEN_BITS) | token_id
    return ctx


def build_token_ngram(ids: list[int], train_tokens: int, length: int):
    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    unigram = [0] * TOKEN_MAX
    for i in range(length - 1, train_tokens - 1):
        ctx = pack_context(ids, i + 1, length)
        if ctx is None or i + 1 >= train_tokens:
            continue
        next_id = ids[i + 1]
        counts[ctx][next_id] += 1
        unigram[next_id] += 1
    return counts, unigram


def write_ntt(path: Path, records: dict[tuple[int, int], int],
              context_len: int) -> None:
    items = sorted(records.items(), key=lambda kv: kv[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(NTT_HEADER.pack(NTT_MAGIC, 1, len(items), context_len))
        for (ctx, next_id), weight in items:
            f.write(NTT_RECORD.pack(ctx, next_id, min(0xFFFFFFFF, weight)))


def run(args: argparse.Namespace) -> int:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        print(f"hf-token-teacher: missing dependency: {exc}", file=sys.stderr)
        return 2

    text = Path(args.corpus).read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = tokenizer.encode(text, add_special_tokens=False)
    if args.max_tokens:
        ids = ids[:args.max_tokens]
    if len(ids) < 64:
        print("hf-token-teacher: corpus too short", file=sys.stderr)
        return 1
    if max(ids) >= TOKEN_MAX:
        print("hf-token-teacher: vocabulary exceeds 18 bits", file=sys.stderr)
        return 1
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("hf-token-teacher: CUDA/ROCm unavailable", file=sys.stderr)
        return 1

    dtype = getattr(torch, args.dtype)
    kwargs = {
        "revision": args.revision,
        "trust_remote_code": True,
        "device_map": args.device_map,
    }
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()

    context_len = args.context_tokens
    if context_len != 3:
        print("hf-token-teacher: current NTT1 format supports context_tokens=3",
              file=sys.stderr)
        return 1
    vocab_size = int(tokenizer.vocab_size)
    train_tokens = int(len(ids) * args.train_frac)
    records: dict[tuple[int, int], int] = defaultdict(int)
    ngram_counts, ngram_unigram = build_token_ngram(ids, train_tokens, context_len)
    ngram_total = sum(ngram_unigram)
    eval_n = 0
    teacher_correct = 0
    teacher_bits = 0.0
    ngram_correct = 0
    ngram_bits = 0.0
    overlap = max(8, args.overlap)
    chunk = max(32, args.chunk)

    with torch.inference_mode():
        for start in range(0, len(ids), chunk):
            lo = max(0, start - overlap)
            hi = min(len(ids), start + chunk)
            if hi - lo < 2:
                continue
            logits = model(
                input_ids=torch.tensor([ids[lo:hi]], device=args.device)
            ).logits[0]
            for i in range(max(start, context_len - 1), hi - 1):
                local = i - lo
                if local < 0 or local >= logits.shape[0] - 1:
                    continue
                ctx = pack_context(ids, i + 1, context_len)
                if ctx is None:
                    continue
                actual = ids[i + 1]
                top = torch.topk(logits[local], k=min(args.top_k, logits.shape[-1]))
                top_ids = top.indices.tolist()
                top_probs = torch.softmax(top.values.float(), dim=-1).tolist()
                mass = sum(top_probs)
                if mass <= 0:
                    continue
                if i + 1 < train_tokens:
                    for token_id, probability in zip(top_ids, top_probs):
                        w = int(round(65535.0 * probability / mass))
                        if w > 0:
                            records[(ctx, token_id)] += w
                else:
                    eval_n += 1
                    best = max(zip(top_ids, top_probs), key=lambda kv: kv[1])[0]
                    if best == actual:
                        teacher_correct += 1
                    p_actual = 0.0
                    for token_id, probability in zip(top_ids, top_probs):
                        if token_id == actual:
                            p_actual = probability / mass
                            break
                    kept = min(0.9, mass)
                    p_actual = p_actual * kept + (1.0 - kept) / vocab_size
                    teacher_bits += -math.log2(max(1e-12, p_actual))

                    bucket = ngram_counts.get(ctx, {})
                    count = bucket.get(actual, 0)
                    total = sum(bucket.values())
                    p_ng = (count + 0.1) / (total + 0.1 * vocab_size)
                    ngram_bits += -math.log2(max(1e-12, p_ng))
                    if bucket:
                        best_ng = max(bucket.items(), key=lambda kv: kv[1])[0]
                        if best_ng == actual:
                            ngram_correct += 1
            if args.log_every and (start // chunk) % args.log_every == 0:
                print(f"hf-token-teacher: tokens={min(start + chunk, len(ids))} "
                      f"records={len(records)}", file=sys.stderr)

    if not records:
        print("hf-token-teacher: no records", file=sys.stderr)
        return 1
    out = Path(args.out)
    write_ntt(out, records, context_len)
    card = {
        "model": args.model,
        "revision": args.revision,
        "corpus": str(args.corpus),
        "tokens": len(ids),
        "train_tokens": train_tokens,
        "context_tokens": context_len,
        "token_bits": TOKEN_BITS,
        "vocab": vocab_size,
        "top_k": args.top_k,
        "records": len(records),
        "contexts": len({ctx for ctx, _ in records}),
        "eval_n": eval_n,
        "teacher_top1": 100.0 * teacher_correct / eval_n if eval_n else 0.0,
        "teacher_bpc": teacher_bits / eval_n if eval_n else 0.0,
        "ngram_top1": 100.0 * ngram_correct / eval_n if eval_n else 0.0,
        "ngram_bpc": ngram_bits / eval_n if eval_n else 0.0,
        "load_in_4bit": args.load_in_4bit,
    }
    if args.card:
        path = Path(args.card)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"hf-token-teacher: tokens={len(ids)} records={len(records)} "
          f"contexts={card['contexts']} eval_n={eval_n} "
          f"top1={card['teacher_top1']:.2f}% bpc={card['teacher_bpc']:.4f} "
          f"ngram_bpc={card['ngram_bpc']:.4f} out={out}")
    if args.card:
        print(f"hf-token-teacher: card={args.card}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--card")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--context-tokens", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--train-frac", type=float, default=0.85)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="float16",
                        choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--log-every", type=int, default=8)
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > 256:
        parser.error("--top-k must be 1..256")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
