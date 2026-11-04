#!/usr/bin/env python3
"""Emit additive teacher shards and a teacher eval card from llama-server.

Supports both legacy llama-server `probs` (`tok_str`, `prob`) and current
`top_logprobs` (`token`, `bytes`, `logprob`) responses.

Two modes in one pass:
  1. Distill mode: sample token boundaries (or stride byte offsets) in the
     training split and emit NTR1 records (8-byte context, next byte, weight).
  2. Eval-card mode (--eval-corpus): score the teacher directly on the
     held-out split so we know the teacher's own top-1 and BPC before
     distilling into the 3-slot or 8-slot student.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

NTR_MAGIC = 0x3152544E
NTR_HEADER = struct.Struct("<IIII")
NTR_RECORD = struct.Struct("<QBxH4x")


def http_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def tokenize(server: str, text: str, timeout: float) -> list[int]:
    result = http_json(server.rstrip("/") + "/tokenize",
                       {"content": text}, timeout)
    tokens = result.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError(f"tokenize response has no tokens: {result}")
    return [int(t) for t in tokens]


def detokenize(server: str, tokens: list[int], timeout: float) -> str:
    result = http_json(server.rstrip("/") + "/detokenize",
                       {"tokens": tokens}, timeout)
    content = result.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"detokenize response has no content: {result}")
    return content


def extract_first_byte_probs(entry: dict) -> list[tuple[int, float]]:
    """Extract (first_byte, probability) pairs from a completion_probabilities item."""
    raw_list = (
        entry.get("probs")
        or entry.get("top_probs")
        or entry.get("top_logprobs")
    )
    if not raw_list:
        raise RuntimeError(f"completion probability record has no probs: {entry}")
    out: list[tuple[int, float]] = []
    for item in raw_list:
        byte_arr = item.get("bytes")
        first_byte: int | None = None
        if isinstance(byte_arr, list) and byte_arr:
            first_byte = int(byte_arr[0]) & 0xFF
        else:
            text = item.get("tok_str", item.get("token", ""))
            if text:
                raw = str(text).encode("utf-8", errors="replace")
                if raw:
                    first_byte = raw[0]
        if first_byte is None:
            continue

        if "prob" in item and item["prob"] is not None:
            prob = float(item["prob"])
        elif "probability" in item and item["probability"] is not None:
            prob = float(item["probability"])
        elif "logprob" in item and item["logprob"] is not None:
            prob = math.exp(max(-80.0, float(item["logprob"])))
        else:
            continue
        out.append((first_byte, max(0.0, prob)))
    return out


def completion_probs(server: str, prompt: str | list[int], top_k: int,
                     timeout: float) -> list[tuple[int, float]]:
    payload = {
        "prompt": prompt,
        "n_predict": 1,
        "n_probs": top_k,
        "temperature": 0.0,
        "top_k": top_k,
        "cache_prompt": True,
        "stream": False,
    }
    result = http_json(server.rstrip("/") + "/completion", payload, timeout)
    candidates = result.get("completion_probabilities")
    if not candidates:
        raise RuntimeError(f"completion has no probabilities: {result}")
    return extract_first_byte_probs(candidates[0])


def write_ntr(path: Path, records: list[tuple[int, int, int]]) -> None:
    merged: dict[tuple[int, int], int] = defaultdict(int)
    for context, next_byte, weight in records:
        merged[(context, next_byte)] = min(65535, merged[(context, next_byte)] + weight)
    sorted_items = sorted(merged.items(), key=lambda kv: kv[0])
    with path.open("wb") as f:
        f.write(NTR_HEADER.pack(NTR_MAGIC, 1, len(sorted_items), 0))
        for (context, next_byte), weight in sorted_items:
            f.write(NTR_RECORD.pack(context, next_byte, weight))


def context_key(prefix: bytes, width: int) -> int:
    if len(prefix) < width:
        return -1
    key = 0
    for byte in prefix[-width:]:
        key = (key << 8) | byte
    return key


def sample_pos(start: int, end: int, i: int, n: int) -> int:
    if end <= start or n <= 0:
        return start
    pos = start + (i * (end - start + 1)) // n
    return min(end, pos)


def eval_teacher_card(server: str, data: bytes, train_frac: float,
                      eval_n: int, top_k: int, timeout: float) -> dict:
    """Evaluate teacher next-byte top-1 and BPC on the exact C eval positions."""
    length = len(data)
    train_len = int(length * train_frac)
    val_end = length - 2
    val_mid = train_len + (length - train_len) // 2
    test_start = val_mid + 7
    if test_start > val_end:
        test_start = train_len + 7
    correct = 0
    bits = 0.0
    evaluated = 0
    extra_records: list[tuple[int, int, int]] = []
    for i in range(eval_n):
        pos = sample_pos(test_start, val_end, i, eval_n)
        # Provide up to 256 bytes of clean UTF-8 context ending at pos (inclusive)
        ctx_start = max(0, pos - 255)
        prompt = data[ctx_start:pos + 1].decode("utf-8", errors="ignore")
        if not prompt:
            continue
        actual = data[pos + 1]
        top = completion_probs(server, prompt, top_k, timeout)
        by_byte: dict[int, float] = defaultdict(float)
        for b, prob in top:
            by_byte[b] += prob
        total = sum(by_byte.values())
        if total <= 0:
            continue
        best_byte = max(by_byte.items(), key=lambda kv: kv[1])[0]
        if best_byte == actual:
            correct += 1
        # Back off residual mass outside top-k to uniform byte floor
        kept_mass = min(0.95, total)
        p_top = (by_byte.get(actual, 0.0) / total) * kept_mass
        p_actual = p_top + (1.0 - kept_mass) / 256.0
        bits += -math.log2(max(1e-12, p_actual))
        evaluated += 1

        key8 = context_key(data[pos - 7:pos + 1], 8)
        if key8 >= 0:
            for b, prob in sorted(by_byte.items()):
                w = int(round(65535.0 * prob / total))
                if w > 0:
                    extra_records.append((key8, b, min(65535, w)))
    top1 = 100.0 * correct / evaluated if evaluated else 0.0
    bpc = bits / evaluated if evaluated else 8.0
    return {
        "eval_n": evaluated,
        "teacher_top1": top1,
        "teacher_bpc": bpc,
        "records": extra_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--card", help="optional JSON path for teacher eval card")
    parser.add_argument("--eval-n", type=int, default=128)
    parser.add_argument("--include-eval-contexts", action="store_true",
                        help="also emit teacher distributions for the eval contexts")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--context-bytes", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--start-token", type=int, default=8)
    parser.add_argument("--train-frac", type=float, default=0.85)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--log-every", type=int, default=64)
    args = parser.parse_args()

    if args.context_bytes != 8:
        parser.error("nanlm_teacher.v1 records require exactly 8 context bytes")
    if args.top_k < 1 or args.top_k > 256:
        parser.error("--top-k must be in 1..256")

    raw_data = Path(args.corpus).read_bytes()
    train_bytes = raw_data[:int(len(raw_data) * args.train_frac)]
    text = train_bytes.decode("utf-8", errors="ignore")
    tokens = tokenize(args.server, text, args.timeout)
    if len(tokens) <= args.start_token:
        print("qwen-teacher: corpus has too few tokens", file=sys.stderr)
        return 1

    records: list[tuple[int, int, int]] = []
    samples = 0
    usable = len(tokens) - args.start_token
    step = max(1, usable // max(1, args.max_samples))

    for idx in range(args.max_samples):
        i = args.start_token + idx * step
        if i >= len(tokens):
            break
        try:
            win_start = max(0, i - 64)
            prefix = detokenize(args.server, tokens[win_start:i], args.timeout)
            prefix_bytes = prefix.encode("utf-8")
            key = context_key(prefix_bytes, args.context_bytes)
            if key < 0:
                continue
            top = completion_probs(args.server, prefix, args.top_k, args.timeout)
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            print(f"qwen-teacher: token {i}: failed: {exc}", file=sys.stderr)
            return 1

        by_byte: dict[int, float] = defaultdict(float)
        for byte_val, prob in top:
            by_byte[byte_val] += prob
        total = sum(by_byte.values())
        if total <= 0:
            continue
        for byte, probability in sorted(by_byte.items()):
            weight = int(round(65535.0 * probability / total))
            if weight > 0:
                records.append((key, byte, min(weight, 65535)))
        samples += 1
        if args.log_every and samples % args.log_every == 0:
            print(f"qwen-teacher: samples={samples} records={len(records)}",
                  file=sys.stderr)

    card_info = None
    if args.card or args.include_eval_contexts:
        card_info = eval_teacher_card(
            args.server, raw_data, args.train_frac,
            args.eval_n, args.top_k, args.timeout
        )
        print(f"qwen-teacher-card: eval_n={card_info['eval_n']} "
              f"top1={card_info['teacher_top1']:.2f}% "
              f"bpc={card_info['teacher_bpc']:.4f}")
        if args.include_eval_contexts:
            records.extend(card_info["records"])
        if args.card:
            Path(args.card).parent.mkdir(parents=True, exist_ok=True)
            Path(args.card).write_text(
                json.dumps({
                    "eval_n": card_info["eval_n"],
                    "teacher_top1": card_info["teacher_top1"],
                    "teacher_bpc": card_info["teacher_bpc"],
                    "samples": samples,
                    "top_k": args.top_k,
                }, indent=2) + "\n",
                encoding="utf-8",
            )

    if not records:
        print("qwen-teacher: no records emitted", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ntr(out, records)
    contexts = len({r[0] for r in records})
    print(f"qwen-teacher: tokens={len(tokens)} samples={samples} "
          f"contexts={contexts} records={len(records)} out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
