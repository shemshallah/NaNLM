#!/usr/bin/env python3
"""Emit additive teacher shards from a running llama.cpp server.

The teacher is a Qwen GGUF served by `llama-server`. We query the *raw text
prefix* at token boundaries, ask for the next-token distribution, and map the
top tokens to their first byte. The resulting records are:

    (8-byte byte context, next byte, weight)

That is exactly the nanlm_teacher format consumed by the 8-slot student.

Why a server instead of a model-specific tokenizer:
  * the server owns tokenization, so teacher alignment is exact;
  * the same runner works for Qwen, Llama, or any GGUF chat model;
  * it can run on a GitHub Actions runner, a GPU farm, or a volunteer box.

It is intentionally a *teacher* tool. It does not train or touch the VM.
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


def get_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url)
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


def completion_probs(server: str, prompt: str, top_k: int,
                     timeout: float) -> list[tuple[str, float]]:
    payload = {
        "prompt": prompt,
        "n_predict": 1,
        "n_probs": top_k,
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": True,
        "stream": False,
    }
    result = http_json(server.rstrip("/") + "/completion", payload, timeout)
    candidates = result.get("completion_probabilities")
    if not candidates:
        raise RuntimeError(f"completion has no probabilities: {result}")
    probs = candidates[0].get("probs") or candidates[0].get("top_probs")
    if not probs:
        raise RuntimeError(f"completion probability record has no probs: {candidates[0]}")
    out: list[tuple[str, float]] = []
    for item in probs:
        text = item.get("tok_str", item.get("token", ""))
        prob = item.get("prob", item.get("probability"))
        if text is None or prob is None:
            continue
        out.append((str(text), float(prob)))
    return out


def write_ntr(path: Path, records: list[tuple[int, int, int]]) -> None:
    records.sort(key=lambda r: (r[0], r[1]))
    with path.open("wb") as f:
        f.write(NTR_HEADER.pack(NTR_MAGIC, 1, len(records), 0))
        for context, next_byte, weight in records:
            f.write(NTR_RECORD.pack(context, next_byte, weight))


def context_key(prefix: bytes, width: int) -> int:
    """Most recent width bytes, oldest byte in the most significant position."""
    if len(prefix) < width:
        return -1
    key = 0
    for byte in prefix[-width:]:
        key = (key << 8) | byte
    return key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--context-bytes", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--start-token", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--log-every", type=int, default=32)
    args = parser.parse_args()

    if args.context_bytes != 8:
        parser.error("nanlm_teacher.v1 records require exactly 8 context bytes")
    if args.top_k < 1 or args.top_k > 256:
        parser.error("--top-k must be in 1..256")

    text = Path(args.corpus).read_text(encoding="utf-8")
    tokens = tokenize(args.server, text, args.timeout)
    if len(tokens) <= args.start_token:
        print("qwen-teacher: corpus has too few tokens", file=sys.stderr)
        return 1

    records: list[tuple[int, int, int]] = []
    samples = 0
    # Cache detokenized prefixes; the server is the source of truth for BPE
    # boundaries. This is O(n) HTTP calls but the corpus is a shard, not a
    # whole repository.
    prefix_cache: dict[int, str] = {}
    last_token = min(len(tokens), args.start_token + args.max_samples)
    for i in range(args.start_token, last_token):
        try:
            prefix = prefix_cache.get(i)
            if prefix is None:
                prefix = detokenize(args.server, tokens[:i], args.timeout)
                prefix_cache[i] = prefix
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            print(f"qwen-teacher: token {i}: detokenize failed: {exc}",
                  file=sys.stderr)
            return 1
        prefix_bytes = prefix.encode("utf-8")
        key = context_key(prefix_bytes, args.context_bytes)
        if key < 0:
            continue
        try:
            top = completion_probs(args.server, prefix, args.top_k, args.timeout)
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            print(f"qwen-teacher: token {i}: completion failed: {exc}",
                  file=sys.stderr)
            return 1

        # Marginalize token probabilities to the first byte. Multi-byte tokens
        # are intentionally treated as their first byte: the student predicts
        # bytes, and the next token boundary is not known to it.
        by_byte: dict[int, float] = defaultdict(float)
        for token_text, prob in top:
            raw = token_text.encode("utf-8", errors="replace")
            if not raw:
                continue
            by_byte[raw[0]] += max(0.0, prob)
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
