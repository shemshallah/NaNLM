#!/usr/bin/env python3
"""Pack neutral data-word payloads into dense 32-bit-plane pages.

This is a storage prototype, not a VM loader. It demonstrates the NANPG1
data-page layout described in docs/PACKING.md:

    magic       8 bytes  "NANPG1\0\0"
    nwords      u32
    planes      u32 = 32
    body_hash   32 bytes SHA-256
    body        32 planes, ceil(nwords/32) u32 words each

Each payload word is 32 bits; bit j of all words forms plane j. The page has
no per-word NaN head. A neutral data head is restored on unpack. Code/control
pages must keep full NaN words.

Usage:
    nanlm_pack_pages.py pack INPUT.nan OUTPUT.nanpg [--page-words N]
    nanlm_pack_pages.py unpack INPUT.nanpg OUTPUT.nan
    nanlm_pack_pages.py check INPUT.nan
"""

from __future__ import annotations

import argparse
import hashlib
import re
import struct
import sys
from pathlib import Path

MAGIC = b"NANPG1\x00\x00"
HEADER = struct.Struct("<8sII32s")
NEUTRAL_NEAR = "FFF8"
NEUTRAL_FAR_SWAPPED = "F8FF"
LINE_RE = re.compile(r"^([0-9A-Fa-f]{4})_([0-9A-Fa-f]{8})_([0-9A-Fa-f]{4})\s*(;.*)?$")


def parse_nan(path: Path) -> list[tuple[int, int, int]]:
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        match = LINE_RE.match(line)
        if not match:
            raise SystemExit(f"{path}: bad .nan line: {line}")
        near = int(match.group(1), 16)
        payload = int(match.group(2), 16)
        far_swapped = int(match.group(3), 16)
        words.append((near, payload, far_swapped))
    return words


def pack_planes(payloads: list[int]) -> bytes:
    n = len(payloads)
    words_per_plane = (n + 31) // 32
    body = bytearray(32 * words_per_plane * 4)
    for bit in range(32):
        base = bit * words_per_plane * 4
        for i, payload in enumerate(payloads):
            if (payload >> bit) & 1:
                byte_index = base + (i // 32) * 4
                value = int.from_bytes(body[byte_index:byte_index + 4], "little")
                value |= 1 << (i % 32)
                body[byte_index:byte_index + 4] = value.to_bytes(4, "little")
    return bytes(body)


def unpack_planes(body: bytes, n: int) -> list[int]:
    words_per_plane = (n + 31) // 32
    payloads = [0] * n
    for bit in range(32):
        base = bit * words_per_plane * 4
        for i in range(n):
            byte_index = base + (i // 32) * 4
            value = int.from_bytes(body[byte_index:byte_index + 4], "little")
            if (value >> (i % 32)) & 1:
                payloads[i] |= 1 << bit
    return payloads


def pack_file(src: Path, dst: Path, page_words: int) -> None:
    words = parse_nan(src)
    pages = [words[i:i + page_words] for i in range(0, len(words), page_words)]
    with dst.open("wb") as f:
        for page_index, page in enumerate(pages):
            payloads = [w[1] for w in page]
            body = pack_planes(payloads)
            digest = hashlib.sha256(body).digest()
            f.write(HEADER.pack(MAGIC, len(payloads), 32, digest))
            f.write(body)
    print(f"pack: {len(words)} words -> {len(pages)} pages, "
          f"{dst.stat().st_size} bytes (input {src.stat().st_size} bytes text)")


def unpack_file(src: Path, dst: Path) -> None:
    data = src.read_bytes()
    offset = 0
    lines = []
    while offset < len(data):
        if offset + HEADER.size > len(data):
            raise SystemExit("truncated page header")
        magic, nwords, planes, digest = HEADER.unpack_from(data, offset)
        offset += HEADER.size
        if magic != MAGIC or planes != 32:
            raise SystemExit("bad page header")
        body_size = 32 * ((nwords + 31) // 32) * 4
        body = data[offset:offset + body_size]
        if len(body) != body_size:
            raise SystemExit("truncated page body")
        if hashlib.sha256(body).digest() != digest:
            raise SystemExit("page hash mismatch")
        offset += body_size
        payloads = unpack_planes(body, nwords)
        for payload in payloads:
            lines.append(f"{NEUTRAL_NEAR}_{payload:08X}_{NEUTRAL_FAR_SWAPPED}")
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"unpack: {len(lines)} words -> {dst}")


def check(src: Path, page_words: int = 1 << 16) -> None:
    words = parse_nan(src)
    payloads = [w[1] for w in words]
    body = pack_planes(payloads)
    if unpack_planes(body, len(payloads)) != payloads:
        raise SystemExit("payload round-trip failed")
    expected = 32 * ((len(payloads) + 31) // 32) * 4
    if len(body) != expected:
        raise SystemExit("packed body size mismatch")
    print(f"pack-check: PASS ({len(words)} payloads, {len(body)} packed bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--page-words", type=int, default=1 << 20)
    u = sub.add_parser("unpack")
    u.add_argument("src")
    u.add_argument("dst")
    c = sub.add_parser("check")
    c.add_argument("src")
    args = parser.parse_args()
    if args.cmd == "pack":
        pack_file(Path(args.src), Path(args.dst), args.page_words)
    elif args.cmd == "unpack":
        unpack_file(Path(args.src), Path(args.dst))
    else:
        check(Path(args.src))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
