#!/usr/bin/env python3
"""Find mounted Kaggle datasets/files from a notebook or script.

Kaggle mounts added datasets read-only under /kaggle/input/<dataset-slug>/.
This helper prints matching files, sizes, and the exact --ntt path to use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/kaggle/input")
    p.add_argument("--pattern", default="*.ntt", help="e.g. '*.ntt' or '*.tgz'")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"missing {root}; add a Kaggle Dataset in the notebook sidebar")
        return 1
    found = sorted(root.rglob(args.pattern))
    if not found:
        print(f"no {args.pattern} under {root}")
        print("Kaggle: Add Data -> Your Datasets -> add the token dataset")
        return 1
    rows = []
    for path in found:
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        rows.append({"path": str(path), "bytes": size,
                     "dataset": path.relative_to(root).parts[0]
                     if len(path.relative_to(root).parts) > 1 else ""})
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(f"{row['path']}  {row['bytes']} bytes  "
                  f"dataset={row['dataset']}")
        if args.pattern == "*.ntt":
            print("\nuse:")
            print(f"--ntt {rows[0]['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
