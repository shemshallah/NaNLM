#!/usr/bin/env python3
"""A small, deterministic runner-ring control plane for NaNLM shards.

The production ring can be GitHub Actions, a GPU farm, or volunteer machines.
This tool keeps the scheduling contract simple enough to test on one machine:

  * a manifest chunk is the unit of work;
  * chunks are grouped into jobs;
  * jobs are assigned to a ring position by consistent hash;
  * a lease is (job, runner, expiry, attempt);
  * on failure, the successor in the ring inherits the lease;
  * workers only produce content-addressed shards.

It intentionally does not store model bytes. A ring plan is a tiny JSON control
artifact; shards live in reporam/content-addressed storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA = "nanlm-ring-plan-v1"


def hash_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def load_manifest(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") != "chunk":
                continue
            records.append(record)
    records.sort(key=lambda r: (r["repo"], r["path"], r["chunk_index"]))
    return records


def make_jobs(records: list[dict], chunks_per_job: int, bytes_per_job: int) -> list[dict]:
    jobs = []
    current: list[dict] = []
    current_bytes = 0

    def flush() -> None:
        nonlocal current, current_bytes
        if not current:
            return
        digest = hashlib.sha256()
        for record in current:
            digest.update(record["chunk_sha256"].encode("ascii"))
            digest.update(b"\n")
        jobs.append({
            "id": f"j-{len(jobs):05d}",
            "chunks": [r["chunk_sha256"] for r in current],
            "paths": [r["text"] for r in current],
            "languages": sorted({r["language"] for r in current}),
            "bytes": current_bytes,
            "chunk_count": len(current),
            "work_sha256": digest.hexdigest(),
        })
        current = []
        current_bytes = 0

    for record in records:
        if current and (len(current) >= chunks_per_job or
                        current_bytes + record["bytes"] > bytes_per_job):
            flush()
        current.append(record)
        current_bytes += record["bytes"]
    flush()
    return jobs


def assign(jobs: list[dict], runners: int) -> list[dict]:
    plan_jobs = []
    for job in jobs:
        pos = hash_int(job["id"]) % runners
        plan_jobs.append({
            **job,
            "runner": pos,
            "successor": (pos + 1) % runners,
            "state": "leased",
            "attempt": 1,
        })
    return plan_jobs


def command_plan(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    records = load_manifest(manifest_path)
    if not records:
        print("ring: manifest has no chunks", file=sys.stderr)
        return 1
    if args.runners < 1:
        print("ring: need at least one runner", file=sys.stderr)
        return 1
    jobs = make_jobs(records, args.chunks_per_job, args.bytes_per_job)
    plan = {
        "schema": SCHEMA,
        "generation": args.generation,
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "ring": [f"runner-{i}" for i in range(args.runners)],
        "lease_ttl_seconds": args.lease_ttl,
        "chunks": len(records),
        "bytes": sum(r["bytes"] for r in records),
        "jobs": assign(jobs, args.runners),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ring: runners={args.runners} jobs={len(jobs)} "
          f"chunks={plan['chunks']} bytes={plan['bytes']} plan={out}")
    return 0


def command_simulate(args: argparse.Namespace) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    runners = len(plan["ring"])
    failed = {int(x) for x in args.fail.split(",") if x.strip()} if args.fail else set()
    if any(r < 0 or r >= runners for r in failed):
        print("ring: --fail runner out of range", file=sys.stderr)
        return 1
    if len(failed) >= runners:
        print("ring: every runner failed; cannot complete", file=sys.stderr)
        return 1

    completed = []
    for job in plan["jobs"]:
        runner = job["runner"]
        attempts = 1
        while runner in failed:
            runner = (runner + 1) % runners
            attempts += 1
            if attempts > runners:
                print("ring: no live successor", file=sys.stderr)
                return 1
        completed.append({
            "id": job["id"],
            "runner": runner,
            "attempts": attempts,
            "bytes": job["bytes"],
            "state": "done",
        })
    if len(completed) != len(plan["jobs"]):
        print("ring: incomplete simulation", file=sys.stderr)
        return 1
    result = {
        "schema": "nanlm-ring-run-v1",
        "generation": plan["generation"],
        "plan": args.plan,
        "failed": sorted(failed),
        "done": completed,
        "coverage": len(completed),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ring: simulated done={len(completed)} failed={sorted(failed)}")
    return 0


def command_materialize(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest_dir = Path(plan["manifest"]).resolve().parent
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for job in plan["jobs"]:
        target = out_dir / f"{job['id']}.txt"
        with target.open("w", encoding="utf-8", newline="\n") as out:
            for relative in job["paths"]:
                piece = (manifest_dir / relative).read_text(encoding="utf-8")
                out.write(piece)
                if not piece.endswith("\n"):
                    out.write("\n")
        meta = {
            "id": job["id"],
            "runner": job["runner"],
            "chunks": job["chunks"],
            "bytes": target.stat().st_size,
            "work_sha256": job["work_sha256"],
        }
        (out_dir / f"{job['id']}.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"ring: materialized {len(plan['jobs'])} jobs in {out_dir}")
    return 0


def command_verify(args: argparse.Namespace) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    shard_dir = Path(args.shards)
    missing = []
    for job in plan["jobs"]:
        path = shard_dir / f"{job['id']}.shard"
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(job["id"])
    if missing:
        print(f"ring: missing shards: {', '.join(missing[:8])}", file=sys.stderr)
        return 1
    print(f"ring: verified {len(plan['jobs'])} shard files")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--manifest", required=True)
    p.add_argument("--runners", type=int, default=8)
    p.add_argument("--chunks-per-job", type=int, default=8)
    p.add_argument("--bytes-per-job", type=int, default=1024 * 1024)
    p.add_argument("--lease-ttl", type=int, default=900)
    p.add_argument("--generation", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=command_plan)

    s = sub.add_parser("simulate")
    s.add_argument("--plan", required=True)
    s.add_argument("--fail", default="", help="comma-separated runner indexes")
    s.add_argument("--out", required=True)
    s.set_defaults(func=command_simulate)

    m = sub.add_parser("materialize")
    m.add_argument("--plan", required=True)
    m.add_argument("--out", required=True)
    m.set_defaults(func=command_materialize)

    v = sub.add_parser("verify")
    v.add_argument("--plan", required=True)
    v.add_argument("--shards", required=True)
    v.set_defaults(func=command_verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
