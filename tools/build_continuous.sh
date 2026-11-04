#!/bin/sh
# Continuous-training plumbing gate:
#
#   code fixture -> license/secret/filter -> chunks -> manifest
#                -> ring plan -> materialized jobs
#                -> additive shards -> merge (A+B == B+A)
#                -> typed slot memory -> .nas -> nanlm_vm -> exact diff
#
# This is a plumbing and provenance gate. It is not a quality claim: the
# fixture is tiny and the model is still a 3-byte memory.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT="$ROOT/build/continuous"
rm -rf "$OUT"
mkdir -p "$OUT"

make -C "$ROOT" \
    build/nanlm_shard build/nanlm_shard_slot build/nanlm_asm build/nanlm_vm \
    >/dev/null

# 1. Ingest a pinned local repository. The fixture contains deliberate
#    minified, generated, and secret-bearing files that must be rejected.
python3 "$ROOT/tools/nanlm_ingest.py" \
    --repo "$ROOT/data/code_fixture" \
    --out "$OUT/ingest" \
    --concat "$OUT/corpus.txt" \
    --include-docs \
    > "$OUT/ingest.log"
cat "$OUT/ingest.log"

python3 - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
report = json.loads((out / "ingest" / "report.json").read_text())
assert report["chunks"] > 0, report
assert report["files_seen"] > report["files_kept"]
for reason in ("minified", "generated", "secret"):
    assert report["skipped"].get(reason, 0) >= 1, (reason, report["skipped"])
assert "MIT" in report["licenses"], report["licenses"]
manifest = [json.loads(line) for line in
            (out / "ingest" / "manifest.jsonl").read_text().splitlines()]
assert all(r["type"] == "chunk" for r in manifest)
assert all(r["license"] == "MIT" for r in manifest)
print("ingest: license, secret, generated, and minified gates PASS")
PY

# 2. Build two count shards over the training range, then prove the merge is
#    commutative byte-for-byte.
SIZE=$(wc -c < "$OUT/corpus.txt")
TRAIN_END=$((SIZE * 85 / 100))
MID=$((TRAIN_END / 2))
"$ROOT/build/nanlm_shard" build "$OUT/corpus.txt" "$OUT/a.shard" 0 "$MID" >/dev/null
"$ROOT/build/nanlm_shard" build "$OUT/corpus.txt" "$OUT/b.shard" "$MID" "$TRAIN_END" >/dev/null
"$ROOT/build/nanlm_shard" merge "$OUT/ab.shard" "$OUT/a.shard" "$OUT/b.shard" >/dev/null
"$ROOT/build/nanlm_shard" merge "$OUT/ba.shard" "$OUT/b.shard" "$OUT/a.shard" >/dev/null
cmp "$OUT/ab.shard" "$OUT/ba.shard"
"$ROOT/build/nanlm_shard" stats "$OUT/ab.shard"
echo "continuous: A+B == B+A PASS"

# 3. Compile merged counts to typed slot memory, run on the VM, diff exactly.
"$ROOT/build/nanlm_shard_slot" \
    --shard "$OUT/ab.shard" \
    --corpus "$OUT/corpus.txt" \
    --out "$OUT/model.nas" \
    --expected "$OUT/model.expected" \
    --manifest "$OUT/model.manifest" \
    > "$OUT/model.log"
cat "$OUT/model.log"
"$ROOT/build/nanlm_asm" "$OUT/model.nas" "$OUT/model.nan"
"$ROOT/build/nanlm_vm" --img "$OUT/model.nan" --steps 300000000 \
    > "$OUT/model.out" 2> "$OUT/model.err" || true
grep -q "rc=3, .*trap=''" "$OUT/model.err"
diff -u "$OUT/model.expected" "$OUT/model.out"
echo "continuous: shard -> typed slot -> VM PASS ($(wc -l < "$OUT/model.out") queries)"

# 4. Ring control plane: plan, fail a runner, fail over to its successor,
#    materialize jobs, and build one real shard per surviving job.
python3 "$ROOT/tools/nanlm_ring.py" plan \
    --manifest "$OUT/ingest/manifest.jsonl" \
    --runners 4 --chunks-per-job 2 \
    --out "$OUT/ring/plan.json" > "$OUT/ring.log"
cat "$OUT/ring.log"
python3 "$ROOT/tools/nanlm_ring.py" simulate \
    --plan "$OUT/ring/plan.json" --fail 1 \
    --out "$OUT/ring/run.json" >> "$OUT/ring.log"
python3 "$ROOT/tools/nanlm_ring.py" materialize \
    --plan "$OUT/ring/plan.json" --out "$OUT/ring/jobs" >> "$OUT/ring.log"
cat "$OUT/ring.log"

mkdir -p "$OUT/ring/shards"
for job in "$OUT/ring/jobs"/j-*.txt; do
    name=$(basename "$job" .txt)
    "$ROOT/build/nanlm_shard" build "$job" "$OUT/ring/shards/$name.shard" >/dev/null
done
python3 "$ROOT/tools/nanlm_ring.py" verify \
    --plan "$OUT/ring/plan.json" --shards "$OUT/ring/shards"
"$ROOT/build/nanlm_shard" merge "$OUT/ring/merged.shard" \
    "$OUT/ring/shards"/j-*.shard >/dev/null
"$ROOT/build/nanlm_shard" stats "$OUT/ring/merged.shard"
echo "continuous: runner ring plan/failover/materialize/shard PASS"
