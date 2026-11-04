#!/bin/sh
# Generate a content manifest for the published model artifacts.
# The manifest is committed; the bytes may live in git, a release, or a mirror.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

if command -v sha256sum >/dev/null 2>&1; then
    HASH="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
    HASH="shasum -a 256"
else
    echo "manifest: no sha256 tool" >&2
    exit 1
fi

OUT="$ROOT/models/SHA256SUMS"
: > "$OUT"
for f in \
    models/nanlm0.nan models/nanlm0.nas models/nanlm0.manifest \
    models/nanlm_ctx.nan models/nanlm_ctx.nas models/nanlm_ctx.manifest \
    models/nanlm_lm.nan models/nanlm_lm.nas models/nanlm_lm.manifest \
    models/nanlm_slot.nan models/nanlm_slot.nas models/nanlm_slot.manifest \
    data/alice_excerpt.txt data/alice_full.txt data/PROVENANCE.md \
    data/code_fixture/LICENSE data/code_fixture/README.md \
    models/qwen_0.5b/card.json models/qwen_0.5b/qwen_slot8.manifest \
    models/qwen_0.5b/qwen_hybrid_slot8.manifest \
    models/qwen_0.5b/qwen_hybrid_slot3.manifest \
    modal_teacher.py tools/hf_teacher.py docs/GPU.md docs/AMD_GPU.md docs/MODAL.md \
    notebooks/qwen7b_teacher.ipynb
do
    if [ -f "$f" ]; then
        $HASH "$f" >> "$OUT"
    fi
done
echo "manifest: wrote $OUT"
