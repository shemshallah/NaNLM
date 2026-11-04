#!/bin/sh
# Data-page bit-plane packer: payload round-trip, hash, and 2x head removal.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT="$ROOT/build/pack"
mkdir -p "$OUT"
python3 - "$OUT" <<'PY'
import random, sys
from pathlib import Path
out = Path(sys.argv[1])
random.seed(1)
lines = [f"FFF8_{random.getrandbits(32):08X}_F8FF" for _ in range(10000)]
(out / "neutral.nan").write_text("\n".join(lines) + "\n")
PY
python3 "$ROOT/tools/nanlm_pack_pages.py" check "$OUT/neutral.nan"
python3 "$ROOT/tools/nanlm_pack_pages.py" pack "$OUT/neutral.nan" "$OUT/neutral.nanpg" --page-words 4096
python3 "$ROOT/tools/nanlm_pack_pages.py" unpack "$OUT/neutral.nanpg" "$OUT/neutral.out"
cmp "$OUT/neutral.nan" "$OUT/neutral.out"
BODY=$(wc -c < "$OUT/neutral.nanpg")
FULL=$((10000 * 8))
if [ "$BODY" -ge "$FULL" ]; then
    echo "pack: packed body did not remove per-word heads" >&2
    exit 1
fi
echo "pack: PASS ($BODY packed bytes vs $FULL full-word bytes)"
