#!/usr/bin/env bash
# Fetch the SGLang python tree for Dockerfile.canary as a GitHub tarball.
# The serving image's git cannot fetch unauthenticated, so the tree is staged on the
# host and COPYed into the image. Pinned to the commit measured in docs/window-20260916.md.
set -euo pipefail
REF="${SGLANG_CANARY_REF:-f80c91a4b97e52e17fbcff088aa6b47b6270f6b9}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$ROOT/runtime/sglang-canary"
PATCH="$ROOT/patches/sglang-41303-bounded-replay-window.patch"
mkdir -p "$DST"
cd "$DST"
rm -rf python src.tgz
echo "fetching sgl-project/sglang @ $REF ..."
curl -fsSL -m 600 "https://github.com/sgl-project/sglang/archive/${REF}.tar.gz" -o src.tgz
# GitHub names the archive root after the requested full commit. Using the exact
# member path works with both GNU tar (Linux/Spark) and bsdtar (macOS admin host).
tar -xzf src.tgz --strip-components=1 "sglang-${REF}/python"
rm -f src.tgz
echo "applying SGLang #41303 bounded-replay safety backport ..."
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH"
patch --silent --fuzz=0 -p1 < "$PATCH"
python3 "$ROOT/scripts/verify_sglang_bounded_replay.py" "$DST/python"
echo "$REF" > REF
echo "41303 ed8c3139f1f643544b68102cf8c44504e5e55e3d" > PATCHES
du -sh python
echo "staged $DST/python (ref $REF + SGLang #41303 backport); now: docker build -f Dockerfile.canary -t dsv41-4x-spark:canary ."
