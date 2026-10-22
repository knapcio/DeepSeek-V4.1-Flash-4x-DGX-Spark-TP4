#!/usr/bin/env bash
# Fetch the SGLang python tree for Dockerfile.canary as a GitHub tarball.
# The serving image's git cannot fetch unauthenticated, so the tree is staged on the
# host and COPYed into the image. Pinned to the commit measured in docs/window-20260916.md.
set -euo pipefail
REF="${SGLANG_CANARY_REF:-f80c91a4b97e52e17fbcff088aa6b47b6270f6b9}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$ROOT/runtime/sglang-canary"
PATCH_SAFETY="$ROOT/patches/sglang-41303-bounded-replay-window.patch"
PATCH_LAYER20="$ROOT/patches/sglang-42901-layer20-tail.patch"
PATCH_INTERVAL="$ROOT/patches/sglang-44001-prefill-interval-min-bs.patch"
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
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_SAFETY"
patch --silent --fuzz=0 -p1 < "$PATCH_SAFETY"
python3 "$ROOT/scripts/verify_sglang_bounded_replay.py" "$DST/python"
echo "applying SGLang #42901 layer-20 bounded-replay backport ..."
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_LAYER20"
patch --silent --fuzz=0 -p1 < "$PATCH_LAYER20"
python3 "$ROOT/scripts/verify_sglang_layer20_tail.py" "$DST/python"
echo "applying SGLang #44001 prefill-interval batch-size gate ..."
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_INTERVAL"
patch --silent --fuzz=0 -p1 < "$PATCH_INTERVAL"
python3 "$ROOT/scripts/verify_sglang_prefill_interval_min_bs.py" "$DST/python"
echo "$REF" > REF
printf '%s\n' \
  "41303 ed8c3139f1f643544b68102cf8c44504e5e55e3d" \
  "42901 05ce0e980f00aa80864190d6f1cd5d9b685802f1" \
  "44001 9569a3656155a34dc763826b1608f247d3f66832" > PATCHES
du -sh python
echo "staged $DST/python (ref $REF + SGLang #41303/#42901/#44001 backports); now: docker build -f Dockerfile.canary -t dsv41-4x-spark:canary ."
