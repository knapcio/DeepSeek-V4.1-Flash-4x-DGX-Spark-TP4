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
PATCH_PARALLEL_ENCODE="$ROOT/patches/sglang-41259-45410-parallel-native-encode.patch"
PARALLEL_ENCODE_REF="fba3ad4b1f4563143f3352beb148c4c28d4be9ed"
PARALLEL_ENCODE_SHA256="722cc41b1096977af3bda6c505e7c629b9b8f7308174cda1c7d2c44f9b54c097"
PARALLEL_ENCODE_TEST_SHA256="fa1adf5c9b51fdf25cb7998e963a7de905c58cb5aa35395cf16d6c75422ecf9e"
mkdir -p "$DST"
cd "$DST"
rm -rf python upstream-tests src.tgz
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
echo "installing exact SGLang #41259 parallel encoder and #45410 native-DeepSeek integration ..."
PARALLEL_ENCODE_PATH="$DST/python/sglang/srt/entrypoints/openai/parallel_prompt_encode.py"
curl -fsSL -m 120 \
  "https://raw.githubusercontent.com/sgl-project/sglang/${PARALLEL_ENCODE_REF}/python/sglang/srt/entrypoints/openai/parallel_prompt_encode.py" \
  -o "$PARALLEL_ENCODE_PATH"
printf '%s  %s\n' "$PARALLEL_ENCODE_SHA256" "$PARALLEL_ENCODE_PATH" | shasum -a 256 -c -
mkdir -p "$DST/upstream-tests"
PARALLEL_ENCODE_TEST="$DST/upstream-tests/test_parallel_prompt_encode.py"
curl -fsSL -m 120 \
  "https://raw.githubusercontent.com/sgl-project/sglang/${PARALLEL_ENCODE_REF}/test/registered/unit/entrypoints/openai/test_parallel_prompt_encode.py" \
  -o "$PARALLEL_ENCODE_TEST"
printf '%s  %s\n' "$PARALLEL_ENCODE_TEST_SHA256" "$PARALLEL_ENCODE_TEST" | shasum -a 256 -c -
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_PARALLEL_ENCODE"
patch --silent --fuzz=0 -p1 < "$PATCH_PARALLEL_ENCODE"
python3 "$ROOT/scripts/verify_sglang_parallel_prompt_encode.py" "$DST/python"
echo "$REF" > REF
printf '%s\n' \
  "41303 ed8c3139f1f643544b68102cf8c44504e5e55e3d" \
  "42901 05ce0e980f00aa80864190d6f1cd5d9b685802f1" \
  "44001 9569a3656155a34dc763826b1608f247d3f66832" \
  "41259 fba3ad4b1f4563143f3352beb148c4c28d4be9ed helper-sha256=$PARALLEL_ENCODE_SHA256 test-sha256=$PARALLEL_ENCODE_TEST_SHA256" \
  "45410 aeeb943fe71626a7cea183b716b8bb236a285f2e" > PATCHES
du -sh python
echo "staged $DST/python (ref $REF + SGLang #41303/#42901/#44001/#41259/#45410 backports); now: docker build -f Dockerfile.canary -t dsv41-4x-spark:canary ."
