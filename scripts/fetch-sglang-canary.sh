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
PATCH_INLINE_SYSTEM="$ROOT/patches/sglang-43549-dsv41-inline-system.patch"
PARALLEL_ENCODE_REF="fba3ad4b1f4563143f3352beb148c4c28d4be9ed"
PARALLEL_ENCODE_SHA256="722cc41b1096977af3bda6c505e7c629b9b8f7308174cda1c7d2c44f9b54c097"
PARALLEL_ENCODE_TEST_SHA256="fa1adf5c9b51fdf25cb7998e963a7de905c58cb5aa35395cf16d6c75422ecf9e"
PLAIN_TEXT_REF="a0206772e54778229648af1187949b1eb0619234"
PLAIN_TEXT_SHA256="8e675e59b756678309bc2076c3c71b3525cb02030acdec94e83c0e20d285aa62"
PLAIN_TEXT_TEST_SHA256="5702531ee250205f8c1aff916f05c89090cfdec220bf06d467efe1ba8b344020"
REASONING_END_REF="2592a6d211b56f503707760b158a635510baaf9b"
REASONING_END_TEST_SHA256="e1d3bb204617a08abea100c3b5be6de49d8c68483d29de90298c12f10ea204c9"
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
echo "applying SGLang #43549 dsv41 inline-system prefix-cache fix ..."
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_INLINE_SYSTEM"
patch --silent --fuzz=0 -p1 < "$PATCH_INLINE_SYSTEM"
python3 "$ROOT/scripts/verify_sglang_dsv41_inline_system.py" "$DST/python"
echo "installing exact SGLang #45457 plain-text helper and pinned-tree integration ..."
PLAIN_TEXT_PATH="$DST/python/sglang/srt/entrypoints/openai/plain_text_encode.py"
curl -fsSL -m 120 \
  "https://raw.githubusercontent.com/sgl-project/sglang/${PLAIN_TEXT_REF}/python/sglang/srt/entrypoints/openai/plain_text_encode.py" \
  -o "$PLAIN_TEXT_PATH"
printf '%s  %s\n' "$PLAIN_TEXT_SHA256" "$PLAIN_TEXT_PATH" | shasum -a 256 -c -
PLAIN_TEXT_TEST="$DST/upstream-tests/test_plain_text_encode.py"
curl -fsSL -m 120 \
  "https://raw.githubusercontent.com/sgl-project/sglang/${PLAIN_TEXT_REF}/test/registered/unit/entrypoints/openai/test_plain_text_encode.py" \
  -o "$PLAIN_TEXT_TEST"
printf '%s  %s\n' "$PLAIN_TEXT_TEST_SHA256" "$PLAIN_TEXT_TEST" | shasum -a 256 -c -
python3 "$ROOT/runtime/patch_dsv41_plain_text_encode.py" "$DST/python"
python3 "$ROOT/scripts/verify_sglang_dsv41_plain_text.py" "$DST/python"
echo "applying SGLang #45458 token-id-aware reasoning-end integration ..."
REASONING_END_TEST="$DST/upstream-tests/test_reasoning_parser.py"
curl -fsSL -m 120 \
  "https://raw.githubusercontent.com/sgl-project/sglang/${REASONING_END_REF}/test/registered/unit/parser/test_reasoning_parser.py" \
  -o "$REASONING_END_TEST"
printf '%s  %s\n' "$REASONING_END_TEST_SHA256" "$REASONING_END_TEST" | shasum -a 256 -c -
python3 "$ROOT/runtime/patch_dsv41_reasoning_end_token.py" "$DST/python"
python3 "$ROOT/scripts/verify_sglang_dsv41_reasoning_end.py" "$DST/python"
echo "$REF" > REF
printf '%s\n' \
  "41303 ed8c3139f1f643544b68102cf8c44504e5e55e3d" \
  "42901 05ce0e980f00aa80864190d6f1cd5d9b685802f1" \
  "44001 9569a3656155a34dc763826b1608f247d3f66832" \
  "41259 fba3ad4b1f4563143f3352beb148c4c28d4be9ed helper-sha256=$PARALLEL_ENCODE_SHA256 test-sha256=$PARALLEL_ENCODE_TEST_SHA256" \
  "45410 aeeb943fe71626a7cea183b716b8bb236a285f2e" \
  "43549 9e0ce2f3fab3b2a456ac97aea04bc880ae2fe7b1" \
  "45457 $PLAIN_TEXT_REF helper-sha256=$PLAIN_TEXT_SHA256 test-sha256=$PLAIN_TEXT_TEST_SHA256" \
  "45458 $REASONING_END_REF test-sha256=$REASONING_END_TEST_SHA256" > PATCHES
du -sh python
echo "staged $DST/python (ref $REF + SGLang #41303/#42901/#44001/#41259/#45410/#43549/#45457/#45458 backports); now: docker build -f Dockerfile.canary -t dsv41-4x-spark:canary ."
