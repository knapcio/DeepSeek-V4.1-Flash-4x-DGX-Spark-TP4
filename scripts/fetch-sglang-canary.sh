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
PATCH_TOOL_IMAGES_THINKING="$ROOT/patches/sglang-44751-dsv41-tool-images-thinking.patch"
PATCH_DSV41_PADDED_QUERY="$ROOT/patches/sglang-45301-dsv41-padded-query-buffer.patch"
PATCH_DSV41_DP_PADDING="$ROOT/patches/sglang-40424-dsv41-dp-padding.patch"
PATCH_MHC_PREWARM_NO_CHUNK="$ROOT/patches/sglang-43448-mhc-prewarm-no-chunk.patch"
PATCH_DSPARK_ACCEPT="$ROOT/patches/sglang-45663-dspark-accept-outputs.patch"
PATCH_DSPARK_SOFTMAX_CLAMP="$ROOT/patches/sglang-45518-dspark-softmax-clamp.patch"
PATCH_DSPARK_RESTAGE="$ROOT/patches/sglang-45450-dspark-param-restage.patch"
PATCH_MXFP8_DECODE_TACTICS="$ROOT/patches/sglang-45630-mxfp8-decode-tactics.patch"
PATCH_DSV41_INDEXER_INPLACE="$ROOT/patches/sglang-45745-dsv41-indexer-inplace.patch"
PATCH_DSV41_BLOCKED_DEQUANT="$ROOT/patches/sglang-45793-dsv41-blocked-kv-dequant.patch"
PATCH_DSV41_ALIGN_WIDTH="$ROOT/patches/sglang-45794-dsv41-align-indexer-width.patch"
PATCH_DSV41_DEFAULT_THINKING="$ROOT/patches/sglang-45762-dsv41-default-thinking-parser.patch"
PATCH_DSV41_THINKING_PARITY="$ROOT/patches/sglang-45804-dsv41-thinking-parity.patch"
PATCH_DSV41_WINDOW_HISTORY_SRC="$ROOT/patches/sglang-45579-window-history-src.patch"
PATCH_DSV41_FUSED_WINDOW_LAYOUT="$ROOT/patches/sglang-45726-fused-window-layout-integration.patch"
PATCH_DSV41_IDLE_SWA_STORE="$ROOT/patches/sglang-45586-45699-idle-swa-store.patch"
PATCH_DSV41_DRAFT_METADATA_GLUE="$ROOT/patches/sglang-45450-draft-metadata-glue.patch"
PATCH_GREEDY_FLAGS_REFRESH="$ROOT/patches/sglang-44923-greedy-flags-refresh.patch"
PATCH_DETERMINISTIC_EXPLICIT="$ROOT/patches/sglang-deterministic-explicit-override-refuse.patch"
PARALLEL_ENCODE_REF="fba3ad4b1f4563143f3352beb148c4c28d4be9ed"
PARALLEL_ENCODE_SHA256="722cc41b1096977af3bda6c505e7c629b9b8f7308174cda1c7d2c44f9b54c097"
PARALLEL_ENCODE_TEST_SHA256="fa1adf5c9b51fdf25cb7998e963a7de905c58cb5aa35395cf16d6c75422ecf9e"
PLAIN_TEXT_REF="a0206772e54778229648af1187949b1eb0619234"
PLAIN_TEXT_SHA256="8e675e59b756678309bc2076c3c71b3525cb02030acdec94e83c0e20d285aa62"
PLAIN_TEXT_TEST_SHA256="5702531ee250205f8c1aff916f05c89090cfdec220bf06d467efe1ba8b344020"
TOOL_IMAGES_THINKING_REF="a038a0dca05cefca1e0edc4ac43e3bfbec2e064c"
TOOL_IMAGES_THINKING_PATCH_SHA256="0851a8f33fd00da18630a5fd4a275b8248e06ca57bb1dd205d7acc1179c36fa2"
DSV41_PADDED_QUERY_REF="117247ea2f9ba77cc8d47005553d3eed18e3a7bc"
DSV41_PADDED_QUERY_PATCH_SHA256="6cbfa3fef31507cb4bf8d11d38d6a24004375c2b1649fb6a794298a34bf58b19"
DSV41_DP_PADDING_REF="720857de766ad8855286da9246418da66724596c"
DSV41_DP_PADDING_PATCH_SHA256="27cfd0e1511733fc853cd0f55679cab9b536c4422f40f2dac902a73b4d7c46e3"
MHC_PREWARM_NO_CHUNK_REF="422280382336eba5daa5233f16a26f904646442d"
MHC_PREWARM_NO_CHUNK_PATCH_SHA256="8c559f5c7d5a0ecb520d592c5445680119d8565607223c0ba56d5e69852505ec"
REASONING_END_REF="2592a6d211b56f503707760b158a635510baaf9b"
REASONING_END_TEST_SHA256="e1d3bb204617a08abea100c3b5be6de49d8c68483d29de90298c12f10ea204c9"
DSPARK_DRAFT_PARAMS_REF="833e51899fb160b20645a868bf000a98fd3181a4"
DSPARK_DRAFT_PARAMS_SHA256="993dc3d686f4673b774278110fdbf44a8b1d190af1841c597f9fd7e50d12fbce"
DSPARK_SOFTMAX_CLAMP_REF="8e85d6cfa9fc9025942c4faf07c362c1ce3d06f4"
DSPARK_SOFTMAX_CLAMP_PATCH_SHA256="a88ed6c2fe8398a998d16f26c222434ccfc664d3f7efe9c10c1e20268a691930"
DSPARK_PARAM_RESTAGE_REF="20189751de186f394abe850b1ad48087df32eb77"
DSPARK_PARAM_RESTAGE_PATCH_SHA256="30a3d02028d1db1974456bda9383cdebf7bbaa0c6e1b9e3cc010d3b0d2b9acd9"
MXFP8_DECODE_TACTICS_REF="741aea0ab6c51ccfbb862203a5a4e4aeaf7e8466"
MXFP8_DECODE_TACTICS_PATCH_SHA256="152571d8705b2fe9b505bd0441c7fecaf1d85ac981553272d85fcb76c25224ba"
DSV41_INDEXER_INPLACE_REF="3458c1f1900fbbaffc237d74491f30e6f8cf288c"
DSV41_INDEXER_INPLACE_PATCH_SHA256="03ec9856e93ff4c34f10ca08585a06d7f0bf64d8c1742b3854a10d8929d371eb"
DSV41_BLOCKED_DEQUANT_REF="0de6dd957f4da0b0655ef6d41bfd4b0593a08d12"
DSV41_BLOCKED_DEQUANT_PATCH_SHA256="a47f2968fe87d98832910907deea20a2c1334ac18d4bfda11a68ee7ae2513638"
DSV41_ALIGN_WIDTH_REF="b7344253b3136384c522aded59015f79c9bb7a08"
DSV41_ALIGN_WIDTH_PATCH_SHA256="457ad92a20e7ddec054ca9775881213fdfbb26b3f20c24d0ea1e4e980ec4e6a0"
DSV41_DEFAULT_THINKING_REF="f2aad50c4163d047d740fd61defea2d122ac5d30"
DSV41_DEFAULT_THINKING_PATCH_SHA256="97d324a27255233defa8ee343a0df2b2c63785e919ce69b60696289277e5df66"
DSV41_THINKING_PARITY_REF="af882cfe1e7ff244db2a49bb70fdc72199c68bb9"
DSV41_THINKING_PARITY_PATCH_SHA256="47363e10d91939b906bb269ebcd2eb428142930c92c4c04643fba95475f3863c"
DSV41_WINDOW_HISTORY_SRC_REF="ad55c50bc3a45bdf957626eb0a9350ca0f4f5176"
DSV41_WINDOW_HISTORY_SRC_PATCH_SHA256="85241d8d32c5e80b746811b00ee3e870c33e15d71206063568a389f77caf16ef"
DSV41_FUSED_WINDOW_LAYOUT_REF="9d5386fd13f7a91e9e84dc0c0ab61c5b1fbaba98"
DSV41_FUSED_WINDOW_LAYOUT_SHA256="4ca6d7d7d79340bb47b9f960d1cf4828674ede86c46f1858f04fe2b6165eb171"
DSV41_FUSED_WINDOW_LAYOUT_PATCH_SHA256="797c0aaab44f66e2c6a89e1277b2d3cd9c7a6dfb061c2b7ae4b98ed982d8a0ea"
DSV41_IDLE_WINDOW_REF="99cf403c9e84272b80cf36e7cca953d83be4b1d5"
DSV41_IDLE_STORE_REF="66366028d2dd259372df5066d7c809b4460b8f05"
DSV41_IDLE_SWA_STORE_PATCH_SHA256="887219d9ca62158e567fd320b6ba26e109cee2d034f0514d4d0ee53ec2f1196f"
DSV41_DRAFT_METADATA_GLUE_REF="1f8e4f3e6cdc378d94a8a08630a9763c648e88eb"
DSV41_DRAFT_METADATA_GLUE_PATCH_SHA256="cc4d32729b31be42f1446d1cd458fa9c799f2ab175797821858781075f52ae8d"
GREEDY_FLAGS_REFRESH_REF="6e0355d972cb926f137d2b8bb70caf5c9af5d129"
GREEDY_FLAGS_REFRESH_PATCH_SHA256="8dd4c412d1c8c25d8a0adeba743308ae644550761acf0f30c3e21e2a3940514a"
# Local patch (no upstream PR): --enable-deterministic-inference refuses explicit conflicting arguments.
DETERMINISTIC_EXPLICIT_PATCH_SHA256="f6803a0bb4ca780df737b74655ce46e9e0db219c94698c28756e4a15b6257adf"
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
echo "applying SGLang #44751 dsv41 tool-image/thinking fix ..."
printf '%s  %s\n' "$TOOL_IMAGES_THINKING_PATCH_SHA256" "$PATCH_TOOL_IMAGES_THINKING" | shasum -a 256 -c -
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_TOOL_IMAGES_THINKING"
patch --silent --fuzz=0 -p1 < "$PATCH_TOOL_IMAGES_THINKING"
python3 "$ROOT/scripts/verify_sglang_dsv41_tool_images_thinking.py" "$DST/python"
echo "applying SGLang #45301 graph-owned V4.1 padded-query buffer fix ..."
printf '%s  %s\n' "$DSV41_PADDED_QUERY_PATCH_SHA256" "$PATCH_DSV41_PADDED_QUERY" | shasum -a 256 -c -
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_DSV41_PADDED_QUERY"
patch --silent --fuzz=0 -p1 < "$PATCH_DSV41_PADDED_QUERY"
python3 "$ROOT/scripts/verify_sglang_dsv41_padded_query_buffer.py" "$DST/python"
python3 "$ROOT/scripts/test_sglang_dsv41_padded_query_buffer.py" "$DST/python"
echo "applying SGLang #40424 V4.1 DP-padding/idle-row fix ..."
printf '%s  %s\n' "$DSV41_DP_PADDING_PATCH_SHA256" "$PATCH_DSV41_DP_PADDING" | shasum -a 256 -c -
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_DSV41_DP_PADDING"
patch --silent --fuzz=0 -p1 < "$PATCH_DSV41_DP_PADDING"
python3 "$ROOT/scripts/verify_sglang_dsv41_dp_padding.py" "$DST/python"
python3 "$ROOT/scripts/test_sglang_dsv41_dp_padding.py" "$DST/python"
echo "applying SGLang #43448 no-chunk MHC prewarm fallback ..."
printf '%s  %s\n' "$MHC_PREWARM_NO_CHUNK_PATCH_SHA256" "$PATCH_MHC_PREWARM_NO_CHUNK" | shasum -a 256 -c -
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_MHC_PREWARM_NO_CHUNK"
patch --silent --fuzz=0 -p1 < "$PATCH_MHC_PREWARM_NO_CHUNK"
python3 "$ROOT/scripts/verify_sglang_mhc_prewarm_no_chunk.py" "$DST/python"
python3 "$ROOT/scripts/test_sglang_mhc_prewarm_no_chunk.py" "$DST/python"
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
echo "installing exact SGLang #45663 fused draft-parameter kernel and pinned-tree integration ..."
DSPARK_DRAFT_PARAMS_PATH="$DST/python/sglang/kernels/ops/speculative/dspark/draft_params.py"
curl -fsSL -m 120 \
  "https://raw.githubusercontent.com/sgl-project/sglang/${DSPARK_DRAFT_PARAMS_REF}/python/sglang/kernels/ops/speculative/dspark/draft_params.py" \
  -o "$DSPARK_DRAFT_PARAMS_PATH"
printf '%s  %s\n' "$DSPARK_DRAFT_PARAMS_SHA256" "$DSPARK_DRAFT_PARAMS_PATH" | shasum -a 256 -c -
python3 "$ROOT/runtime/patch_dspark_draft_params.py" "$DST/python"
python3 "$ROOT/scripts/verify_sglang_dspark_draft_params.py" "$DST/python"
echo "applying SGLang #45663 DSpark accept-output fusion ..."
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_DSPARK_ACCEPT"
patch --silent --fuzz=0 -p1 < "$PATCH_DSPARK_ACCEPT"
python3 "$ROOT/scripts/verify_sglang_dspark_accept_outputs.py" "$DST/python"
echo "applying SGLang #45518 FlashInfer probability clamp ..."
printf '%s  %s\n' "$DSPARK_SOFTMAX_CLAMP_PATCH_SHA256" "$PATCH_DSPARK_SOFTMAX_CLAMP" | shasum -a 256 -c -
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_DSPARK_SOFTMAX_CLAMP"
patch --silent --fuzz=0 -p1 < "$PATCH_DSPARK_SOFTMAX_CLAMP"
python3 "$ROOT/scripts/verify_sglang_dspark_softmax_clamp.py" "$DST/python"
python3 "$ROOT/scripts/test_sglang_dspark_softmax_clamp.py" "$DST/python"
echo "applying SGLang #45450 sampling-parameter restage skip ..."
printf '%s  %s\n' "$DSPARK_PARAM_RESTAGE_PATCH_SHA256" "$PATCH_DSPARK_RESTAGE" | shasum -a 256 -c -
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_DSPARK_RESTAGE"
patch --silent --fuzz=0 -p1 < "$PATCH_DSPARK_RESTAGE"
python3 "$ROOT/scripts/verify_sglang_dspark_param_restage.py" "$DST/python"
python3 "$ROOT/scripts/test_sglang_dspark_param_restage.py" "$DST/python"
echo "applying SGLang #45630 tuned MXFP8 tactics at decode sizes ..."
printf '%s  %s\n' "$MXFP8_DECODE_TACTICS_PATCH_SHA256" "$PATCH_MXFP8_DECODE_TACTICS" | shasum -a 256 -c -
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_MXFP8_DECODE_TACTICS"
patch --silent --fuzz=0 -p1 < "$PATCH_MXFP8_DECODE_TACTICS"
python3 "$ROOT/scripts/verify_sglang_mxfp8_decode_tactics.py" "$DST/python"
echo "applying SGLang #45745 in-place V4.1 indexer scoring ..."
printf '%s  %s\n' "$DSV41_INDEXER_INPLACE_PATCH_SHA256" "$PATCH_DSV41_INDEXER_INPLACE" | shasum -a 256 -c -
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_DSV41_INDEXER_INPLACE"
patch --silent --fuzz=0 -p1 < "$PATCH_DSV41_INDEXER_INPLACE"
python3 "$ROOT/scripts/verify_sglang_dsv41_indexer_inplace.py" "$DST/python"
echo "applying SGLang #45793 eight-token V4.1 KV dequantization ..."
printf '%s  %s\n' "$DSV41_BLOCKED_DEQUANT_PATCH_SHA256" "$PATCH_DSV41_BLOCKED_DEQUANT" | shasum -a 256 -c -
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_DSV41_BLOCKED_DEQUANT"
patch --silent --fuzz=0 -p1 < "$PATCH_DSV41_BLOCKED_DEQUANT"
python3 "$ROOT/scripts/verify_sglang_dsv41_blocked_kv_dequant.py" "$DST/python"
python3 "$ROOT/scripts/test_sglang_dsv41_blocked_kv_dequant.py"
echo "applying SGLang #45794 aligned V4.1 Torch-indexer score width ..."
printf '%s  %s\n' "$DSV41_ALIGN_WIDTH_PATCH_SHA256" "$PATCH_DSV41_ALIGN_WIDTH" | shasum -a 256 -c -
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_DSV41_ALIGN_WIDTH"
patch --silent --fuzz=0 -p1 < "$PATCH_DSV41_ALIGN_WIDTH"
python3 "$ROOT/scripts/verify_sglang_dsv41_align_indexer_width.py" "$DST/python"
python3 "$ROOT/scripts/test_sglang_dsv41_align_indexer_width.py"
echo "applying SGLang #45762 default-thinking parser parity ..."
printf '%s  %s\n' "$DSV41_DEFAULT_THINKING_PATCH_SHA256" "$PATCH_DSV41_DEFAULT_THINKING" | shasum -a 256 -c -
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_DSV41_DEFAULT_THINKING"
patch --silent --fuzz=0 -p1 < "$PATCH_DSV41_DEFAULT_THINKING"
python3 "$ROOT/scripts/verify_sglang_dsv41_default_thinking.py" "$DST/python"
python3 "$ROOT/scripts/test_sglang_dsv41_default_thinking.py" "$DST/python"
echo "applying SGLang #45804 native DeepSeek Chat/Responses thinking parity ..."
printf '%s  %s\n' "$DSV41_THINKING_PARITY_PATCH_SHA256" "$PATCH_DSV41_THINKING_PARITY" | shasum -a 256 -c -
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_DSV41_THINKING_PARITY"
patch --silent --fuzz=0 -p1 < "$PATCH_DSV41_THINKING_PARITY"
python3 "$ROOT/scripts/verify_sglang_dsv41_thinking_parity.py" "$DST/python"
python3 "$ROOT/scripts/test_sglang_dsv41_thinking_parity.py" "$DST/python"
echo "applying SGLang #45579 request-window history-source reuse ..."
printf '%s  %s\n' "$DSV41_WINDOW_HISTORY_SRC_PATCH_SHA256" "$PATCH_DSV41_WINDOW_HISTORY_SRC" | shasum -a 256 -c -
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_DSV41_WINDOW_HISTORY_SRC"
patch --silent --fuzz=0 -p1 < "$PATCH_DSV41_WINDOW_HISTORY_SRC"
python3 "$ROOT/scripts/verify_sglang_dsv41_window_history_src.py" "$DST/python"
python3 "$ROOT/scripts/test_sglang_dsv41_window_history_src.py"
echo "installing exact SGLang #45726 fused request-window layout kernel ..."
DSV41_FUSED_WINDOW_LAYOUT_PATH="$DST/python/sglang/kernels/ops/kvcache/dsv41_window_layout.py"
curl -fsSL -m 120 \
  "https://raw.githubusercontent.com/sgl-project/sglang/${DSV41_FUSED_WINDOW_LAYOUT_REF}/python/sglang/kernels/ops/kvcache/dsv41_window_layout.py" \
  -o "$DSV41_FUSED_WINDOW_LAYOUT_PATH"
printf '%s  %s\n' "$DSV41_FUSED_WINDOW_LAYOUT_SHA256" "$DSV41_FUSED_WINDOW_LAYOUT_PATH" | shasum -a 256 -c -
printf '%s  %s\n' "$DSV41_FUSED_WINDOW_LAYOUT_PATCH_SHA256" "$PATCH_DSV41_FUSED_WINDOW_LAYOUT" | shasum -a 256 -c -
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_DSV41_FUSED_WINDOW_LAYOUT"
patch --silent --fuzz=0 -p1 < "$PATCH_DSV41_FUSED_WINDOW_LAYOUT"
python3 "$ROOT/scripts/verify_sglang_dsv41_fused_window_layout.py" "$DST/python"
python3 "$ROOT/scripts/test_sglang_dsv41_fused_window_layout.py"
echo "applying SGLang #45586/#45699 idle SWA-store safety ..."
printf '%s  %s\n' "$DSV41_IDLE_SWA_STORE_PATCH_SHA256" "$PATCH_DSV41_IDLE_SWA_STORE" | shasum -a 256 -c -
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_DSV41_IDLE_SWA_STORE"
patch --silent --fuzz=0 -p1 < "$PATCH_DSV41_IDLE_SWA_STORE"
python3 "$ROOT/scripts/verify_sglang_dsv41_idle_swa_store.py" "$DST/python"
python3 "$ROOT/scripts/test_sglang_dsv41_idle_swa_store.py" "$DST/python"
echo "applying SGLang #45450 DSpark draft metadata glue ..."
printf '%s  %s\n' "$DSV41_DRAFT_METADATA_GLUE_PATCH_SHA256" "$PATCH_DSV41_DRAFT_METADATA_GLUE" | shasum -a 256 -c -
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_DSV41_DRAFT_METADATA_GLUE"
patch --silent --fuzz=0 -p1 < "$PATCH_DSV41_DRAFT_METADATA_GLUE"
python3 "$ROOT/scripts/verify_sglang_dsv41_draft_metadata_glue.py" "$DST/python"
python3 "$ROOT/scripts/test_sglang_dsv41_draft_metadata_glue.py" "$DST/python"
echo "applying SGLang #44923 greedy-flag refresh on batch filter ..."
printf '%s  %s\n' "$GREEDY_FLAGS_REFRESH_PATCH_SHA256" "$PATCH_GREEDY_FLAGS_REFRESH" | shasum -a 256 -c -
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_GREEDY_FLAGS_REFRESH"
patch --silent --fuzz=0 -p1 < "$PATCH_GREEDY_FLAGS_REFRESH"
python3 "$ROOT/scripts/verify_sglang_greedy_flags_refresh.py" "$DST/python"
python3 "$ROOT/scripts/test_sglang_greedy_flags_refresh.py" "$DST/python"
echo "applying deterministic-inference explicit-override refusal ..."
printf '%s  %s\n' "$DETERMINISTIC_EXPLICIT_PATCH_SHA256" "$PATCH_DETERMINISTIC_EXPLICIT" | shasum -a 256 -c -
patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_DETERMINISTIC_EXPLICIT"
patch --silent --fuzz=0 -p1 < "$PATCH_DETERMINISTIC_EXPLICIT"
echo "$REF" > REF
printf '%s\n' \
  "41303 ed8c3139f1f643544b68102cf8c44504e5e55e3d" \
  "42901 05ce0e980f00aa80864190d6f1cd5d9b685802f1" \
  "44001 9569a3656155a34dc763826b1608f247d3f66832" \
  "41259 fba3ad4b1f4563143f3352beb148c4c28d4be9ed helper-sha256=$PARALLEL_ENCODE_SHA256 test-sha256=$PARALLEL_ENCODE_TEST_SHA256" \
  "45410 aeeb943fe71626a7cea183b716b8bb236a285f2e" \
  "43549 9e0ce2f3fab3b2a456ac97aea04bc880ae2fe7b1" \
  "44751 $TOOL_IMAGES_THINKING_REF patch-sha256=$TOOL_IMAGES_THINKING_PATCH_SHA256" \
  "45301 $DSV41_PADDED_QUERY_REF patch-sha256=$DSV41_PADDED_QUERY_PATCH_SHA256" \
  "40424 $DSV41_DP_PADDING_REF patch-sha256=$DSV41_DP_PADDING_PATCH_SHA256" \
  "43448 $MHC_PREWARM_NO_CHUNK_REF patch-sha256=$MHC_PREWARM_NO_CHUNK_PATCH_SHA256" \
  "45457 $PLAIN_TEXT_REF helper-sha256=$PLAIN_TEXT_SHA256 test-sha256=$PLAIN_TEXT_TEST_SHA256" \
  "45458 $REASONING_END_REF test-sha256=$REASONING_END_TEST_SHA256" \
  "45663 $DSPARK_DRAFT_PARAMS_REF helper-sha256=$DSPARK_DRAFT_PARAMS_SHA256 accept-patch-sha256=8068b7ec44914c5e79627c88ebd5367b6dfc58cc130da9307ab694ce93404e22" \
  "45518 $DSPARK_SOFTMAX_CLAMP_REF patch-sha256=$DSPARK_SOFTMAX_CLAMP_PATCH_SHA256" \
  "45450 $DSPARK_PARAM_RESTAGE_REF patch-sha256=$DSPARK_PARAM_RESTAGE_PATCH_SHA256" \
  "45630 $MXFP8_DECODE_TACTICS_REF patch-sha256=$MXFP8_DECODE_TACTICS_PATCH_SHA256" \
  "45745 $DSV41_INDEXER_INPLACE_REF patch-sha256=$DSV41_INDEXER_INPLACE_PATCH_SHA256" \
  "45793 $DSV41_BLOCKED_DEQUANT_REF patch-sha256=$DSV41_BLOCKED_DEQUANT_PATCH_SHA256" \
  "45794 $DSV41_ALIGN_WIDTH_REF patch-sha256=$DSV41_ALIGN_WIDTH_PATCH_SHA256" \
  "45762 $DSV41_DEFAULT_THINKING_REF patch-sha256=$DSV41_DEFAULT_THINKING_PATCH_SHA256" \
  "45804 $DSV41_THINKING_PARITY_REF patch-sha256=$DSV41_THINKING_PARITY_PATCH_SHA256" \
  "45579 $DSV41_WINDOW_HISTORY_SRC_REF patch-sha256=$DSV41_WINDOW_HISTORY_SRC_PATCH_SHA256" \
  "45726 $DSV41_FUSED_WINDOW_LAYOUT_REF helper-sha256=$DSV41_FUSED_WINDOW_LAYOUT_SHA256 integration-patch-sha256=$DSV41_FUSED_WINDOW_LAYOUT_PATCH_SHA256" \
  "45586 $DSV41_IDLE_WINDOW_REF 45699 $DSV41_IDLE_STORE_REF patch-sha256=$DSV41_IDLE_SWA_STORE_PATCH_SHA256" \
  "45450-draft-glue $DSV41_DRAFT_METADATA_GLUE_REF patch-sha256=$DSV41_DRAFT_METADATA_GLUE_PATCH_SHA256" \
  "44923 $GREEDY_FLAGS_REFRESH_REF patch-sha256=$GREEDY_FLAGS_REFRESH_PATCH_SHA256" \
  "local-deterministic-explicit patch-sha256=$DETERMINISTIC_EXPLICIT_PATCH_SHA256" > PATCHES
du -sh python
echo "staged $DST/python (ref $REF + SGLang #41303/#42901/#44001/#41259/#45410/#43549/#44751/#45301/#40424/#43448/#45457/#45458/#45663/#45518/#45450/#45630/#45745/#45793/#45794/#45762/#45804/#45579/#45726/#45586/#45699/#45450-draft-glue/#44923 backports); now: docker build -f Dockerfile.canary -t dsv41-4x-spark:canary ."
