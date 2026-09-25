#!/bin/bash
# Build the side-by-side copy of b12x main as the top-level package `b12x_next`.
#
# The image already ships the SG17 b12x as `b12x` (/opt/b12x); RoCEnante needs that copy
# (`b12x.comm.roce.AllReduce.prepare`, removed on main). Both cannot be imported under one name,
# so main is copied and renamed:
#   * every whole-word `b12x` identifier/string in .py/.c/.cpp/.h -> `b12x_next` (imports,
#     CompileJob "module:function" strings, torch.library namespaces `b12x::`, cache paths);
#   * every `B12X_*` environment knob -> `B12X_NEXT_*`, so the JIT compile cache, the preparation
#     (autotune) cache and all tuning knobs are separate from the SG17 copy's
#     (B12X_COMPILE_CACHE_DIR=/state/b12x-compile stays RoCEnante's).
# The source commit is written to b12x_next/SOURCE_COMMIT; adapter/moe_b12x_next.py refuses any
# other commit.
#
# usage: scripts/build_b12x_next.sh [out_dir=runtime/b12x_next] [commit]
set -euo pipefail
OUT=${1:-runtime/b12x_next}
COMMIT=${2:-a7d7d29b2ef8869086e0ceaa787321f17544e3c9}
REPO=${B12X_REPO:-https://github.com/local-inference-lab/b12x}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git clone -q "$REPO" "$tmp/src"
git -C "$tmp/src" checkout -q "$COMMIT"
rm -rf "$OUT/b12x_next"
mkdir -p "$OUT"
cp -R "$tmp/src/b12x" "$OUT/b12x_next"
cp "$tmp/src/LICENSE" "$OUT/LICENSE"
find "$OUT/b12x_next" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$OUT/b12x_next" -type f \( -name '*.py' -o -name '*.c' -o -name '*.cpp' -o -name '*.h' \) -print0 \
  | xargs -0 perl -pi -e 's/\bb12x\b(?!-)/b12x_next/g; s/\bB12X_(?!NEXT_)/B12X_NEXT_/g'
echo "$COMMIT" > "$OUT/b12x_next/SOURCE_COMMIT"
# ds41 patches, in order:
#   compact-n64-m64: admit the M64 tile for compact-N64 (N=576, EP1) prefill capacities (b12x pins them to M16)
#   prequant-input:  prequantized_input() launches of the token-major W4A8-MX front-end skip the
#                    in-kernel input quantization (prefill SP gathers the MXFP8 rows, adapter/prefill_sp.py)
#   barrier-zero:    the per-launch re-zero of barrier_count + barrier_epoch (adjacent in the arena)
#                    is one fill over both instead of two (same bytes; 2 -> 1 launch per MoE layer)
#   det-triton-planner: admit the Triton route planner with deterministic output (it writes only
#                    order-free integer row counts / tile prefix and resets the barrier words, so
#                    launches that use it skip the barrier fill above; outputs are bit-identical)
HERE="$(cd "$(dirname "$0")" && pwd)"
PATCHES=("$HERE/b12x_next-compact-n64-m64.patch" "$HERE/b12x_next-prequant-input.patch"
         "$HERE/b12x_next-barrier-zero.patch" "$HERE/b12x_next-det-triton-planner.patch")
for PATCH in "${PATCHES[@]}"; do
  (cd "$OUT" && patch -p0 --forward --quiet < "$PATCH")
done
grep -q "_compact_n64_tiles" "$OUT/b12x_next/moe/fused_moe/_tuning.py" || { echo "compact-n64-m64 patch did not apply" >&2; exit 1; }
grep -q "def prequantized_input" "$OUT/b12x_next/moe/fused_moe/_impl.py" || { echo "prequant-input patch did not apply" >&2; exit 1; }
grep -q "def _zero_barrier_state" "$OUT/b12x_next/moe/fused_moe/_impl.py" || { echo "barrier-zero patch did not apply" >&2; exit 1; }
grep -q "_compact_w4a8_query(query) and not query.deterministic_output" "$OUT/b12x_next/moe/fused_moe/_tuning.py" \
  && { echo "det-triton-planner patch did not apply" >&2; exit 1; }
if command -v sha256sum >/dev/null; then sum=sha256sum; else sum="shasum -a 256"; fi
cat "${PATCHES[@]}" | $sum | cut -c1-16 > "$OUT/b12x_next/SOURCE_PATCH"
# leftovers: only thread names / lock-file names of the form "b12x-..." may keep the old word
left=$(find "$OUT/b12x_next" -name '*.py' -print0 | xargs -0 perl -ne 'print "$ARGV\n" if /\bb12x\b(?!-)/ || /\bB12X_(?!NEXT_)/' | sort -u)
if [ -n "$left" ]; then echo "unrenamed b12x references in: $left" >&2; exit 1; fi
echo "built $OUT/b12x_next from $COMMIT ($(find "$OUT/b12x_next" -name '*.py' | wc -l) python files)"
