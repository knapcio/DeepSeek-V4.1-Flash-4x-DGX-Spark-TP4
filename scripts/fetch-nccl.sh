#!/usr/bin/env bash
# Stage the NCCL source that Dockerfile.canary-roce's nccl-builder stage compiles: NVIDIA/nccl v2.30.7-1 (commit
# 73cf112, the version of the base image's nvidia-nccl-cu13 wheel) as a GitHub tarball in runtime/nccl-src
# (untracked, like runtime/sglang-canary). The stage applies patches/nccl-2393-aarch64-cts-fence.patch (NVIDIA/nccl
# #2393, backported to 2.30.7 by Stanislav Bardyuk) and builds for sm_121 only; the version is NOT bumped.
# The tree is checked file by file against a pinned digest (GitHub does not promise byte-stable tarballs; the tree
# is what is pinned), and the one file the patch edits against its pinned sha256.
#   scripts/fetch-nccl.sh            then: ./start-tp4.sh build (BUILD_DOCKERFILE=Dockerfile.canary-roce)
set -euo pipefail
REF="${NCCL_REF:-73cf112295c33aee2b895f329f592f2a9b4b0f97}"
TREE_SHA256="${NCCL_TREE_SHA256:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$ROOT/runtime/nccl-src"
# pinned for the default REF only (another REF is an experiment: pass NCCL_TREE_SHA256 for it)
if [[ "$REF" == 73cf112295c33aee2b895f329f592f2a9b4b0f97 && -z "$TREE_SHA256" ]]; then
  TREE_SHA256=0761b6c8b49e4350090c232ac4d483f0958024d2e5fd0571f1074ad27b7291eb
fi
[[ -n "$TREE_SHA256" ]] || { echo "fetch-nccl: no tree digest for $REF (set NCCL_TREE_SHA256)" >&2; exit 1; }
P2P_SHA256=7379081678564a0065c60073a6ae3b7a7b259f632805ddc1a6dbac58e47df362   # src/transport/net_ib/p2p.cc at 73cf112
tree_digest() {   # sha256 over "<sha256>  <relative path>" of every file, sorted by path (bytes, C locale)
  python3 - "$1" <<'PY'
import hashlib, os, sys
root = sys.argv[1]; lines = []
for d, _, files in os.walk(root):
    for f in files:
        p = os.path.join(d, f)
        lines.append("%s  %s\n" % (hashlib.sha256(open(p, "rb").read()).hexdigest(), os.path.relpath(p, root)))
print(hashlib.sha256("".join(sorted(lines, key=lambda l: l.split("  ", 1)[1].encode())).encode()).hexdigest())
PY
}
rm -rf "$DST.partial"; mkdir -p "$DST.partial"
echo "fetching NVIDIA/nccl @ $REF ..."
curl -fsSL -m 600 "https://github.com/NVIDIA/nccl/archive/${REF}.tar.gz" -o "$DST.partial/src.tgz"
tar -C "$DST.partial" -xzf "$DST.partial/src.tgz" --strip-components=1
rm -f "$DST.partial/src.tgz"
got=$(tree_digest "$DST.partial")
[[ "$got" == "$TREE_SHA256" ]] || { echo "fetch-nccl: tree digest $got is not the pinned $TREE_SHA256" >&2; exit 1; }
if [[ "$REF" == 73cf112295c33aee2b895f329f592f2a9b4b0f97 ]]; then
  printf '%s  %s\n' "$P2P_SHA256" "$DST.partial/src/transport/net_ib/p2p.cc" | shasum -a 256 -c - >/dev/null
fi
echo "$REF" > "$DST.partial/REF"
rm -rf "$DST"; mv "$DST.partial" "$DST"
echo "staged $DST (NVIDIA/nccl $REF, tree $got); the image applies patches/nccl-2393-aarch64-cts-fence.patch"
