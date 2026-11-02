#!/usr/bin/env python3
"""Apply FlashInfer PR #7676 (issue #7675) to the installed b12x SM120/SM121 dense block-scaled GEMM.

The kernel (flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py, used by mm_mxfp8(backend="b12x"), i.e. by
every DSV41 MXFP8 dense linear through adapter/mxfp8_b12x.py) refills a mainloop shared-memory stage with TMA (async
proxy) as soon as the producer's producer_acquire returns. The MMA warps read that stage with ldmatrix (generic proxy)
and release it through the pipeline's mbarrier. Nothing orders those generic-proxy reads before the async-proxy TMA
write, so under concurrent GPU work the refill can land before the last reads and a tile is computed from the next
pipeline lap's data. The fix (Thireus, flashinfer-ai/flashinfer#7676, commit c1d1214) is one proxy fence right after
producer_acquire in the mainloop producer loop:

    mainloop_pipeline.producer_acquire(mainloop_producer_state)
+   cute.arch.fence_proxy("async.shared", space="cta")

This script makes exactly that edit to the installed file. It refuses anything but the pinned files:

    pristine  sha256 PRE[...]   -> insert the line once at the unique anchor, check the result is POST[...]
    patched   sha256 POST[...]  -> nothing to do (idempotent)
    anything else               -> exit 2 (the image's FlashInfer moved: re-review before shipping)

The kernel is compiled by CuTe DSL in-process at first use (the mxfp8 b12x runner passes no on-disk cache module), so
the edited source is what every boot compiles; nothing is compiled here.

usage: apply_fi7676.py [--check] [FILE]
  FILE     default: the installed flashinfer package's copy (located without importing flashinfer)
  --check  only report: exit 0 when patched, 1 when pristine, 2 when unknown
"""
import hashlib
import importlib.util
import os
import sys

REL = os.path.join("gemm", "kernels", "dense_blockscaled_gemm_sm120_b12x.py")

PINS = {
    # flashinfer 0.6.18 (git 69ff11fc; PyPI wheel sha256 d5d26edb...), the DS v2.6 / v2.6.2 / v2.6.3 image's copy
    "eee668e5c143f1c31690ad8a7b44f65050fee3232df1e1c484a3789c2ef0d960":
        "d583991c917f8cbd609f412a82651445b3f9fa1e4cac9adf24e7176e3c741d05",
    # flashinfer main as of 2026-11-02 (git blob 4451824a, the PR's base) -> the PR head's blob e8b94b00 exactly
    "6739702e27afad21767b71024678b35c6892e3f9c74aaff375fb3f6734399f86":
        "4924ff095448f59b204b9f5ba8627ec9769eddd23777b34a1212bfc754b0e8b6",
}
POSTS = {v: k for k, v in PINS.items()}

ANCHOR = (
    "                for _k_tile in range(0, k_tile_iter_cnt, 1, unroll=2):  # type: ignore[call-overload]\n"
    "                    mainloop_pipeline.producer_acquire(mainloop_producer_state)\n"
)
FENCE = '                    cute.arch.fence_proxy("async.shared", space="cta")\n'


def default_target():
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("apply_fi7676: flashinfer not installed")
    return os.path.join(list(spec.submodule_search_locations)[0], REL)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def patch_text(text):
    """The PR #7676 edit on the kernel source text. Raises ValueError unless the anchor occurs exactly once and the
    fence is not already there."""
    n = text.count(ANCHOR)
    if n != 1:
        raise ValueError(f"anchor found {n} times (expected 1)")
    if (ANCHOR + FENCE) in text:
        raise ValueError("fence already present")
    return text.replace(ANCHOR, ANCHOR + FENCE, 1)


def state(data):
    h = sha256(data)
    if h in POSTS:
        return "patched", h
    if h in PINS:
        return "pristine", h
    return "unknown", h


def main(argv):
    check = "--check" in argv
    args = [a for a in argv if a != "--check"]
    path = args[0] if args else default_target()
    with open(path, "rb") as f:
        data = f.read()
    st, h = state(data)
    print(f"apply_fi7676: {path}: {st} sha256={h}")
    if check:
        return {"patched": 0, "pristine": 1}.get(st, 2)
    if st == "patched":
        return 0
    if st == "unknown":
        print("apply_fi7676: refusing: not a pinned FlashInfer b12x dense GEMM source", file=sys.stderr)
        return 2
    new = patch_text(data.decode("utf-8")).encode("utf-8")
    got = sha256(new)
    if got != PINS[h]:
        print(f"apply_fi7676: refusing: result sha256 {got} != pinned {PINS[h]}", file=sys.stderr)
        return 2
    tmp = path + ".fi7676.tmp"
    with open(tmp, "wb") as f:
        f.write(new)
    os.replace(tmp, path)
    print(f"apply_fi7676: patched (FlashInfer PR #7676 fence) sha256={got}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
