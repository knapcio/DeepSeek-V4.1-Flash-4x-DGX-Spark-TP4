"""DSV41_PREFILL_NATIVE_EXTRA: SM120 sparse-MLA prefill that reads SGLang's native 128/256-token extra pages.

Installed next to flash_mla_sm120.py (sglang/kernels/ops/attention/dsv41_pce.py). Default OFF: with the variable
unset or 0 the adapter only evaluates `routes()` (False) and v2.6's path runs unchanged; nothing here touches
FlashInfer or loads the module.

With DSV41_PREFILL_NATIVE_EXTRA=1, a prefill / wide-verify call (more than 64 query rows) whose extra cache has
128- or 256-token pages goes to a separate FlashInfer module compiled AT IMAGE BUILD from the image's own
sparse_mla_sm120 sources (runtime/flashinfer_pce/gen_pce_sources.py + `python3 -m ... dsv41_pce build`). The kernels
are the stock dual-cache prefill kernels instantiated with PAGE_BLOCK_SIZE_EXTRA = 128 / 256, so the extra cache is no
longer copied into the persistent 64-token split buffer (`flash_mla_sm120_split:<dev>:extra`, N x ratio x 37,440 B:
~2.98 GB per rank at the v2.6 pool) and the per-call page mark + split kernels for the extra source disappear. The
SWA (main) source is still split exactly as before. Extra pages of 64 / 2 tokens keep the stock module.

Identity, fail-closed (REVIEW_sol K8 P2-1), checked on the first call (the boot's own autotune prefill):
  * the flag is 0 / 1 / unset, anything else raises;
  * the generated sources equal their manifest;
  * the installed FlashInfer source closure (both .cu inputs, the glue headers, every sparse_mla_sm120 header) equals
    the closure recorded in the manifest, which the generator already required to be the approved one;
  * the module is the PREBUILT shared object next to the sources (no JIT at serving time): module.json's tag equals the
    manifest's, and the .so's sha256 equals module.json's. The tag and the .so sha are logged in "module ready".
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

ENV = "DSV41_PREFILL_NATIVE_EXTRA"
SRC_ENV = "DSV41_PCE_SRC_DIR"
DEFAULT_SRC = "/opt/dsv41/runtime/flashinfer_pce/gen"
NATIVE_PBSX = frozenset((128, 256))
SOURCES = ("pce_paged_attention.cu", "pce_prefill.cu", "pce_binding.cu")
WIDE_VERIFY_MAX_ROWS = 96   # the c16 target-verify graph: bs 16 x 6 rows

_hits: dict = {}


@lru_cache(maxsize=1)
def enabled() -> bool:
    raw = os.environ.get(ENV, "0").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        logger.warning("%s=1: prefill reads native 128/256-token extra pages (no extra split buffer)", ENV)
        return True
    raise ValueError(f"{ENV} must be 0 or 1, got {raw!r}")


def routes(extra_kv) -> bool:
    """True when this prefill call's extra cache goes to the native-page module."""
    return extra_kv is not None and extra_kv.ndim >= 3 and int(extra_kv.shape[1]) in NATIVE_PBSX and enabled()


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _generator(src: Path):
    for p in (Path(__file__).with_name("gen_pce_sources.py"), Path(src).parent / "gen_pce_sources.py"):
        if p.is_file():
            spec = importlib.util.spec_from_file_location("dsv41_pce_gen", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise RuntimeError(f"{ENV}: gen_pce_sources.py not found next to {__file__} or {src}")


def check_sources(src: Path, csrc: Path, include: Path) -> dict:
    """Generated sources == manifest, installed FlashInfer closure == manifest closure; return the manifest."""
    src = Path(src)
    manifest = json.loads((src / "manifest.json").read_text())
    for name in SOURCES:
        want = manifest["outputs"].get(name)
        if want is None or _sha(src / name) != want:
            raise RuntimeError(f"{ENV}: {src / name} does not match its manifest (regenerate the module sources)")
    gen = _generator(src)
    try:
        got = gen.closure(Path(csrc), Path(include))
        gen.check_closure(got, manifest.get("closure") or {})
    except gen.ClosureError as e:
        raise RuntimeError(f"{ENV}: installed FlashInfer is not the source the module was generated and approved "
                           f"from ({e}); rebuild the image") from None
    return manifest


def module_name(manifest: dict) -> str:
    return f"dsv41_pce_sparse_mla_sm120_{manifest['module_tag']}"


def check_module(src: Path, manifest: dict) -> Path:
    """The prebuilt .so for this manifest, its sha256 checked against module.json; returns its path."""
    src = Path(src)
    try:
        mj = json.loads((src / "module.json").read_text())
    except (OSError, ValueError) as e:
        raise RuntimeError(f"{ENV}: no prebuilt module record {src / 'module.json'} ({e}); the image build must run "
                           f"`dsv41_pce build`") from None
    if mj.get("module_tag") != manifest["module_tag"] or mj.get("name") != module_name(manifest):
        raise RuntimeError(f"{ENV}: prebuilt module {mj.get('name')} / {mj.get('module_tag')} is not this manifest's "
                           f"{module_name(manifest)}")
    so = src / f"{mj['name']}.so"
    if not so.is_file() or _sha(so) != mj.get("so_sha256"):
        raise RuntimeError(f"{ENV}: {so} is missing or its sha256 is not module.json's {str(mj.get('so_sha256'))[:12]}")
    return so


def _flashinfer_dirs():
    from flashinfer.jit import env as jit_env

    return Path(jit_env.FLASHINFER_CSRC_DIR), Path(jit_env.FLASHINFER_INCLUDE_DIR)


@lru_cache(maxsize=1)
def _module():
    import tvm_ffi

    src = Path(os.environ.get(SRC_ENV, DEFAULT_SRC))
    manifest = check_sources(src, *_flashinfer_dirs())
    so = check_module(src, manifest)
    module = tvm_ffi.load_module(str(so))
    logger.warning("%s module ready: tag=%s so_sha256=%s generator=%s", ENV, manifest["module_tag"], _sha(so),
                   manifest["generator_version"])
    return module


def build(src: Path) -> dict:
    """Image build: compile the module with FlashInfer's own JIT spec and flags, keep the .so next to the sources."""
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    src = Path(src)
    manifest = check_sources(src, *_flashinfer_dirs())
    flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    if not any("sm_121a" in f for f in flags):
        raise RuntimeError(f"{ENV}: nvcc flags {flags} do not target sm_121a (set FLASHINFER_CUDA_ARCH_LIST=12.1a)")
    name = module_name(manifest)
    spec = gen_jit_spec(name, [src / n for n in SOURCES], extra_cuda_cflags=flags)
    spec.build()
    so = src / f"{name}.so"
    shutil.copyfile(spec.jit_library_path, so)
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()
    mj = {"name": name, "module_tag": manifest["module_tag"], "so_sha256": _sha(so), "nvcc_flags": flags,
          "nvcc": nvcc[-1] if nvcc else None, "cuda_arch_list": os.environ.get("FLASHINFER_CUDA_ARCH_LIST")}
    (src / "module.json").write_text(json.dumps(mj, indent=1, sort_keys=True) + "\n")
    return mj


def paged_attention(q, kv_64, indices, output, out_lse, sm_scale, *, topk_length, attn_sink,
                    extra_kv, extra_indices, extra_topk_length) -> None:
    """Dual-cache DSV4 prefill with native extra pages; same arguments and in-place outputs as the stock op."""
    from flashinfer.mla._sparse_mla_sm120 import _MODEL_TYPE_DSV4, _resolve_model_type

    model_type = _resolve_model_type(q.shape[-1], "auto")
    if model_type != _MODEL_TYPE_DSV4:
        raise ValueError(f"{ENV}: DSV4 only (d_qk={q.shape[-1]})")
    if output.shape[-1] != 512:
        raise ValueError(f"{ENV}: d_v must be 512, got {output.shape[-1]}")
    pbsx, rows = int(extra_kv.shape[1]), int(q.shape[0])
    # Route evidence, host side (so at CUDA-graph CAPTURE time for the verify graphs): one line per page size and
    # wide-verify row count (<= 96), one per page size for longer prefill calls.
    key = (pbsx, rows if rows <= WIDE_VERIFY_MAX_ROWS else "prefill")
    if key not in _hits:
        _hits[key] = True
        logger.warning("DSV41_PCE_HIT extra_page_block_size=%d rows=%d", pbsx, rows)
    _module().dsv41_pce_paged_attention(
        q, kv_64, indices, output, out_lse, float(sm_scale), int(model_type),
        topk_length, attn_sink, extra_kv, extra_indices, extra_topk_length,
    )


def main(argv) -> int:
    """build: image build (no GPU). verify: sources, closure and .so sha (no GPU). check: verify + load (GPU node)."""
    if len(argv) == 2 and argv[1] in ("build", "verify", "check"):
        src = Path(os.environ.get(SRC_ENV, DEFAULT_SRC))
        if argv[1] == "build":
            mj = build(src)
            print(f"PCE BUILD OK {mj['name']} so_sha256={mj['so_sha256']}")
            return 0
        manifest = check_sources(src, *_flashinfer_dirs())
        so = check_module(src, manifest)
        if argv[1] == "check":
            import tvm_ffi

            tvm_ffi.load_module(str(so))
        print(f"PNX MODULE OK {manifest['module_tag']} {_sha(so)}")
        return 0
    print("usage: python3 -m sglang.kernels.ops.attention.dsv41_pce build|verify|check", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
