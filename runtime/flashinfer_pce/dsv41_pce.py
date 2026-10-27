"""DSV41_PREFILL_NATIVE_EXTRA: SM120 sparse-MLA prefill that reads SGLang's native 128/256-token extra pages.

Installed next to flash_mla_sm120.py (sglang/kernels/ops/attention/dsv41_pce.py). Default OFF: with the variable
unset or 0 nothing here is imported by the serving path and v2.6 runs unchanged.

With DSV41_PREFILL_NATIVE_EXTRA=1, a prefill / large-verify call (more than 64 query rows) whose extra cache has
128- or 256-token pages goes to a separate FlashInfer JIT module built from the image's own sparse_mla_sm120 sources
(runtime/flashinfer_pce/gen_pce_sources.py, run at image build). The kernels are the stock dual-cache prefill
kernels instantiated with PAGE_BLOCK_SIZE_EXTRA = 128 / 256, so the extra cache is no longer copied into the
persistent 64-token split buffer (`flash_mla_sm120_split:<dev>:extra`, N x ratio x 37,440 B: ~2.98 GB per rank
at the v2.6 pool) and the per-call page mark + split kernels for the extra source disappear. The SWA (main) source
is still split exactly as before. Extra pages of 64 / 2 tokens keep the stock module.

Fail-closed: a value other than 0 / 1, generated sources that do not match their manifest, or an installed
FlashInfer whose sparse_mla_sm120 sources differ from the ones the module was generated from raise on the first
call, which is the boot's own autotune prefill.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

ENV = "DSV41_PREFILL_NATIVE_EXTRA"
SRC_ENV = "DSV41_PCE_SRC_DIR"
DEFAULT_SRC = "/opt/dsv41/runtime/flashinfer_pce/gen"
NATIVE_PBSX = frozenset((128, 256))
SOURCES = ("pce_paged_attention.cu", "pce_prefill.cu", "pce_binding.cu")
_INPUTS = ("sparse_mla_sm120_prefill.cu", "sparse_mla_sm120.cu")

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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sources(src: Path, csrc: Path) -> dict:
    """Verify the generated sources against their manifest and the installed FlashInfer inputs; return the manifest."""
    manifest = json.loads((src / "manifest.json").read_text())
    for name in SOURCES:
        want = manifest["outputs"].get(name)
        if want is None or _sha(src / name) != want:
            raise RuntimeError(f"{ENV}: {src / name} does not match its manifest (regenerate the module sources)")
    for name in _INPUTS:
        got = _sha(csrc / name)
        if got != manifest["inputs"].get(name):
            raise RuntimeError(
                f"{ENV}: installed FlashInfer {csrc / name} ({got[:12]}) is not the source the module was generated "
                f"from ({str(manifest['inputs'].get(name))[:12]}); rebuild the image")
    return manifest


@lru_cache(maxsize=1)
def _module():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    src = Path(os.environ.get(SRC_ENV, DEFAULT_SRC))
    manifest = check_sources(src, Path(jit_env.FLASHINFER_CSRC_DIR))
    flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    spec = gen_jit_spec(
        f"dsv41_pce_sparse_mla_sm120_{manifest['module_tag']}",
        [src / name for name in SOURCES],
        extra_cuda_cflags=flags,
    )
    module = spec.build_and_load()
    logger.warning("%s module ready: tag=%s generator=%s", ENV, manifest["module_tag"],
                   manifest["generator_version"])
    return module


def paged_attention(q, kv_64, indices, output, out_lse, sm_scale, *, topk_length, attn_sink,
                    extra_kv, extra_indices, extra_topk_length) -> None:
    """Dual-cache DSV4 prefill with native extra pages; same arguments and in-place outputs as the stock op."""
    from flashinfer.mla._sparse_mla_sm120 import _MODEL_TYPE_DSV4, _resolve_model_type

    model_type = _resolve_model_type(q.shape[-1], "auto")
    if model_type != _MODEL_TYPE_DSV4:
        raise ValueError(f"{ENV}: DSV4 only (d_qk={q.shape[-1]})")
    if output.shape[-1] != 512:
        raise ValueError(f"{ENV}: d_v must be 512, got {output.shape[-1]}")
    pbsx = int(extra_kv.shape[1])
    if pbsx not in _hits:
        # one marker per page size: the window's evidence that the route engaged
        _hits[pbsx] = True
        logger.warning("DSV41_PCE_HIT extra_page_block_size=%d rows=%d", pbsx, int(q.shape[0]))
    _module().dsv41_pce_paged_attention(
        q, kv_64, indices, output, out_lse, float(sm_scale), int(model_type),
        topk_length, attn_sink, extra_kv, extra_indices, extra_topk_length,
    )
