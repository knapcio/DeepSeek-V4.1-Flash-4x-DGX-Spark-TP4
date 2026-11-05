"""DSV41_KV_LAYOUT=v41c: SM120 sparse-MLA readers for V4.1 FP4 compressed KV records (288 B per token).

Installed next to flash_mla_sm120.py as sglang/kernels/ops/attention/dsv41_v41c.py. With the variable unset or `v4`
nothing here runs: the adapter only routes an extra cache whose records are 288 bytes, which only exists with v41c.

Records (SGLang KVLayout.V41_FP4, page of P tokens): P x 256 bytes of e2m1 codes (even index in the low nibble), then
P x 32 bytes of e4m3 scales (one per 16 values); the model's own rounding (fake_quant_compressed_kv). R1 (owner 10-27):
every reader turns a record into the 584-byte V4 row SGLang's V4 store would have written for the same values
(ue8m0 per 64 of max(1e-4, amax) / 448, values times the exact power-of-two inverse, RN e4m3 clipped to +-448, rope as
bf16), so the attention math sees v2.6's bytes. The v41w window (2026-10-27 12:45) confirmed SGLang's writer equals the
model's rounding byte for byte on sm_121a and this transcode equals SGLang's V4 rows (c1).

Readers (DSV41_V41C_READER):

group (default, 2026-10-28 fast verify): rows <= 96 (every decode / DSpark verify graph, bs 1..16) transcode the records
  a call reads ONCE into a persistent V4 scratch of 64-token pages and run the STOCK FlashInfer kernels on it (decode-dsv4
  at <= 64 rows, the prefill kernel above), exactly v2.6's call with the scratch as the extra cache:
  - once per layer GROUP, not per layer: the layers between two kv_source / index_source layers read the same extra
    cache with the same top-k (V4.1: only kv_source layers write it, only index_source layers refill the top-k
    buffer), so the scratch and the rewritten indices of the group's first layer are reused by the rest (8 transcodes
    per step instead of 38). Layer ids come from adapter/v41c_pool.py (tag_layer); without a tag every call transcodes;
  - only live entries (index >= 0 and position < extra_topk_length); everything else stays -1 (the kernel masks it);
  - dedupe (DSV41_V41C_DEDUP_ROWS, default 6 = DSpark's 5 drafts + 1, from DSV41_V41C_DEDUP_MIN_ROWS rows, default 24):
    the rows of one verify block share one V4 row per distinct token (sorted, ranked, scattered back; static shapes);
  - one fused Triton launch per group (plan + transcode), two with dedupe; no host sync, shapes from (rows, K) only.
  The math warps read the same V4 bytes as the IO-warp reader for every live entry (same rule, tests/test_v41c_fast.py);
  a masked entry contributes exactly 0 either way.
native (2026-10-29, off by default; DSV41_V41C_READER=native): no transcode pass. The kernels read the 288-byte records
  themselves and build the V4 operand bytes in shared memory (diagnostics/v41n-native-reader-20261029/DESIGN.md):
  - decode (<= 64 rows): v41n_decode_dsv4 = the stock decode-dsv4 kernel + converter warps; the IO warp bulk-copies the
    raw records into the V4 buffer rows, the converter warps convert them in place while the math warps work on the
    other buffer, the math warps run the stock code on the stock bytes. Same launcher, split heuristic and AutoTuner
    tactic as the stock call (the tactic changes the reduction order);
  - prefill / verify (> 64 rows), captured verify graphs (c16) and small eager calls: v41n_prefill = FlashInfer's
    dual-cache MG prefill (generated from the vendored source by gen_v41n_prefill.py) reading the FP4 pages in place:
    the math warps convert their own records at the start of an extra tile, rope from a shared buffer. No 64-token page
    split of the extra cache, no scratch, no host sync;
  - eager prefill chunks (>= DSV41_V41N_PAGE_MIN_ROWS rows, default 512; DSV41_V41N_PREFILL=auto|kernel|page): the
    page pass (native_page_view): every touched record converted ONCE per layer group into compacted V4 sub-pages,
    then the stock prefill. The in-kernel reader converts per (row, entry), which costs O(rows x K) on an eager chunk
    (dbd4644: +12 % on an 8192-row chunk); the page pass costs O(unique records). Scratch capped
    (DSV41_V41N_PAGE_CAP_MB, default 1024): above it the in-kernel path runs. Shapes it does not serve (not 8 / 16 heads, SWA top-k != 128, missing
    lengths, FP4 pages other than 128 / 256) take the group path.
swan (DSV41_SWA_NATIVE_DECODE=1, 2026-11-02, off by default; independent of DSV41_KV_LAYOUT): the decode (<= 64 rows)
  reads SGLang's SWA pool in its own 256-token footer pages instead of the canary's per-call 64-token page split (mask
  fill + page mark + page split, 43 chains per c1 step; the 10-30 probe priced one extra chain at 0.11-0.14 ms per step).
  The decode kernel's main-section address math is generic in the page size, so at PAGE_BLOCK_SIZE 256 it bulk-copies
  the bytes the split copy would hold into the same shared-memory rows. Stock layout (V4 extra cache or none):
  swan_decode_dsv4 = FlashInfer's launcher with the stock kernel body at 256-token pages; v41n (FP4 extra cache, reader
  native): v41n_decode_dsv4 at 256-token pages. Both with the stock call's tactic (native_tactic). Served shapes:
  SWAN_HEADS x SWAN_TOPK; anything else, AutoTuner tuning mode, the group / io readers and > 64 rows take the split.
io (f092906): decode (<= 64 query rows): FlashInfer decode-dsv4 with the transcode in its IO warp (runtime/flashinfer_v41c,
  EXTRA_FP4), compiled at image build next to the image's own approved sparse_mla_sm120 closure (the K8 mechanism), .so
  sha256 pinned. The split-K tactic is resolved exactly as stock: the same AutoTuner op name, cache keys, runner class
  name and hot cache, so a kept tactic table gives the stock tactic for each shape (the tactic changes the reduction
  order). Measured 2026-10-28: the IO warp transcodes every record once per (query row, 16-head block, layer), which made
  decode / verify 13-17 % slower than v2.6 (diagnostics/ds-v41kv-20261027/SPEED-ANALYSIS.md).
  prefill / wide verify (> 64 rows): the prefill kernel's IO warps are register-capped and its math warps read rope from
  global memory, so records are transcoded into a V4 scratch first, then the STOCK prefill reads 64-token V4 pages:
  rows <= 96 (the captured bs 11..16 verify graphs): one V4 row per (query row, top-k entry), static shape, slot
    r * K + j; indices rewritten to those slots (holes stay -1; entry order unchanged, so the math is v2.6's);
rows > 96 (eager prefill chunks): only the touched 64-token sub-pages, compacted; indices remapped
  map[idx // 64] * 64 + idx % 64; scratch of touched sub-pages x 37,440 B, one host sync per plan. The group reader
  plans once per layer group (8 per chunk instead of 38) on the live entries only (group_page_view).
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

ENV = "DSV41_KV_LAYOUT"
DIR_ENV = "DSV41_V41C_DIR"
DEFAULT_DIR = "/opt/dsv41/runtime/flashinfer_v41c"
PCE_GEN_DEFAULT = "/opt/dsv41/runtime/flashinfer_pce/gen_pce_sources.py"
BPT_FP4, DATA, SCALE = 288, 256, 32
BPT_V4, ROW, FOOT = 584, 576, 8
SUB = 64                                 # V4 page size the stock kernels read
SUB_PAGE_BYTES = -(-SUB * BPT_V4 // ROW) * ROW   # 37,440
TOKEN_MODE_MAX_ROWS = 96
READER_ENV = "DSV41_V41C_READER"         # group (default) | io (f092906) | native (v41n)
DEDUP_ROWS_ENV = "DSV41_V41C_DEDUP_ROWS"  # rows per dedupe block (0 / 1 = off)
DEDUP_MIN_ENV = "DSV41_V41C_DEDUP_MIN_ROWS"
DEDUP_MAX_ENTRIES = 4096                 # one dedupe program sorts rows_per_block x K <= 4096 entries
TOK_R = 4                                # records per Triton transcode program (sm_121a: 96-128 regs, no spills)
SOURCES = ("csrc/v41c_decode_dsv4.cu", "csrc/v41c_binding.cu", "csrc/v41n_decode_dsv4.cu", "csrc/v41n_prefill.cu",
           "csrc/v41n_diag.cu", "csrc/swan_decode_dsv4.cu")
PATCHED = "include/flashinfer_v41c/sparse_mla_sm120/decode_dsv4_kernel.cuh"
# headers that are not FlashInfer files: the v41n converter, the generated native prefill kernel (gen_v41n_prefill.py)
NEW_HEADERS = ("include/flashinfer_v41c/sparse_mla_sm120/v41n_convert.cuh",
               "include/flashinfer_v41c/sparse_mla_sm120/v41n_prefill_kernel.cuh")
_BI = 64


# ------------------------------------------------------------------------------------------------ flag
@lru_cache(maxsize=1)
def enabled() -> bool:
    raw = os.environ.get(ENV, "").strip()
    if raw in ("", "v4"):
        return False
    if raw == "v41c":
        logger.warning("%s=v41c: compressed KV records are V4.1 FP4 (288 B); readers transcode to V4 rows", ENV)
        return True
    raise ValueError(f"{ENV} must be unset, v4 or v41c, got {raw!r}")


@lru_cache(maxsize=1)
def reader() -> str:
    """group (default): group-shared transcode + stock kernels; io: f092906's IO-warp decode / per-layer token mode;
    native: v41n in-kernel FP4 readers (no transcode pass)."""
    raw = os.environ.get(READER_ENV, "").strip()
    if raw in ("", "group"):
        return "group"
    if raw == "io":
        return "io"
    if raw == "native":
        return "native"
    raise ValueError(f"{READER_ENV} must be unset, group, io or native, got {raw!r}")


@lru_cache(maxsize=1)
def dedup_rule() -> tuple:
    """(rows per dedupe block, minimum rows). Block 6 = one DSpark verify request (5 drafts + 1); any block is exact."""
    def _int(name, default):
        raw = os.environ.get(name, "").strip()
        try:
            val = int(raw) if raw else default
        except ValueError:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from None
        if val < 0:
            raise ValueError(f"{name} must be >= 0, got {val}")
        return val
    return _int(DEDUP_ROWS_ENV, 6), _int(DEDUP_MIN_ENV, 24)


def dedup_block(rows: int, k: int) -> int:
    """Rows per dedupe program for a (rows, K) call, or 0 (no dedupe). Depends on shapes only."""
    g, min_rows = dedup_rule()
    if g <= 1 or rows < max(min_rows, 2) or rows % g or g * k > DEDUP_MAX_ENTRIES:
        return 0
    return g


def is_fp4(extra) -> bool:
    """An extra cache in V4.1 FP4 records (uint8 view, last dim 288). Refuses one without the flag."""
    if extra is None or extra.shape[-1] != BPT_FP4:
        return False
    if not enabled():
        raise RuntimeError(f"an extra cache with {BPT_FP4}-byte records reached the reader without {ENV}=v41c")
    return True


# ------------------------------------------------------------------------------------------------ identity
def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pce_generator():
    for p in (Path(__file__).resolve().parent.parent / "flashinfer_pce" / "gen_pce_sources.py", Path(PCE_GEN_DEFAULT)):
        if p.is_file():
            spec = importlib.util.spec_from_file_location("dsv41_pce_gen_for_v41c", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise RuntimeError(f"{ENV}: gen_pce_sources.py (the approved FlashInfer closure) not found")


def vendored(d: Path) -> dict:
    d = Path(d)
    files = [d / s for s in SOURCES] + sorted(p for p in (d / "include").rglob("*") if p.is_file())
    return {p.relative_to(d).as_posix(): _sha(p) for p in files if "__pycache__" not in p.parts}


def check_sources(d: Path, csrc: Path, include: Path) -> dict:
    """Vendored sources == VENDORED.json; every vendored header but the patched one == the approved FlashInfer file;
    the installed FlashInfer closure == the approved closure. Returns {vendored, closure, module_tag}."""
    d = Path(d)
    gen = _pce_generator()
    approved = json.loads((Path(gen.__file__).with_name("APPROVED_CLOSURE.json")).read_text())["files"]
    want = json.loads((d / "VENDORED.json").read_text())["files"]
    got = vendored(d)
    if got != want:
        bad = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
        raise RuntimeError(f"{ENV}: vendored v41c sources differ from VENDORED.json: {bad[:6]}")
    for rel, sha in got.items():
        if rel.startswith("include/flashinfer_v41c/sparse_mla_sm120/") and rel != PATCHED and rel not in NEW_HEADERS:
            key = "include/flashinfer/attention/sparse_mla_sm120/" + rel.split("sparse_mla_sm120/", 1)[1]
            if approved.get(key) != sha:
                raise RuntimeError(f"{ENV}: vendored {rel} is not the approved FlashInfer {key}")
    try:
        clo = gen.closure(Path(csrc), Path(include))
        gen.check_closure(clo, approved)
    except gen.ClosureError as e:
        raise RuntimeError(f"{ENV}: installed FlashInfer is not the approved closure ({e})") from None
    h = hashlib.sha256()
    for k in sorted(got):
        h.update(f"v {k}\0{got[k]}\0".encode())
    for k in sorted(clo):
        h.update(f"c {k}\0{clo[k]}\0".encode())
    return {"vendored": got, "closure": clo, "module_tag": h.hexdigest()[:12]}


def module_name(tag: str) -> str:
    return f"dsv41_v41c_decode_dsv4_{tag}"


def check_module(d: Path, ident: dict) -> Path:
    d = Path(d)
    try:
        mj = json.loads((d / "lib" / "module.json").read_text())
    except (OSError, ValueError) as e:
        raise RuntimeError(f"{ENV}: no prebuilt module record {d / 'lib' / 'module.json'} ({e})") from None
    if mj.get("module_tag") != ident["module_tag"] or mj.get("name") != module_name(ident["module_tag"]):
        raise RuntimeError(f"{ENV}: prebuilt module {mj.get('name')} is not {module_name(ident['module_tag'])}")
    so = d / "lib" / f"{mj['name']}.so"
    if not so.is_file() or _sha(so) != mj.get("so_sha256"):
        raise RuntimeError(f"{ENV}: {so} missing or its sha256 is not module.json's")
    return so


def _flashinfer_dirs():
    from flashinfer.jit import env as jit_env

    return Path(jit_env.FLASHINFER_CSRC_DIR), Path(jit_env.FLASHINFER_INCLUDE_DIR)


@lru_cache(maxsize=1)
def _module():
    import tvm_ffi

    d = Path(os.environ.get(DIR_ENV, DEFAULT_DIR))
    ident = check_sources(d, *_flashinfer_dirs())
    so = check_module(d, ident)
    mod = tvm_ffi.load_module(str(so))
    logger.warning("%s=v41c decode module ready: tag=%s so_sha256=%s", ENV, ident["module_tag"], _sha(so))
    return mod


def build(d: Path) -> dict:
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    d = Path(d)
    ident = check_sources(d, *_flashinfer_dirs())
    flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    if not any("sm_121a" in f for f in flags):
        raise RuntimeError(f"{ENV}: nvcc flags {flags} do not target sm_121a (set FLASHINFER_CUDA_ARCH_LIST=12.1a)")
    name = module_name(ident["module_tag"])
    spec = gen_jit_spec(name, [d / s for s in SOURCES], extra_cuda_cflags=flags, extra_include_paths=[d / "include"])
    spec.build()
    (d / "lib").mkdir(exist_ok=True)
    so = d / "lib" / f"{name}.so"
    shutil.copyfile(spec.jit_library_path, so)
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()
    mj = {"name": name, "module_tag": ident["module_tag"], "so_sha256": _sha(so), "nvcc_flags": flags,
          "nvcc": nvcc[-1] if nvcc else None}
    (d / "lib" / "module.json").write_text(json.dumps(mj, indent=1, sort_keys=True) + "\n")
    return mj


# ------------------------------------------------------------------------------------------------ decode
@lru_cache(maxsize=1)
def _runner():
    from flashinfer.mla import _sparse_mla_sm120 as S

    base = S._get_sparse_mla_decode_dsv4_module().runner_cls
    mod = _module()

    class Runner(base):
        def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
            q, indices, mid_out, mid_lse, output, out_lse = inputs[:6]
            topk_length, attn_sink, extra_indices, extra_topk_length = (
                inputs[6:10] if len(inputs) > 6 else (kwargs.get("topk_length"), kwargs.get("attn_sink"),
                                                       kwargs.get("extra_indices"), kwargs.get("extra_topk_length")))
            topk = indices.shape[-1]
            extra_topk = extra_indices.shape[-1] if extra_indices is not None else 0
            num_splits = (topk + _BI - 1) // _BI + (extra_topk + _BI - 1) // _BI
            mod.v41c_decode_dsv4(q, kwargs["kv_cache"], indices, mid_out, mid_lse, output, out_lse, num_splits,
                                 kwargs["sm_scale"], topk_length, attn_sink, kwargs.get("extra_kv_cache"),
                                 extra_indices, extra_topk_length, tactic if tactic > 0 else -1)
            return output

    # the AutoTuner keys on the runner class name: the stock name keeps the stock tactic table
    Runner.__name__, Runner.__qualname__ = base.__name__, base.__qualname__
    return Runner()


@lru_cache(maxsize=1)
def group_receipt() -> str:
    """First group-reader call (eager warm-up, before any capture): the same identity check and receipt line as the
    io reader (vendored sources, approved FlashInfer closure the stock kernels come from, .so sha256), plus the
    reader line. The prebuilt module is loaded, not used."""
    _module()
    g, min_rows = dedup_rule()
    line = (f"{ENV}=v41c reader=group: group-shared transcode + stock FlashInfer kernels, dedupe {g} rows "
            f"from {min_rows} rows")
    logger.warning(line)
    return line


def decode(q, kv_cache, indices, mid_out, mid_lse, output, out_lse, sm_scale, *, topk_length, attn_sink,
           extra_kv, extra_indices, extra_topk_length):
    """<= 64 rows with an FP4 extra cache.

    group: the group-shared V4 scratch (group_view) as the extra cache of v2.6's own call, FlashInfer's stock
    paged-attention entry (decode-dsv4, the stock runner, the stock tactic for the shape).
    io: FlashInfer's sparse_mla_sm120_decode_dsv4 wrapper logic with the v41c runner (same tactic resolution).
    native: the same wrapper logic with the v41n runner (native_decode)."""
    if reader() == "native":
        return native_decode(q, kv_cache, indices, mid_out, mid_lse, output, out_lse, sm_scale,
                             topk_length=topk_length, attn_sink=attn_sink, extra_kv=extra_kv,
                             extra_indices=extra_indices, extra_topk_length=extra_topk_length)
    if reader() == "group":
        from flashinfer.mla._sparse_mla_sm120 import _sparse_mla_sm120_paged_attention

        group_receipt()
        extra_v4, new_idx = group_view(extra_kv, extra_indices, extra_topk_length)
        _sparse_mla_sm120_paged_attention(q, kv_cache, indices, output, out_lse, sm_scale, d_v=output.shape[-1],
                                          topk_length=topk_length, attn_sink=attn_sink, extra_kv_cache=extra_v4,
                                          extra_indices=new_idx, extra_topk_length=extra_topk_length,
                                          mid_out=mid_out, mid_lse=mid_lse)
        return output
    from flashinfer.autotuner import AutoTuner
    from flashinfer.mla import _sparse_mla_sm120 as S

    runner = _runner()
    inputs = [q, indices, mid_out, mid_lse, output, out_lse, topk_length, attn_sink, extra_indices, extra_topk_length]
    kw = {"sm_scale": sm_scale, "kv_cache": kv_cache, "extra_kv_cache": extra_kv}
    extra_topk = extra_indices.shape[-1] if extra_indices is not None else 0
    num_splits = (indices.shape[-1] + _BI - 1) // _BI + (extra_topk + _BI - 1) // _BI
    hot_key = (S._decode_dsv4_map_to_token_bucket(q.shape[0]), q.shape[1], indices.shape[-1], extra_topk, num_splits,
               runner.get_cache_key_extras(inputs))
    tuner = AutoTuner.get()
    if not tuner.is_tuning_mode:
        cached = S._decode_dsv4_hot_cache.get(hot_key)
        if cached is not None:
            runner(inputs=inputs, tactic=cached, **kw)
            return output
    S._decode_dsv4_maybe_load_cache()
    chosen, tactic = tuner.choose_one("sparse_mla_sm120_decode_dsv4", [runner], S._decode_dsv4_tuning_config(),
                                      inputs, **kw)
    if int(tactic) > 0:
        S._decode_dsv4_hot_cache[hot_key] = int(tactic)
    chosen(inputs=inputs, tactic=tactic, **kw)
    return output


# ------------------------------------------------------------------------------------------------ transcode
def _torch():
    import torch

    return torch


def decode_records(data, scales):
    """[n, 256] + [n, 32] u8 -> [n, 512] fp32 (exact: e2m1 x e4m3 has <= 6 significant bits)."""
    torch = _torch()
    e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=data.device)
    codes = torch.stack((data & 15, data >> 4), dim=-1).reshape(data.shape[0], 512).long()
    val = e2m1[codes & 7]
    val = torch.where((codes & 8) != 0, -val, val)
    s = scales.contiguous().view(torch.float8_e4m3fn).float()
    return (val.reshape(-1, 32, 16) * s.reshape(-1, 32, 1)).reshape(-1, 512)


def v4_rows(values):
    """fp32 [n, 512] -> (data [n, 576] u8, scales [n, 8] u8): SGLang's V4 store rule (c1 / c2)."""
    torch = _torch()
    v = values.float()
    n = v.shape[0]
    nope = v[:, :448].reshape(n, 7, 64)
    amax = nope.abs().amax(-1, keepdim=True)
    s = torch.clamp_min(amax, 1e-4) / 448.0
    bits = s.contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    inv = ((254 - e) << 23).view(torch.float32)
    q = (nope * inv).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8).reshape(n, 448)
    rope = v[:, 448:].to(torch.bfloat16).contiguous().view(torch.uint8).reshape(n, 128)
    sc = torch.zeros(n, FOOT, dtype=torch.uint8, device=v.device)
    sc[:, :7] = e.reshape(n, 7).to(torch.uint8)
    return torch.cat([q, rope], dim=-1), sc


def fp4_page_view(extra):
    """(base u8 tensor [N, page_stride], pbs, page_stride) of a uint8 FP4 extra cache view."""
    torch = _torch()
    n, pbs = int(extra.shape[0]), int(extra.shape[1])
    stride = int(extra.stride(0))
    return torch.as_strided(extra, (n, stride), (stride, 1)), pbs, stride


def gather_records(extra, tokens):
    """FP4 (data, scales) of token ids `tokens` (int64, >= 0) from a page view."""
    raw, pbs, _ = fp4_page_view(extra)
    page, loc = tokens // pbs, tokens % pbs
    rows = raw[page]
    torch = _torch()
    dofs = (loc * DATA).unsqueeze(-1) + torch.arange(DATA, device=raw.device)
    sofs = (pbs * DATA + loc * SCALE).unsqueeze(-1) + torch.arange(SCALE, device=raw.device)
    return rows.gather(1, dofs), rows.gather(1, sofs)


def write_v4_rows(dst_pages, slots, data, scales):
    """Write V4 rows into 64-token V4 pages (dst [P, 37440] u8) at slots."""
    torch = _torch()
    page, loc = slots // SUB, slots % SUB
    flat = dst_pages.view(-1)
    base = page * SUB_PAGE_BYTES
    d_off = (base + loc * ROW).unsqueeze(-1) + torch.arange(ROW, device=flat.device)
    s_off = (base + SUB * ROW + loc * FOOT).unsqueeze(-1) + torch.arange(FOOT, device=flat.device)
    flat[d_off.reshape(-1)] = data.reshape(-1)
    flat[s_off.reshape(-1)] = scales.reshape(-1)


def transcode_tokens_torch(extra, tokens, dst_pages, slots):
    """Reference (and CPU) path: records of `tokens` -> V4 rows at `slots` of dst_pages."""
    if tokens.numel() == 0:
        return
    data, scales = gather_records(extra, tokens.long())
    d4, s4 = v4_rows(decode_records(data, scales))
    write_v4_rows(dst_pages, slots.long(), d4, s4)


def sub_pages_view(buf, n_pages):
    """[n_pages, 64, 1, 584] view of a [>= n_pages, 37440] u8 buffer (the stock kernels' 64-token page ABI)."""
    torch = _torch()
    return torch.as_strided(buf, (n_pages, SUB, 1, BPT_V4), (SUB_PAGE_BYTES, BPT_V4, BPT_V4, 1))


def plan_token_mode(eidx):
    """rows <= 96: slot r * K + j for every entry; holes stay -1 in new_idx. Returns (tokens, slots, new_idx, n_pages).

    Static shapes only (these run inside the captured verify graphs: no boolean-mask indexing, no host sync). A hole
    transcodes token 0 into its own slot r * K + j, which no index references (new_idx is -1 there)."""
    torch = _torch()
    rows, k = eidx.shape
    slot = torch.arange(rows * k, device=eidx.device, dtype=torch.int32).reshape(rows, k)
    valid = eidx >= 0
    new_idx = torch.where(valid, slot, torch.full_like(slot, -1))
    tokens = torch.where(valid, eidx, torch.zeros_like(eidx))
    return tokens.reshape(-1), slot.reshape(-1), new_idx, -(-rows * k // SUB)


def plan_page_mode(eidx, n_tokens_pool):
    """rows > 96: touched 64-token sub-pages compacted. Returns (tokens, slots, new_idx, n_pages)."""
    torch = _torch()
    n_sub = -(-n_tokens_pool // SUB)
    valid = eidx >= 0
    sub = torch.where(valid, eidx // SUB, torch.zeros_like(eidx)).long()
    mark = torch.zeros(n_sub, dtype=torch.int32, device=eidx.device)
    mark[sub[valid]] = 1
    cmap = torch.cumsum(mark, 0, dtype=torch.int32) - 1
    touched = torch.nonzero(mark, as_tuple=False).flatten()
    n_pages = int(touched.numel())                    # eager prefill only: one host sync per call
    new_idx = torch.where(valid, cmap[sub] * SUB + eidx % SUB, torch.full_like(eidx, -1)).to(torch.int32)
    tokens = (touched.unsqueeze(-1) * SUB + torch.arange(SUB, device=eidx.device)).flatten()
    slots = (torch.arange(n_pages, device=eidx.device).unsqueeze(-1) * SUB
             + torch.arange(SUB, device=eidx.device)).flatten()
    keep = tokens < n_tokens_pool
    return tokens[keep], slots[keep], new_idx, n_pages


_scratch: dict = {}


def _token_scratch(dev, n_pages, k):
    """Persistent token-mode scratch (static address for captured verify graphs).

    Sized once for the largest token-mode call at this K (TOKEN_MODE_MAX_ROWS rows), so no captured graph ever holds
    an address that a later, larger capture would reallocate."""
    torch = _torch()
    buf = _scratch.get(dev)
    if buf is None or buf.shape[0] < n_pages:
        size = max(n_pages, -(-TOKEN_MODE_MAX_ROWS * k // SUB))
        with torch.inference_mode(False):
            buf = torch.zeros(size, SUB_PAGE_BYTES, dtype=torch.uint8, device=dev)
        _scratch[dev] = buf
    return buf


def transcode_for_prefill(extra, eidx, transcode=None):
    """(V4 64-token page view, rewritten indices) for the stock prefill from an FP4 extra cache."""
    torch = _torch()
    if transcode is None:
        transcode = transcode_tokens_triton if extra.is_cuda else transcode_tokens_torch
    rows = eidx.shape[0]
    n_pool = int(extra.shape[0]) * int(extra.shape[1])
    if rows <= TOKEN_MODE_MAX_ROWS:
        tokens, slots, new_idx, n_pages = plan_token_mode(eidx)
        buf = _token_scratch(eidx.device, n_pages, eidx.shape[1])
    else:
        tokens, slots, new_idx, n_pages = plan_page_mode(eidx, n_pool)
        buf = torch.zeros(max(n_pages, 1), SUB_PAGE_BYTES, dtype=torch.uint8, device=eidx.device)
    transcode(extra, tokens, buf, slots)
    return sub_pages_view(buf, max(n_pages, 1)), new_idx


# Triton kernel: one program per record. Same rule as v4_rows (division by IEEE div_rn, RN e4m3 via the hardware
# satfinite conversion after an explicit clip); qualified against the torch reference on the GPU (mini-window).
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _fp4_to_v4_kernel(src, src_stride, PBS: tl.constexpr, tok_ptr, dst, slot_ptr, n):
        i = tl.program_id(0)
        if i >= n:
            return
        tok = tl.load(tok_ptr + i).to(tl.int64)
        slot = tl.load(slot_ptr + i).to(tl.int64)
        page_base = src + (tok // PBS) * src_stride
        loc = tok % PBS
        k = tl.arange(0, 512)
        byte = tl.load(page_base + loc * 256 + k // 2).to(tl.int32)
        code = tl.where(k % 2 == 0, byte & 15, byte >> 4)
        m = code & 7
        mag = tl.where(m < 4, m.to(tl.float32) * 0.5,
                       tl.where(m == 4, 2.0, tl.where(m == 5, 3.0, tl.where(m == 6, 4.0, 6.0))))
        val = tl.where((code & 8) != 0, -mag, mag)
        sbyte = tl.load(page_base + PBS * 256 + loc * 32 + k // 16)
        s = sbyte.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        v = tl.reshape(val * s, (8, 64))
        amax = tl.max(tl.abs(v), axis=1)
        sc = tl.math.div_rn(tl.maximum(amax, 1e-4), 448.0)
        bits = sc.to(tl.int32, bitcast=True)
        e = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        inv = ((254 - e) << 23).to(tl.float32, bitcast=True)
        q = tl.minimum(tl.maximum(v * inv[:, None], -448.0), 448.0).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        dpage = dst + (slot // 64) * 37440
        dl = slot % 64
        t = tl.arange(0, 8)[:, None]
        c = tl.arange(0, 64)[None, :]
        tl.store(dpage + dl * 576 + t * 64 + c, q, mask=t < 7)
        rope = tl.reshape(v, (512,))
        rb = rope.to(tl.bfloat16).to(tl.uint16, bitcast=True)
        lo = (rb & 255).to(tl.uint8)
        hi = (rb >> 8).to(tl.uint8)
        kk = tl.arange(0, 512)
        rmask = kk >= 448
        tl.store(dpage + dl * 576 + 448 + (kk - 448) * 2, lo, mask=rmask)
        tl.store(dpage + dl * 576 + 448 + (kk - 448) * 2 + 1, hi, mask=rmask)
        tt = tl.arange(0, 8)
        tl.store(dpage + 64 * 576 + dl * 8 + tt, tl.where(tt < 7, e, 0).to(tl.uint8))

    def transcode_tokens_triton(extra, tokens, dst_pages, slots):
        n = int(tokens.numel())
        if n == 0:
            return
        raw, pbs, stride = fp4_page_view(extra)
        _fp4_to_v4_kernel[(n,)](raw, stride, pbs, tokens.to(_torch().int32).contiguous(), dst_pages,
                                slots.to(_torch().int32).contiguous(), n)
except ImportError:  # CPU-only environments (offline tests): the torch reference is the path
    def transcode_tokens_triton(extra, tokens, dst_pages, slots):
        raise RuntimeError(f"{ENV}: triton is required on the GPU path")


# ------------------------------------------------------------------------------------------------ group reader
# Layer tags: adapter/v41c_pool.py wraps DeepseekV4AttnBackend._forward_attention and calls tag_layer(layer_id,
# group_start) before every ratio-1/2 layer's attention (clear_tag after). group_start = the latest kv_source /
# index_source layer of the same ratio at or before the layer: from there to the next one, the extra cache and its
# top-k buffer are not written, so a transcode made at group_start serves every later layer of the group.
_tag: list = [None]
_group: dict = {}
stats = {"plan": 0, "reuse": 0, "dedup": 0, "page": 0, "native_decode": 0, "native_prefill": 0,
         "native_fallback": 0, "native_head_skip": 0, "native_page": 0, "native_page_cap": 0,
         "swan_decode": 0, "swan_v41n": 0, "swan_fallback": 0}


def tag_layer(layer_id: int, group_start: int) -> None:
    _tag[0] = (int(layer_id), int(group_start))


def clear_tag() -> None:
    _tag[0] = None


def reset_group() -> None:
    _group.clear()
    _tag[0] = None


def _live(eidx, topk_len):
    """Entries the kernel can read: index >= 0 and position < extra_topk_length (the kernel masks every other one)."""
    torch = _torch()
    rows, k = eidx.shape
    live = eidx >= 0
    if topk_len is not None:
        live = live & (torch.arange(k, device=eidx.device).unsqueeze(0) < topk_len.reshape(rows, 1))
    return live


def plan_group_torch(eidx, topk_len):
    """No dedupe: (tokens, slots, new_idx). Slot r * K + j per live entry, -1 elsewhere (tokens too). Static shapes."""
    torch = _torch()
    rows, k = eidx.shape
    live = _live(eidx, topk_len)
    slot = torch.arange(rows * k, device=eidx.device, dtype=torch.int32).reshape(rows, k)
    minus = torch.full_like(slot, -1)
    return (torch.where(live, eidx.to(torch.int32), minus).reshape(-1), slot.reshape(-1),
            torch.where(live, slot, minus))


def plan_dedup_torch(eidx, topk_len, g):
    """Dedupe over blocks of g rows: (tokens, slots, new_idx, counts). Static shapes.

    Per block b (n = g * K entries): live entries sorted by (token, position); the r-th distinct token gets slot
    b * n + r; every live entry of the block is rewritten to its token's slot. tokens[b * n + r] = that token for
    r < counts[b], -1 after. The Triton kernel computes exactly this (same sort key, same ranks)."""
    torch = _torch()
    rows, k = eidx.shape
    nb, n = rows // g, g * k
    live = _live(eidx, topk_len).reshape(nb, n)
    key = torch.where(live, eidx.reshape(nb, n).long(), torch.full_like(live, -1, dtype=torch.long))
    pos = torch.arange(n, device=eidx.device, dtype=torch.long)
    s, _ = torch.sort(((key + 1) << 13) | pos, dim=-1)
    skey, spos = (s >> 13) - 1, s & 8191
    prev = torch.cat([torch.full((nb, 1), -2, dtype=torch.long, device=eidx.device), skey[:, :-1]], dim=1)
    first = (skey != prev) & (skey >= 0)
    rank = torch.cumsum(first.to(torch.int32), dim=-1, dtype=torch.int32) - 1
    slot = torch.arange(nb, device=eidx.device, dtype=torch.int32).unsqueeze(1) * n + rank
    new_flat = torch.empty((nb, n), dtype=torch.int32, device=eidx.device)
    new_flat.scatter_(1, spos, torch.where(skey >= 0, slot, torch.full_like(slot, -1)))
    toks = torch.full((nb, n + 1), -1, dtype=torch.int32, device=eidx.device)
    toks.scatter_(1, torch.where(first, rank.long(), torch.full_like(spos, n)), skey.to(torch.int32))
    slots = torch.arange(nb * n, device=eidx.device, dtype=torch.int32)
    return toks[:, :n].reshape(-1), slots, new_flat.reshape(rows, k), first.sum(dim=-1, dtype=torch.int32)


def _transcode_list_torch(extra, tokens, buf, slots):
    """CPU reference: transcode tokens >= 0 into slots (boolean indexing: CPU only, never on a captured path)."""
    keep = tokens >= 0
    transcode_tokens_torch(extra, tokens[keep].long(), buf, slots[keep].long())


def _group_sig(extra, eidx, topk_len, mode, buf):
    return (str(eidx.device), int(extra.data_ptr()), tuple(extra.shape), tuple(extra.stride()), int(eidx.data_ptr()),
            tuple(eidx.shape), tuple(eidx.stride()), None if topk_len is None else int(topk_len.data_ptr()), mode,
            None if buf is None else int(buf.data_ptr()))


def _group_hit(sig):
    """Consume the layer tag; (tag, cached (view, new_idx) or None). Reuse needs a tagged layer that is not its
    group's start, the previous call at the layer before it, the same group start and the same tensors / shapes."""
    tag, _tag[0] = _tag[0], None
    if (tag is not None and tag[0] != tag[1] and _group.get("sig") == sig and _group.get("start") == tag[1]
            and _group.get("layer") == tag[0] - 1):
        _group["layer"] = tag[0]
        stats["reuse"] += 1
        return tag, (_group["view"], _group["new_idx"])
    _group.clear()                 # release the previous group's tensors before this plan allocates
    return tag, None


def _group_keep(tag, sig, view, new_idx, **extra_refs):
    _group.update(sig=sig, layer=None if tag is None else tag[0], start=None if tag is None else tag[1], view=view,
                  new_idx=new_idx, **extra_refs)
    stats["plan"] += 1


def _rows_2d(eidx, topk_len):
    torch = _torch()
    eidx = eidx.reshape(eidx.shape[0], eidx.shape[-1])
    if eidx.dtype != torch.int32:
        eidx = eidx.to(torch.int32)
    eidx = eidx.contiguous()
    if topk_len is not None:
        topk_len = topk_len.reshape(-1)
        if topk_len.shape[0] != eidx.shape[0]:
            raise ValueError(f"{ENV}: extra_topk_length has {topk_len.shape[0]} rows, indices {eidx.shape[0]}")
        topk_len = topk_len.to(torch.int32).contiguous()
    return eidx, topk_len


def group_view(extra, eidx, topk_len=None):
    """(V4 64-token page view, rewritten int32 indices [rows, K]) for <= 96 rows, reused across a layer group.

    Shapes depend on (rows, K) only, no host sync: safe inside the captured decode / verify graphs. The scratch is the
    persistent token-mode buffer (one static address for every graph, sized once for 96 rows)."""
    eidx, topk_len = _rows_2d(eidx, topk_len)
    rows, k = eidx.shape
    if rows > TOKEN_MODE_MAX_ROWS:
        raise ValueError(f"{ENV}: group_view takes <= {TOKEN_MODE_MAX_ROWS} rows, got {rows}")
    n_pages = -(-rows * k // SUB)
    buf = _token_scratch(eidx.device, n_pages, k)
    g = dedup_block(rows, k)
    sig = _group_sig(extra, eidx, topk_len, g, buf)
    tag, hit = _group_hit(sig)
    if hit is not None:
        return hit
    if extra.is_cuda:
        new_idx = (_dedup_triton(extra, eidx, topk_len, buf, g) if g else _plan_group_triton(extra, eidx, topk_len, buf))
    else:
        if g:
            tokens, slots, new_idx, _ = plan_dedup_torch(eidx, topk_len, g)
        else:
            tokens, slots, new_idx = plan_group_torch(eidx, topk_len)
        _transcode_list_torch(extra, tokens, buf, slots)
    view = sub_pages_view(buf, n_pages)
    _group_keep(tag, sig, view, new_idx)
    stats["dedup"] += bool(g)
    return view, new_idx


def group_page_view(extra, eidx, topk_len=None):
    """> 96 rows (eager prefill chunks): f092906's page mode (the touched 64-token sub-pages, compacted; one host sync
    for their count) on the live entries only, reused across the layer group like group_view. Never captured."""
    torch = _torch()
    eidx, topk_len = _rows_2d(eidx, topk_len)
    sig = _group_sig(extra, eidx, topk_len, "page", None)
    tag, hit = _group_hit(sig)
    if hit is not None:
        return hit
    live_idx = torch.where(_live(eidx, topk_len), eidx, torch.full_like(eidx, -1))
    tokens, slots, new_idx, n_pages = plan_page_mode(live_idx, int(extra.shape[0]) * int(extra.shape[1]))
    buf = torch.zeros(max(n_pages, 1), SUB_PAGE_BYTES, dtype=torch.uint8, device=eidx.device)
    (transcode_tokens_triton if extra.is_cuda else transcode_tokens_torch)(extra, tokens, buf, slots)
    view = sub_pages_view(buf, max(n_pages, 1))
    _group_keep(tag, sig, view, new_idx, buf=buf)
    stats["page"] += 1
    return view, new_idx


# Group reader kernels. Same rule as v4_rows / the IO-warp reader, with the two roundings written in integer
# arithmetic (RN-even e4m3 of the clipped scaled value, RN-even bf16 of the exact rope value) and the e4m3 scale decoded
# exactly from its bits, so the Triton interpreter (CPU tests) computes the bytes the GPU does; the only float op that
# rounds is the IEEE division div_rn(max(amax, 1e-4), 448), as in the IO-warp reader (__fdiv_rn).
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _v41c_e4m3_value(sb):
        """e4m3fn byte (int32 0..255) -> fp32, exact."""
        e = (sb >> 3) & 15
        m = sb & 7
        nv = (((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
        sv = m.to(tl.float32) * 0.001953125
        v = tl.where(e == 0, sv, nv)
        return tl.where((sb & 128) != 0, -v, v)

    @triton.jit
    def _v41c_e4m3_rn(a):
        """fp32, |a| <= 448 -> e4m3fn byte (int32), RN-even, sign kept (-0.0 -> 0x80)."""
        b = a.to(tl.int32, bitcast=True)
        sign = (b >> 31) & 1
        x = b & 0x7FFFFFFF
        ex = (x >> 23) - 127
        mant = (x & 0x7FFFFF) | 0x800000
        normal = ex >= -6
        sh = tl.where(normal, 20, tl.minimum(14 - ex, 30))
        q0 = mant >> sh
        rem = mant & ((1 << sh) - 1)
        half = 1 << (sh - 1)
        up = (rem > half) | ((rem == half) & ((q0 & 1) == 1))
        code = q0 + up.to(tl.int32) + tl.where(normal, (ex + 6) << 3, 0)
        code = tl.where(x == 0, 0, code)
        return (sign << 7) | code

    @triton.jit
    def _v41c_rows(src, src_stride, PBS: tl.constexpr, tok, live, slot, dst, R: tl.constexpr):
        """Transcode R records (tok [R], live [R]) into the V4 rows at slot [R] of 64-token pages at dst."""
        k = tl.arange(0, 512)
        t64 = tl.where(live, tok, 0).to(tl.int64)
        page = src + (t64 // PBS) * src_stride
        loc = t64 % PBS
        lm = live[:, None]
        byte = tl.load(page[:, None] + loc[:, None] * 256 + (k // 2)[None, :], mask=lm, other=0).to(tl.int32)
        code = tl.where((k % 2 == 0)[None, :], byte & 15, byte >> 4)
        mm = code & 7
        mag = tl.where(mm < 4, mm.to(tl.float32) * 0.5,
                       tl.where(mm == 4, 2.0, tl.where(mm == 5, 3.0, tl.where(mm == 6, 4.0, 6.0))))
        val = tl.where((code & 8) != 0, -mag, mag)
        sb = tl.load(page[:, None] + PBS * 256 + loc[:, None] * 32 + (k // 16)[None, :], mask=lm, other=0)
        v = val * _v41c_e4m3_value(sb.to(tl.int32))
        v3 = tl.reshape(v, (R, 8, 64))
        amax = tl.max(tl.abs(v3), axis=2)
        sc = tl.math.div_rn(tl.maximum(amax, 1e-4), 448.0)
        bits = sc.to(tl.int32, bitcast=True)
        e = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        inv = ((254 - e) << 23).to(tl.float32, bitcast=True)
        q = _v41c_e4m3_rn(tl.minimum(tl.maximum(v3 * inv[:, :, None], -448.0), 448.0)).to(tl.uint8)
        s64 = slot.to(tl.int64)
        row = dst + (s64 // 64) * 37440 + (s64 % 64) * 576
        t = tl.arange(0, 8)[None, :, None]
        c = tl.arange(0, 64)[None, None, :]
        tl.store(row[:, None, None] + t * 64 + c, q, mask=live[:, None, None] & (t < 7))
        ub = v.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
        rb = (ub + 0x7FFF + ((ub >> 16) & 1)) >> 16
        rm = lm & (k >= 448)[None, :]
        ro = row[:, None] + 448 + (k - 448)[None, :] * 2
        tl.store(ro, (rb & 255).to(tl.uint8), mask=rm)
        tl.store(ro + 1, ((rb >> 8) & 255).to(tl.uint8), mask=rm)
        tt = tl.arange(0, 8)[None, :]
        sc_row = dst + (s64 // 64) * 37440 + 64 * 576 + (s64 % 64) * 8
        tl.store(sc_row[:, None] + tt, tl.where(tt < 7, e, 0).to(tl.uint8), mask=lm)

    @triton.jit
    def _v41c_plan_kernel(src, src_stride, PBS: tl.constexpr, eidx, tlen, HAS_LEN: tl.constexpr, K, new_idx, dst,
                          R: tl.constexpr):
        """No dedupe, one launch per group: grid (rows, cdiv(K, R)); slot r * K + j per live entry."""
        r = tl.program_id(0)
        j = tl.program_id(1) * R + tl.arange(0, R)
        inr = j < K
        tok = tl.load(eidx + r * K + j, mask=inr, other=-1)
        live = inr & (tok >= 0)
        if HAS_LEN:
            live = live & (j < tl.load(tlen + r))
        slot = r * K + j
        tl.store(new_idx + r * K + j, tl.where(live, slot, -1), mask=inr)
        if tl.sum(live.to(tl.int32), axis=0) > 0:
            _v41c_rows(src, src_stride, PBS, tok, live, slot, dst, R)

    @triton.jit
    def _v41c_dedup_kernel(eidx, tlen, HAS_LEN: tl.constexpr, K, G, new_idx, toks, cnt, tmp, N: tl.constexpr):
        """Grid (rows // G,): one program per block of G rows (n = G * K <= N entries), plan_dedup_torch's rule."""
        b = tl.program_id(0)
        n = G * K
        p = tl.arange(0, N)
        inr = p < n
        row = b * G + p // K
        j = p % K
        tok = tl.load(eidx + row * K + j, mask=inr, other=-1)
        live = inr & (tok >= 0)
        if HAS_LEN:
            live = live & (j < tl.load(tlen + row, mask=inr, other=0))
        key = tl.where(live, tok, -1).to(tl.int64)
        s = tl.sort(((key + 1) << 13) | p.to(tl.int64), 0)
        skey = (s >> 13) - 1
        spos = (s & 8191).to(tl.int32)
        tl.store(tmp + b * N + p, skey)
        tl.debug_barrier()
        prev = tl.load(tmp + b * N + p - 1, mask=p > 0, other=-2)
        first = (skey != prev) & (skey >= 0)
        rank = tl.cumsum(first.to(tl.int32), 0) - 1
        slot = b * n + rank
        tl.store(new_idx + (b * G + spos // K) * K + spos % K, tl.where(skey >= 0, slot, -1), mask=spos < n)
        tl.store(toks + b * N + p, tl.full([N], -1, tl.int32))
        tl.debug_barrier()
        tl.store(toks + b * N + rank, skey.to(tl.int32), mask=first)
        tl.store(cnt + b, tl.sum(first.to(tl.int32), 0))

    @triton.jit
    def _v41c_list_kernel(src, src_stride, PBS: tl.constexpr, toks, cnt, N, n, dst, R: tl.constexpr):
        """Grid (blocks, cdiv(n, R)): transcode block b's distinct tokens [0, cnt[b]) into slots b * n + r."""
        b = tl.program_id(0)
        i0 = tl.program_id(1) * R
        c = tl.load(cnt + b)
        if i0 < c:
            i = i0 + tl.arange(0, R)
            live = i < c
            tok = tl.load(toks + b * N + i, mask=live, other=-1)
            live = live & (tok >= 0)
            _v41c_rows(src, src_stride, PBS, tok, live, b * n + i, dst, R)

    @triton.jit
    def _v41n_mark_kernel(eidx, tlen, HAS_LEN: tl.constexpr, K, n, mark, BLOCK: tl.constexpr):
        """mark[idx // 64] = 1 for every live entry (index >= 0, position < length). Same value from every writer."""
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        inr = p < n
        tok = tl.load(eidx + p, mask=inr, other=-1)
        live = inr & (tok >= 0)
        if HAS_LEN:
            live = live & ((p % K) < tl.load(tlen + p // K, mask=inr, other=0))
        tl.store(mark + tok // 64, tl.full([BLOCK], 1, tl.int32), mask=live)

    @triton.jit
    def _v41n_remap_kernel(eidx, tlen, HAS_LEN: tl.constexpr, K, n, cmap, new_idx, BLOCK: tl.constexpr):
        """new = cmap[idx // 64] * 64 + idx % 64 for a live entry, -1 otherwise (plan_page_mode's rewrite)."""
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        inr = p < n
        tok = tl.load(eidx + p, mask=inr, other=-1)
        live = inr & (tok >= 0)
        if HAS_LEN:
            live = live & ((p % K) < tl.load(tlen + p // K, mask=inr, other=0))
        c = tl.load(cmap + tok // 64, mask=live, other=0)
        tl.store(new_idx + p, tl.where(live, c * 64 + tok % 64, -1), mask=inr)

    @triton.jit
    def _v41n_page_kernel(src, src_stride, PBS: tl.constexpr, mark, cmap, n_pool, dst, R: tl.constexpr):
        """Grid (pool sub-pages, 64 // R): a marked 64-token sub-page's records -> its compacted V4 sub-page."""
        sub = tl.program_id(0)
        if tl.load(mark + sub) != 0:
            j = tl.program_id(1) * R + tl.arange(0, R)
            tok = sub * 64 + j
            live = tok < n_pool
            slot = tl.load(cmap + sub) * 64 + j
            _v41c_rows(src, src_stride, PBS, tok, live, slot, dst, R)

    def _v41n_page_plan_triton(extra, eidx, topk_len, n_sub):
        torch = _torch()
        rows, k = eidx.shape
        n = rows * k
        mark = torch.zeros(max(n_sub, 1), dtype=torch.int32, device=eidx.device)
        blk = 1024
        _v41n_mark_kernel[(triton.cdiv(n, blk),)](eidx, eidx if topk_len is None else topk_len, topk_len is not None,
                                                   k, n, mark, BLOCK=blk)
        return mark

    def _v41n_page_finish_triton(extra, eidx, topk_len, mark, cmap, n_pool, buf):
        torch = _torch()
        rows, k = eidx.shape
        n = rows * k
        new_idx = torch.empty((rows, k), dtype=torch.int32, device=eidx.device)
        blk = 1024
        _v41n_remap_kernel[(triton.cdiv(n, blk),)](eidx, eidx if topk_len is None else topk_len, topk_len is not None,
                                                    k, n, cmap, new_idx, BLOCK=blk)
        raw, pbs, stride = fp4_page_view(extra)
        _v41n_page_kernel[(int(mark.numel()), SUB // TOK_R)](raw, stride, pbs, mark, cmap, n_pool, buf, R=TOK_R,
                                                              num_warps=4)
        return new_idx

    def _plan_group_triton(extra, eidx, topk_len, buf):
        torch = _torch()
        rows, k = eidx.shape
        new_idx = torch.empty((rows, k), dtype=torch.int32, device=eidx.device)
        raw, pbs, stride = fp4_page_view(extra)
        _v41c_plan_kernel[(rows, triton.cdiv(k, TOK_R))](
            raw, stride, pbs, eidx, eidx if topk_len is None else topk_len, topk_len is not None, k, new_idx, buf,
            R=TOK_R, num_warps=4)
        return new_idx

    def _dedup_triton(extra, eidx, topk_len, buf, g):
        torch = _torch()
        rows, k = eidx.shape
        nb, n = rows // g, g * k
        npow = triton.next_power_of_2(n)
        dev = eidx.device
        new_idx = torch.empty((rows, k), dtype=torch.int32, device=dev)
        toks = torch.empty((nb, npow), dtype=torch.int32, device=dev)
        cnt = torch.empty((nb,), dtype=torch.int32, device=dev)
        tmp = torch.empty((nb, npow), dtype=torch.int64, device=dev)
        _v41c_dedup_kernel[(nb,)](eidx, eidx if topk_len is None else topk_len, topk_len is not None, k, g, new_idx,
                                  toks, cnt, tmp, N=npow, num_warps=8)
        raw, pbs, stride = fp4_page_view(extra)
        _v41c_list_kernel[(nb, triton.cdiv(n, TOK_R))](raw, stride, pbs, toks, cnt, npow, n, buf, R=TOK_R,
                                                       num_warps=4)
        return new_idx
except ImportError:
    def _plan_group_triton(extra, eidx, topk_len, buf):
        raise RuntimeError(f"{ENV}: triton is required on the GPU path")

    def _v41n_page_plan_triton(extra, eidx, topk_len, n_sub):
        raise RuntimeError(f"{ENV}: triton is required on the GPU path")

    def _v41n_page_finish_triton(extra, eidx, topk_len, mark, cmap, n_pool, buf):
        raise RuntimeError(f"{ENV}: triton is required on the GPU path")

    def _dedup_triton(extra, eidx, topk_len, buf, g):
        raise RuntimeError(f"{ENV}: triton is required on the GPU path")


def prefill(q2, kv_64, idx, softmax_scale, *, topk_length, attn_sink, extra_fp4, extra_idx,
            extra_topk_length, head_dim_v):
    """> 64 rows with an FP4 extra cache: transcode to V4 64-token pages, then FlashInfer's stock prefill.
    native: v41n_prefill reads the FP4 pages in place (native_prefill)."""
    if reader() == "native":
        return native_prefill(q2, kv_64, idx, softmax_scale, topk_length=topk_length, attn_sink=attn_sink,
                              extra_fp4=extra_fp4, extra_idx=extra_idx, extra_topk_length=extra_topk_length,
                              head_dim_v=head_dim_v)
    torch = _torch()
    from flashinfer.mla._sparse_mla_sm120 import _sparse_mla_sm120_paged_attention

    num_tokens, num_heads = q2.shape[0], q2.shape[1]
    if reader() == "group":
        group_receipt()
        view_of = group_view if extra_idx.shape[0] <= TOKEN_MODE_MAX_ROWS else group_page_view
        extra_v4, new_idx = view_of(extra_fp4, extra_idx, extra_topk_length)
    else:
        extra_v4, new_idx = transcode_for_prefill(extra_fp4, extra_idx)
    output = q2.new_empty((num_tokens, num_heads, head_dim_v), dtype=torch.bfloat16)
    out_lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=q2.device)
    _sparse_mla_sm120_paged_attention(q2, kv_64, idx, output, out_lse, softmax_scale, topk_length=topk_length,
                                      attn_sink=attn_sink, extra_kv_cache=extra_v4, extra_indices=new_idx,
                                      extra_topk_length=extra_topk_length, mid_out=None, mid_lse=None)
    return output


# ------------------------------------------------------------------------------------------------ native (v41n)
NATIVE_PREFILL_HEADS = (8, 16)
NATIVE_PREFILL_SWA_TOPK = 128
NATIVE_PREFILL_PBS = (128, 256)


@lru_cache(maxsize=1)
def native_receipt() -> str:
    """First native call (eager warm-up, before any capture): the module identity check + receipt line."""
    _module()
    line = (f"{ENV}=v41c reader=native: in-kernel V4.1 FP4 readers (decode converter warps, prefill in place), "
            f"no transcode pass")
    logger.warning(line)
    return line


@lru_cache(maxsize=1)
def native_page_receipt() -> str:
    """First eager page pass: receipt line (Triton kernels + the stock prefill; the v41n module is not needed)."""
    mode, rows, cap = native_prefill_rule()
    line = (f"{ENV}=v41c reader=native: eager prefill chunks take the page pass ({PREFILL_ENV}={mode}, >= {rows} rows, "
            f"scratch cap {cap >> 20} MiB): touched records converted once per layer group, stock prefill")
    logger.warning(line)
    return line


@lru_cache(maxsize=1)
def _native_runner():
    from flashinfer.mla import _sparse_mla_sm120 as S

    base = S._get_sparse_mla_decode_dsv4_module().runner_cls
    mod = _module()

    class Runner(base):
        def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
            q, indices, mid_out, mid_lse, output, out_lse = inputs[:6]
            topk_length, attn_sink, extra_indices, extra_topk_length = (
                inputs[6:10] if len(inputs) > 6 else (kwargs.get("topk_length"), kwargs.get("attn_sink"),
                                                       kwargs.get("extra_indices"), kwargs.get("extra_topk_length")))
            topk = indices.shape[-1]
            extra_topk = extra_indices.shape[-1] if extra_indices is not None else 0
            num_splits = (topk + _BI - 1) // _BI + (extra_topk + _BI - 1) // _BI
            mod.v41n_decode_dsv4(q, kwargs["kv_cache"], indices, mid_out, mid_lse, output, out_lse, num_splits,
                                 kwargs["sm_scale"], topk_length, attn_sink, kwargs.get("extra_kv_cache"),
                                 extra_indices, extra_topk_length, tactic if tactic > 0 else -1,
                                 int(kwargs.get("active_heads", 0)), int(kwargs.get("diag_mode", 0)))
            return output

    # the AutoTuner keys on the runner class name: the stock name keeps the stock tactic table (same split, same
    # reduction order as the stock call for the shape)
    Runner.__name__, Runner.__qualname__ = base.__name__, base.__qualname__
    return Runner()


DIAG_ENV = "DSV41_V41N_DIAG"           # qualification timing diagnostics only: noconv | allheads (default: off)
_heads_tag: list = [None]             # heads the caller reads (adapter: layer.tp_q_head_num) for the current call


def tag_heads(n) -> None:
    """The caller keeps output heads [0, n) only (SGLang pads TP4's 16 heads to 64 zero queries for the decode kernel
    and slices o[:, :16]); the native decode then launches and merges only the head blocks holding them."""
    _heads_tag[0] = None if n is None else int(n)


def clear_heads() -> None:
    _heads_tag[0] = None


@lru_cache(maxsize=1)
def native_diag() -> str:
    raw = os.environ.get(DIAG_ENV, "").strip()
    if raw not in ("", "noconv", "allheads"):
        raise ValueError(f"{DIAG_ENV} must be unset, noconv or allheads, got {raw!r}")
    if raw == "noconv":
        logger.warning("%s=noconv: the native decode SKIPS the conversion (wrong outputs): timing diagnostic only",
                       DIAG_ENV)
    return raw


TACTICS_ENV = "DSV41_V41N_TACTICS"   # optional frozen tactic table (JSON {repr(key): tactic}); a miss fails closed
native_tactics: dict = {}             # every shape key this process resolved -> tactic (recorded, never re-resolved)


@lru_cache(maxsize=1)
def _frozen_tactics():
    path = os.environ.get(TACTICS_ENV, "").strip()
    if not path:
        return None
    table = json.loads(Path(path).read_text())
    if not isinstance(table, dict) or not all(isinstance(v, int) for v in table.values()):
        raise RuntimeError(f"{TACTICS_ENV}={path}: not a JSON object of integer tactics")
    return table


def native_tactic(S, inputs, kw, hot_key) -> int:
    """The decode-dsv4 tactic (chunks_per_block) the STOCK call resolves for this shape, without tuning and without
    writing the stock caches. The tactic changes the split partition and so the merge order: native must run A's.
    - AutoTuner in tuning mode: refuse (tune with the stock reader, then freeze its table);
    - DSV41_V41N_TACTICS set: that frozen table, fail closed on a missing shape;
    - else the stock hot cache, else the stock runner's AutoTuner lookup (non-tuning: cache lookup, -1 = the C++
      heuristic on a miss), exactly what the stock wrapper resolves.
    The first resolution per shape is recorded in native_tactics; a later different answer raises."""
    from flashinfer.autotuner import AutoTuner

    key = repr(hot_key)
    tuner = AutoTuner.get()
    if tuner.is_tuning_mode:
        raise RuntimeError(f"{READER_ENV}=native never autotunes: tune with the stock reader and freeze its table "
                           f"({TACTICS_ENV})")
    table = _frozen_tactics()
    if table is not None:
        if key not in table:
            raise RuntimeError(f"{READER_ENV}=native: no frozen tactic for shape {key} in {TACTICS_ENV}")
        tactic = int(table[key])
    else:
        cached = S._decode_dsv4_hot_cache.get(hot_key)
        if cached is not None:
            tactic = int(cached)
        else:
            S._decode_dsv4_maybe_load_cache()
            stock = S._get_sparse_mla_decode_dsv4_module().runner_cls()
            _, tactic = tuner.choose_one("sparse_mla_sm120_decode_dsv4", [stock], S._decode_dsv4_tuning_config(),
                                         inputs, **kw)
            tactic = int(tactic)
    prev = native_tactics.setdefault(key, tactic)
    if prev != tactic:
        raise RuntimeError(f"{READER_ENV}=native: tactic for {key} changed from {prev} to {tactic}")
    return tactic


def _dtype_is(t, name) -> bool:
    return str(getattr(t, "dtype", "")).endswith(name)


def native_decode_supported(q, kv_cache, indices, output, *, extra_kv, extra_indices, extra_topk_length) -> bool:
    """Contracts of v41n_decode_dsv4 (static: shapes and dtypes only)."""
    if extra_kv is None or extra_indices is None or extra_kv.shape[-1] != BPT_FP4 or extra_kv.dim() < 3:
        return False
    return (q.dim() == 3 and int(q.shape[2]) == 512 and int(output.shape[-1]) == 512
            and _dtype_is(q, "bfloat16") and _dtype_is(output, "bfloat16") and _dtype_is(indices, "int32")
            and _dtype_is(extra_indices, "int32") and _dtype_is(extra_kv, "uint8") and _dtype_is(kv_cache, "uint8")
            and (extra_topk_length is None or _dtype_is(extra_topk_length, "int32")))


def native_decode(q, kv_cache, indices, mid_out, mid_lse, output, out_lse, sm_scale, *, topk_length, attn_sink,
                  extra_kv, extra_indices, extra_topk_length):
    """<= 64 rows, native: the v41n kernel with the stock call's tactic (native_tactic); contract violations take the
    group path (counted in stats["native_fallback"])."""
    from flashinfer.mla import _sparse_mla_sm120 as S

    if not native_decode_supported(q, kv_cache, indices, output, extra_kv=extra_kv, extra_indices=extra_indices,
                                   extra_topk_length=extra_topk_length):
        from flashinfer.mla._sparse_mla_sm120 import _sparse_mla_sm120_paged_attention

        stats["native_fallback"] += 1
        group_receipt()
        extra_v4, new_idx = group_view(extra_kv, extra_indices, extra_topk_length)
        _sparse_mla_sm120_paged_attention(q, kv_cache, indices, output, out_lse, sm_scale, d_v=output.shape[-1],
                                          topk_length=topk_length, attn_sink=attn_sink, extra_kv_cache=extra_v4,
                                          extra_indices=new_idx, extra_topk_length=extra_topk_length,
                                          mid_out=mid_out, mid_lse=mid_lse)
        return output
    native_receipt()
    runner = _native_runner()
    inputs = [q, indices, mid_out, mid_lse, output, out_lse, topk_length, attn_sink, extra_indices, extra_topk_length]
    kw = {"sm_scale": sm_scale, "kv_cache": kv_cache, "extra_kv_cache": extra_kv}
    extra_topk = extra_indices.shape[-1] if extra_indices is not None else 0
    num_splits = (indices.shape[-1] + _BI - 1) // _BI + (extra_topk + _BI - 1) // _BI
    hot_key = (S._decode_dsv4_map_to_token_bucket(q.shape[0]), q.shape[1], indices.shape[-1], extra_topk, num_splits,
               runner.get_cache_key_extras(inputs))
    tactic = native_tactic(S, inputs, kw, hot_key)
    diag = native_diag()
    heads = _heads_tag[0]
    active = heads if (heads is not None and 0 < heads < int(q.shape[1]) and diag != "allheads") else 0
    stats["native_decode"] += 1
    if active:
        stats["native_head_skip"] += 1
    runner(inputs=inputs, tactic=tactic, active_heads=active, diag_mode=2 if diag == "noconv" else 0, **kw)
    return output


def native_convert_dump(extra, tokens, mode: int):
    """Qualification diagnostic: the native conversion code (mode 0 = decode path, 1 = prefill path) on the GPU for
    `tokens` (int32 [n], < 0 = masked) of an FP4 extra cache; returns [n, 584] uint8 in the V4 row ABI (448 e4m3 nope,
    128 bytes of bf16 rope, 7 ue8m0 + pad)."""
    torch = _torch()
    out = torch.empty((int(tokens.numel()), BPT_V4), dtype=torch.uint8, device=tokens.device)
    _module().v41n_convert_dump(extra, tokens.to(torch.int32).contiguous(), out, int(mode))
    return out


def native_prefill_supported(q2, kv_64, idx, *, topk_length, extra_fp4, extra_idx, extra_topk_length,
                             head_dim_v=512) -> bool:
    """The one configuration v41n_prefill serves (static: shapes, dtypes, contiguity and presence only, capture-safe).
    The IO warps read whole 64-entry tiles of the extra indices, so their width must be a positive multiple of 64;
    the kernel writes 512 values per head, so head_dim_v must be 512. The C++ entry checks the same and refuses."""
    if extra_fp4 is None or extra_fp4.shape[-1] != BPT_FP4 or extra_fp4.dim() < 3:
        return False
    width = int(extra_idx.shape[-1])
    return (q2.dim() == 3 and q2.is_contiguous() and idx.is_contiguous() and extra_idx.is_contiguous()
            and int(q2.shape[1]) in NATIVE_PREFILL_HEADS and int(q2.shape[2]) == 512 and int(head_dim_v) == 512
            and int(idx.shape[-1]) == NATIVE_PREFILL_SWA_TOPK and kv_64.dim() >= 3 and int(kv_64.shape[1]) == SUB
            and int(extra_fp4.shape[1]) in NATIVE_PREFILL_PBS and width > 0 and width % 64 == 0
            and int(idx.shape[0]) == int(q2.shape[0]) and int(extra_idx.shape[0]) == int(q2.shape[0])
            and _dtype_is(q2, "bfloat16") and _dtype_is(idx, "int32") and _dtype_is(extra_idx, "int32")
            and _dtype_is(kv_64, "uint8") and _dtype_is(extra_fp4, "uint8")
            and topk_length is not None and extra_topk_length is not None
            and _dtype_is(topk_length, "int32") and _dtype_is(extra_topk_length, "int32"))


PREFILL_ENV = "DSV41_V41N_PREFILL"              # auto (default) | kernel | page
PAGE_MIN_ROWS_ENV = "DSV41_V41N_PAGE_MIN_ROWS"   # auto: eager calls with at least this many rows take the page pass
PAGE_CAP_ENV = "DSV41_V41N_PAGE_CAP_MB"          # the page pass's scratch limit; above it the in-kernel path runs
PAGE_MIN_ROWS_DEFAULT = 512
PAGE_CAP_MB_DEFAULT = 1024


@lru_cache(maxsize=1)
def native_prefill_rule() -> tuple:
    """(mode, min rows, scratch cap bytes). The page pass is for EAGER prefill chunks: the in-kernel reader converts
    every record once per (query row, top-k entry), O(rows x K): an 8192-row chunk is 38 layers x 8192 rows x 512
    entries = 160 M record conversions on the math warps' critical path (dbd4644: +12 %). The page pass converts each
    touched record once per layer group, O(unique records), then runs the stock prefill."""
    raw = os.environ.get(PREFILL_ENV, "auto").strip() or "auto"
    if raw not in ("auto", "kernel", "page"):
        raise ValueError(f"{PREFILL_ENV} must be auto, kernel or page, got {raw!r}")
    rows = int(os.environ.get(PAGE_MIN_ROWS_ENV, "") or PAGE_MIN_ROWS_DEFAULT)
    cap = int(os.environ.get(PAGE_CAP_ENV, "") or PAGE_CAP_MB_DEFAULT) << 20
    return raw, rows, cap


def _capturing() -> bool:
    torch = _torch()
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def native_prefill_mode(rows: int) -> str:
    """kernel (in-kernel FP4 reader) or page (one conversion pass + the stock prefill). Never page under capture: the
    page pass sizes its scratch from a host read (eager only), so captured verify graphs keep the in-kernel reader."""
    mode, min_rows, _ = native_prefill_rule()
    if mode == "kernel" or _capturing():
        return "kernel"
    if mode == "page":
        return "page"
    return "page" if rows >= min_rows else "kernel"


def native_page_view(extra, eidx, topk_len=None):
    """Eager prefill: (V4 64-token page view, rewritten indices), or None when the scratch would exceed the cap.

    The touched 64-token sub-pages of the FP4 pool, each converted ONCE into a compacted V4 sub-page by the group
    reader's exact rule (_v41c_rows), indices rewritten cmap[idx // 64] * 64 + idx % 64 (plan_page_mode's mapping),
    reused across the layer group. GPU: mark kernel, one cumsum, one host read of the count, remap kernel, page
    transcode kernel; no index-sized torch temporaries (group_page_view plans with ~12 torch ops over rows x K
    entries, ~4 ms per group at 8192 rows: the G arm's +33 ms on the dbd4644 chunk)."""
    torch = _torch()
    eidx, topk_len = _rows_2d(eidx, topk_len)
    sig = _group_sig(extra, eidx, topk_len, "v41n-page", None)
    tag, hit = _group_hit(sig)
    if hit is not None:
        return hit
    n_pool = int(extra.shape[0]) * int(extra.shape[1])
    n_sub = -(-n_pool // SUB)
    if extra.is_cuda:
        mark = _v41n_page_plan_triton(extra, eidx, topk_len, n_sub)
    else:
        live = _live(eidx, topk_len)
        mark = torch.zeros(max(n_sub, 1), dtype=torch.int32, device=eidx.device)
        mark[(eidx[live] // SUB).long()] = 1
    cmap = torch.cumsum(mark, 0, dtype=torch.int32) - 1
    n_pages = int(cmap[-1].item()) + 1 if n_sub else 0     # eager only: one host read per group
    _, _, cap = native_prefill_rule()
    if max(n_pages, 1) * SUB_PAGE_BYTES > cap:
        stats["native_page_cap"] += 1
        return None
    buf = torch.zeros(max(n_pages, 1), SUB_PAGE_BYTES, dtype=torch.uint8, device=eidx.device)
    if extra.is_cuda:
        new_idx = _v41n_page_finish_triton(extra, eidx, topk_len, mark, cmap, n_pool, buf)
    else:
        live = _live(eidx, topk_len)
        sub = torch.where(live, eidx // SUB, torch.zeros_like(eidx)).long()
        new_idx = torch.where(live, cmap[sub] * SUB + eidx % SUB, torch.full_like(eidx, -1)).to(torch.int32)
        touched = torch.nonzero(mark[:n_sub], as_tuple=False).flatten()
        tokens = (touched.unsqueeze(-1) * SUB + torch.arange(SUB, device=eidx.device)).flatten()
        slots = (cmap[touched].long().unsqueeze(-1) * SUB + torch.arange(SUB, device=eidx.device)).flatten()
        keep = tokens < n_pool
        transcode_tokens_torch(extra, tokens[keep], buf, slots[keep])
    view = sub_pages_view(buf, max(n_pages, 1))
    _group_keep(tag, sig, view, new_idx, buf=buf)
    stats["native_page"] += 1
    return view, new_idx


def native_prefill(q2, kv_64, idx, softmax_scale, *, topk_length, attn_sink, extra_fp4, extra_idx,
                   extra_topk_length, head_dim_v):
    """> 64 rows, native. Eager chunks (native_prefill_mode == page): the touched records converted once per layer
    group (native_page_view), then FlashInfer's stock prefill: the group reader's page path with a planner that costs
    O(unique records). Captured verify graphs and small eager calls: v41n_prefill over the FP4 pages in place.
    Unsupported shapes take the group path."""
    torch = _torch()
    if native_prefill_mode(int(q2.shape[0])) == "page":
        got = native_page_view(extra_fp4, extra_idx, extra_topk_length)
        if got is not None:
            from flashinfer.mla._sparse_mla_sm120 import _sparse_mla_sm120_paged_attention

            native_page_receipt()
            extra_v4, new_idx = got
            output = q2.new_empty((q2.shape[0], q2.shape[1], head_dim_v), dtype=torch.bfloat16)
            out_lse = torch.empty((q2.shape[0], q2.shape[1]), dtype=torch.float32, device=q2.device)
            _sparse_mla_sm120_paged_attention(q2, kv_64, idx, output, out_lse, softmax_scale, topk_length=topk_length,
                                              attn_sink=attn_sink, extra_kv_cache=extra_v4, extra_indices=new_idx,
                                              extra_topk_length=extra_topk_length, mid_out=None, mid_lse=None)
            return output
    if not native_prefill_supported(q2, kv_64, idx, topk_length=topk_length, extra_fp4=extra_fp4,
                                    extra_idx=extra_idx, extra_topk_length=extra_topk_length,
                                    head_dim_v=head_dim_v):
        from flashinfer.mla._sparse_mla_sm120 import _sparse_mla_sm120_paged_attention

        stats["native_fallback"] += 1

        group_receipt()
        view_of = group_view if extra_idx.shape[0] <= TOKEN_MODE_MAX_ROWS else group_page_view
        extra_v4, new_idx = view_of(extra_fp4, extra_idx, extra_topk_length)
        output = q2.new_empty((q2.shape[0], q2.shape[1], head_dim_v), dtype=torch.bfloat16)
        out_lse = torch.empty((q2.shape[0], q2.shape[1]), dtype=torch.float32, device=q2.device)
        _sparse_mla_sm120_paged_attention(q2, kv_64, idx, output, out_lse, softmax_scale, topk_length=topk_length,
                                          attn_sink=attn_sink, extra_kv_cache=extra_v4, extra_indices=new_idx,
                                          extra_topk_length=extra_topk_length, mid_out=None, mid_lse=None)
        return output
    native_receipt()
    mod = _module()
    stats["native_prefill"] += 1
    output = q2.new_empty((q2.shape[0], q2.shape[1], head_dim_v), dtype=torch.bfloat16)
    out_lse = torch.empty((q2.shape[0], q2.shape[1]), dtype=torch.float32, device=q2.device)
    mod.v41n_prefill(q2, kv_64, idx, output, out_lse, float(softmax_scale), topk_length, attn_sink, extra_fp4,
                     extra_idx, extra_topk_length)
    return output


# ------------------------------------------------------------------------------------------------ swan
SWAN_ENV = "DSV41_SWA_NATIVE_DECODE"   # 1: the decode reads the SWA pool's own 256-token pages (default off)
SWAN_PBS = 256                          # SGLang's SWA page size
SWAN_MIN_PAGE_BYTES = SWAN_PBS * BPT_V4  # data 256 x 576 + footer 256 x 8 (149,760 with SGLang's 576 padding)
SWAN_HEADS = (16, 64)                   # TP4's native 16 heads; 64 = the padded layout of images without it
SWAN_TOPK = (128, 192)                  # SWA_WINDOW; 192 = the padded DSpark shape the stock grid also carries


@lru_cache(maxsize=1)
def swan_enabled() -> bool:
    raw = os.environ.get(SWAN_ENV, "").strip().lower()
    if raw in ("", "0", "off", "false", "no"):
        return False
    if raw in ("1", "on", "true", "yes"):
        return True
    raise ValueError(f"{SWAN_ENV} must be unset, 0 or 1, got {raw!r}")


@lru_cache(maxsize=1)
def swan_receipt() -> str:
    """First swan call (eager warm-up, before any capture): the module identity check + receipt line."""
    _module()
    line = (f"{SWAN_ENV}=1: decode (<= 64 rows) reads the SWA pool's {SWAN_PBS}-token pages natively, no per-call "
            f"64-token split; heads {SWAN_HEADS} x SWA top-k {SWAN_TOPK}, others take the split")
    logger.warning(line)
    return line


@lru_cache(maxsize=None)
def _swan_fallback_logged(reason: str) -> None:
    logger.warning("%s=1: a decode call takes the 64-token split (%s); logged once per reason", SWAN_ENV, reason)


def _swan_fallback(reason: str):
    stats["swan_fallback"] += 1
    _swan_fallback_logged(reason)
    return None


def swan_view(kv_u8):
    """The SWA pool as [pages, 256, 1, 584] with its own page stride, when it is the 256-token footer layout the kernel
    addresses natively (data at page * stride + t * 576, footer at page * stride + 256 * 576 + t * 8: exactly what the
    canary's split reads); else None. Bulk copies need 16-byte aligned rows: base and page stride multiples of 16."""
    torch = _torch()
    if kv_u8 is None or kv_u8.dtype != torch.uint8 or kv_u8.dim() not in (3, 4):
        return None
    if int(kv_u8.shape[1]) != SWAN_PBS or int(kv_u8.shape[-1]) != BPT_V4 or (kv_u8.dim() == 4 and int(kv_u8.shape[2]) != 1):
        return None
    stride = int(kv_u8.stride(0))
    if stride < SWAN_MIN_PAGE_BYTES or stride % 16 or kv_u8.data_ptr() % 16:
        return None
    return torch.as_strided(kv_u8, (int(kv_u8.shape[0]), SWAN_PBS, 1, BPT_V4), (stride, BPT_V4, BPT_V4, 1),
                            kv_u8.storage_offset())


def _swan_tuning() -> bool:
    try:
        from flashinfer.autotuner import AutoTuner
    except Exception:  # noqa: BLE001
        return True                          # no tuner to ask: never guess, take the split
    return bool(AutoTuner.get().is_tuning_mode)


def swan_route(q2, kv_u8, idx, extra_kv, extra_idx, extra_topk_length, head_dim_v, max_rows):
    """("stock" | "v41n", native SWA view) for a decode call swan serves, else None (the caller splits as before).
    Stock: no extra cache or a V4 (584-byte) one. v41n: a V4.1 FP4 extra cache with DSV41_V41C_READER=native."""
    if not swan_enabled():
        return None
    if int(q2.shape[0]) > max_rows:
        return _swan_fallback("rows")
    if _swan_tuning():
        # tuning runs the stock path so the tactic table is the stock one (the swan runner never tunes)
        return _swan_fallback("autotuner tuning mode")
    heads, topk = int(q2.shape[1]), int(idx.shape[-1])
    if heads not in SWAN_HEADS or topk not in SWAN_TOPK or idx.dim() != 2 or int(head_dim_v) != 512:
        return _swan_fallback(f"shape heads={heads} topk={topk}")
    if not (q2.dim() == 3 and int(q2.shape[2]) == 512 and _dtype_is(q2, "bfloat16") and _dtype_is(idx, "int32")):
        return _swan_fallback("dtype")
    view = swan_view(kv_u8)
    if view is None:
        return _swan_fallback("SWA layout")
    if extra_kv is not None and int(extra_kv.shape[-1]) == BPT_FP4:
        if not (enabled() and reader() == "native"):
            return _swan_fallback(f"FP4 extra cache with reader {reader() if enabled() else 'off'}")
        return ("v41n", view)
    if extra_kv is not None and (int(extra_kv.shape[-1]) != BPT_V4 or extra_idx is None or extra_idx.dim() != 2
                                 or not _dtype_is(extra_idx, "int32") or not _dtype_is(extra_kv, "uint8")):
        return _swan_fallback("extra layout")
    if extra_topk_length is not None and not _dtype_is(extra_topk_length, "int32"):
        return _swan_fallback("extra lengths dtype")
    return ("stock", view)


@lru_cache(maxsize=1)
def _swan_runner():
    from flashinfer.mla import _sparse_mla_sm120 as S

    base = S._get_sparse_mla_decode_dsv4_module().runner_cls
    mod = _module()

    class Runner(base):
        def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
            q, indices, mid_out, mid_lse, output, out_lse = inputs[:6]
            topk_length, attn_sink, extra_indices, extra_topk_length = inputs[6:10]
            topk = indices.shape[-1]
            extra_topk = extra_indices.shape[-1] if extra_indices is not None else 0
            num_splits = (topk + _BI - 1) // _BI + ((extra_topk + _BI - 1) // _BI if extra_topk > 0 else 0)
            mod.swan_decode_dsv4(q, kwargs["kv_cache"], indices, mid_out, mid_lse, output, out_lse, num_splits,
                                 kwargs["sm_scale"], topk_length, attn_sink, kwargs.get("extra_kv_cache"),
                                 extra_indices, extra_topk_length, tactic if tactic > 0 else -1)
            return output

    # the AutoTuner keys on the runner class name: the stock name resolves the stock tactic for the shape
    Runner.__name__, Runner.__qualname__ = base.__name__, base.__qualname__
    return Runner()


def swan_call(q, kv_u8, indices, topk_length, attn_sink, head_dim_v, softmax_scale, extra_k_cache, extra_indices,
              extra_topk_length):
    """flash_mla_sm120._flash_mla_flashinfer's decode with the SWA pool read in its own pages: the canary's own
    argument handling and buffers (output, lse, split-K scratch of the same shapes), then the swan kernel with the
    stock tactic. None = not served (the canary continues with the split, unchanged)."""
    if not swan_enabled():
        return None
    torch = _torch()
    from flashinfer.mla._sparse_mla_sm120 import _DECODE_MAX_TOKENS

    q2 = q.squeeze(1) if q.ndim == 4 else q
    idx = indices.squeeze(1) if indices.dim() == 3 else indices
    extra_kv = (extra_k_cache.view(torch.uint8) if extra_k_cache is not None and extra_k_cache.dtype != torch.uint8
                else extra_k_cache)
    extra_idx = extra_indices.squeeze(1) if extra_indices is not None and extra_indices.dim() == 3 else extra_indices
    route = swan_route(q2, kv_u8, idx, extra_kv, extra_idx, extra_topk_length, head_dim_v, _DECODE_MAX_TOKENS)
    if route is None:
        return None
    kind, view = route
    swan_receipt()
    B, H = int(q2.shape[0]), int(q2.shape[1])
    dev = q.device
    output = torch.empty(B, H, head_dim_v, dtype=torch.bfloat16, device=dev)
    out_lse = torch.empty(B, H, dtype=torch.float32, device=dev)
    topk = idx.shape[-1]
    extra_topk = extra_idx.shape[-1] if extra_idx is not None else 0
    num_splits = (topk + _BI - 1) // _BI + ((extra_topk + _BI - 1) // _BI if extra_topk > 0 else 0)
    mid_out = torch.empty(B, H, num_splits, head_dim_v, dtype=torch.bfloat16, device=dev)
    mid_lse = torch.empty(B, H, num_splits, dtype=torch.float32, device=dev)
    if kind == "v41n":
        stats["swan_v41n"] += 1
        native_decode(q2, view, idx, mid_out, mid_lse, output, out_lse, softmax_scale, topk_length=topk_length,
                      attn_sink=attn_sink, extra_kv=extra_kv, extra_indices=extra_idx,
                      extra_topk_length=extra_topk_length)
        return (output.unsqueeze(1), None)
    from flashinfer.mla import _sparse_mla_sm120 as S

    runner = _swan_runner()
    inputs = [q2, idx, mid_out, mid_lse, output, out_lse, topk_length, attn_sink, extra_idx, extra_topk_length]
    kw = {"sm_scale": softmax_scale, "kv_cache": view, "extra_kv_cache": extra_kv}
    hot_key = (S._decode_dsv4_map_to_token_bucket(B), H, topk, extra_topk, num_splits,
               runner.get_cache_key_extras(inputs))
    tactic = native_tactic(S, inputs, kw, hot_key)
    stats["swan_decode"] += 1
    runner(inputs=inputs, tactic=tactic, **kw)
    return (output.unsqueeze(1), None)


def main(argv) -> int:
    if len(argv) == 2 and argv[1] in ("build", "verify", "check"):
        d = Path(os.environ.get(DIR_ENV, DEFAULT_DIR))
        if argv[1] == "build":
            mj = build(d)
            print(f"V41C BUILD OK {mj['name']} so_sha256={mj['so_sha256']}")
            return 0
        ident = check_sources(d, *_flashinfer_dirs())
        so = check_module(d, ident)
        if argv[1] == "check":
            import tvm_ffi

            tvm_ffi.load_module(str(so))
        print(f"V41C MODULE OK {ident['module_tag']} {_sha(so)}")
        return 0
    print("usage: python3 -m sglang.kernels.ops.attention.dsv41_v41c build|verify|check", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
