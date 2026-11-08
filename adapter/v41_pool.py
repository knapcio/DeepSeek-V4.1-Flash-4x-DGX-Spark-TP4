"""DSV41_KV_LAYOUT=v41 (v2.7, FlashInfer >= 0.7.2): the DeepSeek-V4.1 KV layouts on SM121, read natively.

The main (SWA) cache, the paged DSpark draft SWA cache included, is ``KVLayout.V41``: 512 FP8 values per token in
groups of 32 with UE8M0 scales, RoPE included, 528 B per token. That is how DeepSeek's own V4.1 reference
(``inference/model.py`` ``_window_kv``: ``act_quant(kv, 32, "ue8m0", ...)``, "RoPE tail included") holds the window
K. The ratio-1 / ratio-2 compressed caches are ``V41_FP4`` (288 B), the reference's ``fp4_act_quant(latent, 16, e4m3)``
records, which stock ``resolve_compressed_kv_layout`` pairs with a V41 main cache. SGLang's writers already follow
the pool's layout (``fused_store_cache`` / ``fused_k_norm_rope_flashmla``). The reader is FlashInfer's SM120 DSv4.1
route in ``flash_mla_sm120.canary.py`` (528-byte main caches; a port of sglang#45851).

Default off. sitecustomize checks the flag before importing this module; ``v41c`` is the v2.6.5 path
(adapter/v41c_pool.py) and is refused on FlashInfer >= 0.7 by boot.py.

What changes when on (and nothing else):

* ``deepseek_v4_memory_pool.select_dsv4_kv_layout`` (and the name ``kv_cache_configurator`` imported from it): on an
  SM12x GPU it returns ``(KVLayout.V41, None)``. Stock asserts an SM100 GPU for the V4.1 layouts (sglang#45851's
  selector hunk, luz's "A1"). Elsewhere the stock function runs unchanged.
* ``DSV4PoolConfigurator``: the stock sizing prices every latent slot at the V4 slot, 584 B (``kv_bytes`` = 448 +
  128 + 8). It is re-priced at the V41 slot, 528 B: the paged SWA term (40 target layers; the draft's paged SWA layers
  with encoder replay), the encoder request window, and the c4 / c128 terms (V4.1-Flash has none). The ratio-1/2 term
  is priced at the 288-byte V41_FP4 record (as v41c did). This is luz's "A7"; upstream has no equivalent.
* ``DeepSeekV4TokenToKVPool``: after construction the main cache must be V41 (528 B per token) and the ratio-1/2
  pools V41_FP4 (288 B). One receipt line per pool: ``DSV41_V41_POOL ...``.

Fail closed: every SGLang function this changes or relies on is pinned by the sha256 of its source text (the same
``source_pin`` as v41c_pool), the KVLayout values are asserted, FlashInfer must be >= 0.7.2 and the SM120 FlashMLA
backend FlashInfer, ``SGLANG_DSV4_KV_LAYOUT`` must be unset / auto / v41 and ``SGLANG_DSV4_COMPRESSED_KV_LAYOUT``
auto, and unified_kv must be off. Before re-pricing, the stock numbers are recomputed from the formula at 584 B and
must equal what the stock ``__init__`` produced.

Credits: the V4.1 layouts, their writers and the (V41, V41_FP4) pool support are SGLang's (dsv4.1 branch). The SM12x
selector and reader route follow sgl-project/sglang#45851 (05yuki). The SM120 DSv4.1 kernels are FlashInfer's
(flashinfer-ai/flashinfer#5197, #6616, lucifer1004). The pricing gap was found from luz-deepseek-flash-all-in-one's
"A1 / A2 / A7" notes (luxingcom).
"""
from __future__ import annotations

import logging
import os

from v41c_pool import FP4_BYTES, LOW_RATIOS, V4_BYTES, low_ratio_bytes, source_pin

logger = logging.getLogger(__name__)

ENV = "DSV41_KV_LAYOUT"
VALUE = "v41"
POOL_MODULE = "sglang.srt.mem_cache.deepseek_v4_memory_pool"
CONFIG_MODULE = "sglang.srt.model_executor.pool_configurator"
KVC_MODULE = "sglang.srt.mem_cache.kv_cache_configurator"
BACKEND_MODULE = "sglang.srt.layers.attention.deepseek_v4_backend"
MODULES = (POOL_MODULE, CONFIG_MODULE, KVC_MODULE)   # added to sitecustomize's finder when the flag is v41
V41_BYTES = 528                  # KVLayout.V41: 512 fp8 (RoPE included) + 16 ue8m0 scales
V41_PAGE_BYTES = {256: 135168, 128: 67584, 64: 33792}
FP4_PAGE_BYTES = {256: 73728, 128: 36864}
MIN_FLASHINFER = (0, 7, 2)

# sha256 of the stock source text (source_pin) of everything this module wraps or whose behaviour it assumes, in the
# staged SGLang tree of v2.6.5 (f80c91a4b + the recipe's patch train, runtime/sglang-canary).
PINS = {
    POOL_MODULE: {
        "select_dsv4_kv_layout": "bcf6a07d5aca6ce7080df785d68de7f82175744364df757f9f76b6c2077371d6",
        "resolve_compressed_kv_layout": "ea3cb4e5d022d40573f0c2e66bfb598cfd383ae56b110b2f1a06447048a38a06",
        "DeepSeekV4TokenToKVPool.compressed_kv_layout": "a26b02459190c0f67325fd3e03e186cfb07564bbfe3181e5f6145b365b7b5890",
        "DeepSeekV4TokenToKVPool._init_compressed_pools": "0d8c4434e5ed405f6a1702011b4800f8daec048caa6ce98e786110f41cf0310c",
        "DeepSeekV4TokenToKVPool._make_kv_pool": "2bf7bdc4dfdfe8119c4323c0ddff0150957d562d489b36afa5d48515a031517b",
        "DeepSeekV4SingleKVPool.get_bytes_per_token": "b96098689317d5a111df93f2f17d67cdeea07c0ca7b33ddcc3c3fd1684c0911f",
        "DeepSeekV4SingleKVPool.create_buffer": "fbc46d81ace879cf3d9af25e93848b4b62d3e957b4e231120dae38f23836eac0",
        "DeepSeekV4SingleKVPool.set_key_buffer_fused": "4ce1edee872787e9cfc18387c76debea7a7ca1fa6bb05bae442f5c04557a2bcc",
        # the SWA writers take the pool's layout (V41: in-kernel e4m3 g32 ue8m0, RoPE included)
        "DeepSeekV4TokenToKVPool.set_swa_key_buffer_radix_fused": "7d00681ba3a705093cee551d179167564858bfa0ce50daf9cd7ed2d438337216",
        "DeepSeekV4TokenToKVPool.set_swa_key_buffer_radix_fused_norm_rope": "12a00674ba270c8a31fc1b06588c5c5e2c5f2a5d71c6969b33ff6ef62f4d4e2d",
        "DeepSeekV4TokenToKVPool.set_extra_key_buffer_fused": "3cc3857443d27b6cc827e76af2879ea030500f125c4f12f2a9bea625a849a479",
        # the backend views the SWA cache at this width (528 -> the canary's V41 route)
        "DeepSeekV4TokenToKVPool.get_swa_key_bytes_per_token": "351d4a1d36a519613411c49af0451ff8bd94be77c5d9ac53953cf7422767fdad",
    },
    CONFIG_MODULE: {
        "DSV4PoolConfigurator.__init__": "613151c2cd15be1076638ff0c898fd20dc6df6148c430a3ea992e105e794302a",
        "DSV4PoolConfigurator._get_bytes_per_full_token": "2aad8ab3f89d170b633bd39857fa55818e12da836ef717199c1bd0b9a6c3c21a",
        "DSV4PoolConfigurator._get_bytes_per_swa_token": "1c449dd434d7d34783c1991d400a644efd230aa4b58c8ef3b6cc149c39719b42",
    },
    BACKEND_MODULE: {
        "DeepseekV4AttnBackend._low_ratio_compress_fused": "215e611b57cc1e0d4d463c50c1352c87241903f5f7f657e752fd457e9fc310be",
        "DeepseekV4AttnBackend._low_ratio_write_group": "7177b031df284861aafdb0e2eaef8ad0edf9af4c75dc961874afd14d2b503d98",
        "DeepseekV4AttnBackend._forward_attention": "57be88e9f00790bc7e751ac051c2300cdc530be5da4acd2e76e606ba5a5c87e1",
    },
}


def enabled() -> bool:
    return os.environ.get(ENV, "").strip() == VALUE


def check_pins(module, pins=None) -> None:
    pins = PINS[module.__name__] if pins is None else pins
    path = getattr(module, "__file__", None)
    if not path or not os.path.exists(path):
        raise RuntimeError(f"{ENV}=v41: no source file for {module.__name__}; refusing to patch it blind")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    bad = [f"{q} {(source_pin(text, q) or 'missing')[:12]} != {want[:12]}"
           for q, want in pins.items() if source_pin(text, q) != want]
    if bad:
        raise RuntimeError(f"{ENV}=v41: {module.__name__} is not the pinned SGLang source ({'; '.join(bad)}); "
                           "the V4.1 layout patch was written against v2.6.5's tree and refuses another")


def check_layout(KVLayout) -> None:
    v41, fp4 = KVLayout.V41, KVLayout.V41_FP4
    got = (v41.bytes_per_token, v41.data_bytes, v41.scale_bytes, v41.tile_size,
           tuple(v41.page_bytes(p) for p in sorted(V41_PAGE_BYTES)),
           fp4.bytes_per_token, tuple(fp4.page_bytes(p) for p in sorted(FP4_PAGE_BYTES)), KVLayout.V4.bytes_per_token)
    want = (V41_BYTES, 512, 16, 32, tuple(V41_PAGE_BYTES[p] for p in sorted(V41_PAGE_BYTES)),
            FP4_BYTES, tuple(FP4_PAGE_BYTES[p] for p in sorted(FP4_PAGE_BYTES)), V4_BYTES)
    if got != want:
        raise RuntimeError(f"{ENV}=v41: KVLayout values {got} are not the V4.1 record ABI {want}")


def flashinfer_version():
    from importlib.metadata import version

    return version("flashinfer-python")


def check_flashinfer(version=None) -> str:
    v = flashinfer_version() if version is None else version
    try:
        parts = tuple(int(x) for x in v.split("+")[0].split(".")[:3])
    except ValueError:
        parts = ()
    if len(parts) < 3 or parts < MIN_FLASHINFER:
        raise RuntimeError(f"{ENV}=v41 needs FlashInfer >= 0.7.2 (SM120 DSv4.1 kernels); installed {v!r}")
    return v


def check_env(envs) -> None:
    mode = str(os.environ.get("SGLANG_DSV4_KV_LAYOUT", "")).strip().lower()
    if mode not in ("", "auto", "v41"):
        raise ValueError(f"{ENV}=v41 selects the V4.1 layouts itself; SGLANG_DSV4_KV_LAYOUT={mode!r} conflicts with it")
    option = str(envs.SGLANG_DSV4_COMPRESSED_KV_LAYOUT.get()).strip().lower()
    if option not in ("", "auto"):
        raise ValueError(f"{ENV}=v41 keeps the stock compressed layouts (ratio 1/2 V41_FP4); "
                         f"SGLANG_DSV4_COMPRESSED_KV_LAYOUT={option!r} conflicts with it")
    backend = str(envs.SGLANG_SM120_FLASHMLA_BACKEND.get()).strip().lower()
    if backend != "flashinfer":
        raise RuntimeError(f"{ENV}=v41 needs SGLANG_SM120_FLASHMLA_BACKEND=flashinfer (got {backend!r}): only "
                           "FlashInfer's SM120 kernels read the V4.1 layouts")


def _capability_major():
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()[0]


# ------------------------------------------------------------------------------------------------ pool module
def install_pool(module, capability=None, fi_version=None) -> None:
    """Patch sglang.srt.mem_cache.deepseek_v4_memory_pool (called by sitecustomize right after it executes)."""
    if getattr(module, "_dsv41_v41_pool", False):
        return
    check_pins(module)
    KVLayout = module.KVLayout
    check_layout(KVLayout)
    check_env(module.envs)
    check_flashinfer(fi_version)
    stock_select = module.select_dsv4_kv_layout
    major_of = _capability_major if capability is None else capability

    def select_dsv4_kv_layout():
        major = major_of()
        if major == 12:
            check_env(module.envs)
            return KVLayout.V41, None
        if major == 10:
            return stock_select()
        raise RuntimeError(f"{ENV}=v41 is for SM12x (DGX Spark GB10) here; GPU capability major {major}")

    select_dsv4_kv_layout.__wrapped__ = stock_select
    select_dsv4_kv_layout.__doc__ = (stock_select.__doc__ or "") + "\n\n(DSV41_KV_LAYOUT=v41: SM12x -> (V41, None))"
    select_dsv4_kv_layout._dsv41_v41 = True
    module.select_dsv4_kv_layout = select_dsv4_kv_layout

    cls = module.DeepSeekV4TokenToKVPool
    stock_init = cls.__init__

    def __init__(self, *args, **kwargs):
        stock_init(self, *args, **kwargs)
        line = check_pool(self, KVLayout)
        logger.warning(line)
        print(line, flush=True)

    __init__.__wrapped__ = stock_init
    cls.__init__ = __init__
    module._dsv41_v41_pool = True


def install_kv_configurator(module) -> None:
    """sglang.srt.mem_cache.kv_cache_configurator imports select_dsv4_kv_layout by name: point it at the wrapper."""
    import sys

    pool = sys.modules.get(POOL_MODULE)
    if pool is None or not getattr(pool, "_dsv41_v41_pool", False):
        raise RuntimeError(f"{ENV}=v41: {KVC_MODULE} imported before the pool module was patched")
    current = getattr(module, "select_dsv4_kv_layout", None)
    if current is pool.select_dsv4_kv_layout:
        return
    if current is not getattr(pool.select_dsv4_kv_layout, "__wrapped__", None):
        raise RuntimeError(f"{ENV}=v41: {KVC_MODULE}.select_dsv4_kv_layout is neither the stock nor the patched one")
    module.select_dsv4_kv_layout = pool.select_dsv4_kv_layout


def check_pool(pool, KVLayout) -> str:
    """Post-construction check of a DeepSeekV4TokenToKVPool under v41; returns the receipt line."""
    if getattr(pool, "_unified_kv", False):
        raise RuntimeError(f"{ENV}=v41: unified_kv keeps bf16 rows, not paged V4.1 records; unset it")
    if pool.kv_layout is not KVLayout.V41:
        raise RuntimeError(f"{ENV}=v41: the main cache is {pool.kv_layout}, not V41")
    swa_bytes = pool.get_swa_key_bytes_per_token() if hasattr(pool, "get_swa_key_bytes_per_token") else None
    if swa_bytes is not None and swa_bytes != V41_BYTES:
        raise RuntimeError(f"{ENV}=v41: the SWA cache is viewed at {swa_bytes} B per token, not {V41_BYTES}")
    parts = []
    for ratio in LOW_RATIOS:
        p = (getattr(pool, "kv_pools", None) or {}).get(ratio)
        if p is None:
            continue
        want_page = FP4_PAGE_BYTES.get(p.page_size, KVLayout.V41_FP4.page_bytes(p.page_size))
        shapes = {tuple(b.shape) for b in p.kv_buffer}
        if p.kv_layout is not KVLayout.V41_FP4 or p.bytes_per_page_padded != want_page \
                or any(len(s) != 2 or s[1] != want_page for s in shapes) or p.kv_cache_total_dim != FP4_BYTES:
            raise RuntimeError(f"{ENV}=v41: ratio-{ratio} pool is {p.kv_layout} with pages {shapes} "
                               f"(want V41_FP4, {want_page} B per {p.page_size}-token page)")
        parts.append(f"r{ratio}=v41_fp4:page={p.page_size}:layers={len(p.kv_buffer)}"
                     f":bytes={sum(int(b.numel()) * int(b.element_size()) for b in p.kv_buffer)}")
    swa = getattr(pool, "swa_kv_pool", None)
    if swa is not None:
        if swa.kv_layout is not KVLayout.V41 or swa.kv_cache_total_dim != V41_BYTES:
            raise RuntimeError(f"{ENV}=v41: the paged SWA pool is {swa.kv_layout} at {swa.kv_cache_total_dim} B")
        parts.insert(0, f"swa=v41:page={swa.page_size}:layers={len(swa.kv_buffer)}"
                        f":bytes={sum(int(b.numel()) * int(b.element_size()) for b in swa.kv_buffer)}")
    elif getattr(pool, "request_window", None) is not None:
        parts.insert(0, "swa=v41:request_window")
    return (f"DSV41_V41_POOL main=v41 {' '.join(parts) or 'no_paged_pools'} "
            f"full_size={getattr(pool, 'full_size', None)}")


# ------------------------------------------------------------------------------------------------ sizing
def request_window_bytes(conf, kv_bytes, chunked_prefill_size, ceil_align) -> int:
    """The stock DSV4PoolConfigurator.__init__ request-window expression (encoder replay) at a given slot size."""
    slots = conf.requested_max_running_requests_per_worker + 1
    capacity = ceil_align(conf.sliding_window_size + conf.online_c128_mtp_max_draft_tokens, conf.page_size)
    layers = conf.num_layers_total
    scratch = (max(chunked_prefill_size, slots * max(128, conf.online_c128_mtp_max_draft_tokens))
               + slots * 128 + conf.page_size)
    return ((slots * capacity + conf.page_size) * layers * (kv_bytes + 16)
            + 4 * scratch * (kv_bytes + 16)
            + slots * 3 * 16 * conf.attn_head_dim * 8)


def install_configurator(module) -> None:
    """Patch sglang.srt.model_executor.pool_configurator (called by sitecustomize right after it executes)."""
    if getattr(module, "_dsv41_v41_sizing", False):
        return
    check_pins(module)
    cls = module.DSV4PoolConfigurator
    stock_init = cls.__init__

    def __init__(self, kvc):
        stock_init(self, kvc)
        line = resize(self, kvc, module)
        logger.warning(line)
        print(line, flush=True)

    __init__.__wrapped__ = stock_init
    cls.__init__ = __init__
    module._dsv41_v41_sizing = True


def _sizing(conf, kv_bytes, low_bytes, kvc, module):
    """(low_ratio_bytes_per_full_token, request_window_bytes, bytes_per_swa_token, bytes_per_full_token) of an
    initialised configurator at the given latent slot sizes, by the stock formulas."""
    cfg = kvc.model_config
    index_bytes = module.get_dsv4_indexer_bytes_per_token(conf.indexer_head_dim, use_fp4_indexer=True)
    low = low_ratio_bytes(cfg.compress_ratios, list(cfg.hf_config.kv_source_layer_ids), kvc.layer_info.start_layer,
                          kvc.layer_info.end_layer, low_bytes, index_bytes)
    saved = (conf.kv_bytes, conf.low_ratio_bytes_per_full_token)
    try:
        conf.kv_bytes, conf.low_ratio_bytes_per_full_token = kv_bytes, low
        rw = 0
        if conf.encoder_replay:
            rw = request_window_bytes(conf, kv_bytes, module.get_schedule().chunked_prefill_size, module.ceil_align)
        swa = conf._get_bytes_per_swa_token()
        full = conf._get_bytes_per_full_token()
    finally:
        conf.kv_bytes, conf.low_ratio_bytes_per_full_token = saved
    infl = conf._spec_infl if (conf.is_speculative and not conf.encoder_replay) else 1.0
    return low, rw, swa * infl, full * infl


def resize(conf, kvc, module) -> str:
    """Re-price an initialised DSV4PoolConfigurator at the V4.1 slots; returns the receipt line."""
    if getattr(conf, "_unified", False):
        raise RuntimeError(f"{ENV}=v41: unified_kv sizing has no V4.1 paged pools; unset it")
    if conf.kv_bytes != V4_BYTES:
        raise RuntimeError(f"{ENV}=v41: the configurator's latent slot is {conf.kv_bytes} B, not the V4 {V4_BYTES}")
    stock = _sizing(conf, V4_BYTES, V4_BYTES, kvc, module)
    have = (conf.low_ratio_bytes_per_full_token, conf.request_window_bytes, conf.bytes_per_swa_token,
            conf.bytes_per_full_token)
    if any(abs(a - b) > 1e-6 * max(1.0, abs(b)) for a, b in zip(stock, have)):
        raise RuntimeError(f"{ENV}=v41: the stock sizing {have} is not the formula this patch re-prices ({stock})")
    low, rw, swa, full = _sizing(conf, V41_BYTES, FP4_BYTES, kvc, module)
    conf.kv_bytes = V41_BYTES
    conf.low_ratio_bytes_per_full_token, conf.request_window_bytes = low, rw
    conf.bytes_per_swa_token, conf.bytes_per_full_token = swa, full
    if not (0 < full <= have[3] and 0 <= swa <= have[2]):
        raise RuntimeError(f"{ENV}=v41: re-priced sizing {(low, rw, swa, full)} vs stock {have}")
    return (f"DSV41_V41_SIZING slot_bytes={V4_BYTES}->{V41_BYTES} record_bytes={FP4_BYTES} "
            f"low_ratio_bytes_per_full_token={have[0]:.2f}->{low:.2f} "
            f"bytes_per_full_token={have[3]:.2f}->{full:.2f} bytes_per_swa_token={have[2]:.2f}->{swa:.2f} "
            f"request_window_bytes={have[1]}->{rw}")


def check_backend(module) -> None:
    """sglang.srt.layers.attention.deepseek_v4_backend: the write / read call sites this design relies on."""
    check_pins(module)
