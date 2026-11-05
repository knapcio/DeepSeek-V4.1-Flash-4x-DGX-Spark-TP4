"""DSV41_KV_LAYOUT=v41c, step 3: SGLang's DeepSeek-V4 KV pools allocate the ratio-1 / ratio-2 compressed caches as
V4.1 FP4 records (KVLayout.V41_FP4, 288 B per token) next to the 584-byte V4 SWA (main) cache, and the pool sizing
prices those records at 288 B, so the same byte budget holds more tokens.

Default off. sitecustomize checks the flag before importing this module; unset or ``v4`` leaves SGLang untouched
(nothing here is imported), ``v41c`` installs, anything else raises (the same rule as dsv41_v41c.enabled()).

What changes when on (and nothing else):

* ``deepseek_v4_memory_pool.resolve_compressed_kv_layout``: a V4 main cache with compress ratio 1 or 2 resolves to
  ``V41_FP4`` (stock: V4). Every other (main, ratio) pair is the stock answer. SGLang's own writers already follow the
  pool's layout (c1 / c2 decode stores, the extend store with in-kernel RoPE, ``fused_store_cache``); the v41w window
  checked them on GB10 against the model's rounding. The readers are dsv41_v41c (group reader: one transcode per layer
  group into a V4 scratch read by the stock kernels; DSV41_V41C_READER=io: f092906's IO-warp decode / per-layer token
  mode), routed on the 288-byte record size.
* ``deepseek_v4_memory_pool.is_valid_kv_layout_pair``: additionally accepts (V4 main, V41_FP4 extra). Stock pairs
  only V41 with V41_FP4 (the SM100 FlashMLA decode kernel).
* ``DSV4PoolConfigurator``: the ratio-1/2 term of bytes_per_full_token prices one latent at 288 B instead of the
  FlashMLA V4 slot (584 B); the SWA ring / request window and the draft's paged SWA stay at 584 B (SWA stays V4).
* ``DeepSeekV4TokenToKVPool``: after construction the low-ratio pools are checked (V41_FP4, page bytes =
  page_size x 288, SWA V4) and one receipt line is printed per rank: ``DSV41_V41C_POOL ...``.
* ``DeepseekV4AttnBackend._forward_attention`` (group reader only): wrapped to tag the reader with (layer id, layer
  group start) around the call; the call itself is unchanged (install_layer_tags).

Fail closed: every SGLang function this changes or relies on is pinned by the sha256 of its source text
(``source_pin``: the dedented source lines of the def / class, from the module's own file), the KVLayout values are
asserted (288 B records, 73,728 / 36,864-byte pages at 256 / 128 tokens), the SM120 FlashMLA backend must be
FlashInfer (the triton / torch fallbacks cannot read FP4 records), unified_kv must be off, and a forced
SGLANG_DSV4_COMPRESSED_KV_LAYOUT refuses. A mismatch raises at import, so the engine does not start.

Credits: the V4.1 FP4 record layout, its writers and the (V41, V41_FP4) pool support are SGLang's (dsv4.1 branch,
KVLayout / resolve_compressed_kv_layout); the record arithmetic was aligned with local-inference-lab b12x_next's
deepseek_v41 writer (reference only); the readers build on FlashInfer's sparse_mla_sm120.
"""
from __future__ import annotations

import ast
import hashlib
import logging
import os
import textwrap

logger = logging.getLogger(__name__)

ENV = "DSV41_KV_LAYOUT"
POOL_MODULE = "sglang.srt.mem_cache.deepseek_v4_memory_pool"
CONFIG_MODULE = "sglang.srt.model_executor.pool_configurator"
BACKEND_MODULE = "sglang.srt.layers.attention.deepseek_v4_backend"
MODULES = (POOL_MODULE, CONFIG_MODULE)       # the modules sitecustomize must add to its finder (BACKEND is there already)
FP4_BYTES = 288                  # KVLayout.V41_FP4.bytes_per_token: 256 e2m1 bytes + 32 e4m3 scales
V4_BYTES = 584                   # KVLayout.V4: 448 fp8 + 64 bf16 RoPE + 7 ue8m0 + 1 pad
LOW_RATIOS = (1, 2)
FP4_PAGE_BYTES = {256: 73728, 128: 36864}    # ratio-1 / ratio-2 pools at the 256-token model page

# sha256 of the stock source text (source_pin) of everything this module wraps or whose behaviour it assumes, in the
# staged SGLang dsv4.1 tree of v2.6 (f80c91a4b + the recipe's patch train; runtime/sglang-canary, f744d83).
PINS = {
    POOL_MODULE: {
        "resolve_compressed_kv_layout": "ea3cb4e5d022d40573f0c2e66bfb598cfd383ae56b110b2f1a06447048a38a06",
        "DeepSeekV4TokenToKVPool.compressed_kv_layout": "a26b02459190c0f67325fd3e03e186cfb07564bbfe3181e5f6145b365b7b5890",
        "DeepSeekV4TokenToKVPool._init_compressed_pools": "0d8c4434e5ed405f6a1702011b4800f8daec048caa6ce98e786110f41cf0310c",
        "DeepSeekV4TokenToKVPool._make_kv_pool": "2bf7bdc4dfdfe8119c4323c0ddff0150957d562d489b36afa5d48515a031517b",
        "DeepSeekV4SingleKVPool.get_bytes_per_token": "b96098689317d5a111df93f2f17d67cdeea07c0ca7b33ddcc3c3fd1684c0911f",
        "DeepSeekV4SingleKVPool.create_buffer": "fbc46d81ace879cf3d9af25e93848b4b62d3e957b4e231120dae38f23836eac0",
        "DeepSeekV4SingleKVPool.set_key_buffer_fused": "4ce1edee872787e9cfc18387c76debea7a7ca1fa6bb05bae442f5c04557a2bcc",
    },
    CONFIG_MODULE: {
        "DSV4PoolConfigurator.__init__": "613151c2cd15be1076638ff0c898fd20dc6df6148c430a3ea992e105e794302a",
        "DSV4PoolConfigurator._get_bytes_per_full_token": "2aad8ab3f89d170b633bd39857fa55818e12da836ef717199c1bd0b9a6c3c21a",
        "DSV4PoolConfigurator._get_bytes_per_swa_token": "1c449dd434d7d34783c1991d400a644efd230aa4b58c8ef3b6cc149c39719b42",
    },
    # the writers follow the pool's layout (c1 / c2 decode stores take get_extra_key_layout, the extend store passes
    # the un-quantized latent + freqs_cis to a V41_FP4 pool) and the attention call views the extra cache at
    # get_extra_key_bytes_per_token (288 -> the v41c readers); _forward_attention is wrapped (layer tags), not changed
    BACKEND_MODULE: {
        "DeepseekV4AttnBackend._low_ratio_compress_fused": "215e611b57cc1e0d4d463c50c1352c87241903f5f7f657e752fd457e9fc310be",
        "DeepseekV4AttnBackend._low_ratio_write_group": "7177b031df284861aafdb0e2eaef8ad0edf9af4c75dc961874afd14d2b503d98",
        "DeepseekV4AttnBackend._forward_attention": "57be88e9f00790bc7e751ac051c2300cdc530be5da4acd2e76e606ba5a5c87e1",
    },
}


def enabled() -> bool:
    raw = os.environ.get(ENV, "").strip()
    if raw in ("", "v4"):
        return False
    if raw == "v41c":
        return True
    raise ValueError(f"{ENV} must be unset, v4 or v41c, got {raw!r}")


# ------------------------------------------------------------------------------------------------ source pins
def _node(tree: ast.Module, qualname: str):
    body, node = tree.body, None
    for part in qualname.split("."):
        node = next((n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and n.name == part), None)
        if node is None:
            return None
        body = node.body
    return node


def source_pin(text: str, qualname: str):
    """sha256 of the dedented source lines (decorators included) of ``qualname`` in a module's source text, or None."""
    node = _node(ast.parse(text), qualname)
    if node is None:
        return None
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    lines = text.splitlines(keepends=True)[first - 1:node.end_lineno]
    return hashlib.sha256(textwrap.dedent("".join(lines)).encode()).hexdigest()


def check_pins(module, pins=None) -> None:
    pins = PINS[module.__name__] if pins is None else pins
    path = getattr(module, "__file__", None)
    if not path or not os.path.exists(path):
        raise RuntimeError(f"{ENV}=v41c: no source file for {module.__name__}; refusing to patch it blind")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    bad = [f"{q} {(source_pin(text, q) or 'missing')[:12]} != {want[:12]}"
           for q, want in pins.items() if source_pin(text, q) != want]
    if bad:
        raise RuntimeError(f"{ENV}=v41c: {module.__name__} is not the pinned SGLang source ({'; '.join(bad)}); "
                           "the pool patch was written against v2.6's tree and refuses another")


# ------------------------------------------------------------------------------------------------ layout values
def check_layout(KVLayout) -> None:
    fp4 = KVLayout.V41_FP4
    got = (fp4.bytes_per_token, fp4.data_bytes, fp4.scale_bytes, fp4.page_bytes(256), fp4.page_bytes(128),
           KVLayout.V4.bytes_per_token)
    want = (FP4_BYTES, 256, 32, FP4_PAGE_BYTES[256], FP4_PAGE_BYTES[128], V4_BYTES)
    if got != want:
        raise RuntimeError(f"{ENV}=v41c: KVLayout values {got} are not the V4.1 FP4 record ABI {want}")


# ------------------------------------------------------------------------------------------------ pool module
def install_pool(module) -> None:
    """Patch sglang.srt.mem_cache.deepseek_v4_memory_pool (called by sitecustomize right after it executes)."""
    if getattr(module, "_dsv41_v41c_pool", False):
        return
    check_pins(module)
    KVLayout = module.KVLayout
    check_layout(KVLayout)
    backend = str(module.envs.SGLANG_SM120_FLASHMLA_BACKEND.get()).strip().lower()
    if backend != "flashinfer":
        raise RuntimeError(f"{ENV}=v41c needs SGLANG_SM120_FLASHMLA_BACKEND=flashinfer (got {backend!r}): only the "
                           "FlashInfer route has the FP4 record readers")
    stock_resolve = module.resolve_compressed_kv_layout
    stock_valid = module.is_valid_kv_layout_pair
    V4, FP4 = KVLayout.V4, KVLayout.V41_FP4

    def resolve_compressed_kv_layout(kv_layout, compress_ratio, option=None):
        if kv_layout is V4 and compress_ratio in LOW_RATIOS:
            opt = None if option is None else str(option).strip().lower()
            if opt not in (None, "", "auto"):
                raise ValueError(f"{ENV}=v41c sets the ratio-1/2 layout itself; "
                                 f"SGLANG_DSV4_COMPRESSED_KV_LAYOUT={option!r} conflicts with it")
            return FP4
        return stock_resolve(kv_layout, compress_ratio, option)

    def is_valid_kv_layout_pair(kv, extra_kv):
        return (kv is V4 and extra_kv is FP4) or stock_valid(kv, extra_kv)

    resolve_compressed_kv_layout.__doc__ = (stock_resolve.__doc__ or "") + "\n\n(DSV41_KV_LAYOUT=v41c: V4 + ratio 1/2 -> V41_FP4)"
    module.resolve_compressed_kv_layout = resolve_compressed_kv_layout
    module.is_valid_kv_layout_pair = is_valid_kv_layout_pair

    cls = module.DeepSeekV4TokenToKVPool
    stock_init = cls.__init__

    def __init__(self, *args, **kwargs):
        stock_init(self, *args, **kwargs)
        line = check_pool(self, KVLayout)
        logger.warning(line)
        print(line, flush=True)

    __init__.__wrapped__ = stock_init
    cls.__init__ = __init__
    module._dsv41_v41c_pool = True


def check_pool(pool, KVLayout) -> str:
    """Post-construction check of a DeepSeekV4TokenToKVPool under v41c; returns the receipt line."""
    if getattr(pool, "_unified_kv", False):
        raise RuntimeError(f"{ENV}=v41c: unified_kv keeps bf16 rows, not paged FP4 records; unset it")
    if pool.kv_layout is not KVLayout.V4:
        raise RuntimeError(f"{ENV}=v41c keeps the SWA cache V4; the main cache is {pool.kv_layout}")
    parts = []
    for ratio in LOW_RATIOS:
        p = (getattr(pool, "kv_pools", None) or {}).get(ratio)
        if p is None:
            continue
        want_page = FP4_PAGE_BYTES.get(p.page_size, KVLayout.V41_FP4.page_bytes(p.page_size))
        shapes = {tuple(b.shape) for b in p.kv_buffer}
        if p.kv_layout is not KVLayout.V41_FP4 or p.bytes_per_page_padded != want_page \
                or any(len(s) != 2 or s[1] != want_page for s in shapes) or p.kv_cache_total_dim != FP4_BYTES:
            raise RuntimeError(f"{ENV}=v41c: ratio-{ratio} pool is {p.kv_layout} with pages {shapes} "
                               f"(want V41_FP4, {want_page} B per {p.page_size}-token page)")
        pages = next(iter(shapes))[0] if shapes else 0
        nbytes = sum(int(b.numel()) * int(b.element_size()) for b in p.kv_buffer)
        parts.append(f"r{ratio}=v41_fp4:page={p.page_size}:page_bytes={want_page}:layers={len(p.kv_buffer)}"
                     f":pages={pages}:bytes={nbytes}")
    if not parts:
        return f"DSV41_V41C_POOL swa=v4 low_ratio_pools=0 (no ratio-1/2 kv_source layers in this pool)"
    return f"DSV41_V41C_POOL swa=v4 {' '.join(parts)} full_size={getattr(pool, 'full_size', None)}"


def check_backend(module) -> None:
    """sglang.srt.layers.attention.deepseek_v4_backend: the pins (the write / read call sites this design relies on),
    then the layer tags of the group reader (install_layer_tags; not with DSV41_V41C_READER=io). The native reader
    keeps the tags: its fallback for shapes v41n_prefill does not serve is the group path."""
    check_pins(module)
    if reader_mode() in ("group", "native"):
        install_layer_tags(module)


# ------------------------------------------------------------------------------------------------ layer tags
READER_ENV = "DSV41_V41C_READER"
READER_MODULE = "sglang.kernels.ops.attention.dsv41_v41c"


def reader_mode() -> str:
    """The same rule as dsv41_v41c.reader(): unset / group -> group, io -> io, native -> native, anything else
    raises."""
    raw = os.environ.get(READER_ENV, "").strip()
    if raw in ("", "group"):
        return "group"
    if raw == "io":
        return "io"
    if raw == "native":
        return "native"
    raise ValueError(f"{READER_ENV} must be unset, group, io or native, got {raw!r}")


def group_starts(cfg) -> dict:
    """{layer: group start} for the ratio-1/2 layers of a DeepSeek-V4.1 text config.

    group start = the latest kv_source / index_source layer of the same ratio at or before the layer. Only kv_source
    layers write a ratio-1/2 extra cache and only index_source layers refill that ratio's top-k buffer (models/
    deepseek_v4.py builds the compressor / indexer only there), so from a group start to the next one the extra cache
    and the top-k a layer reads are the same. Missing fields -> {} (no tags: the reader transcodes every call)."""
    ratios = list(getattr(cfg, "compress_ratios", None) or [])
    kv = sorted(getattr(cfg, "kv_source_layer_ids", None) or [])
    ix = sorted(getattr(cfg, "index_source_layer_ids", None) or [])
    out = {}
    for layer, ratio in enumerate(ratios):
        if ratio not in LOW_RATIOS:
            continue
        ks = [s for s in kv if s <= layer and s < len(ratios) and ratios[s] == ratio]
        xs = [s for s in ix if s <= layer and s < len(ratios) and ratios[s] == ratio]
        if ks and xs:
            out[layer] = max(ks[-1], xs[-1])
    return out


def _reader_module():
    import importlib

    return importlib.import_module(READER_MODULE)


def install_layer_tags(module) -> None:
    """Wrap DeepseekV4AttnBackend._forward_attention (pinned): tag the reader with (layer_id, group start) for the
    call, clear the tag after it. Python only (eager and capture time); a captured graph replays the kernels the
    tagged capture recorded."""
    cls = module.DeepseekV4AttnBackend
    if getattr(cls._forward_attention, "_dsv41_v41c_tags", False):
        return
    stock = cls._forward_attention
    state = {}

    def _forward_attention(self, q, k, v, layer, forward_batch, *args, **kwargs):
        starts = self.__dict__.get("_dsv41_v41c_starts")
        if starts is None:
            try:
                starts = group_starts(self.model_runner.model_config.hf_text_config)
            except AttributeError:
                starts = {}
            self._dsv41_v41c_starts = starts
        lid = getattr(layer, "layer_id", None)
        start = starts.get(lid)
        if start is None:
            return stock(self, q, k, v, layer, forward_batch, *args, **kwargs)
        rd = state.get("reader")
        if rd is None:
            rd = state["reader"] = _reader_module()
        rd.tag_layer(lid, start)
        native = state.setdefault("native", reader_mode() == "native")
        if native:
            # the heads the model keeps (o[:, :tp_q_head_num]); the native decode skips the padded head blocks
            rd.tag_heads(getattr(layer, "tp_q_head_num", None))
        try:
            return stock(self, q, k, v, layer, forward_batch, *args, **kwargs)
        finally:
            rd.clear_tag()
            if native:
                rd.clear_heads()

    _forward_attention.__wrapped__ = stock
    _forward_attention._dsv41_v41c_tags = True
    cls._forward_attention = _forward_attention


# ------------------------------------------------------------------------------------------------ sizing
def low_ratio_bytes(compress_ratios, kv_source_layer_ids, start, end, kv_bytes, index_bytes) -> float:
    """The stock DSV4PoolConfigurator.low_ratio_bytes_per_full_token formula with a given latent slot size."""
    return sum((kv_bytes + index_bytes) / compress_ratios[l] for l in kv_source_layer_ids
               if start <= l < end and compress_ratios[l] in LOW_RATIOS)


def install_configurator(module) -> None:
    """Patch sglang.srt.model_executor.pool_configurator (called by sitecustomize right after it executes)."""
    if getattr(module, "_dsv41_v41c_sizing", False):
        return
    check_pins(module)
    cls = module.DSV4PoolConfigurator
    stock_init = cls.__init__
    index_bytes_of = module.get_dsv4_indexer_bytes_per_token

    def __init__(self, kvc):
        stock_init(self, kvc)
        line = resize(self, kvc, index_bytes_of)
        logger.warning(line)
        print(line, flush=True)

    __init__.__wrapped__ = stock_init
    cls.__init__ = __init__
    module._dsv41_v41c_sizing = True


def resize(conf, kvc, index_bytes_of) -> str:
    """Re-price the ratio-1/2 term of an initialised DSV4PoolConfigurator at FP4_BYTES; returns the receipt line."""
    if getattr(conf, "_unified", False):
        raise RuntimeError(f"{ENV}=v41c: unified_kv sizing has no FP4 record pool; unset it")
    if conf.kv_bytes != V4_BYTES:
        raise RuntimeError(f"{ENV}=v41c: the configurator's latent slot is {conf.kv_bytes} B, not the V4 {V4_BYTES}")
    cfg = kvc.model_config
    ratios = cfg.compress_ratios
    sources = list(cfg.hf_config.kv_source_layer_ids)
    start, end = kvc.layer_info.start_layer, kvc.layer_info.end_layer
    index_bytes = index_bytes_of(conf.indexer_head_dim, use_fp4_indexer=True)
    stock = low_ratio_bytes(ratios, sources, start, end, conf.kv_bytes, index_bytes)
    if abs(stock - conf.low_ratio_bytes_per_full_token) > 1e-6:
        raise RuntimeError(f"{ENV}=v41c: the stock low-ratio term {conf.low_ratio_bytes_per_full_token} is not the "
                           f"formula this patch re-prices ({stock})")
    old_low, old_full = conf.low_ratio_bytes_per_full_token, conf.bytes_per_full_token
    if old_low == 0:
        return "DSV41_V41C_SIZING low_ratio_layers=0 (no ratio-1/2 kv_source layers in this stage: sizing unchanged)"
    conf.low_ratio_bytes_per_full_token = low_ratio_bytes(ratios, sources, start, end, FP4_BYTES, index_bytes)
    # the same expression as the stock __init__: the coefficient, then the speculative-draft inflation (1.0 if none)
    conf.bytes_per_full_token = conf._get_bytes_per_full_token() * conf._spec_infl
    if not 0 < conf.bytes_per_full_token < old_full:
        raise RuntimeError(f"{ENV}=v41c: re-priced bytes_per_full_token {conf.bytes_per_full_token} vs {old_full}")
    return (f"DSV41_V41C_SIZING low_ratio_bytes_per_full_token={old_low:.2f}->{conf.low_ratio_bytes_per_full_token:.2f} "
            f"bytes_per_full_token={old_full:.2f}->{conf.bytes_per_full_token:.2f} "
            f"capacity_x={old_full / conf.bytes_per_full_token:.4f} record_bytes={FP4_BYTES}")
