"""Exact SWA-cone replay on a lost resume point (DSV41_SWA_CONE_REPLAY=1).

Problem. A cached prefix is reusable only where its last sliding window (128 tokens) still
holds SWA slots (unified_cache/components/swa.py:319-342). When that window is evicted the
match stops at the last surviving window, often token 0, and the request re-prefills
everything after it although the prefix's full KV (the ratio-1/2 latents and index keys of
the kv_source layers) is still resident: 55-130 s at 300-700K tokens.

What this does. The stock match already reports how far the full KV reaches:
req.swa_branching_seqlen = F, the page-aligned full-KV hit beyond the SWA boundary m
(swa.py:350-357). When F - m is large, the request instead matches on full KV only at
s = F - N (N = DSV41_SWA_CONE_TOKENS, default 8192, page-aligned) and prefills [s, L) with:
  1. prefix = the resident full KV of [0, s), lock on the tree node that ends at F, so the
     resident [s, F) stays put until the request's first insert re-points to it;
  2. the replay rows [s, F) get fresh full + SWA slots (stock allocation); before the forward
     the resident ratio-1/2 latents and index keys of [s, F) are copied page for page into
     those fresh slots (byte copy of every full-page-addressed pool buffer);
  3. the replay rows' compressor writes (latent + index key, one slot vector:
     deepseek_v4_backend.py `_low_ratio_write_group`) are redirected to the padding slot 0
     (allocator/paged.py:338: "The padded slot 0 is used for writing dummy outputs from padded
     tokens"), so the copies keep the resident bytes: every main-KV / index-key read of a
     position < F returns the resident value, as on a normal prefix hit at F;
  4. the request's SWA window is floored at s (`swa_replay_start`, the engine's own floor used
     by the late-layer tail), so no row reads an SWA slot below s (tombstoned pages map to the
     shared SWA sink slot 0: allocator/swa.py:97-105). Not needed for exactness (rows below s are
     outside the dependency cone of every row >= F); it keeps the replay rows deterministic and
     off the sink slot, whose bytes padded writes overwrite at any time.
The first insert (cache_unfinished_req) dedups [s, F) against the tree: the tree keeps its
resident pages, the request's duplicate slots are freed and its req_to_token re-points to the
resident ones (unified_tree_core.py:1246-1257, unified_radix_cache.py:1196-1199).

Why it is exact. Layers 0..E-1 (E = late_layer_start = 20, deepseek_v4.py:4114-4136) run every
prefill row; each has its own per-layer SWA KV (MQALayer.wkv) and reads the shared latents /
index keys of its kv_source layer only at positions <= the row (compress_lens = (pos+1)//ratio,
deepseek_v4_backend.py `_low_ratio_index_topk_dense`). Row r's output of layer l is a function
of its own input, the layer-l SWA KV of rows [r-127, r] (each a function of that row's input to
layer l), the resident latents/keys and per-row state (embedding, Engram hash of token ids with
history from the fill ids: ngram_embedding_manager.py:187-198, mHC streams). Unrolled, a row's
output of layer E-1 depends on inputs of rows >= r - 127 * E only, so every row >= F is exact
once N >= 127 * E = 2,540, whatever the replay's first rows see below s. The late layers run only the last
128 rows of the extend (decoder bounded replay, window floored at the tail start:
deepseek_v4_backend.py `_build_late_layer_tail_metadata`), which are >= F because the match
holds back one window (schedule_batch.py:1548-1569, unified_radix_cache.py:3405-3407).
DSpark injects only those tail rows into the draft's SWA (dspark_worker_v2.py:599-618; the
draft is compress_ratio 0: deepseek_v4_dspark.py:114-122). The ratio-2 pair state starts at an
even, page-aligned s and only odd rows complete a pair (`_low_ratio_compression_metadata`).
Exactness class: the same as any prefix hit off a chunk boundary or any re-prefill: the
inside rows [F-127, F) are recomputed with another chunk shape, so kernels whose result
depends on the row count can round differently, plus the indexer top-k order noise every
prefill has (memory 2026-10-29). It is not bit-identical to a cold prefill in general.

Contaminated SWA never reaches the tree. Row r's layer-l SWA KV is exact iff r >= s + 127 * l,
so rows r < s + 127 * (E - 1) hold approximate early-layer SWA KV. The first chunk must cover all of [s, F)
and be a chunked pass that ends before the prompt end (enforced at admission: otherwise the request
falls back to the stock match; a whole-prompt pass may finish at its first token and reach the tree
through cache_finished_req, which does not free out-of-window SWA before its insert), so the
request's first insert (cache_unfinished_req) frees its SWA below
page_floor(F - 1 - 256) = F - 512 (the branch insert key ends at F; common.py:51-104) before the tree can adopt live SWA
(swa.py:388-466), and F - 512 >= s + 127 * 19 for N >= 3,072 (min_replay_tokens).

Scope and guards. Only the dense SM12x prefill path (the sparse chunk cache of other GPUs ignores
the floor: deepseek_v4_backend.py:3946-3953; not load-bearing, but not what was checked), decoder
SWA bounded replay on, paged SWA (no request window),
no HiCache / radix sessions / mamba / CP; text-only requests without prompt logprobs. Every
decision reads replicated scheduler state (tree, request, chunk budget): rank-invariant.

Knobs: DSV41_SWA_CONE_REPLAY=1 (off by default), DSV41_SWA_CONE_TOKENS (N, default 8192 = the
production profile's chunk, must be a page multiple >= min_replay_tokens), DSV41_SWA_CONE_MIN_SAVE (default 8192: replay only when
it saves at least this many prefill rows). Test only: DSV41_SWA_CONE_TEST=1 makes a request
whose rid starts with "swacone-evict" evict every evictable SWA slot before its match, and a
rid containing "-ref" skip the cone (the stock re-prefill reference).
"""
import hashlib
import inspect
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

ENV = "DSV41_SWA_CONE_REPLAY"
ENV_TOKENS = "DSV41_SWA_CONE_TOKENS"
ENV_MIN_SAVE = "DSV41_SWA_CONE_MIN_SAVE"
ENV_TEST = "DSV41_SWA_CONE_TEST"
DEFAULT_TOKENS = 8192
DEFAULT_MIN_SAVE = 8192

REQ_ATTR = "_dsv41_swa_cone"        # Plan on a Req between its match and its first prepare_for_extend
BATCH_ATTR = "_dsv41_swa_cone"      # ConeBatch on a ScheduleBatch, popped by ForwardBatch.init_new
FB_ATTR = "_dsv41_swa_cone"         # ConeBatch on the target ForwardBatch (metadata wrapper)
_MARK = "_dsv41_swa_cone_hook"
RECEIPT_EVERY = 64

# sha256 of inspect.getsource() of the engine pieces the argument above relies on, on the
# pinned SGLang dsv4.1 tree (f80c91a4b + recipe patches). On drift: not installed (stock).
PINS = {
    "schedule_batch.Req.init_next_round_input":
        "1932d7fdf86159e584102564849deea0b3884940049c5873a545ba904ceaec65",
    "schedule_batch.ScheduleBatch.prepare_for_extend":
        "392257094ffca433d64a2b1530be45135a3858503175ba2b513928499bc006b9",
    "schedule_policy.PrefillAdder._select_prefill_admission":
        "67313f2a413e91d604b62614c9d5098cf50f4e89a517d572a5b1bc8715f380bf",
    "forward_batch_info.ForwardBatch.init_new":
        "ef1f1a38a998222302d3af00303ba724c20c4bc2b3f6f4687878a9c5e47285d1",
    "deepseek_v4_backend.DeepseekV4AttnBackend.init_forward_metadata_prefill":
        "5d648034d97191fbeff62d5d6965771cddaac0b221d34d6d51bce2c8c766cdd6",
    "deepseek_v4_backend.DeepseekV4AttnBackend._build_forward_metadata":
        "f6210231f978dc167a153dabcbb59e2522fe07758d4a2caf8c902cf30fde84f0",
    "deepseek_v4_backend.DeepseekV4AttnBackend.make_core_attn_metadata":
        "7b95aa8840b13c3c635e3f68c3fc1bd6d435614c71d5aba634f7e0b9e1edbdf6",
    "deepseek_v4_backend.DeepseekV4AttnBackend.forward_low_ratio_sources":
        "6aa8ffd77738c5c69e6480dcb1fd96174d3f563fa844abe3fab5f7572ed58e96",
    "deepseek_v4_backend.DeepseekV4AttnBackend._low_ratio_compress_torch":
        "85f5e9d4711310ad03966525a38fa4755af0fe4a389ce7c0a87e8b423971a650",
    "deepseek_v4_backend.DeepseekV4AttnBackend._low_ratio_write_group":
        "b5c6bb747e1339b18d17c0ea8e2d3ae73ec403b731c2293d0ab9701ba21c09da",
    "deepseek_v4_backend._low_ratio_compression_metadata":
        "6091f3a22b7b7d866a4d2b7b88bee6bc4abfe55192de0136c402cdd3f206b044",
    "unified_tree_core.UnifiedTreeCore.match_full_device_prefix":
        "72b28eb6114dcec72450c87fd9555abe07561bda6b8ece52cbdfa8ed4cf955a7",
    "unified_tree_core.UnifiedTreeCore.collect_full_device_indices":
        "940eb4c6a4c22570b44afdcd550a53ed580c55d9ca8d3d8551f4343884dec92f",
    "unified_radix_cache.UnifiedRadixCache.cache_unfinished_req":
        "d64c72cfc5f6b129b08a6a59f072946e2c923c541f3be1e2262379b1721121c1",
    "swa.SWAComponent.update_component_on_insert_overlap":
        "6409aa3e2ec0515c9aad08b4d0d30eb5b1cea6ece0a681da817bacc35db3a68c",
    "common.free_swa_out_of_window_slots":
        "4478aa6081ca843d22332f4ee586cfe9c7f18b8b3fc3c5fcac7801c5ceb227e4",
}


class _State:
    installed: set = set()
    active = False          # cleared on pin drift or an error: stock behaviour from then on
    tokens = DEFAULT_TOKENS
    min_save = DEFAULT_MIN_SAVE
    test = False
    early_layers: Optional[int] = None   # late_layer_start of the target model (first forward)
    planned = 0
    reverted = 0
    replayed = 0
    rows = 0
    errors = 0


STATE = _State()


def _truthy(v):
    return str(v).strip().lower() not in ("", "0", "off", "false", "no")


def enabled(env=None):
    env = os.environ if env is None else env
    return _truthy(env.get(ENV, "0"))


def configure(env=None):
    """Read the knobs into STATE; returns False (and logs) when they are invalid."""
    env = os.environ if env is None else env
    STATE.active = enabled(env)
    STATE.test = _truthy(env.get(ENV_TEST, "0"))
    try:
        STATE.tokens = int(env.get(ENV_TOKENS, DEFAULT_TOKENS))
        STATE.min_save = int(env.get(ENV_MIN_SAVE, DEFAULT_MIN_SAVE))
    except ValueError:
        logger.warning("DSV41_SWA_CONE not installed: %s / %s must be integers", ENV_TOKENS, ENV_MIN_SAVE)
        STATE.active = False
    return STATE.active


# ------------------------------------------------------------------ the length rule
def min_replay_tokens(window: int, early_layers: int, page: int) -> int:
    """Smallest page-aligned replay length N for which the cone replay is exact.

    cone : a row's output of the last early layer depends on the inputs of rows >= r - (window - 1)
           * early_layers only (one window per layer), so rows >= F are exact once
           N >= (window - 1) * early_layers. Row r's layer-l SWA KV is exact iff
           r >= s + (window - 1) * l.
    guard: the first insert keeps the request's SWA live from page_floor(F - 1 - max(window,
           page)) >= F - max(window, page) - page + 1 on; those rows must be exact at every early
           layer's SWA input (r >= s + (window - 1) * (early_layers - 1)) before the tree may
           adopt them -> N >= (window - 1) * (early_layers - 1) + max(window, page) + page.
    """
    cone = (window - 1) * early_layers
    guard = (window - 1) * (early_layers - 1) + max(window, page) + page
    need = max(cone, guard)
    return -(-need // page) * page


def plan_window(m: int, full_hit: Optional[int], tokens: int, min_save: int, page: int) -> Optional[int]:
    """Replay start s, or None when the stock match is kept.

    m: the stock (SWA-valid) match length; full_hit: page-aligned full-KV hit F (None when the
    full KV does not reach past the SWA boundary)."""
    if full_hit is None or tokens <= 0 or page <= 0:
        return None
    if full_hit % page or tokens % page:
        return None
    s = full_hit - tokens
    if s <= m or s - m < min_save:
        return None
    return s


# ------------------------------------------------------------------ scheduler side
@dataclass
class Plan:
    s: int
    full_hit: int
    src_pages: List[int]                 # resident full page ids of [s, F), in order
    stock: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.full_hit - self.s


_STOCK_FIELDS = ("prefix_indices", "last_node", "last_host_node", "best_match_node",
                 "host_hit_length", "swa_host_hit_length", "swa_branching_seqlen")


def _snapshot(req) -> dict:
    snap = {k: getattr(req, k, None) for k in _STOCK_FIELDS}
    snap["cache_protected_len"] = req.kv.cache_protected_len
    return snap


def revert(req) -> bool:
    """Put back the stock match (before admission commits anything)."""
    plan = getattr(req, REQ_ATTR, None)
    if plan is None:
        return False
    for k in _STOCK_FIELDS:
        setattr(req, k, plan.stock[k])
    req.kv.cache_protected_len = plan.stock["cache_protected_len"]
    setattr(req, REQ_ATTR, None)
    STATE.reverted += 1
    return True


def _eligible_cache(cache) -> bool:
    tc = getattr(cache, "tree_core", None)
    if tc is None or getattr(cache, "disable", True):
        return False
    if getattr(tc, "enable_hicache", False) or getattr(tc, "enable_session_radix_cache", False):
        return False
    if getattr(cache, "is_mamba_enabled", False):
        return False
    if not getattr(cache, "decoder_swa_bounded_replay", False):
        return False
    if getattr(cache, "linker", None) is not None:
        return False
    # The adoption guard relies on cache_unfinished_req freeing the request's out-of-window SWA
    # before its insert (unified_radix_cache.py cache_unfinished_req); off = no such free.
    from sglang.srt.environ import envs
    if not envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.get():
        return False
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    swa = cache.components.get(ComponentType.SWA)
    return swa is not None and bool(swa.sliding_window_size)


def _eligible_req(req) -> bool:
    if getattr(req, "session", None) is not None:
        return False
    if getattr(req, "multimodal_inputs", None) is not None or getattr(req, "input_embeds", None) is not None:
        return False
    if getattr(req, "positional_embed_overrides", None) is not None:
        return False
    if getattr(req, "host_hit_length", 0):
        return False
    if req.return_logprob and req.logprob_start_len not in (-1, len(req.origin_input_ids)):
        return False
    if getattr(req, "is_dllm", lambda: False)():
        return False
    if STATE.test and "-ref" in str(req.rid):
        return False
    return True


def plan_request(req, cache) -> Optional[Plan]:
    """After the stock match: switch the request to the cone match when it saves enough.
    Mutates the request only when every check passed."""
    setattr(req, REQ_ATTR, None)
    if not STATE.active or STATE.early_layers is None:
        return None
    if not _tree_pins_ok():
        return None
    if not _eligible_cache(cache) or not _eligible_req(req):
        return None
    page = cache.page_size
    window = cache.sliding_window_size
    tokens = STATE.tokens
    if tokens < min_replay_tokens(window, STATE.early_layers, page):
        return None
    m = len(req.prefix_indices)
    full_hit = req.swa_branching_seqlen
    s = plan_window(m, full_hit, tokens, STATE.min_save, page)
    if s is None:
        return None
    from sglang.srt.mem_cache.radix_cache import RadixKey

    tc = cache.tree_core
    key = RadixKey(token_ids=req.full_untruncated_fill_ids, extra_key=req.extra_key,
                   limit=full_hit, cache_salt=req.cache_salt)
    matched, node_id, pinned = tc.match_full_device_prefix(key)
    if matched != full_hit or pinned < full_hit:
        return None
    idx = tc.collect_full_device_indices(node_id, tc.root_node.id)
    if idx.numel() < full_hit:
        return None
    seg = idx[s:full_hit].to("cpu").view(-1, page)
    import torch
    base = seg[:, :1]
    if not bool(((seg - base) == torch.arange(page, dtype=seg.dtype)).all()) or bool((base % page != 0).any()):
        return None                                   # not whole contiguous pages: keep stock
    plan = Plan(s=s, full_hit=full_hit, src_pages=(base[:, 0] // page).tolist(), stock=_snapshot(req))
    req.prefix_indices = idx[:s]
    req.last_node = node_id
    req.last_host_node = node_id
    req.best_match_node = node_id
    req.kv.cache_protected_len = s
    # swa_branching_seqlen stays F: the first insert stops at F and the tree's tombstoned
    # window at F is rebuilt from the request's exact rows (stock branch semantics).
    setattr(req, REQ_ATTR, plan)
    STATE.planned += 1
    # One line per replay (rare: only a lost resume point with resident full KV): the evidence a gate or an
    # operator reads, with the copy line below.
    logger.warning("DSV41_SWA_CONE plan: rid=%s m=%d s=%d F=%d pages=%d", req.rid, m, s, full_hit,
                   len(plan.src_pages))
    if STATE.planned == 1 or STATE.planned % RECEIPT_EVERY == 0:
        _receipt()
    return plan


TREE_PINS = ("unified_tree_core.UnifiedTreeCore.match_full_device_prefix",
             "unified_tree_core.UnifiedTreeCore.collect_full_device_indices",
             "unified_radix_cache.UnifiedRadixCache.cache_unfinished_req",
             "swa.SWAComponent.update_component_on_insert_overlap",
             "common.free_swa_out_of_window_slots")
_TREE_OK = []


def _tree_pins_ok() -> bool:
    """Checked once, at the first candidate (the tree modules are imported by then)."""
    if not _TREE_OK:
        _TREE_OK.append(_check(list(TREE_PINS)))
    return _TREE_OK[0]


def _receipt():
    logger.warning("DSV41_SWA_CONE receipt: tokens=%d min_save=%d early_layers=%s planned=%d reverted=%d "
                   "replayed=%d rows=%d errors=%d", STATE.tokens, STATE.min_save, STATE.early_layers,
                   STATE.planned, STATE.reverted, STATE.replayed, STATE.rows, STATE.errors)


def _test_evict(req, cache):
    if not (STATE.test and str(req.rid).startswith("swacone-evict")):
        return
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams
    n = int(cache.swa_evictable_size())
    if n > 0:
        cache.evict(EvictParams(swa_num_tokens=n))
    logger.warning("DSV41_SWA_CONE test: rid=%s evicted %d evictable SWA tokens before the match", req.rid, n)


@dataclass
class ConeBatch:
    """Per forward: (row start in the extend, replay rows n, extend rows of the request, s,
    resident page ids) for every request that replays in this forward."""
    entries: List[Tuple[int, int, int, int, List[int]]]
    page: int


def build_cone_batch(reqs, prefix_lens, extend_lens, page) -> Optional[ConeBatch]:
    """Collect (and consume) the plans of the requests whose first chunk this batch runs."""
    entries, start = [], 0
    for req, pre, ext in zip(reqs, prefix_lens, extend_lens):
        plan = getattr(req, REQ_ATTR, None)
        if plan is not None:
            setattr(req, REQ_ATTR, None)
            if pre != plan.s or ext < plan.n:
                # Admission guarantees both; anything else would break the exactness argument.
                raise RuntimeError(f"DSV41_SWA_CONE: rid={req.rid} prefix {pre} / extend {ext} do not "
                                   f"cover the planned replay [{plan.s}, {plan.full_hit})")
            entries.append((start, plan.n, ext, plan.s, list(plan.src_pages)))
        start += ext
    return ConeBatch(entries=entries, page=page) if entries else None


# ------------------------------------------------------------------ forward side
def full_page_buffers(pool):
    """Every buffer the KV pool addresses by FULL page id, as (2-D row tensor, rows per full page).
    Mirrors DeepSeekV4TokenToKVPool.get_contiguous_buf_infos (the PD / HiCache page transfer)."""
    if getattr(pool, "_unified_kv", False) or getattr(pool, "request_window", None) is not None:
        raise RuntimeError("DSV41_SWA_CONE needs the paged (non-unified) DSV4 KV layout")
    page = pool.page_size
    out = []
    for ratio, kv_pool in pool.kv_pools.items():
        if kv_pool is None:
            continue
        for buf in kv_pool.kv_buffer:
            assert buf.ndim == 2
            out.append((buf, 1))
    for ratio, ipool in pool.index_pools.items():
        per = 1
        if ratio in (1, 2):
            slots = page // ratio
            assert slots % ipool.page_size == 0
            per = slots // ipool.page_size
        for buf in ipool.contiguous_page_row_buffers():
            assert buf.ndim == 2
            out.append((buf, per))
    return out


def copy_full_pages(buffers, src_pages, dst_pages):
    """dst page rows := src page rows in every buffer (byte copy; gather before scatter)."""
    import torch
    if src_pages.numel() == 0:
        return 0
    copied = 0
    for buf, per in buffers:
        if per == 1:
            rs, rd = src_pages, dst_pages
        else:
            off = torch.arange(per, device=src_pages.device, dtype=src_pages.dtype)
            rs = (src_pages[:, None] * per + off).reshape(-1)
            rd = (dst_pages[:, None] * per + off).reshape(-1)
        buf.index_copy_(0, rd, buf.index_select(0, rs))
        copied += 1
    return copied


def replay_copy(pool, cone: ConeBatch, out_cache_loc):
    """Copy the resident latents / index keys of every replay into the request's fresh slots."""
    import torch
    buffers = full_page_buffers(pool)
    page = cone.page
    for start, n, _ext, _s, src in cone.entries:
        dst = out_cache_loc[start:start + n:page].to(torch.int64) // page
        srcp = torch.tensor(src, dtype=torch.int64, device=out_cache_loc.device)
        assert dst.numel() == srcp.numel()
        copied = copy_full_pages(buffers, srcp, dst)
        STATE.replayed += 1
        STATE.rows += n
        logger.warning("DSV41_SWA_CONE copy: s=%d rows=%d pages=%d buffers=%d", _s, n, dst.numel(), copied)


def floor_vector(cone: ConeBatch, padded_rows: int, device):
    """Per-row SWA floor: s on the replaying requests' rows, 0 elsewhere (0 = no floor)."""
    import torch
    floor = torch.zeros(padded_rows, dtype=torch.int32, device=device)
    for start, _n, ext, s, _src in cone.entries:
        floor[start:start + ext] = s
    return floor


def mask_compressor_writes(core, cone: ConeBatch):
    """Replay rows write their latent / index key to the padding slot 0, never to a real slot."""
    for start, n, _ext, _s, _src in cone.entries:
        for name in ("c1_out_loc", "c2_out_loc"):
            loc = getattr(core, name, None)
            if loc is not None:
                loc[start:start + n] = 0


# ------------------------------------------------------------------ pins and install
def source_sha(obj):
    obj = inspect.unwrap(obj)
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


def _resolve(name):
    import importlib
    mods = {
        "schedule_batch": "sglang.srt.managers.schedule_batch",
        "schedule_policy": "sglang.srt.managers.schedule_policy",
        "forward_batch_info": "sglang.srt.model_executor.forward_batch_info",
        "deepseek_v4_backend": "sglang.srt.layers.attention.deepseek_v4_backend",
        "unified_tree_core": "sglang.srt.mem_cache.unified_cache.unified_tree_core",
        "unified_radix_cache": "sglang.srt.mem_cache.unified_radix_cache",
        "swa": "sglang.srt.mem_cache.unified_cache.components.swa",
        "common": "sglang.srt.mem_cache.common",
    }
    mod, *path = name.split(".")
    obj = importlib.import_module(mods[mod])
    for p in path:
        obj = getattr(obj, p)
    return obj


def pin_drift(names=None, pins=None, originals=None):
    """Pinned names whose source differs (empty = OK). originals: name -> stock object to hash
    instead of the module attribute (another adapter may have replaced it already)."""
    pins = PINS if pins is None else pins
    originals = originals or {}
    out = []
    for name in (pins if names is None else names):
        try:
            obj = originals.get(name) or _resolve(name)
            if source_sha(obj) != pins[name]:
                out.append(name)
        except Exception:
            out.append(name)
    return out


def _check(names, originals=None) -> bool:
    drift = pin_drift(names, originals=originals)
    if drift:
        STATE.active = False
        logger.warning("DSV41_SWA_CONE not installed: engine source drift in %s", ", ".join(drift))
        return False
    return True


def _disable(where, exc):
    STATE.errors += 1
    STATE.active = False
    logger.warning("DSV41_SWA_CONE disabled after an error in %s: %r", where, exc)


def install_schedule_batch(module, env=None):
    """Req.init_next_round_input (plan) and ScheduleBatch.prepare_for_extend (batch plan)."""
    if not configure(env):
        return False
    if "schedule_batch" in STATE.installed:
        return True
    if not _check(["schedule_batch.Req.init_next_round_input",
                   "schedule_batch.ScheduleBatch.prepare_for_extend"]):
        return False
    req_cls, batch_cls = module.Req, module.ScheduleBatch
    orig_init = req_cls.init_next_round_input
    orig_prep = batch_cls.prepare_for_extend

    def init_next_round_input(self, tree_cache=None, cow_mamba=None):
        if tree_cache is not None and STATE.active and STATE.test:
            _test_evict(self, tree_cache)
        result = orig_init(self, tree_cache, cow_mamba)
        if tree_cache is not None:
            if STATE.active:
                try:
                    plan_request(self, tree_cache)
                except Exception as exc:       # the stock match is still intact
                    setattr(self, REQ_ATTR, None)
                    _disable("plan", exc)
            else:
                setattr(self, REQ_ATTR, None)
        return result

    def prepare_for_extend(self):
        result = orig_prep(self)
        cone = None
        if any(getattr(r, REQ_ATTR, None) is not None for r in self.reqs):
            cone = build_cone_batch(self.reqs, self.prefix_lens, self.extend_lens, self.tree_cache.page_size)
        setattr(self, BATCH_ATTR, cone)
        return result

    for fn, orig in ((init_next_round_input, orig_init), (prepare_for_extend, orig_prep)):
        setattr(fn, _MARK, True)
        fn.__wrapped__ = orig
    req_cls.init_next_round_input = init_next_round_input
    batch_cls.prepare_for_extend = prepare_for_extend
    STATE.installed.add("schedule_batch")
    logger.warning("DSV41_SWA_CONE installed (scheduler): N=%d min_save=%d test=%s",
                   STATE.tokens, STATE.min_save, STATE.test)
    return True


_STOCK_SELECT = {}


def guarded_select(orig, adder, req, args, kwargs, add_req_result):
    """The admission's first chunk must cover the whole replay [s, F) AND be a chunked pass (end
    before the prompt end); otherwise the request goes back to its stock match (before anything is
    committed) and is selected again, with the longer extend debited exactly as add_one_req computes
    total_tokens.

    Why chunked: only a chunked first pass is guaranteed to reach the tree through
    cache_unfinished_req, which frees the request's SWA below page_floor(F - 1 - max(window, page))
    BEFORE its insert. A whole-prompt pass can finish at its first sampled token (stop / EOS /
    max_new_tokens 1 / abort); then cache_finished_req inserts the branch at F with
    swa_evicted_seqlen still 0, and the insert recovers every tombstoned node of [s, F) with the
    replay rows' own SWA, whose early-layer KV is inexact below s + (window - 1) * (E - 1)
    (tests/test_swa_cone.py test_finish_at_prefill_would_adopt_cone_swa)."""
    res = orig(adder, req, *args, **kwargs)
    plan = getattr(req, REQ_ATTR, None)
    if plan is None or isinstance(res, add_req_result):
        return res
    if res.extend_len >= plan.n and getattr(res, "is_chunked", False) is True:
        return res
    delta = plan.s - len(plan.stock["prefix_indices"])
    revert(req)
    logger.warning("DSV41_SWA_CONE revert: rid=%s first chunk %d (chunked=%s) does not cover replay %d "
                   "in a chunked pass: stock match kept", getattr(req, "rid", None), res.extend_len,
                   getattr(res, "is_chunked", None), plan.n)
    if "total_tokens" in kwargs:
        kwargs = dict(kwargs, total_tokens=kwargs["total_tokens"] + delta)
    return orig(adder, req, *args, **kwargs)


def pin_adder(module=None, env=None):
    """Before any other adapter wraps PrefillAdder: remember (and pin) the stock selection."""
    if not configure(env):
        return False
    if module is None:
        import importlib
        module = importlib.import_module("sglang.srt.managers.schedule_policy")
    fn = module.PrefillAdder._select_prefill_admission
    _STOCK_SELECT["fn"] = fn
    return _check(["schedule_policy.PrefillAdder._select_prefill_admission"],
                  originals={"schedule_policy.PrefillAdder._select_prefill_admission": fn})


def install_adder(module=None, env=None):
    """PrefillAdder._select_prefill_admission: a first pass that is not a chunk covering the replay -> stock match."""
    if not configure(env):
        return False
    if "adder" in STATE.installed:
        return True
    if module is None:
        import importlib
        module = importlib.import_module("sglang.srt.managers.schedule_policy")
    if "fn" not in _STOCK_SELECT and not pin_adder(module, env):
        return False
    cls = module.PrefillAdder
    orig = cls._select_prefill_admission
    add_req_result = module.AddReqResult

    def _select_prefill_admission(self, req, *args, **kwargs):
        return guarded_select(orig, self, req, args, kwargs, add_req_result)

    setattr(_select_prefill_admission, _MARK, True)
    _select_prefill_admission.__wrapped__ = orig
    cls._select_prefill_admission = _select_prefill_admission
    STATE.installed.add("adder")
    return True


def install_forward_batch(module, env=None):
    """ForwardBatch.init_new: the page copy on the forward stream, the plan onto the target batch."""
    if not configure(env):
        return False
    if "forward_batch" in STATE.installed:
        return True
    if not _check(["forward_batch_info.ForwardBatch.init_new"]):
        return False
    cls = module.ForwardBatch
    orig = cls.init_new.__func__

    def init_new(klass, batch, model_runner, *args, **kwargs):
        fb = orig(klass, batch, model_runner, *args, **kwargs)
        if getattr(model_runner, "is_draft_worker", False):
            return fb
        if STATE.early_layers is None:
            try:
                late = getattr(getattr(model_runner.model, "model", None), "late_layer_start", None)
                STATE.early_layers = int(late) if late is not None else None
                if late is None:
                    STATE.active = False
                    logger.warning("DSV41_SWA_CONE inactive: decoder SWA bounded replay is off")
            except Exception as exc:
                _disable("model check", exc)
        cone = batch.__dict__.pop(BATCH_ATTR, None) if hasattr(batch, "__dict__") else None
        if cone is None:
            return fb
        if not fb.forward_mode.is_extend_without_speculative():
            raise RuntimeError("DSV41_SWA_CONE: a replay plan reached a non-prefill forward")
        replay_copy(model_runner.token_to_kv_pool, cone, batch.out_cache_loc)
        setattr(fb, FB_ATTR, cone)
        return fb

    setattr(init_new, _MARK, True)
    init_new.__wrapped__ = orig
    cls.init_new = classmethod(init_new)
    STATE.installed.add("forward_batch")
    return True


def install_backend(module, env=None):
    """DeepseekV4AttnBackend.init_forward_metadata_prefill: floor + compressor write mask."""
    if not configure(env):
        return False
    if "backend" in STATE.installed:
        return True
    if not _check([n for n in PINS if n.startswith("deepseek_v4_backend.")]):
        return False
    cls = module.DeepseekV4AttnBackend
    orig = cls.init_forward_metadata_prefill

    def init_forward_metadata_prefill(self, *args, **kwargs):
        fb = kwargs.get("forward_batch")
        cone = getattr(fb, FB_ATTR, None) if fb is not None else None
        if cone is None or kwargs.get("swa_replay_start") is not None or getattr(self, "is_dspark_draft", False):
            return orig(self, *args, **kwargs)
        from sglang.srt.runtime_context import get_platform
        if not get_platform().is_sm120 or self.token_to_kv_pool.request_window is not None:
            # The sparse chunk cache (non-SM12x) ignores the floor; the request window is
            # another SWA layout. Neither is what the argument covers.
            raise RuntimeError("DSV41_SWA_CONE: replay planned on an unsupported prefill path")
        if getattr(fb, "attn_cp_metadata", None) is not None:
            raise RuntimeError("DSV41_SWA_CONE: replay planned under context parallelism")
        out_cache_loc = kwargs["out_cache_loc"]
        kwargs["swa_replay_start"] = floor_vector(cone, out_cache_loc.shape[0], out_cache_loc.device)
        md = orig(self, *args, **kwargs)
        mask_compressor_writes(md.core_attn_metadata, cone)
        return md

    setattr(init_forward_metadata_prefill, _MARK, True)
    init_forward_metadata_prefill.__wrapped__ = orig
    cls.init_forward_metadata_prefill = init_forward_metadata_prefill
    STATE.installed.add("backend")
    logger.warning("DSV41_SWA_CONE installed (backend): replay rows floored and write-masked")
    return True
