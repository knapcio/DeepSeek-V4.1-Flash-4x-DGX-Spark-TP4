"""Keep a prompt's last SWA window live through decode when the tree does not own it yet (DSV41_SWA_PROMPT_TAIL_KEEP).

Ported from Eric Jinho Jang (dealign.ai), jjang-ai/dealignai-deepseek-v4.1-flash-1017-4x-dgx-spark, commits
c175335 / a9f87fe, adapter/swa_evict_internal_first.py `install_prompt_tail_floor` (AGPL-3.0-or-later; a modification
of this repository's AGPL files). Their text: "A prompt shorter than one prefill chunk is not inserted into the radix
tree before it finishes, so during decode `free_swa_out_of_window_slots` frees its SWA up to seqlen - window --
including the prompt's end. The finished insert then tombstones the prompt end, and a revisit (or a turn that
diverges after the prompt) cannot end its match there: measured 0 cached tokens on the 2nd visit of a 6k prompt, hits
only from the 3rd visit on. `swa_retain_floor` is SGLang's own hook for this (the mamba path uses checkpoint -
window); the decode-time eviction promises not to free past it." Their measurement on the ring (506fb10 line): a 6k
chat turn 2 went from 0 % to 93.7 % cached; qeval unchanged.

What this port changes against theirs (reviewed on the pinned SGLang dsv4.1 tree, CPU-tested on the real
UnifiedRadixCache, tests/test_swa_prompt_tail.py):

  1. Decode only. Theirs overrides UnifiedRadixCache.swa_retain_floor, which the prompt-end and chunked inserts also
     consult (SWAComponent._free_out_of_window_slots before every cache_unfinished_req insert): there it would make the
     tree adopt N live SWA tokens per prompt end instead of one page, and change the first insert of a cone replay.
     Here only ScheduleBatch._evict_swa (the decode-time free, maybe_evict_swa) sees the floor; every insert is stock.
  2. Only where the bug can happen. When the prompt-end insert ran (cache_unfinished_req chunked=False after the last
     prefill chunk), the tree owns [0, page_floor(prompt)) and the request can never free it
     (Req.kv.swa_dead_lo = max(cache_protected_len, ...)): the stock path already keeps the prompt end matchable (CPU
     test: 2nd visit = page_floor(prompt) cached, with or without this module). The floor applies only to a request
     whose cache_protected_len is below page_floor(prompt), i.e. the prompt end is still request-owned during decode.
     With a normal prompt-end insert this module is an exact no-op.
  3. Bounded hold. free_swa_out_of_window_slots keeps one contiguous frontier, so a floor at the prompt end keeps EVERY
     later decode token's SWA slot too: the hold per running request is N + output length, not N (theirs says "<= N";
     CPU test: 32-40 vs 5-8 slots at the end of a 30-token output). Unbounded, a few long thinking outputs would eat
     the --swa-prefix-tails headroom (v2.6.5: 182,784 slots = request cap 35,328 + 384 tails x 384) and push other
     cached prefixes out. The floor is dropped (stock frees from then on) once seqlen - floor > N + HOLD
     (DSV41_SWA_PROMPT_TAIL_HOLD, default 8192): <= N + 8192 slots per request.

Exactness. Only WHEN request-owned SWA slots are freed changes (later, never earlier): no kernel, model input, KV value
or pool size changes; which tokens a later request matches is the engine's normal admission (match length depends on
eviction timing already). All inputs (the request's lengths, cache_protected_len, the page size) are replicated
scheduler state, so every TP rank takes the same decision (rank-invariant). The SWA attention window of the request
itself is unchanged: slots below seqlen - window are never read by its own decode.

Knobs (off unless DSV41_SWA_PROMPT_TAIL_KEEP is a positive integer):
  DSV41_SWA_PROMPT_TAIL_KEEP=<tokens>  N, 1..8192 (theirs: 1024): live SWA kept below page_floor(prompt)
  DSV41_SWA_PROMPT_TAIL_HOLD=<tokens>  extra decode tokens the floor may hold, default 8192, 0..1048576
Receipt: one "DSV41_SWA_PROMPT_TAIL receipt" line at the first applied floor and every 4096 decode-time floors, with
the number of requests it held and how many hit the hold bound. A boot whose receipt never appears had no request with
a request-owned prompt end (the module did nothing).

Not covered (no-op): chunk cache (no tree), mamba / hybrid trees, EAGLE bigram keys, HiCache, radix sessions; on pin
drift of the engine pieces below the module refuses to install (stock behaviour) and logs why.
"""
import hashlib
import inspect
import logging
import os

logger = logging.getLogger(__name__)

ENV_KEEP = "DSV41_SWA_PROMPT_TAIL_KEEP"
ENV_HOLD = "DSV41_SWA_PROMPT_TAIL_HOLD"
DEFAULT_HOLD = 8192
MAX_KEEP = 8192
MAX_HOLD = 1 << 20
REQ_ATTR = "_dsv41_prompt_tail"      # set on a Req the first time its floor applies
_MARK = "_dsv41_prompt_tail_hook"
RECEIPT_EVERY = 4096

# sha256 of inspect.getsource() of the engine pieces this module relies on, on the pinned SGLang dsv4.1 tree
# (f80c91a4b + the recipe's patches; = the swa_thin / swa_cone pins' tree). On drift: not installed (stock).
PINS = {
    # replaced: the decode-time free; the replacement is this body with retain_floor = min(stock floor, ours)
    "schedule_batch.ScheduleBatch._evict_swa":
        "3858c94799a55ed2a613e9ad31c6771725b52bec5142b619f0b97cc16636a239",
    # the decode condition that calls _evict_swa(req, seqlen - 1)
    "schedule_batch.ScheduleBatch.maybe_evict_swa":
        "acf10ed2d2064abb57a725e138a08e5115acdf2587a150aac2d2e8e7404fd6c7",
    # retain_floor semantics: min(evict_threshold, retain_floor), one contiguous page-aligned frontier
    "common.free_swa_out_of_window_slots":
        "4478aa6081ca843d22332f4ee586cfe9c7f18b8b3fc3c5fcac7801c5ceb227e4",
    # the stock floor (None without mamba) that ours is combined with
    "unified_radix_cache.UnifiedRadixCache.swa_retain_floor":
        "54676595c5c8280d3dcb946ddd0b372b7f0d7a8f1af4264dd978bd67618dabca",
    # the prompt-end insert raises cache_protected_len to the matched prompt (the no-op condition)
    "unified_radix_cache.UnifiedRadixCache.cache_unfinished_req":
        "d64c72cfc5f6b129b08a6a59f072946e2c923c541f3be1e2262379b1721121c1",
    # the finished insert adopts live SWA from the request's swa_evicted_seqlen
    "unified_radix_cache.UnifiedRadixCache.cache_finished_req":
        "5a3ee37b0ad8425db2c7918ccab5f02e78608d9d3beb723cfa667825fe631cc8",
    "swa.SWAComponent.prepare_for_caching_req":
        "0abd37666d62d9baf572b1a63cb664149e9d7b856623a9abb87b4197cb7afbd1",
}


class _State:
    installed = False
    active = False
    keep = 0
    hold = DEFAULT_HOLD
    applied = 0        # decode-time frees that used the floor
    requests = 0       # distinct requests the floor held
    released = 0       # requests whose hold bound was reached (stock from then on)
    errors = 0


STATE = _State()


def config(env=None):
    """(keep, hold), keep 0 = off. Raises ValueError on a malformed or out-of-range value."""
    env = os.environ if env is None else env
    raw = str(env.get(ENV_KEEP, "0")).strip().lower()
    if raw in ("", "0", "off", "false", "no"):
        return 0, DEFAULT_HOLD
    keep = int(raw)
    if not 0 < keep <= MAX_KEEP:
        raise ValueError(f"{ENV_KEEP}={raw}: must be 1..{MAX_KEEP}")
    hold = int(str(env.get(ENV_HOLD, DEFAULT_HOLD)).strip() or DEFAULT_HOLD)
    if not 0 <= hold <= MAX_HOLD:
        raise ValueError(f"{ENV_HOLD}={hold}: must be 0..{MAX_HOLD}")
    return keep, hold


def enabled(env=None):
    try:
        return config(env)[0] > 0
    except ValueError:
        return True   # install() refuses and logs: a typo must not pass silently as "off"


def source_sha(obj):
    obj = inspect.unwrap(obj)
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


def pin_targets(schedule_batch, common, urc, swa):
    return {
        "schedule_batch.ScheduleBatch._evict_swa": schedule_batch.ScheduleBatch._evict_swa,
        "schedule_batch.ScheduleBatch.maybe_evict_swa": schedule_batch.ScheduleBatch.maybe_evict_swa,
        "common.free_swa_out_of_window_slots": common.free_swa_out_of_window_slots,
        "unified_radix_cache.UnifiedRadixCache.swa_retain_floor": urc.UnifiedRadixCache.swa_retain_floor,
        "unified_radix_cache.UnifiedRadixCache.cache_unfinished_req": urc.UnifiedRadixCache.cache_unfinished_req,
        "unified_radix_cache.UnifiedRadixCache.cache_finished_req": urc.UnifiedRadixCache.cache_finished_req,
        "swa.SWAComponent.prepare_for_caching_req": swa.SWAComponent.prepare_for_caching_req,
    }


def pin_drift(targets, pins=None):
    pins = PINS if pins is None else pins
    return sorted(name for name, fn in targets.items() if source_sha(fn) != pins.get(name))


def prompt_tail_floor(cache, req, keep, hold):
    """The retain floor for this decode-time free, or None (stock).

    None unless: a radix tree with SWA (not chunk cache, mamba, EAGLE keys, HiCache, sessions), a prompt of at least
    one page, the prompt end still request-owned (cache_protected_len < page_floor(prompt)), and the hold bound not
    reached (seqlen - floor <= keep + hold)."""
    if cache.is_chunk_cache():
        return None
    tc = getattr(cache, "tree_core", None)
    if tc is None or getattr(cache, "is_mamba_enabled", False) or getattr(tc, "is_eagle", False):
        return None
    if getattr(tc, "enable_hicache", False) or getattr(tc, "enable_session_radix_cache", False):
        return None
    page = cache.page_size
    end = (len(req.origin_input_ids) // page) * page
    if end <= 0 or req.kv.cache_protected_len >= end:
        return None
    floor = max(0, end - keep)
    if req.seqlen - floor > keep + hold:
        if getattr(req, REQ_ATTR, None) == "held":
            setattr(req, REQ_ATTR, "released")
            STATE.released += 1
        return None
    return floor


def _receipt():
    logger.warning("DSV41_SWA_PROMPT_TAIL receipt: keep=%d hold=%d applied=%d requests=%d released=%d errors=%d",
                   STATE.keep, STATE.hold, STATE.applied, STATE.requests, STATE.released, STATE.errors)


def install(module, env=None):
    """Replace module.ScheduleBatch._evict_swa (module = sglang.srt.managers.schedule_batch). Returns True if armed."""
    try:
        keep, hold = config(env)
    except ValueError as exc:
        logger.warning("DSV41_SWA_PROMPT_TAIL not installed: %s", exc)
        return False
    if keep <= 0:
        return False
    cls = module.ScheduleBatch
    if getattr(cls._evict_swa, _MARK, False):
        return True
    try:
        from sglang.srt.mem_cache import common
        from sglang.srt.mem_cache import unified_radix_cache as urc
        from sglang.srt.mem_cache.unified_cache.components import swa
        drift = pin_drift(pin_targets(module, common, urc, swa))
    except Exception as exc:  # an engine without these pieces: stock
        logger.warning("DSV41_SWA_PROMPT_TAIL not installed: %r", exc)
        return False
    if drift:
        logger.warning("DSV41_SWA_PROMPT_TAIL not installed: engine source drift in %s", ", ".join(drift))
        return False
    original = cls._evict_swa
    free = module.free_swa_out_of_window_slots

    def _evict_swa(self, req, pre_len):
        if STATE.active:
            try:
                floor = prompt_tail_floor(self.tree_cache, req, STATE.keep, STATE.hold)
            except Exception as exc:  # never fail a decode over bookkeeping: stock from now on
                STATE.errors += 1
                STATE.active = False
                logger.warning("DSV41_SWA_PROMPT_TAIL disabled after an error: %r", exc)
                floor = None
            if floor is not None:
                # = the pinned stock body with retain_floor = min(stock floor, ours)
                assert self.tree_cache.supports_swa(), "prefix cache must support swa"
                base = self.tree_cache.swa_retain_floor(req)
                free(
                    req,
                    pre_len,
                    sliding_window_size=self.tree_cache.sliding_window_size,
                    page_size=self.tree_cache.page_size,
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    is_chunk_cache=self.tree_cache.is_chunk_cache(),
                    retain_floor=floor if base is None else min(base, floor),
                )
                STATE.applied += 1
                if getattr(req, REQ_ATTR, None) is None:
                    setattr(req, REQ_ATTR, "held")
                    STATE.requests += 1
                if STATE.applied == 1 or STATE.applied % RECEIPT_EVERY == 0:
                    _receipt()
                return None
        return original(self, req, pre_len)

    setattr(_evict_swa, _MARK, True)
    _evict_swa.__wrapped__ = original
    cls._evict_swa = _evict_swa
    STATE.installed = True
    STATE.active = True
    STATE.keep, STATE.hold = keep, hold
    logger.warning("DSV41_SWA_PROMPT_TAIL installed: keep=%d hold=%d: decode keeps the last %d prompt tokens' SWA "
                   "while the prompt end is request-owned (bounded at %d held tokens)", keep, hold, keep, keep + hold)
    return True
