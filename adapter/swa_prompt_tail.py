"""Keep the last N prompt tokens' SWA live at the prompt-end insert, so a revisit can match there (DSV41_SWA_PROMPT_TAIL_KEEP).

Ported from Eric Jinho Jang (dealign.ai), jjang-ai/dealignai-deepseek-v4.1-flash-1017-4x-dgx-spark, commits
c175335 / a9f87fe, adapter/swa_evict_internal_first.py `install_prompt_tail_floor` (AGPL-3.0-or-later; a modification
of this repository's AGPL files). Their text: "A prompt shorter than one prefill chunk is not inserted into the radix
tree before it finishes, so during decode `free_swa_out_of_window_slots` frees its SWA up to seqlen - window --
including the prompt's end. The finished insert then tombstones the prompt end, and a revisit (or a turn that
diverges after the prompt) cannot end its match there: measured 0 cached tokens on the 2nd visit of a 6k prompt, hits
only from the 3rd visit on. `swa_retain_floor` is SGLang's own hook for this (the mamba path uses checkpoint -
window); the decode-time eviction promises not to free past it." Their floor: page_floor(prompt) - N, N = 1024;
measured on the ring (506fb10 line): a 6k chat turn 2 from 0 % to 93.7 % cached, qeval unchanged.

What the floor does on the pinned SGLang dsv4.1 tree (CPU-tested on the real UnifiedRadixCache and
ScheduleBatch.maybe_evict_swa, tests/test_swa_prompt_tail.py):
  * At the prompt-end insert (cache_unfinished_req chunked=False after the last prefill chunk, which the stock
    scheduler does for every request) the pre-insert free (SWAComponent._free_out_of_window_slots) keeps
    max(window, page) = ONE page of live SWA below page_floor(prompt): the tree's prompt-end checkpoint can only be
    matched at page_floor(prompt) itself. A revisit that diverges INSIDE the last page or earlier (a chat turn whose
    new question replaces the old one: the common prefix ends a few dozen tokens before the old prompt's end) finds no
    live window to end its match on and re-prefills from 0; its own branch insert then leaves a checkpoint at the
    divergence, so the 3rd visit hits. That is the measured pattern (2nd visit 0, hits from the 3rd). With the floor
    the prompt-end checkpoint keeps N live SWA tokens, so any divergence in the last N tokens (page-aligned) matches.
  * In decode, after that insert, the request can never free below cache_protected_len = page_floor(prompt), and
    their floor (below it) then stops EVERY decode-time free: free_swa_out_of_window_slots keeps one contiguous
    frontier, so the request holds N + its whole output in SWA until it finishes (test: 40 vs 8 slots at the end of a
    30-token output; ~N + 30K slots for a long thinking answer), and the finished insert hands all of it to the tree.
    On v2.6.5 (SWA pool 182,784 = request cap 35,328 + 384 tails x 384) a few long outputs would eat the tails'
    headroom and evict other prefixes' resume points.

This port therefore applies the floor:
  1. at the prompt-end insert (inside cache_unfinished_req with chunked=False): theirs, unchanged. Chunked inserts
     (chunk boundaries, the cone replay's first insert) are stock: the floor never lowers a chunked free, so SWA the
     cone replay computed from an approximate window never reaches the tree (adapter/swa_cone.py's argument holds);
  2. at decode-time frees only while the prompt end is still request-owned (cache_protected_len <
     page_floor(prompt): no prompt-end insert ran; their docstring's case, not seen on the stock scheduler), and only
     until the request holds N + DSV41_SWA_PROMPT_TAIL_HOLD (8192) tokens past the floor (then stock frees resume);
  3. never otherwise: once the tree owns the prompt end, decode frees are stock.
Cost: each cached prompt end holds up to N + one page of SWA in the tree instead of one page (1280 vs 256 slots at
N = 1024); evictable like any checkpoint (DSV41_SWA_THIN's demotion and the LRU apply unchanged).

Exactness. Only WHEN request-owned SWA slots are freed changes (later, never earlier) and therefore how many live SWA
slots the tree adopts at the prompt end: no kernel, model input, KV value or pool size changes; a slot is adopted only
with the KV its own request computed (exact); which prefix a later request reuses is the engine's normal admission (it
already depends on eviction timing). All inputs (lengths, cache_protected_len, the page size, whether the insert is
chunked) are replicated scheduler state: every TP rank decides the same (rank-invariant).

Knobs (off unless DSV41_SWA_PROMPT_TAIL_KEEP is a positive integer):
  DSV41_SWA_PROMPT_TAIL_KEEP=<tokens>  N, 1..8192 (theirs: 1024): live SWA kept below page_floor(prompt)
  DSV41_SWA_PROMPT_TAIL_HOLD=<tokens>  decode-time bound of rule 2, default 8192, 0..1048576
Receipt: "DSV41_SWA_PROMPT_TAIL receipt: ... prompt_end=<inserts that kept the tail> decode=<decode-time frees under
rule 2> ..." at the first application and every 1024 applications.

Not covered (stock): chunk cache, mamba / hybrid trees, EAGLE bigram keys, HiCache, radix sessions. On pin drift of the
engine pieces below the module refuses to install (stock behaviour) and logs why.
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
REQ_ATTR = "_dsv41_prompt_tail"      # "held" / "released" on a Req under rule 2
PHASE_ATTR = "_dsv41_prompt_tail_insert"   # on the cache during cache_unfinished_req: "prompt_end" | "chunked"
_MARK = "_dsv41_prompt_tail_hook"
RECEIPT_EVERY = 1024

# sha256 of inspect.getsource() of the engine pieces this module relies on, on the pinned SGLang dsv4.1 tree
# (f80c91a4b + the recipe's patches; = the swa_thin / swa_cone pins' tree). On drift: not installed (stock).
PINS = {
    # wrapped: the stock floor (None without mamba) that ours is combined with
    "unified_radix_cache.UnifiedRadixCache.swa_retain_floor":
        "54676595c5c8280d3dcb946ddd0b372b7f0d7a8f1af4264dd978bd67618dabca",
    # wrapped (phase marker only): the pre-insert free, the insert, the re-match raising cache_protected_len
    "unified_radix_cache.UnifiedRadixCache.cache_unfinished_req":
        "d64c72cfc5f6b129b08a6a59f072946e2c923c541f3be1e2262379b1721121c1",
    # the insert-path callers of swa_retain_floor
    "swa.SWAComponent._free_out_of_window_slots":
        "6802b56535aad64e83ea9ae1d7d16e035381c7ad4b2f029341d66760a4a58eef",
    "swa.SWAComponent.cleanup_after_caching_req":
        "a93ea2c3374f4a4187787eb9c23b764c1dee9fc4b996474de8625ec73a209513",
    # the insert keeps live SWA from the request's swa_evicted_seqlen on
    "swa.SWAComponent.prepare_for_caching_req":
        "0abd37666d62d9baf572b1a63cb664149e9d7b856623a9abb87b4197cb7afbd1",
    # the finished insert adopts live SWA from the request's swa_evicted_seqlen
    "unified_radix_cache.UnifiedRadixCache.cache_finished_req":
        "5a3ee37b0ad8425db2c7918ccab5f02e78608d9d3beb723cfa667825fe631cc8",
    # retain_floor semantics: min(evict_threshold, retain_floor), one contiguous page-aligned frontier
    "common.free_swa_out_of_window_slots":
        "4478aa6081ca843d22332f4ee586cfe9c7f18b8b3fc3c5fcac7801c5ceb227e4",
    # the decode-time caller (retain_floor=self.tree_cache.swa_retain_floor(req)) and its condition
    "schedule_batch.ScheduleBatch._evict_swa":
        "3858c94799a55ed2a613e9ad31c6771725b52bec5142b619f0b97cc16636a239",
    "schedule_batch.ScheduleBatch.maybe_evict_swa":
        "acf10ed2d2064abb57a725e138a08e5115acdf2587a150aac2d2e8e7404fd6c7",
}


class _State:
    installed = False
    active = False
    keep = 0
    hold = DEFAULT_HOLD
    prompt_end = 0     # prompt-end inserts whose pre-insert free used the floor
    decode = 0         # decode-time frees under rule 2
    requests = 0       # distinct requests held under rule 2
    released = 0       # rule-2 requests that reached the hold bound
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


def pin_targets(urc, swa, common, schedule_batch):
    return {
        "unified_radix_cache.UnifiedRadixCache.swa_retain_floor": urc.UnifiedRadixCache.swa_retain_floor,
        "unified_radix_cache.UnifiedRadixCache.cache_unfinished_req": urc.UnifiedRadixCache.cache_unfinished_req,
        "swa.SWAComponent._free_out_of_window_slots": swa.SWAComponent._free_out_of_window_slots,
        "swa.SWAComponent.cleanup_after_caching_req": swa.SWAComponent.cleanup_after_caching_req,
        "swa.SWAComponent.prepare_for_caching_req": swa.SWAComponent.prepare_for_caching_req,
        "unified_radix_cache.UnifiedRadixCache.cache_finished_req": urc.UnifiedRadixCache.cache_finished_req,
        "common.free_swa_out_of_window_slots": common.free_swa_out_of_window_slots,
        "schedule_batch.ScheduleBatch._evict_swa": schedule_batch.ScheduleBatch._evict_swa,
        "schedule_batch.ScheduleBatch.maybe_evict_swa": schedule_batch.ScheduleBatch.maybe_evict_swa,
    }


def pin_drift(targets, pins=None):
    pins = PINS if pins is None else pins
    return sorted(name for name, fn in targets.items() if source_sha(fn) != pins.get(name))


def prompt_tail_floor(cache, req, keep, hold, phase):
    """Our retain floor for this free, or None (stock). phase: getattr(cache, PHASE_ATTR, None)."""
    if phase == "chunked" or cache.is_chunk_cache():
        return None
    tc = getattr(cache, "tree_core", None)
    if tc is None or getattr(cache, "is_mamba_enabled", False) or getattr(tc, "is_eagle", False):
        return None
    if getattr(tc, "enable_hicache", False) or getattr(tc, "enable_session_radix_cache", False):
        return None
    page = cache.page_size
    end = (len(req.origin_input_ids) // page) * page
    if end <= 0 or req.kv.cache_protected_len >= end:
        return None                       # nothing request-owned at the prompt end: stock
    floor = max(0, end - keep)
    if phase == "prompt_end":
        return floor                      # rule 1
    if req.seqlen - floor > keep + hold:  # rule 2, bounded
        if getattr(req, REQ_ATTR, None) == "held":
            setattr(req, REQ_ATTR, "released")
            STATE.released += 1
        return None
    return floor


def _receipt():
    logger.warning("DSV41_SWA_PROMPT_TAIL receipt: keep=%d hold=%d prompt_end=%d decode=%d requests=%d released=%d "
                   "errors=%d", STATE.keep, STATE.hold, STATE.prompt_end, STATE.decode, STATE.requests,
                   STATE.released, STATE.errors)


def install(module, env=None):
    """Wrap UnifiedRadixCache.swa_retain_floor + cache_unfinished_req (module = sglang.srt.managers.schedule_batch,
    whose import hook runs first; unified_radix_cache is imported here). Returns True when armed."""
    try:
        keep, hold = config(env)
    except ValueError as exc:
        logger.warning("DSV41_SWA_PROMPT_TAIL not installed: %s", exc)
        return False
    if keep <= 0:
        return False
    try:
        from sglang.srt.mem_cache import common
        from sglang.srt.mem_cache import unified_radix_cache as urc
        from sglang.srt.mem_cache.unified_cache.components import swa
        cls = urc.UnifiedRadixCache
        if getattr(cls.swa_retain_floor, _MARK, False):
            return True
        drift = pin_drift(pin_targets(urc, swa, common, module))
    except Exception as exc:  # an engine without these pieces: stock
        logger.warning("DSV41_SWA_PROMPT_TAIL not installed: %r", exc)
        return False
    if drift:
        logger.warning("DSV41_SWA_PROMPT_TAIL not installed: engine source drift in %s", ", ".join(drift))
        return False
    orig_floor = cls.swa_retain_floor
    orig_cache = cls.cache_unfinished_req

    def swa_retain_floor(self, req):
        base = orig_floor(self, req)
        if not STATE.active:
            return base
        phase = getattr(self, PHASE_ATTR, None)
        try:
            floor = prompt_tail_floor(self, req, STATE.keep, STATE.hold, phase)
        except Exception as exc:  # never fail a free over bookkeeping: stock from now on
            STATE.errors += 1
            STATE.active = False
            logger.warning("DSV41_SWA_PROMPT_TAIL disabled after an error: %r", exc)
            return base
        if floor is None:
            return base
        if phase == "prompt_end":
            STATE.prompt_end += 1
        else:
            STATE.decode += 1
            if getattr(req, REQ_ATTR, None) is None:
                setattr(req, REQ_ATTR, "held")
                STATE.requests += 1
        n = STATE.prompt_end + STATE.decode
        if n == 1 or n % RECEIPT_EVERY == 0:
            _receipt()
        return floor if base is None else min(base, floor)

    def cache_unfinished_req(self, req, chunked=False, **kwargs):
        prev = getattr(self, PHASE_ATTR, None)
        setattr(self, PHASE_ATTR, "chunked" if chunked else "prompt_end")
        try:
            return orig_cache(self, req, chunked=chunked, **kwargs)
        finally:
            setattr(self, PHASE_ATTR, prev)

    for fn, orig in ((swa_retain_floor, orig_floor), (cache_unfinished_req, orig_cache)):
        setattr(fn, _MARK, True)
        fn.__wrapped__ = orig
        for attr in ("_dsv41_swa_thin",):         # keep the other hooks' install marks visible on the outer wrapper
            if getattr(orig, attr, False):
                setattr(fn, attr, True)
    cls.swa_retain_floor = swa_retain_floor
    cls.cache_unfinished_req = cache_unfinished_req
    STATE.installed = True
    STATE.active = True
    STATE.keep, STATE.hold = keep, hold
    logger.warning("DSV41_SWA_PROMPT_TAIL installed: keep=%d hold=%d: the prompt-end insert keeps the last %d prompt "
                   "tokens' SWA; decode frees stay stock once the tree owns the prompt end", keep, hold, keep)
    return True
