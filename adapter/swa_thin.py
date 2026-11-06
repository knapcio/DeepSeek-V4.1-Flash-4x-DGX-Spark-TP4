"""Keep long documents' SWA resume points through other long prefills (DSV41_SWA_THIN=1).

Why. The unified radix cache can reuse a cached prefix only at a node whose last
sliding window (128 tokens) still holds SWA slots. The SWA pool is small (cap mode:
84,480 slots on the TP4 profile, ~22 KB per slot per rank) and every chunked-prefill
step leaves one such checkpoint behind: cache_unfinished_req(chunked=True) inserts the
prefix at the chunk boundary and free_swa_out_of_window_slots keeps the trailing
max(window, page) tokens there, page-aligned down -> 2 pages = 512 slots per 8192-token
chunk. A 950K-token prefill writes ~110 of them (56K slots), so prefilling a few long
documents pushes the OLDER documents' checkpoints out of the SWA LRU wholesale, and a
document without any resident checkpoint is re-prefilled from token 0 although its full
KV is still resident (issue16 soak: 200K / 400K documents lost after every cold phase,
33 / 67 s each; 800K / 950K follow in some orders, 146 / 176 s).

What. When a request's prefill inserts past the node it stood on, that node's checkpoint
is superseded by the request's own newer one, and it is moved to the evict-first end of
the SWA LRU, unless another request uses it (locked), it is a branch point (more than one
child) or it is not on the request's new path. Two inserts do this:
  * every chunked step (cache_unfinished_req chunked=True): chunk boundary C_k when the
    prefill reaches C_k+1 (on the first step: the node the prefix match ended on);
  * the prompt-end insert (cache_unfinished_req chunked=False, after the last prefill
    chunk) of a request that made no chunked step: the node its prefix match ended on. In
    an agent session that is the previous turn's resume point, and the new turn's
    prompt-end / branch and output checkpoints replace it, so it stops competing with other
    sessions' resume points. After chunked steps the prompt-end insert demotes nothing: the
    last chunk boundary is the resume point of a later request that diverges inside the
    final chunk (another question on the same document), as in the chunk-only version.
DSV41_SWA_THIN=chunk does the first only (the 2026-11-02 version). The prompt-end and
output-end checkpoints a request leaves, and every branch point, keep their normal LRU
position. A demoted checkpoint that a later match ends on is refreshed to MRU by the
engine's own MATCH_END refresh, so a checkpoint that turns out to be a shared-prefix branch
point regains normal standing. Under SWA pressure superseded checkpoints go first (most
recently demoted first), so idle sessions and other documents keep their resume points.

Exactness. Only the order of the SWA LRU list changes. Which SWA slots get evicted under
pressure is not observable except through how much of a prompt the next match reuses,
and the engine already admits any eviction order (LRU order depends on arrival timing).
Model code, kernels, the pool sizes and the tree's lock/ledger accounting are untouched
(the demotion moves list pointers only; evictable/protected sizes stay the same). All TP
ranks run the same deterministic tree operations, so the order stays rank-invariant.

Not changed: HiCache host LRUs, radix sessions (partitioned LRU), mamba trees: the hook
is a no-op there. DSV41_SWA_THIN=0 / unset = stock behaviour; 1 = both inserts; chunk = chunked
steps only.
"""
import hashlib
import inspect
import logging
import os

logger = logging.getLogger(__name__)
_WARNED = []

ENV = "DSV41_SWA_THIN"
_MARK = "_dsv41_swa_thin"
_REQ_SEEN = "_dsv41_swa_thin_seen"   # set on a Req after its first hooked insert (chunked or not)

# sha256 of inspect.getsource() of the engine pieces this hook relies on, on the pinned
# SGLang dsv4.1 tree (f80c91a4b + the recipe's patches). On drift the hook raises RuntimeError at
# install (it used to fall back to stock behaviour with a log line).
PINS = {
    "UnifiedRadixCache.cache_unfinished_req":
        "d64c72cfc5f6b129b08a6a59f072946e2c923c541f3be1e2262379b1721121c1",
    "UnifiedLRUList._add_node_after":
        "7b84b69969bc04730bec910f80927121f6de0a96499d8801b4a21f24b7abcae1",
    "UnifiedLRUList._remove_node":
        "e0101fd38b9dc80111fa9fd6b043faf7e442da4586105f912bea4fc097e0d8fb",
}


RECEIPT_EVERY = 1024  # one log line per this many hooked inserts (chunked steps + prompt ends)


class _Stats:
    installed = False
    active = False
    prompt_end = False   # also demote at the prompt-end insert (mode "all")
    calls = 0            # chunked steps seen
    prompt_calls = 0     # prompt-end inserts seen (mode "all")
    demoted = 0
    errors = 0


STATS = _Stats()


def mode(env=None):
    """None (off), "chunk" (chunked steps only) or "all" (chunked steps + prompt-end insert)."""
    env = os.environ if env is None else env
    v = str(env.get(ENV, "0")).strip().lower()
    if v in ("", "0", "off", "false", "no"):
        return None
    # Override site: any value other than "chunk" was read as "all", so a mistyped mode silently
    # ran the wider mode. A value the user set must take effect, so an unknown value now raises.
    if v not in ("chunk", "all", "1", "on", "true", "yes"):
        raise RuntimeError(f"{ENV}={v!r}: unknown mode (use 0, chunk or all); it would be read as all; "
                           f"refusing to boot")
    return "chunk" if v == "chunk" else "all"


def enabled(env=None):
    return mode(env) is not None


def source_sha(obj):
    obj = inspect.unwrap(obj)
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


def pin_drift(cache_cls, lru_cls, pins=None):
    """Names whose source no longer matches PINS (empty list = OK)."""
    pins = PINS if pins is None else pins
    found = {
        "UnifiedRadixCache.cache_unfinished_req": cache_cls.cache_unfinished_req,
        "UnifiedLRUList._add_node_after": lru_cls._add_node_after,
        "UnifiedLRUList._remove_node": lru_cls._remove_node,
    }
    return [name for name, fn in found.items() if source_sha(fn) != pins.get(name)]


def _node(cache, handle):
    if handle is None:
        return None
    if isinstance(handle, int):
        return cache.tree_core.node_by_id(handle)
    return handle


def _move_to_lru_end(lru, node):
    """Unlink node and relink it next to the tail sentinel (evicted first)."""
    pt = lru._pt
    lru._remove_node(node)
    lru._add_node_after(lru.tail.lru_prev[pt], node)


def demote_superseded(cache, old_handle, new_handle, swa_type):
    """Move the SWA checkpoint at old_handle to the evict-first end of the SWA LRU.

    old_handle: the request's last node before this chunk's insert (the previous chunk
    boundary); new_handle: its last node after the insert and re-match. Returns the number
    of tree nodes moved (0 when any precondition fails).
    """
    tc = cache.tree_core
    if getattr(tc, "enable_session_radix_cache", False) or getattr(tc, "enable_hicache", False):
        return 0
    if getattr(cache, "is_mamba_enabled", False):
        return 0
    swa = cache.components.get(swa_type)
    if swa is None:
        return 0
    lru = tc.lru_lists.get(swa_type) if hasattr(tc.lru_lists, "get") else tc.lru_lists[swa_type]
    if lru is None or getattr(lru, "mid", None) is not None:
        return 0
    old = _node(cache, old_handle)
    new = _node(cache, new_handle)
    root = tc.root_node
    if old is None or new is None or old is new or old is root:
        return 0
    # The checkpoint must be a proper ancestor of the request's new position: the same
    # request moved past it. A rotation decline or a rematch elsewhere leaves it alone.
    cur = new.parent
    while cur is not None and cur is not root and cur is not old:
        cur = cur.parent
    if cur is not old:
        return 0
    span = swa.sliding_window_size + tc.page_size
    covered = 0
    moved = 0
    x = old
    while x is not root and x is not None and covered < span:
        cd = x.component_data[swa_type]
        if (cd.value is not None and cd.lock_ref == 0 and lru.in_list(x)
                and len(x.children) == 1):
            _move_to_lru_end(lru, x)
            moved += 1
        covered += len(x.key)
        x = x.parent
    return moved


def install(module, env=None):
    """Wrap module.UnifiedRadixCache.cache_unfinished_req (module = unified_radix_cache)."""
    m = mode(env)
    if m is None:
        return False
    cls = module.UnifiedRadixCache
    if getattr(cls.cache_unfinished_req, _MARK, False):
        return True
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedLRUList

    drift = pin_drift(cls, UnifiedLRUList)
    # Override site: engine source drift left a set DSV41_SWA_THIN silently not installed. A gate the
    # user set must take effect, so drift now stops the boot.
    if drift:
        raise RuntimeError(f"{ENV}={m!r}: engine source drift in {', '.join(drift)}; the hook would not be "
                           f"installed; refusing to boot")
    original = cls.cache_unfinished_req
    swa_type = ComponentType.SWA

    def cache_unfinished_req(self, req, chunked=False, **kwargs):
        old = None
        if STATS.active:
            if chunked:
                old = req.last_node
            elif STATS.prompt_end and not getattr(req, _REQ_SEEN, False):
                old = req.last_node   # the prefix-match node: no chunked step moved past it
        result = original(self, req, chunked=chunked, **kwargs)
        if STATS.active:
            # Fall-through site: a request that cannot carry the seen-mark is treated as not seen
            # (its next insert may demote once more). Kept; one WARNING line per process.
            try:
                setattr(req, _REQ_SEEN, True)
            except Exception as exc:  # never fail a prefill over bookkeeping
                if not _WARNED:
                    _WARNED.append(True)
                    logger.warning("WARNING DSV41_SWA_THIN: cannot mark request seen (%r); prompt-end "
                                   "tracking skipped for such requests (logged once)", exc)
        if old is not None and STATS.active:
            if chunked:
                STATS.calls += 1
            else:
                STATS.prompt_calls += 1
            try:
                STATS.demoted += demote_superseded(self, old, req.last_node, swa_type)
            # Override site: an error in demote_superseded turned a set DSV41_SWA_THIN off for the rest
            # of the process. A gate the user set must not stop taking effect silently, so the error
            # now raises where it happens.
            except Exception as exc:
                STATS.errors += 1
                STATS.active = False
                raise RuntimeError(f"{ENV}: error in demote_superseded ({exc!r}); the hook would be disabled "
                                   f"for the rest of the process; refusing to continue") from exc
            n = STATS.calls + STATS.prompt_calls
            if n == 1 or n % RECEIPT_EVERY == 0:
                logger.warning("DSV41_SWA_THIN receipt: mode=%s chunk_steps=%d prompt_steps=%d demoted=%d errors=%d",
                               "all" if STATS.prompt_end else "chunk", STATS.calls, STATS.prompt_calls,
                               STATS.demoted, STATS.errors)
        return result

    setattr(cache_unfinished_req, _MARK, True)
    cache_unfinished_req.__wrapped__ = original
    cls.cache_unfinished_req = cache_unfinished_req
    STATS.installed = True
    STATS.active = True
    STATS.prompt_end = m == "all"
    logger.warning("DSV41_SWA_THIN installed: mode=%s: superseded %s SWA checkpoints are evicted before "
                   "other cached prefixes' resume points", m,
                   "chunk-boundary and match-end" if m == "all" else "chunk-boundary")
    return True
