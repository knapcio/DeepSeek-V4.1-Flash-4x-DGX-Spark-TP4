"""Keep the caching allocator's blocks between the chunks of ONE long prefill; give them back when it ends.

DSV41_PREFILL_KEEP_CACHE=1 (default 0 = adapter/prefill_empty_cache.py unchanged: empty_cache() after every extend
forward with a sequence >= DSV41_PREFILL_EMPTY_CACHE_TOKENS).

Why. The per-chunk empty_cache() exists because a chunk's prefix-dependent transients (the dense indexer's fp32 logits
[rows, prefix / 4], its bool masks and the candidate-block copy) grow every chunk: the next chunk asks for a block a
little larger than the cached one, the allocator mallocs a new segment and keeps the old one, and reserved memory
grows with the square of the prompt (REPORT 15/17). The v2.7 trace shows what that costs: 15 cudaFree (81 ms host,
the first one also drains the GPU) and 14 cudaMalloc per 8192-row chunk, 42 ms of GPU idle between chunks and
7 + 3.5 ms inside the next one.

What changes, three parts, all allocation-only (no value changes anywhere):

1. Width padding (DSV41_PREFILL_KEEP_CACHE_PAD, default 1 with the gate; adapter/indexer_chunked_v3.py). The dense
   indexer's growing tensors are sized by the request's FINAL compressed width instead of the current one: the
   logits [rows, W] (DeepGEMM writes column j only for j < ke - ks and computes only that KV range, so a wider row
   stride changes no value and no work), the causal bool mask and the candidate-block maxima (taken on the padded,
   already -inf columns instead of an F.pad copy). The final width comes from ForwardBatch.dsv41_final_lens (below).
   Width per path: the split path pads to ceil(final_lc, lcm(4, block)); the unsplit path to the width of the last
   chunk of this prompt that stays below the split threshold (last_unsplit_lc), the widest it reaches anyway. So each of these
   tensors is, in every chunk, as large as it is in that path's last full chunk today: the PEAK is unchanged
   (bound: <= rows_rank x chunk / ratio x 4 B = 16 MiB at chunk 8192, TP4, when the last chunk is short), and the
   allocation sizes stop changing from chunk to chunk.
2. Plumbing: ForwardBatch.init_new attaches dsv41_final_lens = [max(extend end, len(origin_input_ids))] per request
   (the expression ScheduleBatch uses for orig_seq_lens, on the host). Rank-invariant: every TP rank's scheduler
   holds the same requests. Missing -> no padding and the old per-chunk empty_cache.
3. The policy (Policy below, pure and unit-tested). After a long extend forward:
     - the prompt of some request in the batch continues and the guard passes -> keep the cache (no empty_cache);
     - otherwise -> empty_cache() exactly as today: the last chunk of every prefill, an unknown final length,
       the first long chunk whose starting cache state is unknown, and any guard trip.
   Guard (memory neutrality, measured every kept chunk from host-side allocator counters, no device sync):
     slack = reserved - peak_allocated of this forward = cached bytes this chunk did not need even at its peak.
     A chunk that started from an emptied cache measures the allocator's own rounding/fragmentation, frag_ref.
     Keep only while slack <= frag_ref + DSV41_PREFILL_KEEP_CACHE_SLACK_MB (default 256): then
     reserved <= peak_allocated + frag_ref + SLACK, i.e. what the same chunk would reserve from an empty cache
     plus at most SLACK. A trip empties the cache (today's state); a trip by more than 4 x SLACK also stops keeping
     for the rest of that prefill. Between prefills nothing is kept: the last chunk always empties, so decode and
     the KV-pool budget see exactly today's memory.
   What the guard cannot bound: the excess INSIDE a chunk whose cached blocks turn out not to fit (it is seen after
   the chunk and then emptied). With part 1 the remaining growing allocations are small (k_slots / index-K gather /
   tail masks, ~20-60 B per prefix token); the fleet receipts (num_device_alloc per kept chunk = 0, head
   MemAvailable low-water vs A/A') are what proves it (diagnostics/prefill-fixed-20261108/DESIGN.md).

Receipts: one line per long prefill per rank ("DSV41 keep-cache prefill: ..."): chunks, kept, empties by reason,
new segments in kept chunks, reserved high-water, max slack, frag_ref.
"""
import logging
import math
import os

logger = logging.getLogger(__name__)

_OFF = ("", "0", "off", "false", "no")
MIB = 1 << 20

try:  # DSV41_AB_VARIANTS in-boot A/B (test only)
    import ab_variant as _ab
except ImportError:  # CPU tests without the adapter dir
    _ab = None


def _ab_active(name):
    return _ab is not None and getattr(_ab, "ACTIVE", False) and name in getattr(_ab, "KNOWN", ())


def env(name, default):
    if _ab_active(name):
        return _ab.env(name, default)
    return os.environ.get(name, default)


def truthy(value):
    return value is not None and str(value).strip().lower() not in _OFF


def enabled():
    return truthy(env("DSV41_PREFILL_KEEP_CACHE", "0"))


def pad_enabled():
    return enabled() and truthy(env("DSV41_PREFILL_KEEP_CACHE_PAD", "1"))


def wanted_at_install():
    return truthy(os.environ.get("DSV41_PREFILL_KEEP_CACHE", "0")) or _ab_active("DSV41_PREFILL_KEEP_CACHE")


def slack_bytes():
    try:
        return max(0, int(float(env("DSV41_PREFILL_KEEP_CACHE_SLACK_MB", "256")) * MIB))
    except ValueError:
        return 256 * MIB


# ------------------------------------------------------------------ plumbing: final prompt lengths on the host
def final_lens_of(reqs):
    """Per request: the prompt length this prefill ends at (ScheduleBatch's orig_seq_lens expression), or None."""
    out = []
    for r in reqs:
        rng = getattr(r, "extend_range", None)
        origin = getattr(r, "origin_input_ids", None)
        if rng is None or origin is None:
            return None
        out.append(max(int(rng.end), len(origin)))
    return out


def prompt_continues(seq_lens, final_lens):
    """True if some request's prompt goes on after this chunk, False if every one ends here, None if unknown."""
    if seq_lens is None or final_lens is None or len(seq_lens) != len(final_lens) or not len(final_lens):
        return None
    return any(int(s) < int(f) for s, f in zip(seq_lens, final_lens))


def install_forward_batch(module):
    """ForwardBatch.init_new: attach dsv41_final_lens on extend batches of the target model."""
    cls = getattr(module, "ForwardBatch", None)
    if cls is None or not hasattr(cls, "init_new"):
        logger.warning("DSV41_PREFILL_KEEP_CACHE: ForwardBatch.init_new not found; keep-cache inactive")
        return False
    if getattr(cls, "_dsv41_keep_cache", False):
        return True
    orig = cls.init_new.__func__

    def init_new(klass, batch, model_runner, *args, **kwargs):
        fb = orig(klass, batch, model_runner, *args, **kwargs)
        try:
            mode = getattr(fb, "forward_mode", None)
            if mode is not None and mode.is_extend() and not getattr(model_runner, "is_draft_worker", False):
                fb.dsv41_final_lens = final_lens_of(getattr(batch, "reqs", None) or [])
        except Exception as exc:  # noqa: BLE001 - bookkeeping never breaks a batch
            logger.warning("DSV41_PREFILL_KEEP_CACHE final lens: %r", exc)
        return fb

    cls.init_new = classmethod(init_new)
    cls._dsv41_keep_cache = True
    logger.warning("DSV41_PREFILL_KEEP_CACHE: ForwardBatch carries the requests' final prompt lengths")
    return True


# ------------------------------------------------------------------ part 1: padded indexer widths
def lcm(a, b):
    return a * b // math.gcd(a, b)


def ceil_align(x, a):
    return (x + a - 1) // a * a


def padded_width(lc, final_lc, align, cap=None):
    """Width to allocate for a row of compressed scores: the current lc rounded up to `align`, raised to the final
    compressed width (capped at `cap`, the widest this path reaches for this prompt), never below the current one."""
    want = lc if final_lc is None else max(lc, min(final_lc, cap) if cap is not None else final_lc)
    return ceil_align(max(want, 1), align)


def last_unsplit_lc(seq, step, final_seq, ratio, split_min_lc):
    """Compressed width of the widest chunk of this prompt that the unsplit indexer path will score: chunks end at
    seq, seq + step, ... and at final_seq; the SG18 split takes every chunk whose width (end // ratio) reaches
    split_min_lc. Assumes the remaining chunks keep this chunk's size (the guard covers a scheduler that changes it)."""
    limit = split_min_lc * ratio - 1          # last sequence end that is still unsplit
    if final_seq <= limit:
        return final_seq // ratio
    if seq > limit or step <= 0:
        return seq // ratio
    return (seq + (limit - seq) // step * step) // ratio


# ------------------------------------------------------------------ part 3: the policy
class Policy:
    """Keep / empty decision per long extend forward, from host-side allocator counters.

    stats: dict with reserved (bytes), peak (allocated high-water of this forward), allocs / frees (cumulative
    cudaMalloc / cudaFree counts). before(): counters at forward start; after(): returns (empty: bool, reason)."""

    def __init__(self, slack=256 * MIB, big_factor=4):
        self.slack = slack
        self.big = big_factor * slack
        self.frag_ref = None
        self.flushed_allocs = None   # num_device_alloc right after our last empty_cache
        self.session = None
        self.fresh = False
        self._start = None
        self.sessions = []           # finished-session receipts (tests read them)

    def _new_session(self):
        self.session = {"chunks": 0, "kept": 0, "empty": {}, "new_segments_kept": 0, "reserved_max": 0,
                        "slack_max": 0, "disabled": False, "frag_ref": None}

    def before(self, stats):
        self._start = dict(stats)
        # fresh = nothing was allocated from the driver since our last empty_cache: the cache is empty of transients
        self.fresh = self.flushed_allocs is not None and stats["allocs"] == self.flushed_allocs
        if self.session is None:
            self._new_session()

    def after(self, more, stats):
        s = self.session
        start = self._start or stats
        new_segments = stats["allocs"] - start["allocs"]
        slack = stats["reserved"] - stats["peak"]
        s["chunks"] += 1
        s["reserved_max"] = max(s["reserved_max"], stats["reserved"])
        if self.fresh:
            self.frag_ref = slack
            s["frag_ref"] = slack
        else:
            s["slack_max"] = max(s["slack_max"], slack)
        if more is not True:
            reason = "end" if more is False else "unknown-final"
        elif s["disabled"]:
            reason = "disabled"
        elif self.frag_ref is None:
            reason = "no-reference"
        elif not self.fresh and slack > self.frag_ref + self.slack:
            reason = "trip"
            if slack > self.frag_ref + self.big:
                s["disabled"] = True
                reason = "trip-big"
        else:
            s["kept"] += 1
            if not self.fresh:
                s["new_segments_kept"] += new_segments
            return False, "keep"
        s["empty"][reason] = s["empty"].get(reason, 0) + 1
        if more is not True:
            self.sessions.append(s)
            self.session = None
        return True, reason

    def emptied(self, stats):
        """Call right after torch.cuda.empty_cache() with fresh counters."""
        self.flushed_allocs = stats["allocs"]

    def reset(self):
        self.session = None
        self.fresh = False
        self.flushed_allocs = None


def read_stats(torch):
    st = torch.cuda.memory_stats()
    return {"reserved": int(st.get("reserved_bytes.all.current", 0)),
            "peak": int(st.get("allocated_bytes.all.peak", 0)),
            "allocs": int(st.get("num_device_alloc", 0)),
            "frees": int(st.get("num_device_free", 0))}


def receipt(s):
    empties = ",".join(f"{k}={v}" for k, v in sorted(s["empty"].items())) or "-"
    ref = "unknown" if s["frag_ref"] is None else f"{s['frag_ref'] / MIB:.0f} MiB"
    return (f"DSV41 keep-cache prefill: chunks={s['chunks']} kept={s['kept']} empties[{empties}] "
            f"new_segments_in_kept={s['new_segments_kept']} reserved_max={s['reserved_max'] / MIB:.0f} MiB "
            f"slack_max={s['slack_max'] / MIB:.0f} MiB frag_ref={ref} disabled={s['disabled']}")
