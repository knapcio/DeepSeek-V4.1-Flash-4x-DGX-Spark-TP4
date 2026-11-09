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
3. The policy (Policy below, pure and unit-tested). A kept cache belongs to ONE prompt (a session: request ids, final
   lengths, sequence ends, first-chunk rows).
     - before a long extend forward: a held cache that this batch does not continue (other requests or finals, a gap or
       rewind, more rows than the first chunk) is emptied FIRST; a new prompt whose cache is not known to be empty is
       emptied first too, so its first chunk measures the guard reference on this prompt;
     - after it: keep only if every request continues and the guard passes; otherwise empty_cache() exactly as today
       (last chunk, a mixed batch where some request ends, unknown final length, a wider-than-chunk batch, a trip);
     - release: Scheduler hooks empty a held cache once its prompt can no longer continue (aborted / retracted chunked
       request, pause, idle loop) and the hook does when the gate reads off. Rank-local by design: empty_cache decides
       no shape and joins no collective.
   Guard (memory neutrality, measured every kept chunk from host-side allocator counters, no device sync):
     slack = reserved - peak_allocated of this forward = cached bytes this chunk did not need even at its peak.
     The prompt's first chunk (from an emptied cache) measures the allocator's own rounding/fragmentation, frag_ref.
     Keep only while slack <= frag_ref + DSV41_PREFILL_KEEP_CACHE_SLACK_MB (default 256): then
     reserved <= peak_allocated + frag_ref + SLACK, i.e. what the same chunk would reserve from an empty cache
     plus at most SLACK. A trip empties the cache (the next chunk re-measures); a trip by more than 4 x SLACK also stops
     keeping for the rest of that prompt. Between prompts nothing is kept.
   DSV41_PREFILL_KEEP_CACHE_PAD_MAX_ROWS (default 8192 = the chunk): a wider batch (bypass admitted beside a chunk
   without solo) runs with stock widths (ForwardBatch.dsv41_pad_ok = False) and is emptied after.
   What the guard cannot bound: the excess INSIDE a chunk whose cached blocks turn out not to fit (seen after the chunk,
   then emptied). With part 1 the remaining growing allocations are small (k_slots / index-K gather / tail masks,
   ~20-60 B per prefix token); the fleet receipts (new_segments_in_kept, head MemAvailable low-water vs A/A') decide
   (diagnostics/prefill-fixed-20261108/DESIGN.md).

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
    """The in-boot A/B is armed AND one of its variants names this key (an armed A/B that does not name it leaves the
    process exactly as without the A/B: nothing installed, the base env read)."""
    if _ab is None or not getattr(_ab, "ACTIVE", False) or name not in getattr(_ab, "KNOWN", ()):
        return False
    return any(name in spec for spec in getattr(_ab, "_specs", ()) or ())


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


def pad_max_rows():
    """Rows of a long extend up to which the indexer pads to the final width (the chunk size). A wider batch (a
    bypass admitted beside a chunk without solo) runs with stock widths and is emptied after."""
    try:
        return int(env("DSV41_PREFILL_KEEP_CACHE_PAD_MAX_ROWS", "8192"))
    except ValueError:
        return 8192


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


def rids_of(reqs):
    out = [getattr(r, "rid", None) for r in reqs]
    return None if any(x is None for x in out) else out


def install_forward_batch(module):
    """ForwardBatch.init_new: attach dsv41_final_lens and dsv41_rids on prefill extend batches of the target model
    (not on speculative verify / draft extends: is_extend_without_speculative)."""
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
            if (mode is not None and mode.is_extend_without_speculative()
                    and not getattr(model_runner, "is_draft_worker", False)):
                reqs = getattr(batch, "reqs", None) or []
                fb.dsv41_final_lens = final_lens_of(reqs)
                fb.dsv41_rids = rids_of(reqs)
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
class Batch:
    """What the policy needs to know about one long extend forward (all host values).

    ids: request ids (None = unknown, then the final lengths stand in for them); finals: final prompt lengths (None =
    unknown); seqs: sequence end after this chunk; extends: rows of this chunk per request (None = unknown)."""
    __slots__ = ("ids", "finals", "seqs", "extends")

    def __init__(self, ids, finals, seqs, extends=None):
        self.ids = None if ids is None else tuple(ids)
        self.finals = None if finals is None else tuple(int(f) for f in finals)
        self.seqs = tuple(int(x) for x in seqs)
        self.extends = None if extends is None else tuple(int(x) for x in extends)
        if self.finals is not None and len(self.finals) != len(self.seqs):
            self.finals = None
        if self.ids is not None and len(self.ids) != len(self.seqs):
            self.ids = None
        if self.extends is not None and len(self.extends) != len(self.seqs):
            self.extends = None

    def identity(self):
        return self.ids if self.ids is not None else ("finals",) + (self.finals or ())

    def rows(self, prev_ends=None):
        if self.extends is not None:
            return sum(self.extends)
        if prev_ends is not None:
            return sum(s - e for s, e in zip(self.seqs, prev_ends))
        return sum(self.seqs)          # unknown: assume no prefix (first chunk)

    def more(self):
        """True if every request's prompt goes on after this chunk, False if every one ends here, 'mixed' if some end
        and some go on, None if the final lengths are unknown."""
        if self.finals is None or not self.seqs:
            return None
        cont = [s < f for s, f in zip(self.seqs, self.finals)]
        if all(cont):
            return True
        return False if not any(cont) else "mixed"


class Policy:
    """Keep / empty decisions for the long extend forwards of ONE prompt at a time, from host-side allocator counters.

    A kept cache belongs to a session: the request ids, their final lengths, the sequence ends after the last chunk and
    the rows of the session's first chunk. Per long extend forward:
      begin(batch, stats) -> (empty_before, reason): the caller empties the cache first when it is True: the cache held
          for a session that this batch does not continue (another prompt, a mixed batch, a gap or a rewind, more rows
          than the first chunk), or a new session whose first chunk would not start from an empty cache (the guard's
          reference must be measured on this prompt);
      before(stats): counters at forward start (after any empty);
      after(batch, stats) -> (empty_after, reason).
    release() -> True if a cache is held (the caller empties it): an aborted / retracted / paused chunked request, an
    idle scheduler, the gate switched off. Stats: reserved (bytes), peak (allocated high-water of this forward),
    allocs / frees (cumulative cudaMalloc / cudaFree counts)."""

    def __init__(self, slack=256 * MIB, big_factor=4, max_rows=8192):
        self.slack = slack
        self.max_rows = max_rows
        self.big = big_factor * slack
        self.frag_ref = None
        self.flushed_allocs = None   # num_device_alloc right after our last empty_cache
        self.session = None
        self.fresh = False
        self._start = None
        self.sessions = []           # finished-session receipts (the hook prints and pops them)
        self.no_keep = None          # (identity, finals) of a prompt whose chunks interleave with decode steps
        self._no_keep_now = False

    @property
    def holding(self):
        return self.session is not None and self.session["holding"]

    def _new_session(self):
        self.frag_ref = None
        self.session = {"chunks": 0, "kept": 0, "empty": {}, "new_segments_kept": 0, "reserved_max": 0,
                        "slack_max": 0, "disabled": False, "frag_ref": None, "holding": False,
                        "identity": None, "finals": None, "ends": None, "rows": None}

    def _close(self, reason):
        s = self.session
        if s is not None:
            s["empty"][reason] = s["empty"].get(reason, 0) + 1
            s["holding"] = False
            self.sessions.append(s)
        self.session = None

    def continues(self, batch, need_holding=True, ignore_rows=False):
        s = self.session
        if s is None or (need_holding and not s["holding"]) or batch.finals is None or s["ends"] is None:
            return False
        if batch.identity() != s["identity"] or batch.finals != s["finals"]:
            return False
        ends = s["ends"]
        if batch.extends is not None:
            if any(q - x != e for q, x, e in zip(batch.seqs, batch.extends, ends)):
                return False        # not the next chunk of these requests (gap, rewind, re-prefill)
        if any(q <= e for q, e in zip(batch.seqs, ends)):
            return False
        return ignore_rows or s["rows"] is None or batch.rows(ends) <= s["rows"]

    def mid_prompt(self):
        """A prompt's prefill is under way (a session with at least one chunk, held or not)."""
        return self.session is not None and self.session["chunks"] > 0

    def mark_no_keep(self):
        """A decode / verify step ran between the chunks of the current prompt: stop keeping for the rest of it
        (mixed prefill + decode must never hold the prefill's peak through the decode windows). Returns True if a
        cache is held (the caller empties it)."""
        s = self.session
        if s is None or not s["chunks"]:
            return False
        self.no_keep = (s["identity"], s["finals"])
        s["no_keep"] = True
        if s["holding"]:
            s["holding"] = False
            s["empty"]["decode"] = s["empty"].get("decode", 0) + 1
            return True
        return False

    def begin(self, batch, stats):
        # padding needs every request of the batch to continue (a batch where some request ends is emptied after it,
        # so padding it to the largest final width would only raise its own peak: A last + B first)
        self._all_continue = batch.more() is True
        if self.no_keep is not None:
            if (batch.identity(), batch.finals) == self.no_keep:
                # the rest of a prompt that interleaves with decode: today's behaviour (empty after every chunk,
                # stock widths), no empty before (the cache was emptied after the previous chunk)
                self._no_keep_now = True
                if self.session is None:
                    self._new_session()
                    self.session["no_keep"] = True
                return False, "no-keep"
            self.no_keep = None
            if self.session is not None and self.session.get("no_keep"):
                self._close("switch")
        self._no_keep_now = False
        out = self._begin(batch, stats)
        s = self.session
        if s is not None and batch.rows(s["ends"] if self.continues(batch, need_holding=False) else None) \
                > self.max_rows:
            s["wide"] = True           # stock widths for this chunk (pad_ok False) and empty after it
        return out

    def pad_ok(self):
        """The indexer may pad this forward to the final width: every request of the batch continues (begin) and the
        batch is not wider than a chunk."""
        if not getattr(self, "_all_continue", False) or self._no_keep_now:
            return False
        return self.session is None or not self.session.get("wide", False)

    def _begin(self, batch, stats):
        if batch.finals is None and not self.holding:
            return False, "no-final"   # cannot keep anyway: the forward runs as today and empties after
        if self.holding:
            if self.continues(batch):
                return False, "continue"
            wider = self.continues(batch, ignore_rows=True)
            self._close("switch")
            self._new_session()
            if wider:
                # more rows than the session's first chunk (bypass without solo): run it from an empty cache and
                # empty after it too, so the following normal chunks do not carry the wider chunk's blocks
                self.session["wide"] = True
            return True, "switch"
        if self.session is not None and self.session["chunks"] and not self.continues(batch, need_holding=False):
            self._close("switch")      # the cache is already empty (the last chunk tripped); receipts per prompt
        if self.session is None:
            self._new_session()
        if self.flushed_allocs is None or stats["allocs"] != self.flushed_allocs:
            return True, "start"
        return False, "start-fresh"

    def before(self, stats):
        self._start = dict(stats)
        # fresh = nothing was allocated from the driver since our last empty_cache: the cache is empty of transients
        self.fresh = self.flushed_allocs is not None and stats["allocs"] == self.flushed_allocs
        if self.session is None:
            self._new_session()

    def after(self, batch, stats):
        s = self.session
        start = self._start or stats
        new_segments = stats["allocs"] - start["allocs"]
        slack = stats["reserved"] - stats["peak"]
        first = s["chunks"] == 0
        s["chunks"] += 1
        s["reserved_max"] = max(s["reserved_max"], stats["reserved"])
        remeasure = first or s.pop("remeasure", False)
        if self.fresh and remeasure:
            self.frag_ref = slack
            s["frag_ref"] = slack
        else:
            s["slack_max"] = max(s["slack_max"], slack)
        more = batch.more()
        if more is not True:
            reason = {False: "end", None: "unknown-final"}.get(more, "mixed")
        elif self._no_keep_now:
            reason = "decode-mixed"
        elif s.pop("wide", False):
            reason = "wide"
        elif s["disabled"]:
            reason = "disabled"
        elif self.frag_ref is None:
            reason = "no-reference"
        elif not remeasure and slack > self.frag_ref + self.slack:
            reason = "trip"
            if slack > self.frag_ref + self.big:
                s["disabled"] = True
                reason = "trip-big"
        else:
            s["kept"] += 1
            if not remeasure:
                s["new_segments_kept"] += new_segments
            if s["rows"] is None:
                s["rows"] = batch.rows(s["ends"])
            s["identity"], s["finals"], s["ends"] = batch.identity(), batch.finals, batch.seqs
            s["holding"] = True
            return False, "keep"
        s["holding"] = False
        if more is not True:
            self._close(reason)
        else:
            s["empty"][reason] = s["empty"].get(reason, 0) + 1
            s["remeasure"] = True      # the next chunk starts from the empty cache: a new reference
            # after a trip the next chunk starts from an empty cache and continues the session; after a wider
            # chunk the next normal chunk sets the row count
            if first and reason != "wide":
                s["rows"] = batch.rows()
            s["identity"], s["finals"], s["ends"] = batch.identity(), batch.finals, batch.seqs
        return True, reason

    def emptied(self, stats):
        """Call right after torch.cuda.empty_cache() with fresh counters."""
        self.flushed_allocs = stats["allocs"]

    def release(self):
        """True if a cache is held for a session (the caller empties it and calls emptied())."""
        if self.holding:
            self._close("released")
            return True
        if self.session is not None and self.session["chunks"]:
            self._close("released")
        return False

    def reset(self):
        self.session = None
        self.fresh = False
        self.flushed_allocs = None
        self.frag_ref = None


# ------------------------------------------------------------------ one policy per process, shared by the hooks
_POLICY = [None]


def policy():
    if _POLICY[0] is None:
        _POLICY[0] = Policy(slack=slack_bytes(), max_rows=pad_max_rows())
    return _POLICY[0]


def batch_of(forward_batch):
    lens = getattr(forward_batch, "seq_lens_cpu", None)
    seqs = [] if lens is None else [int(x) for x in (lens.tolist() if hasattr(lens, "tolist") else lens)]
    ext = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if ext is not None and hasattr(ext, "tolist"):
        ext = ext.tolist()
    return Batch(getattr(forward_batch, "dsv41_rids", None), getattr(forward_batch, "dsv41_final_lens", None),
                 seqs, ext)


def release(torch, why="released"):
    """Empty the cache if a prompt's cache is held. Rank-local and allowed: empty_cache decides no shape and joins
    no collective. Returns True if it emptied."""
    p = _POLICY[0]
    if p is None or not p.holding:
        return False
    p.release()
    torch.cuda.empty_cache()
    p.emptied(read_stats(torch))
    while p.sessions:
        print(receipt(p.sessions.pop(), why), flush=True)
    return True


def release_for_decode(torch):
    """A decode / verify forward of the target model while a prompt is mid-prefill: stop keeping for that prompt and
    empty a held cache now. Returns True if it emptied."""
    p = _POLICY[0]
    if p is None or not p.mid_prompt():
        return False
    if not p.mark_no_keep():
        return False
    torch.cuda.empty_cache()
    p.emptied(read_stats(torch))
    return True


def scheduler_owns_session(scheduler, batch):
    """True while the held session may still continue: the scheduler's chunked request, or a request of the batch
    about to run, is one of the session's requests."""
    p = _POLICY[0]
    if p is None or not p.holding:
        return True
    ids = set(p.session["identity"] or ())
    chunked = getattr(scheduler, "chunked_req", None)
    if chunked is not None and getattr(chunked, "rid", None) in ids:
        return True
    reqs = getattr(batch, "reqs", None) if batch is not None else None
    return bool(reqs) and any(getattr(r, "rid", None) in ids for r in reqs)


def install_scheduler(module, torch=None):
    """Release a held cache when the prompt it belongs to can no longer continue: after get_next_batch_to_run (an
    aborted, retracted or finished-elsewhere chunked request, an idle loop), after process_pending_chunked_abort,
    pause_generation and on_idle. Installed after decode_yield (which pins get_next_batch_to_run's source)."""
    if torch is None:
        import torch
    cls = getattr(module, "Scheduler", None)
    if cls is None or getattr(cls, "_dsv41_keep_cache", False):
        return cls is not None
    done = []
    if hasattr(cls, "get_next_batch_to_run"):
        orig_next = cls.get_next_batch_to_run

        def get_next_batch_to_run(self, *args, **kwargs):
            plan = orig_next(self, *args, **kwargs)
            try:
                batch = getattr(plan, "batch_to_run", plan)
                if not scheduler_owns_session(self, batch):
                    release(torch, "orphaned")
            except Exception as exc:  # noqa: BLE001 - bookkeeping never breaks the loop
                logger.warning("DSV41_PREFILL_KEEP_CACHE release: %r", exc)
            return plan

        get_next_batch_to_run.__wrapped__ = orig_next
        cls.get_next_batch_to_run = get_next_batch_to_run
        done.append("get_next_batch_to_run")
    for name, why in (("process_pending_chunked_abort", "abort"), ("pause_generation", "pause"), ("on_idle", "idle")):
        orig = getattr(cls, name, None)
        if orig is None:
            continue

        def wrap(orig=orig, why=why):
            def fn(self, *args, **kwargs):
                out = orig(self, *args, **kwargs)
                try:
                    # the fleet run (run-20261109-011349) reached on_idle between the chunks of ONE prefill:
                    # release only when neither the chunked request nor a batch belongs to the held prompt
                    if not scheduler_owns_session(self, None):
                        release(torch, why)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("DSV41_PREFILL_KEEP_CACHE release (%s): %r", why, exc)
                return out
            fn.__wrapped__ = orig
            return fn

        setattr(cls, name, wrap())
        done.append(name)
    cls._dsv41_keep_cache = True
    logger.warning("DSV41_PREFILL_KEEP_CACHE: held cache released from Scheduler.%s", ", ".join(done))
    return True


def read_stats(torch):
    st = torch.cuda.memory_stats()
    return {"reserved": int(st.get("reserved_bytes.all.current", 0)),
            "peak": int(st.get("allocated_bytes.all.peak", 0)),
            "allocs": int(st.get("num_device_alloc", 0)),
            "frees": int(st.get("num_device_free", 0))}


def receipt(s, why=None):
    empties = ",".join(f"{k}={v}" for k, v in sorted(s["empty"].items())) or "-"
    ref = "unknown" if s["frag_ref"] is None else f"{s['frag_ref'] / MIB:.0f} MiB"
    return (f"DSV41 keep-cache prefill: chunks={s['chunks']} kept={s['kept']} empties[{empties}] "
            f"new_segments_in_kept={s['new_segments_kept']} reserved_max={s['reserved_max'] / MIB:.0f} MiB "
            f"slack_max={s['slack_max'] / MIB:.0f} MiB frag_ref={ref} disabled={s['disabled']}"
            + (f" released_by={why}" if why else ""))
