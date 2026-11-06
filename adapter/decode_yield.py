"""Decode yield: running streams keep a share of the GPU while a long prompt prefills (DSV41_DECODE_YIELD=<pct>).

Why. A prompt longer than one chunk is prefilled in 8192-token chunks of ~1.35-1.7 s each (more with a longer
prefix). Between two chunks the engine runs only `--prefill-decode-interval` decode steps (2 in the production
profile), ~0.06-0.18 s. So while one agent's 170-680K prompt prefills, every other running stream gets ~2 decode
steps per ~1.6 s, ~10 % of its normal step rate, for 0.5-2 minutes (production head log 2026-11-03/04: 20 chunks per
40-step "Decode batch" line in every long prefill with streams running, gen throughput 3-28 tok/s instead of 100-160).

What. After the engine schedules a NON-FINAL chunk of a chunked prompt while decode rows are running, the next
prefill is deferred until the running streams have had decode time worth `pct / (100 - pct)` of that chunk's time
(pct = 50: as much decode time as chunk time). Both times come from a fixed cost model (integers, below), never a
clock: chunk_us = tokens / 8192 * (1,354,500 + 0.487 * prefix) and step_us = 26,300 + 15,200 * running requests,
fitted to the production head log (diagnostics/ds-decode-yield-20261104/analysis/fit_models.py). The deferral is the
engine's own one (`_should_defer_prefill`, the path `--prefill-decode-interval` already takes between chunks in
production): the long prompt's chunk stays in flight, untouched, and each deferred round runs one ordinary decode
batch of the running requests. Each deferred round charges step_us(current batch size) against the budget. The yield
ends early (no idle round) when no decode row is left or the chunked prompt is gone (aborted / finished). It is never
shorter than the configured interval (the production cadence) and never longer than 128 decode steps.

Not armed after: a final chunk (the prompt joins the running batch and decodes at once), a solo short pass of
DSV41_PREFILL_BYPASS_SOLO (its batch carries no chunked request), a chunk with no decode row running (a lone long
prompt prefills exactly as before), and anything under DP attention, dLLM or a prefill-only batch.

Exactness. Only WHEN the next chunk runs changes. The chunk boundaries, the chunk batches and the decode batches are
the ones the engine builds anyway (the decode batches merely run more often between two chunks), so every request's
computation is the same; no kernel, model code, batch content or memory accounting is touched.

Rank invariance. Every TP rank runs its own scheduler, and they agree only by computing the same decisions from the
same state. Every input here is replicated scheduler state: the scheduled batch (its mode, chunked request, extend
token count, the chunk's start position), the running batch size, the request id, and integer constants. No clock, no
memory reading, no environment read after install, no float arithmetic.

Settings (identical on every rank; read once at install):
  DSV41_DECODE_YIELD          unset / 0 / off = not installed (sitecustomize checks before importing this module);
                              1..90 = the decode share pct; anything else refuses to arm (stock behaviour).
  DSV41_DECODE_YIELD_RID_ARMS TEST ONLY. 1 = a chunked request whose rid starts with "dyield-s<NN>-" uses share NN
                              (0..90; 0 = the stock interval only) instead of the default, so one boot can A/B several
                              shares (the rid is replicated to every rank). Unset in serving.
Receipt: "DSV41_DECODE_YIELD_RECEIPT {json}" on the first yield of every chunked request and every 256th yield
(counts and the last yield's inputs; no time, no rank), identical on all ranks.
"""
import hashlib
import inspect
import json
import logging
import os

logger = logging.getLogger(__name__)

ENV = "DSV41_DECODE_YIELD"
ENV_RID_ARMS = "DSV41_DECODE_YIELD_RID_ARMS"
_MARK = "_dsv41_decode_yield"
RID_PREFIX = "dyield-s"
MAX_PCT = 90
MAX_STEPS = 128
RECEIPT_EVERY = 256

# Cost model (integers, microseconds), least squares on the production head log 2026-11-03 12:39 .. 2026-11-04 11:13
# (analysis/fit_models.py): 655 chunk gaps over prefix 0-664K, 5,101 decode-line gaps over batch size 1-8.
CHUNK_REF_TOKENS = 8192
CHUNK_BASE_US = 1_354_500        # an 8192-token chunk at prefix 0
CHUNK_PREFIX_NUM = 487           # + 0.487 us per prefix token (per 8192-token chunk)
CHUNK_PREFIX_DEN = 1000
STEP_BASE_US = 26_300            # decode step (DSpark verify) = 26.3 ms + 15.2 ms per running request
STEP_PER_REQ_US = 15_200

# sha256 of inspect.getsource(inspect.unwrap(...)) of the engine pieces this relies on, pinned SGLang dsv4.1 tree
# f80c91a4b + the recipe's patches (incl. #44001 min-bs). On drift: not installed (stock behaviour), logged.
PINS = {
    "Scheduler.get_next_batch_to_run":
        "adf3e4689e0e3b55f29616ca52b8ca0409f1330468b0f4a24a753502a6ae1084",
    "Scheduler._should_defer_prefill":
        "0ac4b94664ec65be18da4dca0dd8ca93ee12eda8ebf566d6ae8d9e7489075fe7",
    "Scheduler._arm_prefill_decode_interval":
        "54db911b31a3cfcb1f4e2463d9d9e39b9e31cc3579c9df487cf48265b2be6933",
}


class _Stats:
    installed = False
    pct = 0
    rid_arms = False
    yields = 0          # yields armed (one per non-final chunk with decode rows running)
    steps = 0           # deferred rounds charged to a yield (one decode batch each)
    cancels = 0         # yields ended early: no decode row left or no chunked prompt
    last_rid = None
    last = None


STATS = _Stats()


def parse_pct(raw):
    """None = off; int 1..MAX_PCT = on; ValueError = refuse."""
    v = (raw or "").strip().lower()
    if v in ("", "0", "off", "false", "no"):
        return None
    pct = int(v)                       # ValueError on garbage
    if not 1 <= pct <= MAX_PCT:
        raise ValueError(f"{ENV} must be 1..{MAX_PCT} (percent decode share), got {raw!r}")
    return pct


def requested(env=None):
    env = os.environ if env is None else env
    try:
        return parse_pct(env.get(ENV)) is not None
    except ValueError:
        return True       # install() logs the refusal


def chunk_us(tokens, prefix):
    return tokens * (CHUNK_BASE_US + prefix * CHUNK_PREFIX_NUM // CHUNK_PREFIX_DEN) // CHUNK_REF_TOKENS


def step_us(bs):
    return STEP_BASE_US + STEP_PER_REQ_US * bs


def yield_budget_us(pct, tokens, prefix, bs, interval):
    """Decode time owed to the running streams after one chunk: pct / (100 - pct) of the chunk, at least `interval`
    decode steps, at most MAX_STEPS (both at the batch size the chunk ran beside). Integers only."""
    c = chunk_us(tokens, prefix)
    s = step_us(bs)
    want = -(-pct * c // (100 - pct))                 # ceil
    lo = max(int(interval), 0) * s
    hi = MAX_STEPS * s
    return min(max(want, lo), hi)


def share_for(rid, default_pct, rid_arms):
    if rid_arms and isinstance(rid, str) and rid.startswith(RID_PREFIX):
        head = rid[len(RID_PREFIX):].split("-", 1)[0]
        if head.isdigit() and 0 <= int(head) <= MAX_PCT:
            return int(head)
    return default_pct


def source_sha(obj):
    obj = inspect.unwrap(obj)
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


def pin_drift(cls, pins=None):
    pins = PINS if pins is None else pins
    found = {
        "Scheduler.get_next_batch_to_run": cls.get_next_batch_to_run,
        "Scheduler._should_defer_prefill": cls._should_defer_prefill,
        "Scheduler._arm_prefill_decode_interval": cls._arm_prefill_decode_interval,
    }
    return [name for name, fn in found.items() if source_sha(fn) != pins.get(name)]


def _receipt(event, info):
    try:
        logger.warning("DSV41_DECODE_YIELD_RECEIPT %s", json.dumps(dict(
            event=event, pct=STATS.pct, yields=STATS.yields, steps=STATS.steps, cancels=STATS.cancels, **info),
            sort_keys=True))
    except Exception:  # noqa: BLE001 - evidence only
        pass


def _decode_rows(running_batch):
    return (running_batch is not None and not running_batch.is_empty()
            and not getattr(running_batch, "is_prefill_only", False))


def arm_after(scheduler, plan):
    """Called with the plan get_next_batch_to_run returned. Arms a yield for a non-final chunk beside decode rows."""
    batch = plan.batch_to_run
    if batch is None or getattr(scheduler, "require_mlp_sync", False) or scheduler.dllm_config is not None:
        return
    long = getattr(batch, "chunked_req", None)
    if (long is None or long is not scheduler.chunked_req or batch is plan.running_batch
            or getattr(batch.forward_mode, "name", None) != "EXTEND"):
        return
    running = plan.running_batch
    if not _decode_rows(running):
        return
    pct = share_for(getattr(long, "rid", None), STATS.pct, STATS.rid_arms)
    if pct <= 0:
        return
    tokens = int(batch.extend_num_tokens or 0)
    if tokens <= 0:
        return
    rng = getattr(long, "extend_range", None)
    prefix = int(rng.start) if rng is not None else len(long.prefix_indices)
    bs = running.batch_size()
    budget = yield_budget_us(pct, tokens, prefix, bs, scheduler.prefill_decode_interval)
    scheduler._dsv41_yield_left_us = budget
    STATS.yields += 1
    info = {"rid": str(long.rid), "prefix": prefix, "tokens": tokens, "bs": bs, "share": pct, "budget_us": budget}
    STATS.last = info
    if STATS.last_rid != info["rid"]:
        STATS.last_rid = info["rid"]
        _receipt("first", info)
    elif STATS.yields % RECEIPT_EVERY == 0:
        _receipt("every", info)


def defer_step(scheduler, running_batch):
    """True/False when a yield decides this round, None to fall through to the stock rule."""
    left = getattr(scheduler, "_dsv41_yield_left_us", 0)
    if left <= 0:
        return None
    if not _decode_rows(running_batch) or scheduler.chunked_req is None:
        scheduler._dsv41_yield_left_us = 0
        scheduler._prefill_decode_interval_remaining = 0
        STATS.cancels += 1
        return False
    scheduler._dsv41_yield_left_us = left - step_us(running_batch.batch_size())
    # the stock interval counter (armed by the same chunk) is part of this yield
    scheduler._prefill_decode_interval_remaining = 0
    STATS.steps += 1
    return True


def install(module, env=None):
    """Wrap Scheduler.get_next_batch_to_run and Scheduler._should_defer_prefill (module = managers.scheduler)."""
    env = os.environ if env is None else env
    # Override site: a DSV41_DECODE_YIELD value outside 1..MAX_PCT used to log "refusing to arm"
    # and keep the stock scheduler. A user-set invalid value now raises.
    try:
        pct = parse_pct(env.get(ENV))
    except ValueError as exc:
        raise RuntimeError(f"DSV41_DECODE_YIELD: {ENV}={env.get(ENV)!r} cannot arm: {exc}") from None
    if pct is None:
        return False
    rid_raw = (env.get(ENV_RID_ARMS) or "").strip().lower()
    # Override site: an invalid DSV41_DECODE_YIELD rid-arms value used to refuse to arm silently
    # for the whole feature. A user-set invalid value now raises.
    if rid_raw not in ("", "0", "1", "off", "on"):
        raise RuntimeError(
            f"DSV41_DECODE_YIELD: {ENV_RID_ARMS}={env.get(ENV_RID_ARMS)!r} must be 0 or 1 "
            f"({ENV}={env.get(ENV)!r} is set)")
    cls = getattr(module, "Scheduler", None)
    # Override site: a missing Scheduler class used to refuse to arm with a log line. The gate
    # DSV41_DECODE_YIELD is set when this runs, so the gate that does not take effect now raises.
    if cls is None:
        raise RuntimeError(f"DSV41_DECODE_YIELD: {ENV}={env.get(ENV)!r} is set but the module has no Scheduler class")
    if getattr(cls.get_next_batch_to_run, _MARK, False):
        return True
    try:
        drift = pin_drift(cls)
    except (AttributeError, TypeError, OSError) as exc:
        drift = [repr(exc)]
    # Override site: engine source drift used to refuse to arm with a log line. The gate
    # DSV41_DECODE_YIELD is set when this runs, so the gate that does not take effect now raises.
    if drift:
        raise RuntimeError(
            f"DSV41_DECODE_YIELD: {ENV}={env.get(ENV)!r} is set but the engine source drifted in "
            f"{', '.join(drift)}")
    original_next = cls.get_next_batch_to_run
    original_defer = cls._should_defer_prefill

    def get_next_batch_to_run(self, *args, **kwargs):
        plan = original_next(self, *args, **kwargs)
        arm_after(self, plan)
        return plan

    def _should_defer_prefill(self, running_batch):
        decided = defer_step(self, running_batch)
        if decided is None:
            return original_defer(self, running_batch)
        return decided

    for fn, orig in ((get_next_batch_to_run, original_next), (_should_defer_prefill, original_defer)):
        setattr(fn, _MARK, True)
        fn.__wrapped__ = orig
    cls.get_next_batch_to_run = get_next_batch_to_run
    cls._should_defer_prefill = _should_defer_prefill
    STATS.installed = True
    STATS.pct = pct
    STATS.rid_arms = rid_raw in ("1", "on")
    logger.warning("DSV41_DECODE_YIELD armed: share=%d%%: after a non-final chunk beside running streams, decode "
                   "runs for %d/%d of the chunk's modelled time before the next chunk (min the configured interval, "
                   "max %d steps)%s", pct, pct, 100 - pct, MAX_STEPS,
                   "; TEST rid arms on (dyield-s<NN>-)" if STATS.rid_arms else "")
    return True
