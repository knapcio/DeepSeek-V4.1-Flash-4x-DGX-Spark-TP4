"""Sampled DSpark steps compute the draft probabilities on a side stream while the verify graph runs. Default OFF.

DSV41_SAMPLED_DRAFT_PROBS_EARLY=1 | check | 0;  DSV41_SAMPLED_DRAFT_PROBS_EARLY_MAX_BS=2 (larger batches: stock).

Why. A sampled step's eager accept starts with draft_probs = SoftmaxTemp(corrected_logits, temperatures) (dspark_verify.
accept_draft_tokens): repeat_interleave, bf16 -> fp32 cast, FlashInfer online softmax (map + reduce), clamp. Its
inputs are the draft graph's persistent corrected_out rows and the staged draft temperatures, both final when the
proposal returns, i.e. ~28 ms before the accept runs, but the work sits on the critical path after the verify graph.
Trace 2026-11-08 (T 1.0 / top-p 0.95, c1 prose, rank 0): 10.7 + 36.5 + 25.6 + 30.0 + 5.2 = ~108 us of GPU plus five
dispatch gaps per step (diagnostics/c3-sampled-accept-20261109/DESIGN.md).

What. Right after DraftBlockProposer.propose returns for a batch that is not all-greedy, the SAME call
(SoftmaxTemp.execute on the same tensors, same arguments) is launched on a side stream that first waits for the
forward stream, so it overlaps the target verify graph. accept_draft_tokens' SoftmaxTemp.execute then returns that
tensor (the forward stream waits on the side stream's event first) instead of recomputing it, once per step, only
when the call's logits / temperatures / rows_per_request are exactly the ones launched early.

Exactness (bitwise, Class A). Same function, same input tensors (nothing writes corrected_out or the staged
temperatures between the proposal and the accept: the next draft replay is after the accept), same kernels; only the
stream and the launch time differ. No random numbers are drawn, so the RNG stream is untouched. check mode recomputes
on the forward stream, returns the recomputed tensor (production behaviour unchanged) and counts on device every
element whose bits differ (contract: 0), logged every DSV41_SAMPLED_DRAFT_PROBS_EARLY_LOG_EVERY uses and at idle.

Rank invariance: no collective is added, skipped or moved; every rank computes the same values either way.
CUDA graphs: eager work only; never launched while a stream is capturing.
Memory: the [bs*5, V] fp32 probabilities (2.6 MB per request) live from the proposal to the accept instead of only
inside the accept; MAX_BS bounds it (default 2: <= 5.2 MB).
"""
import os

import torch

KEY = "DSV41_SAMPLED_DRAFT_PROBS_EARLY"
_OFF = ("", "0", "off", "false", "no")
MAX_BS = int(os.environ.get("DSV41_SAMPLED_DRAFT_PROBS_EARLY_MAX_BS", "2") or 2)
LOG_EVERY = int(os.environ.get("DSV41_SAMPLED_DRAFT_PROBS_EARLY_LOG_EVERY", "2000") or 0)
_CPU_TEST = os.environ.get("DSV41_SAMPLED_DRAFT_PROBS_EARLY_CPU", "0") == "1"   # tests only: no streams on CPU

try:  # DSV41_AB_VARIANTS in-boot A/B (test only): read per call on the replayed variant
    import ab_variant as _ab
except ImportError:  # pragma: no cover
    _ab = None


def parse_mode(value):
    v = (value or "0").strip().lower()
    if v in _OFF:
        return "off"
    if v == "check":
        return "check"
    if v in ("1", "on", "true", "yes"):
        return "on"
    raise RuntimeError(f"{KEY}={value!r}: expected 0 / 1 / check")


def installed():
    return parse_mode(os.environ.get(KEY, "0")) != "off"


def mode():
    if _ab is not None and getattr(_ab, "ACTIVE", False):
        return parse_mode(_ab.env(KEY, "0"))
    return parse_mode(os.environ.get(KEY, "0"))


def _capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


_S = {"pending": None, "side": {}, "orig": None}
STATS = {"early": 0, "used": 0, "check": 0, "unused": 0}
REASONS = {}
_C = {"counters": None}            # check mode, on device: [uses, elements, mismatched elements]
_LOGGED = {"n": 0}


def _rank():
    try:
        import torch.distributed as dist
        return dist.get_rank() if dist.is_initialized() else 0
    except Exception:  # noqa: BLE001
        return 0


def _skip(reason):
    REASONS[reason] = REASONS.get(reason, 0) + 1


def _lines(at=None):
    tail = "" if at is None else f" at={at}"
    out = [f"[sampled_draft_probs_early] receipt rank={_rank()} early={STATS['early']} used={STATS['used']} "
           f"check={STATS['check']} unused={STATS['unused']} skipped={dict(sorted(REASONS.items()))}{tail}"]
    c = _C["counters"]
    if c is not None:
        v = c.tolist()                              # host sync: log lines only (check mode)
        out.append(f"[sampled_draft_probs_early] check: rank={_rank()} uses {v[0]} elements {v[1]} "
                   f"mismatched elements {v[2]}{tail}")
    return out


def _tick():
    n = STATS["used"] + STATS["check"]
    if LOG_EVERY and n % LOG_EVERY == 0:
        for line in _lines():
            print(line, flush=True)


def flush(at="idle"):
    n = STATS["early"]
    if n == _LOGGED["n"]:
        return
    _LOGGED["n"] = n
    try:
        for line in _lines(at):
            print(line, flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[sampled_draft_probs_early] terminal receipt failed: {exc!r}", flush=True)


def _key(logits, temperatures, rows_per_request):
    return (logits.data_ptr(), tuple(logits.shape), tuple(logits.stride()), logits.dtype, logits.device,
            temperatures.data_ptr(), tuple(temperatures.shape), temperatures.dtype, int(rows_per_request))


def _side(device):
    s = _S["side"].get(device)
    if s is None:
        s = _S["side"][device] = torch.cuda.Stream(device=device)
    return s


def launch_early(draft_block, sampling_info, execute):
    """After the proposal: start SoftmaxTemp(corrected_logits, temperatures) on the side stream (sampled batches)."""
    if _S["pending"] is not None:
        STATS["unused"] += 1                        # last step's tensor never consumed (e.g. a greedy accept)
    _S["pending"] = None
    if mode() == "off":
        return
    if sampling_info is None or sampling_info.is_all_greedy:
        return                                      # the greedy accept never builds draft probabilities
    cl = getattr(draft_block, "corrected_logits", None)
    temps = getattr(draft_block, "temperatures", None)
    if cl is None or temps is None or cl.dim() != 3:
        _skip("no_logits")
        return
    if not cl.is_cuda and not _CPU_TEST:
        _skip("cpu")
        return
    if _capturing():
        _skip("capturing")
        return
    bs, rows, vocab = cl.shape
    if bs > MAX_BS:
        _skip("bs")
        return
    logits = cl.reshape(bs * rows, vocab)           # accept_draft_tokens' own reshape (a view: same pointer)
    if cl.is_cuda:
        main = torch.cuda.current_stream(cl.device)
        side = _side(cl.device)
        side.wait_stream(main)                      # the draft graph's corrected_out / staged temperatures
        with torch.cuda.stream(side):
            probs = execute(logits=logits, temperatures=temps, rows_per_request=rows)
        done = torch.cuda.Event()
        done.record(side)
    else:                                           # _CPU_TEST
        probs, done = execute(logits=logits, temperatures=temps, rows_per_request=rows), None
    _S["pending"] = (_key(logits, temps, rows), probs, done, (logits, temps))
    STATS["early"] += 1


def take(logits, temperatures, rows_per_request, execute):
    """accept_draft_tokens' SoftmaxTemp call: the early tensor when it is this call's, else None."""
    p = _S["pending"]
    if p is None:
        return None
    try:
        key = _key(logits, temperatures, rows_per_request)
    except Exception:  # noqa: BLE001 - not tensors: not ours
        return None
    if key != p[0]:
        return None
    _S["pending"] = None
    m = mode()
    if m == "off":
        _skip("mode_off_at_use")
        return None
    probs = p[1]
    if p[2] is not None:
        main = torch.cuda.current_stream(logits.device)
        main.wait_event(p[2])
        probs.record_stream(main)                   # allocated on the side stream, used (and freed) on main
    if m == "check":
        STATS["check"] += 1
        ref = execute(logits=logits, temperatures=temperatures, rows_per_request=rows_per_request)
        c = _C["counters"]
        if c is None or c.device != ref.device:
            c = _C["counters"] = torch.zeros(3, dtype=torch.int64, device=ref.device)
        same_shape = ref.shape == probs.shape and ref.dtype == probs.dtype
        bad = ((ref.view(torch.int32) != probs.view(torch.int32)).sum() if same_shape
               else torch.full((), ref.numel(), dtype=torch.int64, device=ref.device))
        one = torch.ones((), dtype=torch.int64, device=c.device)
        c.add_(torch.stack([one, torch.full_like(one, ref.numel()), bad.to(torch.int64)]))
        _tick()
        return ref
    STATS["used"] += 1
    _tick()
    return probs


# ------------------------------------------------------------------------------------------------- install --
def install_draft(module):
    """sglang.srt.speculative.dspark_components.dspark_draft: DraftBlockProposer.propose."""
    if not installed():
        return False
    cls = getattr(module, "DraftBlockProposer", None)
    if cls is None or not hasattr(cls, "propose"):
        print("[sampled_draft_probs_early] NOT installed: DraftBlockProposer.propose gone", flush=True)
        return False
    if getattr(cls, "_dsv41_draft_probs_early", False):
        return True
    cls._dsv41_draft_probs_early = True
    orig = cls.propose

    def propose(self, *args, **kw):
        out = orig(self, *args, **kw)
        execute = _S["orig"]
        if execute is not None and not args:
            launch_early(getattr(out, "draft_block", None), kw.get("sampling_info"), execute)
        return out

    propose.__wrapped__ = orig
    cls.propose = propose
    print("[sampled_draft_probs_early] armed on DraftBlockProposer.propose", flush=True)
    return True


def install_verify(module):
    """sglang.srt.speculative.dspark_components.dspark_verify: its SoftmaxTemp name (used only by
    accept_draft_tokens for the draft probabilities) through take(). The engine's SoftmaxTemp class is untouched."""
    if not installed():
        return False
    st = getattr(module, "SoftmaxTemp", None)
    if st is None or not hasattr(st, "execute") or not hasattr(module, "accept_draft_tokens"):
        print("[sampled_draft_probs_early] NOT installed: dspark_verify.SoftmaxTemp gone", flush=True)
        return False
    if getattr(st, "_dsv41_draft_probs_early", False):
        return True
    execute = st.execute
    _S["orig"] = execute

    class SoftmaxTempEarly:
        _dsv41_draft_probs_early = True
        _dsv41_wrapped = st

        @staticmethod
        def execute(*args, **kw):
            if not args and set(kw) == {"logits", "temperatures", "rows_per_request"}:
                got = take(kw["logits"], kw["temperatures"], kw["rows_per_request"], execute)
                if got is not None:
                    return got
            return execute(*args, **kw)

    for name in ("torch", "triton", "flashinfer"):
        if hasattr(st, name):
            setattr(SoftmaxTempEarly, name, getattr(st, name))
    module.SoftmaxTemp = SoftmaxTempEarly
    print("[sampled_draft_probs_early] armed on dspark_verify.SoftmaxTemp (draft probabilities)", flush=True)
    return True


def install_scheduler(module):
    """sglang.srt.managers.scheduler: terminal receipts on the busy -> idle transition."""
    if not installed():
        return
    cls = module.Scheduler
    if getattr(cls, "_dsv41_dpe_receipt", False):
        return
    name = next((n for n in ("on_idle", "self_check_during_idle") if hasattr(cls, n)), None)
    if name is None:
        return
    original = getattr(cls, name)

    def idle_hook(self, *args, **kwargs):
        flush("idle")
        return original(self, *args, **kwargs)

    idle_hook.__wrapped__ = original
    setattr(cls, name, idle_hook)
    cls._dsv41_dpe_receipt = True
