"""Non-final prefill chunks run the late layers on ONE row (DSV41_PREFILL_CED_SKIP=1, default OFF).

Ported from Christopher Owen's spark-ds41f (Apache-2.0): vLLM patches 0229-0255 ("skip its nonfinal rows", 0255 "CED
skip: keep one decoder row of unread nonfinal chunks") and 0526-0527 (replay a prompt's last CED rows wherever its
chunks end; ced.py::nonfinal_limits). Adapted here to SGLang's decoder SWA bounded replay. Analysis and cost:
diagnostics/ced-late-tail-profile-20261108/PLAN.md and diagnostics/fable-ideas-20261108/prefill.md idea 3 (the v2.7
trace: the late-layer tail costs 86-93 ms per 8192-row chunk; a one-row pass ~25 ms).

What the engine does today. Under ``enable_decoder_swa_bounded_replay`` (boot.py always passes it) every extend runs
layers 0..19 over all rows; layer 20 (the last kv_source layer) writes its compressed KV for all rows, then layers
20..39 run over each request's last SWA_WINDOW = 128 extend rows only (``late_layer_tail_layout``, window floored at
seq_len - 128: rows below the floor are never written at those layers). That tail runs for EVERY chunk, also for a
chunk whose prompt goes on for more rows afterwards.

What this changes. For a request whose prompt has at least SWA_WINDOW rows still to come after this chunk
(final_len - seq_len >= 128, the "unread nonfinal chunk"), the tail is that chunk's LAST row only, with its window
floored at its own position (seq_len - 1: it reads no SWA slot of the late layers below itself, so nothing unwritten).
All other requests (the chunk that ends a prompt, the chunk before a final chunk of < 128 rows, bypass prompts, any
request whose final length is unknown) keep the stock tail and floor, byte for byte. Everything downstream (tail
rows, positions, logits rows, the DSpark aux rows, the late-layer indexer's publish rows) is derived by the engine
from the layout's outputs, so it follows.

Why the outputs the prompt uses are unchanged.
* Layers 0..19 and layer 20's compressed-KV write see the same rows as before: untouched.
* What the late layers persist is their own SWA KV (per-layer MQA wkv). It is read only by rows inside the window:
  - the next chunk's tail: floored at its own seq_len - 128 >= this chunk's end (it has >= 128 rows, or the final
    chunk after it is >= 128 rows: that is the qualifying rule);
  - the final chunk's tail and every decode step: positions >= final_len - 128, which lie in chunks that do NOT
    qualify (final_len - seq_len < 128), so their tails are the stock ones;
  - a later request's prefix hit: #41303 (patches/sglang-41303-bounded-replay-window.patch) holds one full window
    back from every radix match, so a hit re-prefills >= 128 rows and its own tail never reads the cached prefix's
    late rows; the SWA cone replay (swa_cone.py) re-prefills >= 3,072 rows from s and floors its window at s.
* The skipped chunk's own logits row (discarded: the request is still chunked) is computed from a one-row window.
* The DSpark draft receives the tail rows' aux hidden states (layers 37-39) for its SWA (compress ratio 0, window
  128): a skipped chunk injects 1 row instead of 128. Those rows lie >= 128 positions before the prompt end, outside
  the draft's decode window. If the draft's injection itself attends back across the final chunk's start, its
  context can differ: that is why the in-boot A/B gates on a T 0 qeval bracket and a 64-token greedy continuation,
  not only on the first token (the prefill's own output, which must be bit-equal).
* Rank invariance: the flags come from host lists every TP rank holds identically (the scheduler's requests);
  under the in-boot A/B the gate is the replayed variant's (switched on the same iteration on every rank).
* Not touched: context parallelism (refused: stock), the draft model's own backend, speculative extends.

Plumbing: ForwardBatch.dsv41_final_lens (adapter/prefill_keep_cache.py install_forward_batch, installed when this gate
or the keep-cache gate is wanted). Missing -> stock tail.

Install: sitecustomize.py, the deepseek_v4_backend branch, AFTER extend_nosync (this wraps whatever is bound:
the stock method or extend_nosync's dispatcher, and the module-level late_layer_tail_layout name both resolve).

Receipts: "[prefill_ced_skip] armed: ..." once; "[prefill_ced_skip] forward <k>: <n> request(s) tail 128 -> 1 (rows
saved <r>; totals skipped <S> rows_saved <R>)" for the first 4 skipping forwards, then every 64th, per rank;
"[prefill_ced_skip] NOT armed: <reason>" when the engine does not match (the stock path runs).
"""
import inspect
import logging
import os

logger = logging.getLogger(__name__)

GATE = "DSV41_PREFILL_CED_SKIP"
_OFF = ("", "0", "off", "false", "no")
_TAG = "[prefill_ced_skip]"

try:  # DSV41_AB_VARIANTS in-boot A/B (test only)
    import ab_variant as _ab
except ImportError:  # CPU tests without the adapter dir
    _ab = None

_STATE = {"installed": False, "static": False, "forwards": 0, "skipped": 0, "rows_saved": 0, "unused": 0,
          "why_not": None}
_CTX = {"skip": None, "used": False}


def truthy(value):
    return value is not None and str(value).strip().lower() not in _OFF


def _ab_names_gate():
    if _ab is None or not getattr(_ab, "ACTIVE", False) or GATE not in getattr(_ab, "KNOWN", ()):
        return False
    return any(GATE in spec for spec in getattr(_ab, "_specs", ()) or ())


def wanted_at_install(environ=None):
    environ = os.environ if environ is None else environ
    return truthy(environ.get(GATE, "0")) or _ab_names_gate()


def on():
    """Per call: the in-boot A/B variant's value while armed and naming the gate, else the boot-time gate."""
    if _ab_names_gate():
        return truthy(_ab.env(GATE, "0"))
    return _STATE["static"]


# ------------------------------------------------------------------ pure parts (CPU-tested)
def plan(extend_lens, seq_lens, final_lens, window):
    """Per request: True = the tail shrinks to one row. None when nothing qualifies or the inputs are incomplete."""
    if final_lens is None or extend_lens is None or seq_lens is None:
        return None
    if not (len(final_lens) == len(extend_lens) == len(seq_lens)) or not len(extend_lens):
        return None
    flags = []
    for n, s, f in zip(extend_lens, seq_lens, final_lens):
        if f is None:
            return None
        n, s, f = int(n), int(s), int(f)
        flags.append(n > 1 and f - s >= window)
    return flags if any(flags) else None


def host_layout(extend_lens, seq_lens, window, skip):
    """Host lists of the CED layout: (token indices into the extend, per-request tail lengths, per-row floors). For an
    unskipped request: the stock min(window, n) rows and the stock floor max(0, seq - window); skipped: the last row,
    floored at its own position seq - 1."""
    tails = [1 if k else min(window, int(n)) for n, k in zip(extend_lens, skip)]
    idx, floors, start = [], [], 0
    for n, s, t, k in zip(extend_lens, seq_lens, tails, skip):
        n, s = int(n), int(s)
        idx.extend(range(start + n - t, start + n))
        floors.extend([s - 1 if k else max(0, s - window)] * t)
        start += n
    return idx, tails, floors


def stock_host_layout(extend_lens, seq_lens, window):
    """The engine's late_layer_tail_layout on the host (tests: the CED layout with no skip must equal it)."""
    return host_layout(extend_lens, seq_lens, window, [False] * len(extend_lens))


def _h2d(values, dtype, device):
    import torch
    device = torch.device(device)
    if device.type == "cuda":
        return torch.tensor(values, dtype=dtype, pin_memory=True).to(device, non_blocking=True)
    return torch.tensor(values, dtype=dtype, device=device)


def ced_layout(extend_lens_cpu, seq_lens_cpu, tail_len, device, skip):
    """late_layer_tail_layout's contract (token indices int64, tail lengths list, floors int32) with skipped
    requests reduced to their last row. No host sync: host lists, one pinned non-blocking copy each."""
    import torch
    idx, tails, floors = host_layout(extend_lens_cpu, seq_lens_cpu, tail_len, skip)
    if len(extend_lens_cpu) == 1:
        n, t = int(extend_lens_cpu[0]), tails[0]
        return (torch.arange(n - t, n, device=device), tails,
                torch.full((t,), floors[0], dtype=torch.int32, device=device))
    return _h2d(idx, torch.int64, device), tails, _h2d(floors, torch.int32, device)


# ------------------------------------------------------------------ engine hook
def _note(skip, extend_lens, window):
    n = sum(1 for k in skip if k)
    saved = sum(min(window, int(e)) - 1 for e, k in zip(extend_lens, skip) if k)
    _STATE["forwards"] += 1
    _STATE["skipped"] += n
    _STATE["rows_saved"] += saved
    k = _STATE["forwards"]
    if k <= 4 or k % 64 == 0:
        print(f"{_TAG} forward {k}: {n} request(s) tail {window} -> 1 (rows saved {saved}; totals skipped "
              f"{_STATE['skipped']} rows_saved {_STATE['rows_saved']})", flush=True)


def _engine_ok(module):
    cls = getattr(module, "DeepseekV4AttnBackend", None)
    if cls is None or "_build_late_layer_tail_metadata" not in cls.__dict__:
        return "DeepseekV4AttnBackend._build_late_layer_tail_metadata not found"
    if getattr(module, "SWA_WINDOW", None) != 128:
        return f"SWA_WINDOW is {getattr(module, 'SWA_WINDOW', None)!r}, the analysis assumes 128"
    if not callable(getattr(module, "late_layer_tail_layout", None)):
        return "late_layer_tail_layout is not bound in the backend module"
    meth = cls.__dict__["_build_late_layer_tail_metadata"]
    try:
        src = inspect.getsource(getattr(meth, "__wrapped__", meth))
    except (OSError, TypeError):
        return "cannot read the tail metadata builder's source"
    for need in ("late_layer_tail_layout(", "tail_len=SWA_WINDOW", "num_tokens=sum(tail_lens_cpu)",
                 "extend_seq_lens_cpu=tail_lens_cpu", "swa_replay_start=swa_replay_start"):
        if need not in src:
            return f"the tail metadata builder no longer contains {need!r}"
    return None


def install_backend(module, environ=None):
    """sglang.srt.layers.attention.deepseek_v4_backend: wrap the tail metadata builder and the layout name."""
    if _STATE["installed"]:
        return True
    if not wanted_at_install(environ):
        return False
    why = _engine_ok(module)
    if why is not None:
        _STATE["why_not"] = why
        print(f"{_TAG} NOT armed: {why}", flush=True)
        return False
    cls = module.DeepseekV4AttnBackend
    build = cls.__dict__["_build_late_layer_tail_metadata"]
    layout = module.late_layer_tail_layout
    window = module.SWA_WINDOW
    cp_active = getattr(module, "is_cp_active", None)

    def _build_late_layer_tail_metadata(self, forward_batch):
        skip = None
        if on() and not getattr(self, "is_dspark_draft", False):
            cp = (cp_active(forward_batch) if callable(cp_active)
                  else getattr(forward_batch, "attn_cp_metadata", None) is not None)
            if not cp:
                seq = getattr(forward_batch, "seq_lens_cpu", None)
                seq = seq.tolist() if hasattr(seq, "tolist") else seq
                skip = plan(getattr(forward_batch, "extend_seq_lens_cpu", None), seq,
                            getattr(forward_batch, "dsv41_final_lens", None), window)
        if skip is None:
            return build(self, forward_batch)
        _CTX["skip"], _CTX["used"] = skip, False
        try:
            md = build(self, forward_batch)
            used = _CTX["used"]
        finally:
            _CTX["skip"], _CTX["used"] = None, False
        if used:
            _note(skip, forward_batch.extend_seq_lens_cpu, window)
        else:
            _STATE["unused"] += 1
        return md

    def late_layer_tail_layout(*, extend_lens_cpu, seq_lens_cpu, tail_len, device):
        skip = _CTX["skip"]
        if skip is None or len(skip) != len(extend_lens_cpu) or tail_len != window:
            return layout(extend_lens_cpu=extend_lens_cpu, seq_lens_cpu=seq_lens_cpu, tail_len=tail_len,
                          device=device)
        _CTX["used"] = True
        return ced_layout(extend_lens_cpu, seq_lens_cpu, tail_len, device, skip)

    _build_late_layer_tail_metadata.__wrapped__ = getattr(build, "__wrapped__", build)
    _build_late_layer_tail_metadata._dsv41_ced = True
    late_layer_tail_layout.__wrapped__ = getattr(layout, "__wrapped__", layout)
    late_layer_tail_layout._dsv41_ced = True
    setattr(cls, "_build_late_layer_tail_metadata", _build_late_layer_tail_metadata)
    module.late_layer_tail_layout = late_layer_tail_layout
    _STATE["static"] = truthy((os.environ if environ is None else environ).get(GATE, "0"))
    _STATE["installed"] = True
    print(f"{_TAG} armed: non-final prefill chunks with >= {window} prompt rows after them run the late layers "
          f"(20-39) on their last row only (Owen spark-ds41f 0255 keep-one-row); default off", flush=True)
    return True


def stats():
    return dict(_STATE)
