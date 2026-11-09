"""Eager extends without host syncs in their metadata build (DSV41_EXTEND_NOSYNC=1, default OFF). Bit-identical.

Why. With the overlap scheduler a new request's extend is launched while the previous decode step is still on the
GPU. Production v2.7 admits one ~28-token request at c16 every ~1.5 s, and each admission stalls decode by ~165 ms
(diagnostics/fable-ideas-20261108/tier0/RESULTS.md, Z1), the same at c4 (decode step 80 ms) as at c16 (190 ms). That
is the signature of an extend whose host launch starts only after the in-flight decode step has finished: a host sync
at the start of the extend (below) waits for that step, and the extend is then launched host-paced (~160 ms of host
time for ~105-115 ms of GPU work, 10-23 traces). Without the sync the launch runs under the decode step and the stall
becomes the extend's GPU time. DSV41_EXTEND_SYNC_AUDIT=1 (adapter/extend_sync_audit.py) lists whatever syncs remain.

The syncs removed here are integer bookkeeping only: the same values reach the same kernels in the same order.

* ``DeepseekV4AttnBackend._build_late_layer_tail_metadata`` runs for EVERY extend (``enable_decoder_swa_bounded_replay``
  is on in production), from ``init_forward_metadata``, before the first layer:
  - ``torch.tensor(tail_lens_cpu, dtype=torch.int32, device=device)``: a pageable host-to-device copy, which PyTorch
    finishes with a stream synchronize. Replaced by the same values through a pinned buffer, ``non_blocking=True``
    (the caching host allocator keeps the buffer until the copy has run, as SGLang's own ForwardBatch.init_new does).
  - ``torch.repeat_interleave(req_pool_indices, tail_lens)`` without ``output_size``: an implicit ``.item()``. The
    output size is ``sum(tail_lens_cpu)``, already on the host.
* ``late_layer_tail_layout`` (called from the method above; only multi-request extends reach the copy):
  ``torch.tensor([extend_lens_cpu, tail_lens_cpu, seq_lens_cpu], device=device)`` is pageable. Same pinned copy.
* ``EngramHasher.forward`` (sglang.srt.layers.engram), once per extend before layer 0: the token -> request map
  ``torch.repeat_interleave(torch.arange(bs), lens)`` has no ``output_size``. The size is ``sum(extend_seq_lens_cpu)``,
  the host mirror of ``lens``; when that mirror is missing or disagrees with ``extend_num_tokens`` the stock call runs.

Left alone (see diagnostics/i1-admission-20261108/DESIGN.md): cold-only Engram staging copies, prefill_sp's per-new-M
self-checks (M >= 2048), hc_fused's one-time self-check, and prefill_empty_cache's ``torch.cuda.empty_cache()`` after
extends of sequences >= 8192 tokens (after the launch: it does not delay the extend, it blocks the next iteration).

Stream ordering. A blocking copy is complete on the device before the host goes on; a non-blocking one is ordered on
the current (forward) stream only. Every consumer of these tensors is either on that stream or derived from tensors
computed on it, and SGLang's own ForwardBatch.init_new already stages extend_seq_lens / extend_prefix_lens the same
way for the same metadata; a reviewer should still confirm no side stream reads the raw tail lengths.

Mechanism. Each target's source is sha256-pinned (the v2.7 tree: f80c91a4b + the recipe patches); the replacement is
that source with the listed exact edits (each must match once), compiled into the original module's globals. On drift
the install raises: a gate that is set but silently not armed would turn an A/B into an A/A. A per-call dispatcher
picks the edited or the stock function, so the in-boot A/B (adapter/ab_variant.py, union key) switches it per batch;
the switch is broadcast with the requests, so every rank takes the same branch. Only the host-to-device staging and
the output size argument differ between the two branches.

Install: sitecustomize.py's ``sglang.srt.layers.attention.deepseek_v4_backend`` branch, gate checked before the import.
"""
import hashlib
import inspect
import os
import sys
import textwrap

import torch

GATE = "DSV41_EXTEND_NOSYNC"
_OFF = ("", "0", "off", "false", "no")

try:  # DSV41_AB_VARIANTS in-boot A/B (test only)
    import ab_variant as _ab
except ImportError:
    _ab = None


def _h2d(values, dtype, device):
    """torch.tensor(values, dtype=dtype, device=device) without a host sync: pinned staging + non_blocking copy on
    CUDA; elsewhere (CPU / meta tests) the plain constructor."""
    device = torch.device(device)
    if device.type == "cuda":
        return torch.tensor(values, dtype=dtype, pin_memory=True).to(device, non_blocking=True)
    return torch.tensor(values, dtype=dtype, device=device)


# target -> (sha256 of inspect.getsource on the v2.7 tree, [(old, new), ...] exact edits)
TARGETS = {
    "DeepseekV4AttnBackend._build_late_layer_tail_metadata": (
        "9c9e433899fc166cc1299ab15f4ddd7034f085b073a48f3a82330eaec6b6bb4b",
        [
            ("tail_lens = torch.tensor(tail_lens_cpu, dtype=torch.int32, device=device)",
             "tail_lens = _dsv41_h2d(tail_lens_cpu, torch.int32, device)"),
            ("forward_batch.req_pool_indices.to(torch.int64), tail_lens.to(torch.int64)\n",
             "forward_batch.req_pool_indices.to(torch.int64), tail_lens.to(torch.int64),\n"
             "            output_size=sum(tail_lens_cpu),\n"),
        ],
    ),
    "late_layer_tail_layout": (
        "4f914006daa4717c77134c201b76a873ed434e46a8a417b7512bb4e6f5706e62",
        [
            ("lens = torch.tensor([extend_lens_cpu, tail_lens_cpu, seq_lens_cpu], device=device)",
             "lens = _dsv41_h2d([extend_lens_cpu, tail_lens_cpu, seq_lens_cpu], torch.int64, device)"),
        ],
    ),
    # sglang.srt.layers.engram: the hasher runs once per forward before layer 0 (deepseek_v4.py, or
    # adapter/prefill_sp.py under SP); its extend branch sizes the token -> request map from device lengths.
    "EngramHasher.forward": (
        "5d17c65051cd1703ec2979cc57c5e23dbbc502a2c0d62ba04385fef0d7ca1334",
        [
            ("row = torch.repeat_interleave(torch.arange(bs, device=device), lens)",
             "row = torch.repeat_interleave(torch.arange(bs, device=device), lens, **_dsv41_ri_size(forward_batch, bs))"),
        ],
    ),
}


def _ri_size(forward_batch, bs):
    """{'output_size': total} when the host mirror of this extend's lengths is complete and agrees with the
    batch's host token count, else {} (the stock call, with its sync). Both come from the scheduler's lists, the
    same lists ForwardBatch.init_new copied to the device lengths the call repeats over."""
    lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if not isinstance(lens_cpu, (list, tuple)) or len(lens_cpu) != bs:
        _STATE["ri_fallback"] = _STATE.get("ri_fallback", 0) + 1
        return {}
    total = int(sum(lens_cpu))
    expect = getattr(forward_batch, "extend_num_tokens", None)
    if expect is not None and int(expect) != total:
        _STATE["ri_fallback"] = _STATE.get("ri_fallback", 0) + 1
        return {}
    return {"output_size": total}

_STATE = {"installed": False, "static": False, "calls": {}}


def enabled(environ=None):
    environ = os.environ if environ is None else environ
    return str(environ.get(GATE, "0")).strip().lower() not in _OFF


def on():
    """Per call: the in-boot A/B variant's value while armed, else the boot-time gate."""
    if _ab is not None and getattr(_ab, "ACTIVE", False) and GATE in getattr(_ab, "KNOWN", ()):
        return str(_ab.env(GATE, "0")).strip().lower() not in _OFF
    return _STATE["static"]


def sha(src):
    return hashlib.sha256(src.encode()).hexdigest()


def apply_edits(src, edits):
    """Dedented source with every (old, new) applied; each old must occur exactly once."""
    out = textwrap.dedent(src)
    for old, new in edits:
        n = out.count(old)
        if n != 1:
            raise RuntimeError(f"{GATE}: edit anchor found {n} times (expected 1): {old[:80]!r}")
        out = out.replace(old, new)
    return out


def build(fn, name, pin, edits, globals_):
    """Compile the edited copy of fn in globals_ (the defining module's namespace)."""
    src = inspect.getsource(fn)
    if sha(src) != pin:
        raise RuntimeError(f"{GATE}: {name} drifted (sha256 {sha(src)[:16]} != pinned {pin[:16]}); "
                           "unset the gate or re-pin against this engine tree")
    code = compile(apply_edits(src, edits), f"<{GATE}:{name}>", "exec")
    scope = {}
    globals_.setdefault("_dsv41_h2d", _h2d)
    globals_.setdefault("_dsv41_ri_size", _ri_size)
    exec(code, globals_, scope)
    new = scope[fn.__name__]
    new.__qualname__ = fn.__qualname__
    new.__module__ = fn.__module__
    return new


def _dispatch(name, stock, edited):
    calls = _STATE["calls"].setdefault(name, [0, 0])

    def wrapper(*a, **kw):
        if on():
            calls[1] += 1
            return edited(*a, **kw)
        calls[0] += 1
        return stock(*a, **kw)

    wrapper.__wrapped__ = stock
    wrapper.__name__ = stock.__name__
    wrapper.__qualname__ = stock.__qualname__
    wrapper._dsv41_extend_nosync = True
    return wrapper


def install_backend(module, environ=None):
    """deepseek_v4_backend: the late-layer tail metadata and the tail layout, both pinned."""
    if not enabled(environ) or _STATE["installed"]:
        return _STATE["installed"]
    cls = module.DeepseekV4AttnBackend
    meth = "_build_late_layer_tail_metadata"
    stock_meth = cls.__dict__[meth]
    pin, edits = TARGETS[f"DeepseekV4AttnBackend.{meth}"]
    edited_meth = build(stock_meth, f"DeepseekV4AttnBackend.{meth}", pin, edits, module.__dict__)
    stock_layout = module.late_layer_tail_layout
    pin, edits = TARGETS["late_layer_tail_layout"]
    # compiled in the kernels module's namespace (its own imports), bound by name in the backend
    layout_globals = sys.modules[stock_layout.__module__].__dict__
    edited_layout = build(stock_layout, "late_layer_tail_layout", pin, edits, layout_globals)
    setattr(cls, meth, _dispatch(meth, stock_meth, edited_meth))
    module.late_layer_tail_layout = _dispatch("late_layer_tail_layout", stock_layout, edited_layout)
    _STATE["static"] = enabled(environ)
    _STATE["installed"] = True
    print(f"[{GATE}] armed: extend tail metadata without host syncs (pinned H2D, host-known repeat_interleave "
          "output size); bit-identical", flush=True)
    return True


def install_engram(module, environ=None):
    """sglang.srt.layers.engram: EngramHasher.forward's extend branch with a host-known output size. Must run
    before engram_backend / engram_prefetch wrap the hasher (sitecustomize.py calls it first)."""
    if not enabled(environ) or _STATE.get("engram"):
        return bool(_STATE.get("engram"))
    cls = module.EngramHasher
    stock = cls.__dict__["forward"]
    pin, edits = TARGETS["EngramHasher.forward"]
    edited = build(stock, "EngramHasher.forward", pin, edits, module.__dict__)
    cls.forward = _dispatch("EngramHasher.forward", stock, edited)
    _STATE["static"] = enabled(environ)
    _STATE["engram"] = True
    print(f"[{GATE}] armed: engram hasher extend map with a host-known output size; bit-identical", flush=True)
    return True
