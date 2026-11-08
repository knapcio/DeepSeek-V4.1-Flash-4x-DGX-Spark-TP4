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
}

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
