"""CPU checks of DSV41_EXTEND_NOSYNC (adapter/extend_nosync.py) and DSV41_EXTEND_SYNC_AUDIT
(adapter/extend_sync_audit.py). No GPU, no sglang import: the stock functions are lifted from the engine files.

- the pinned sha256 of every target matches the engine tree given (the v2.7 staged tree, or the image's);
- each edit anchor matches exactly once and the edited source compiles;
- late_layer_tail_layout: edited == stock (values and dtypes) on 600 random batches, 1-16 requests, extends of
  1..700 tokens over cached prefixes, tail_len 128;
- _build_late_layer_tail_metadata: every argument handed to init_forward_metadata_prefill and every tensor the
  metadata keeps (tail lens, low_ratio_req_indices, positions, LateLayerTail) identical, on 300 random batches,
  with and without a request window (the SWA translate path), single- and multi-request;
- no host read of device data: on the meta device (a data-dependent size raises) the edited method runs and the stock
  method raises, which is the repeat_interleave .item() the knob removes; the pageable copies are gone from the
  edited source and _h2d stages CUDA copies in pinned memory with non_blocking;
- drift: a changed engine source refuses to build; a missing / doubled anchor refuses;
- the knob is read per call under the armed in-boot A/B (union key), once otherwise; off-by-default install is a no-op;
- the audit: site attribution skips torch / its own frames, bookkeeping and the summary line, and a non-extend batch
  and a CPU-only process pass straight through.

usage: PYTHONPATH=adapter python3 tests/test_extend_nosync.py [SGLANG_PYTHON_ROOT]
       (root = the directory that holds sglang/; default: the image path, then $DSV41_SGLANG_ROOT)
"""
import ast
import os
import random
import sys
import types

import torch

for k in ("DSV41_EXTEND_NOSYNC", "DSV41_EXTEND_SYNC_AUDIT", "DSV41_AB_VARIANTS"):
    os.environ.pop(k, None)
import extend_nosync as N  # noqa: E402
import extend_sync_audit as A  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("[OK]   " if cond else "[FAIL] ") + name + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def find_root():
    cands = sys.argv[1:2] + [os.environ.get("DSV41_SGLANG_ROOT", ""), "/sgl-workspace/sglang/python",
                             "/Users/zbizbi/Projects/sparks/repos/deepseek/ds41-pr73/runtime/sglang-canary/python"]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "sglang/srt/layers/attention/deepseek_v4_backend.py")):
            return c
    return None


def lift(path, qualname):
    """inspect.getsource-equivalent text of a module function or a class method, by AST."""
    src = open(path).read()
    lines = src.splitlines(keepends=True)
    parts = qualname.split(".")
    body = ast.parse(src).body
    node = None
    for p in parts:
        node = next(n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == p)
        body = node.body
    start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
    return "".join(lines[start:node.end_lineno])


ROOT = find_root()
if ROOT is None:
    print("SKIP: no SGLang tree (pass the python/ root that holds sglang/)")
    sys.exit(0)
BACKEND = os.path.join(ROOT, "sglang/srt/layers/attention/deepseek_v4_backend.py")
KERNELS = os.path.join(ROOT, "sglang/kernels/ops/attention/dsv4_attn_metadata_kernels.py")
SRC = {
    "DeepseekV4AttnBackend._build_late_layer_tail_metadata": lift(BACKEND, "DeepseekV4AttnBackend._build_late_layer_tail_metadata"),
    "late_layer_tail_layout": lift(KERNELS, "late_layer_tail_layout"),
}

# ---- 1. pins and anchors -------------------------------------------------------------------------------------
for name, (pin, edits) in N.TARGETS.items():
    check(f"pin {name}", N.sha(SRC[name]) == pin, f"tree sha {N.sha(SRC[name])}")
    try:
        compile(N.apply_edits(SRC[name], edits), name, "exec")
        check(f"edits apply and compile: {name}", True)
    except Exception as exc:  # noqa: BLE001
        check(f"edits apply and compile: {name}", False, repr(exc))


def make(name, src, globals_, edited):
    text = N.apply_edits(src, N.TARGETS[name][1]) if edited else __import__("textwrap").dedent(src)
    scope = {}
    globals_.setdefault("_dsv41_h2d", N._h2d)
    exec(compile(text, name, "exec"), globals_, scope)
    return next(iter(scope.values()))


# ---- 2. late_layer_tail_layout --------------------------------------------------------------------------------
G_K = {"torch": torch, "Optional": None}
stock_layout = make("late_layer_tail_layout", SRC["late_layer_tail_layout"], dict(G_K), False)
new_layout = make("late_layer_tail_layout", SRC["late_layer_tail_layout"], dict(G_K), True)
rng = random.Random(20261108)


def rand_batch(n):
    ext = [rng.choice((1, rng.randint(1, 40), rng.randint(1, 300), rng.randint(100, 700))) for _ in range(n)]
    seq = [e + rng.choice((0, rng.randint(0, 5000))) for e in ext]
    return ext, seq


def same(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and a.device == b.device and torch.equal(a, b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


bad = 0
for _ in range(600):
    ext, seq = rand_batch(rng.randint(1, 16))
    kw = dict(extend_lens_cpu=ext, seq_lens_cpu=seq, tail_len=128, device=torch.device("cpu"))
    if not same(stock_layout(**kw), new_layout(**kw)):
        bad += 1
check("late_layer_tail_layout edited == stock on 600 random batches", bad == 0, f"{bad} differ")

# ---- 3. _build_late_layer_tail_metadata -----------------------------------------------------------------------
class _Rec(types.SimpleNamespace):
    pass


class LateLayerTail(types.SimpleNamespace):
    pass


def _tail_rows(t, *, token_indices, contiguous_start):
    if contiguous_start is not None:
        return t[contiguous_start:]
    return t[token_indices]


def backend_globals(layout):
    return {"torch": torch, "ForwardBatch": object, "DSV4Metadata": object, "late_layer_tail_layout": layout,
            "SWA_WINDOW": 128, "_tail_rows": _tail_rows, "is_cp_active": lambda fb: False,
            "LateLayerTail": LateLayerTail}


MNAME = "DeepseekV4AttnBackend._build_late_layer_tail_metadata"
stock_meth = make(MNAME, SRC[MNAME], backend_globals(stock_layout), False)
new_meth = make(MNAME, SRC[MNAME], backend_globals(new_layout), True)


class FakeBackend:
    def __init__(self, window, device):
        self.calls = []
        self.device = device
        pool = types.SimpleNamespace()
        pool.request_window = (types.SimpleNamespace() if window else None)
        pool.translate_loc_from_full_to_swa = lambda loc: (loc * 3 + 1).to(torch.int64)
        self.token_to_kv_pool = pool
        self.window = window

    def init_forward_metadata_prefill(self, **kw):
        self.calls.append(kw)
        out_loc = kw["out_cache_loc"]
        layout = types.SimpleNamespace(write_loc=(out_loc + 7).to(torch.int32)) if self.window else None
        core = _Rec(request_window_layout=layout)
        return _Rec(core_attn_metadata=core)

    def _late_layer_tail_cp_layout(self, *a):
        raise AssertionError("CP is off")


def fake_batch(ext, seq, device, pool_rows=64):
    total = sum(ext)
    return types.SimpleNamespace(
        extend_seq_lens_cpu=list(ext),
        seq_lens_cpu=torch.tensor(seq, dtype=torch.int64),
        seq_lens=torch.tensor(seq, dtype=torch.int64, device=device),
        out_cache_loc=torch.randint(0, 1 << 30, (total,), dtype=torch.int64, device=device),
        positions=torch.randint(0, 1 << 20, (total,), dtype=torch.int64, device=device),
        req_pool_indices=torch.tensor(rng.sample(range(pool_rows), len(ext)), dtype=torch.int64, device=device),
    )


def flat(md, rec):
    out = {"low_ratio_req_indices": md.low_ratio_req_indices, "low_ratio_pos_i64": md.low_ratio_pos_i64,
           "swa_out": md.core_attn_metadata.swa_out_cache_loc}
    out.update({f"tail.{k}": v for k, v in vars(md.late_layer_tail).items()})
    out.update({f"call.{k}": v for k, v in rec.items() if k != "forward_batch"})
    return out


bad = []
for trial in range(300):
    n = 1 if trial % 3 == 0 else rng.randint(2, 12)
    ext, seq = rand_batch(n)
    fb = fake_batch(ext, seq, torch.device("cpu"))
    window = trial % 2 == 0
    b0, b1 = FakeBackend(window, "cpu"), FakeBackend(window, "cpu")
    m0, m1 = stock_meth(b0, fb), new_meth(b1, fb)
    f0, f1 = flat(m0, b0.calls[0]), flat(m1, b1.calls[0])
    if f0.keys() != f1.keys() or not all(same(f0[k], f1[k]) for k in f0):
        bad.append((trial, [k for k in f0 if k in f1 and not same(f0[k], f1[k])]))
check("_build_late_layer_tail_metadata edited == stock on 300 random batches (window / SWA translate)",
      not bad, str(bad[:3]))

# ---- 4. no host read on the meta device -----------------------------------------------------------------------
meta = torch.device("meta")


def meta_batch(ext, seq):
    fb = fake_batch(ext, seq, torch.device("cpu"))
    for k in ("seq_lens", "out_cache_loc", "positions", "req_pool_indices"):
        setattr(fb, k, getattr(fb, k).to(meta))
    return fb


for ext, seq, label in (([28], [28], "one 28-token admission"), ([28, 5, 300], [1028, 5, 300], "three requests")):
    try:
        new_meth(FakeBackend(False, meta), meta_batch(ext, seq))
        ok_new = True
    except Exception as exc:  # noqa: BLE001
        ok_new = False
        print("   edited raised on meta:", repr(exc)[:200])
    try:
        stock_meth(FakeBackend(False, meta), meta_batch(ext, seq))
        stock_raised = False
    except Exception:  # noqa: BLE001
        stock_raised = True
    check(f"meta device, {label}: edited runs without a host read", ok_new)
    check(f"meta device, {label}: stock needs a host read (the removed sync)", stock_raised)

edited_text = N.apply_edits(SRC[MNAME], N.TARGETS[MNAME][1]) + N.apply_edits(
    SRC["late_layer_tail_layout"], N.TARGETS["late_layer_tail_layout"][1])
check("edited sources have no pageable torch.tensor(..., device=...) copy left",
      "torch.tensor(tail_lens_cpu" not in edited_text and "torch.tensor([extend_lens_cpu" not in edited_text)
import inspect  # noqa: E402
h2d_src = inspect.getsource(N._h2d)
check("_h2d stages CUDA copies in pinned memory, non_blocking",
      "pin_memory=True" in h2d_src and "non_blocking=True" in h2d_src)
check("_h2d off-CUDA == torch.tensor", same(N._h2d([3, 1, 2], torch.int32, "cpu"),
                                             torch.tensor([3, 1, 2], dtype=torch.int32)))

# ---- 5. drift and anchors refuse -------------------------------------------------------------------------------
try:
    N.apply_edits("def f():\n    return 1\n", N.TARGETS[MNAME][1])
    check("missing anchor refuses", False)
except RuntimeError:
    check("missing anchor refuses", True)
try:
    N.apply_edits("x = 1\nx = 1\n", [("x = 1", "x = 2")])
    check("doubled anchor refuses", False)
except RuntimeError:
    check("doubled anchor refuses", True)


def drifted():
    return 1


try:
    N.build(drifted, "drifted", N.TARGETS[MNAME][0], [], {})
    check("drifted source refuses to build", False)
except RuntimeError as exc:
    check("drifted source refuses to build", "drifted" in str(exc))

# ---- 6. dispatcher: static and in-boot A/B ---------------------------------------------------------------------
seen = []
w = N._dispatch("t", lambda: seen.append("stock") or "s", lambda: seen.append("edited") or "e")
N._STATE["static"] = False
r_off = w()
N._STATE["static"] = True
r_on = w()
check("static gate: off -> stock, on -> edited", (r_off, r_on) == ("s", "e") and seen == ["stock", "edited"])
check("off-by-default install is a no-op", N.install_backend(types.SimpleNamespace(), environ={}) is False)

import ab_variant  # noqa: E402
env = {"DSV41_AB_VARIANTS": "2", "DSV41_AB_V0": "DSV41_EXTEND_NOSYNC=0", "DSV41_AB_V1": "DSV41_EXTEND_NOSYNC=1"}
try:
    armed = ab_variant.configure(env)
    check("ab_variant accepts DSV41_EXTEND_NOSYNC as a union key", armed and env.get("DSV41_EXTEND_NOSYNC") == "1")
    N._STATE["static"] = True          # the union sets the process gate; the variant decides per call
    ab_variant._runtime = 0
    v0 = N.on()
    ab_variant._runtime = 1
    v1 = N.on()
    check("in-boot A/B: variant 0 stock, variant 1 edited, read per call", (v0, v1) == (False, True))
    check("extend_nosync is in the A/B source hash", "extend_nosync" in ab_variant.SOURCES)
finally:
    ab_variant.configure({})
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, ab_variant._Finder)]
    N._STATE["static"] = False

# ---- 7. the audit ----------------------------------------------------------------------------------------------
FS = types.SimpleNamespace
stack = [FS(filename="/usr/lib/python3/site-packages/sglang/srt/managers/scheduler.py", lineno=10, name="run_batch"),
         FS(filename="/sgl-workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py", lineno=1700,
            name="_build_late_layer_tail_metadata"),
         FS(filename="/usr/lib/python3/site-packages/torch/_tensor.py", lineno=5, name="__torch_function__"),
         FS(filename="/opt/dsv41/adapter/extend_sync_audit.py", lineno=99, name="show")]
key = A.site_key(stack)
check("audit site key: innermost engine frame first, torch / audit frames skipped",
      key[0] == "sglang/srt/layers/attention/deepseek_v4_backend.py:1700 _build_late_layer_tail_metadata"
      and len(key) == 2, str(key))
A._STATE.update(extends=0, with_sync=0, inflight=0, sites={}, spans={}, new_site=False)
A.record([(key, ["s"])], 28, 0.160, True)
A.record([], 28, 0.120, False)
A.record([(key, ["s"])], 300, 0.200, True)
s = A.summary()
check("audit bookkeeping", s["extends"] == 3 and s["extends_with_sync"] == 2 and s["launched_behind_gpu_work"] == 2
      and s["sites"][0]["n"] == 2 and s["spans"]["<=32"]["n"] == 2 and s["spans"]["<=32"]["syncs"] == 1,
      str(s))


class FakeMode:
    def __init__(self, ext):
        self.ext = ext

    def is_extend(self):
        return self.ext

    def is_target_verify(self):
        return False


class FakeScheduler:
    def run_batch(self, batch):
        return ("ran", batch.tag)


mod = types.SimpleNamespace(Scheduler=FakeScheduler)
os.environ["DSV41_EXTEND_SYNC_AUDIT"] = "1"
A._STATE["installed"] = False
try:
    check("audit installs on the scheduler class", A.install(mod) and getattr(FakeScheduler.run_batch,
                                                                              "_dsv41_extend_sync_audit", False))
    sch = FakeScheduler()
    out_d = sch.run_batch(types.SimpleNamespace(forward_mode=FakeMode(False), tag="d"))
    out_e = sch.run_batch(types.SimpleNamespace(forward_mode=FakeMode(True), tag="e", extend_num_tokens=28))
    check("audit passes results through unchanged (decode, and extend on a CPU-only process)",
          out_d == ("ran", "d") and out_e == ("ran", "e"))
finally:
    os.environ.pop("DSV41_EXTEND_SYNC_AUDIT", None)

print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
