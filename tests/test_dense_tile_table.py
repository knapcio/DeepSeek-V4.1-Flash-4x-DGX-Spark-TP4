"""CPU checks for adapter/dense_tile_table.py (DSV41_DENSE_TILE_TABLE; torch on CPU, a fake FlashInfer b12x runner, a
fake SGLang linear / quant op) through adapter/mxfp8_b12x.py's dispatch:

1. a tabled (N, K, M): the first eager call runs the production path and the tabled tile on the same inputs; a
   bytewise-identical pair records True and the tile serves the key from then on (one runner call, the explicit tactic
   (tile, (1, 1), False, False, "sm120", None), the production path not called);
2. a mismatch (incl. +0.0 vs -0.0) records False and the production path serves the key for good;
3. a plan drift (the image's default plan for the shape is not the measured one) records False without running the tile;
4. inside a graph capture an unchecked key takes the production path and records nothing;
5. M buckets: M <= 16 -> (16, 64), 17..32 -> (32, 64), M > 32 and untabled shapes -> the production path, no check;
6. the TP agreement: a rank whose pair matched but whose peer's did not keeps the production path on every rank;
7. the gate off per call (an A/B variant without it) -> the production path, nothing recorded; the install receipt.
  CUDA_VISIBLE_DEVICES= PYTHONPATH=adapter python3 tests/test_dense_tile_table.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter"))
for k in [k for k in os.environ if k.startswith(("DSV41_DENSE_", "DSV41_QUANT_EARLY", "DSV41_AB", "DSV41_MXFP8"))]:
    del os.environ[k]
os.environ["DSV41_DENSE_TILE_TABLE"] = "1"

import torch  # noqa: E402

# fake FlashInfer: the b12x runner factory, the plan selector, the workspace
runner_calls = []
plan_tile = {"tile": (64, 64)}
poison = {"on": False}        # the tabled tile's result differs (a mismatch the self-check must catch)


class FakeRunner:
    def __init__(self, enable_pdl, out_dtype):
        self.enable_pdl, self.out_dtype = enable_pdl, out_dtype

    def __call__(self, inputs, **kwargs):
        q, b, x_sf, w_sf, out_dtype, out, ws = inputs
        runner_calls.append((tuple(b.shape), int(q.shape[0]), kwargs.get("tactic")))
        res = (q.float() @ b.float()).to(out_dtype)
        if poison["on"]:
            res = res + 1
        out.copy_(res)
        return out


gemm_base = types.ModuleType("flashinfer.gemm.gemm_base")
gemm_base.DEFAULT_WORKSPACE_SIZE = 16
gemm_base._get_cache_buf = lambda name, size, device: torch.empty(size, dtype=torch.uint8)
gemm_base._b12x_gemm_mxfp8_runner = lambda major, minor, enable_pdl, out_dtype: FakeRunner(enable_pdl, out_dtype)
kmod = types.ModuleType("flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x")
kmod._select_default_dense_gemm_plan = lambda m, n, k, sm, *, is_mxfp8=False, expected_m=None: types.SimpleNamespace(mma_tiler_mn=plan_tile["tile"])
futils = types.ModuleType("flashinfer.utils")
futils.get_device_sm_count = lambda device: 48
for name, mod in (("flashinfer", types.ModuleType("flashinfer")), ("flashinfer.gemm", types.ModuleType("flashinfer.gemm")),
                  ("flashinfer.gemm.gemm_base", gemm_base), ("flashinfer.gemm.kernels", types.ModuleType("flashinfer.gemm.kernels")),
                  ("flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x", kmod), ("flashinfer.utils", futils)):
    sys.modules[name] = mod
sys.modules["flashinfer.gemm"].gemm_base = gemm_base

import dense_tile_table as dt  # noqa: E402
import mxfp8_b12x as mb  # noqa: E402

orig_calls = []


def fake_original(input, weight, weight_scale, input_scale=None, bias=None, output_dtype=None, backend="cutlass", pin_tactic=False):
    orig_calls.append((backend, tuple(weight.shape), int(input.shape[0])))
    out = (input.float() @ weight.float().T).to(torch.bfloat16)
    if getattr(weight, "_negzero", False):
        out = out.clone()
        out.view(-1)[0] = -0.0 if out.view(-1)[0] == 0 else out.view(-1)[0]
    return out


def fake_quant(x, is_sf_swizzled_layout=True, alignment=32):
    return x, torch.zeros(1, dtype=torch.uint8)


def setup(capturing=False, agree=None, gate="1"):
    runner_calls.clear()
    orig_calls.clear()
    dt._CHECKED.clear()
    dt._RUNNERS.clear()
    for k in dt._STATS:
        dt._STATS[k] = 0
    plan_tile["tile"] = (64, 64)
    poison["on"] = False
    cuda = types.SimpleNamespace(is_current_stream_capturing=lambda: capturing, get_device_capability=lambda d: (12, 1))
    dt.torch = types.SimpleNamespace(equal=torch.equal, uint8=torch.uint8, empty=torch.empty, float16=torch.float16,
                                     bfloat16=torch.bfloat16, float32=torch.float32, cuda=cuda)
    mb.torch = types.SimpleNamespace(equal=torch.equal, uint8=torch.uint8, tensor=torch.tensor, int32=torch.int32,
                                     distributed=torch.distributed, cuda=cuda)
    mb._agree = (lambda flag, device: agree(flag)) if agree is not None else (lambda flag, device: bool(flag))
    dt._env = lambda name, default=None: gate if name == dt.ENV else os.environ.get(name, default)
    module = types.SimpleNamespace(flashinfer_mxfp8_blockscaled_linear=fake_original, flashinfer_mxfp8_quantize=fake_quant,
                                   block_fp8_scale_to_mxfp8_e8m0=lambda s, shape, bs: s)
    mb._REJECTED.clear()
    mb.install(module)
    return module.flashinfer_mxfp8_blockscaled_linear


def xw(m=6, n=512, k=5120):
    x = (torch.randn(m, k) * 0.1).to(torch.bfloat16)
    x.device.index
    w = (torch.randn(n, k) * 0.1).to(torch.bfloat16)
    return x, w


def ref(x, w):
    return (x.float() @ w.float().T).to(torch.bfloat16)


def test_identical_pair_serves_tile():
    linear = setup()
    x, w = xw()
    out = linear(x, w, torch.ones(1))
    assert torch.equal(out, ref(x, w))
    assert orig_calls == [("b12x", (512, 5120), 6)] and runner_calls == [((5120, 512), 6, ((16, 64), (1, 1), False, False, "sm120", None))], (orig_calls, runner_calls)
    assert list(dt._CHECKED.values()) == [True] and dt.stats()["same"] == 1
    linear(x, w, torch.ones(1))
    assert len(orig_calls) == 1 and len(runner_calls) == 2        # the tile serves the key, the production path not called
    assert dt._RUNNERS and list(dt._RUNNERS.values())[0].enable_pdl is True


def test_mismatch_keeps_production():
    linear = setup()
    x, w = xw()
    poison["on"] = True
    out = linear(x, w, torch.ones(1))
    assert torch.equal(out, ref(x, w))                             # the production result
    assert list(dt._CHECKED.values()) == [False] and dt.stats()["diff"] == 1
    linear(x, w, torch.ones(1))
    assert len(orig_calls) == 2 and len(runner_calls) == 1
    linear2 = setup()
    x, w = xw()
    x.zero_()                                                      # every output +0.0 on the tile
    w._negzero = True                                              # the production path writes -0.0: torch.equal says equal, bytes differ
    linear2(x, w, torch.ones(1))
    assert list(dt._CHECKED.values()) == [False] and dt.stats()["diff"] == 1


def test_plan_drift_keeps_production_without_running():
    linear = setup()
    plan_tile["tile"] = (16, 128)
    x, w = xw()
    out = linear(x, w, torch.ones(1))
    assert torch.equal(out, ref(x, w))
    assert list(dt._CHECKED.values()) == [False] and dt.stats()["drift"] == 1 and runner_calls == []


def test_capture_unchecked_is_production():
    linear = setup(capturing=True)
    x, w = xw()
    linear(x, w, torch.ones(1))
    assert dt._CHECKED == {} and dt.stats()["skipped_capture"] == 1 and runner_calls == [] and len(orig_calls) == 1


def test_buckets_and_untabled():
    linear = setup()
    x, w = xw(m=24, n=1152)
    linear(x, w, torch.ones(1))
    assert runner_calls[-1][2][0] == (32, 64)
    x, w = xw(m=16, n=1536)
    linear(x, w, torch.ones(1))
    assert runner_calls[-1][2][0] == (16, 64)
    n_before = len(runner_calls)
    x, w = xw(m=33, n=512)
    linear(x, w, torch.ones(1))
    x, w = xw(m=6, n=5120, k=640)
    linear(x, w, torch.ones(1))
    x, w = xw(m=6, n=4096, k=1280)
    linear(x, w, torch.ones(1))
    assert len(runner_calls) == n_before and len(dt._CHECKED) == 2
    assert dt.entry(512, 5120, 6) == ((16, 64), (64, 64)) and dt.entry(512, 5120, 32) == ((32, 64), (64, 64)) \
        and dt.entry(512, 5120, 48) is None and dt.entry(8192, 1280, 6) is None


def test_tp_disagreement():
    linear = setup(agree=lambda flag: False)
    x, w = xw()
    out = linear(x, w, torch.ones(1))
    assert torch.equal(out, ref(x, w)) and list(dt._CHECKED.values()) == [False] and dt.stats()["disagree"] == 1


def test_gate_off_per_call_and_receipt():
    linear = setup(gate="0")
    x, w = xw()
    linear(x, w, torch.ones(1))
    assert dt._CHECKED == {} and runner_calls == [] and orig_calls == [("b12x", (512, 5120), 6)]
    assert dt.wanted_at_install()
    assert set(dt.TABLE) == {(512, 5120), (1152, 5120), (1536, 5120)} and dt.MAX_M == 32
    for rows in dt.TABLE.values():
        assert [r[0] for r in rows] == [16, 32] and [r[1] for r in rows] == [(16, 64), (32, 64)] and all(r[2] == (64, 64) for r in rows)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"test_dense_tile_table: ALL {len(tests)} OK")
