"""DSV41_PREFILL_GEMM_ROW_SPLIT, CPU: block plan, the 128x4 scale slicing, the split GEMM against the unsplit one with
a fake mm_mxfp8 that really reads the scales it is handed, the bit-identity self-check, and the gates.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=adapter python3 tests/test_prefill_gemm_split.py   (or pytest)
"""
import os
import re
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter"))
import prefill_gemm_split as pgs  # noqa: E402

pgs.AGREE[0] = False
ADAPTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter")


def _swizzle_128x4_from_prefill_sp():
    """adapter/prefill_sp.swizzle_128x4, lifted from the source (prefill_sp imports the engine at module level)."""
    src = open(os.path.join(ADAPTER, "prefill_sp.py")).read()
    m = re.search(r"\ndef swizzle_128x4\(.*?(?=\n\n\ndef |\n\ndef )", src, re.S)
    assert m, "swizzle_128x4 not found in prefill_sp.py"
    ns = {"torch": torch}
    exec(m.group(0), ns)
    return ns["swizzle_128x4"]


swizzle = _swizzle_128x4_from_prefill_sp()


def unswizzle(flat, rows, g):
    """Inverse of swizzle_128x4 for `rows` rows (padded to 128) and g scale columns."""
    mt = (rows + 127) // 128
    return flat.view(mt, g // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(mt * 128, g)[:rows]


def test_parse_and_plan():
    assert pgs.parse("0") == (0, 0, 0, None)
    assert pgs.parse("off") == (0, 0, 0, None)
    assert pgs.parse("4096") == (4096, 5120, 8192, {1792, 1152})      # default: the two measured shapes only
    assert pgs.parse("4096", "5120", "", "all") == (4096, 5120, 8192, None)
    assert pgs.parse("4096", "4096", "12288", "1792") == (4096, 4096, 12288, {1792})
    for bad in ("100", "4096:x"):
        try:
            pgs.parse(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    assert pgs.plan_blocks(8192, 4096) == [(0, 4096), (4096, 8192)]
    assert pgs.plan_blocks(10000, 4096) == [(0, 4096), (4096, 8192), (8192, 10000)]
    cfg = pgs.parse("4096")
    assert pgs.eligible(8192, 1792, 5120, cfg) and pgs.eligible(8192, 1152, 5120, cfg)
    assert not pgs.eligible(8192, 5120, 2048, cfg)          # wo_b K=2048: stays one launch
    assert not pgs.eligible(96, 1792, 5120, cfg)            # decode / verify rows
    assert not pgs.eligible(8191, 1792, 5120, cfg)
    assert not pgs.eligible(8192 + 256, 1792, 5120, cfg)    # bypass-admitted 8192 + x: stock, no re-check
    assert pgs.eligible(12288, 1792, 5120, cfg)
    assert not pgs.eligible(8192, 64, 5120, cfg)            # not a measured shape under the default filter
    assert pgs.eligible(8192, 64, 5120, pgs.parse("4096", "5120", "", "all"))
    assert not pgs.eligible(8192, 1792, 5120, pgs.parse("4096", "5120", "", "1152"))


def test_scale_rows_matches_swizzle_of_the_rows():
    torch.manual_seed(0)
    for m, k in ((8192, 5120), (10000, 5120), (1000, 256), (8192, 4096 + 32 * 4)):
        g = (k // 32 + 3) // 4 * 4
        lin = torch.randint(0, 255, (m, g), dtype=torch.uint8)
        full = swizzle(lin)
        for rows in (128, 4096):
            for r0, r1 in pgs.plan_blocks(m, rows):
                got = pgs.scale_rows(full, m, k, r0, r1)
                assert got is not None
                assert torch.equal(got, swizzle(lin[r0:r1])), (m, k, r0, r1)
                got2 = pgs.scale_rows(full.view((m + 127) // 128 * 128, g), m, k, r0, r1)
                assert got2.shape == ((r1 - r0 + 127) // 128 * 128, g) and torch.equal(got2.reshape(-1), got)
    assert pgs.scale_rows(full, m, k, 64, 128) is None                  # not on a tile boundary
    assert pgs.scale_rows(torch.zeros(7, dtype=torch.uint8), 128, 256, 0, 128) is None


def fake_mm(a, b, a_descale, b_descale, out=None, out_dtype=torch.bfloat16, use_8x4_sf_layout=False, backend="b12x",
            perturb_m=None):
    """out = (a * 2^(scale_a - 127) per 32 columns) @ b in float64 (exact for these small ints), rows of the block."""
    m, k = a.shape
    g = (k // 32 + 3) // 4 * 4
    sa = unswizzle(a_descale.reshape(-1), m, g)[:, : k // 32].to(torch.float64) - 127.0
    av = a.to(torch.float64) * torch.pow(2.0, sa).repeat_interleave(32, dim=1)
    res = av @ b.to(torch.float64)
    if perturb_m is not None and m == perturb_m:
        res = res + 1.0                                               # a tactic that changes the numbers at this M
    res = res.to(out_dtype)
    if out is None:
        return res
    out.copy_(res)
    return out


def make_fp8_utils(perturb_m=None, calls=None):
    def raw(*a, **kw):
        if calls is not None:
            calls.append(a[0].shape[0])
        return fake_mm(*a, perturb_m=perturb_m, **kw)

    def quantize(x, is_sf_swizzled_layout=True, alignment=32):
        m, k = x.shape
        g = (k // 32 + 3) // 4 * 4
        lin = torch.full((m, g), 127, dtype=torch.uint8)
        lin[:, : k // 32] = (127 + (torch.arange(m)[:, None] % 3) + torch.arange(k // 32)[None, :] % 2).to(torch.uint8)
        return x.clone(), swizzle(lin)

    def original(input, weight, weight_scale, input_scale=None, bias=None, output_dtype=None, backend="cutlass",
                 pin_tactic=False):
        x2 = input.reshape(-1, input.shape[-1])
        if input_scale is None:
            q, xs = quantize(x2)
        else:
            q, xs = x2, input_scale
        ws_t = weight_scale.t() if weight_scale.ndim == 2 else weight_scale
        out = raw(q, weight.t(), xs, ws_t, out_dtype=output_dtype or torch.bfloat16, backend=backend)
        return out.view(*input.shape[:-1], weight.shape[0])

    mod = types.SimpleNamespace(_raw_flashinfer_mm_mxfp8=raw, flashinfer_mxfp8_quantize=quantize)
    return mod, original


def _inputs(m=8192, k=5120, n=64):
    torch.manual_seed(1)
    x = torch.randint(-3, 4, (m, k)).to(torch.float32)
    w = torch.randint(-2, 3, (n, k)).to(torch.float32)
    ws = torch.zeros(n, k // 32)
    return x, w, ws


def test_split_equals_unsplit_and_is_used_after_the_check():
    os.environ["DSV41_PREFILL_GEMM_ROW_SPLIT"] = "4096"
    os.environ["DSV41_PREFILL_GEMM_ROW_SPLIT_N"] = "all"      # the fake GEMMs use N=64
    try:
        pgs._DECIDED.clear()
        calls = []
        mod, original = make_fp8_utils(calls=calls)
        split = pgs.make_linear_split(mod)
        x, w, ws = _inputs()
        ref = original(x, w, ws, None, None, torch.float32)
        calls.clear()
        first = split(original, x, w, ws, None, None, torch.float32, "b12x", True)      # the check: stock result
        assert torch.equal(first, ref) and calls == [4096, 4096, 8192], calls
        assert pgs._DECIDED and all(pgs._DECIDED.values())
        calls.clear()
        second = split(original, x, w, ws, None, None, torch.float32, "b12x", True)
        assert torch.equal(second, ref) and calls == [4096, 4096], calls
        # pre-quantized input (DSV41_PREFILL_SP_FP8's Mxfp8SwizzledInput): data + swizzled scales sliced per block
        q, xs = mod.flashinfer_mxfp8_quantize(x)
        ref_q = original(q, w, ws, xs, None, torch.float32)
        assert torch.equal(split(original, q, w, ws, xs, None, torch.float32, "b12x", True), ref_q)
        calls.clear()
        assert torch.equal(split(original, q, w, ws, xs, None, torch.float32, "b12x", True), ref_q)
        assert calls == [4096, 4096], calls
        # three blocks (M = 12288); M = 10000 (not a multiple of the block) keeps the stock call, no check
        x3, w3, ws3 = _inputs(m=12288)
        ref3 = original(x3, w3, ws3, None, None, torch.float32)
        split(original, x3, w3, ws3, None, None, torch.float32, "b12x", True)
        calls.clear()
        assert torch.equal(split(original, x3, w3, ws3, None, None, torch.float32, "b12x", True), ref3)
        assert calls == [4096, 4096, 4096], calls
        x4, w4, ws4 = _inputs(m=10000)
        n_keys = len(pgs._DECIDED)
        assert split(original, x4, w4, ws4, None, None, torch.float32, "b12x", True) is None
        assert len(pgs._DECIDED) == n_keys
    finally:
        os.environ.pop("DSV41_PREFILL_GEMM_ROW_SPLIT")
        os.environ.pop("DSV41_PREFILL_GEMM_ROW_SPLIT_N", None)


def test_check_refuses_a_shape_whose_numbers_change_with_m():
    os.environ["DSV41_PREFILL_GEMM_ROW_SPLIT"] = "4096"
    os.environ["DSV41_PREFILL_GEMM_ROW_SPLIT_N"] = "all"      # the fake GEMMs use N=64
    try:
        pgs._DECIDED.clear()
        mod, original = make_fp8_utils(perturb_m=4096)
        split = pgs.make_linear_split(mod)
        x, w, ws = _inputs()
        ref = original(x, w, ws, None, None, torch.float32)
        out = split(original, x, w, ws, None, None, torch.float32, "b12x", True)
        assert torch.equal(out, ref)                       # the checked call returns the stock result
        assert list(pgs._DECIDED.values()) == [False]
        assert split(original, x, w, ws, None, None, torch.float32, "b12x", True) is None   # stock from now on
    finally:
        os.environ.pop("DSV41_PREFILL_GEMM_ROW_SPLIT")
        os.environ.pop("DSV41_PREFILL_GEMM_ROW_SPLIT_N", None)


def test_off_and_ineligible_calls_return_none():
    pgs._DECIDED.clear()
    mod, original = make_fp8_utils()
    split = pgs.make_linear_split(mod)
    x, w, ws = _inputs(m=8192, k=5120)
    assert split(original, x, w, ws, None, None, torch.float32, "b12x", True) is None       # gate off
    os.environ["DSV41_PREFILL_GEMM_ROW_SPLIT"] = "4096"
    os.environ["DSV41_PREFILL_GEMM_ROW_SPLIT_N"] = "all"
    split = pgs.make_linear_split(mod)                       # config is read once per install
    try:
        xs, ws_, wss = _inputs(m=96)
        assert split(original, xs, ws_, wss, None, None, torch.float32, "b12x", True) is None  # decode rows
        xk, wk, wsk = _inputs(m=8192, k=2048)
        assert split(original, xk, wk, wsk, None, None, torch.float32, "b12x", True) is None   # K below MIN_K
        assert split(original, x, w, ws, None, None, torch.float32, "trtllm", True) is None    # other layouts
        assert split(original, x, w, ws, None, torch.zeros(64), torch.float32, "b12x", True) is None  # bias
        assert not pgs._DECIDED
    finally:
        os.environ.pop("DSV41_PREFILL_GEMM_ROW_SPLIT")
        os.environ.pop("DSV41_PREFILL_GEMM_ROW_SPLIT_N", None)


def test_config_read_once_without_ab():
    os.environ["DSV41_PREFILL_GEMM_ROW_SPLIT"] = "4096"
    try:
        pgs._CONFIG[0] = None
        first = pgs.config()
        os.environ["DSV41_PREFILL_GEMM_ROW_SPLIT"] = "0"
        assert pgs.config() is first and first[0] == 4096
    finally:
        os.environ.pop("DSV41_PREFILL_GEMM_ROW_SPLIT")
        pgs._CONFIG[0] = None


def test_wanted_at_install():
    os.environ.pop("DSV41_PREFILL_GEMM_ROW_SPLIT", None)
    assert not pgs.wanted_at_install()
    os.environ["DSV41_PREFILL_GEMM_ROW_SPLIT"] = "4096"
    assert pgs.wanted_at_install()
    os.environ.pop("DSV41_PREFILL_GEMM_ROW_SPLIT")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[OK] {name}", flush=True)
    print("all passed")
