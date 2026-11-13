"""CPU checks for the MXFP4 draft head (adapter/draft_head_fp8.py, DSV41_DRAFT_HEAD_FP4).

1. the quantizer is OCP MXFP4 (e = floor(log2 amax) - 2, nearest e2m1 with ties to the even code, saturation at 6),
   checked against an independent table-driven reference, incl. ties, saturation, zeros, signs, tiny runs;
2. the packed layout (even k in the low nibble) and the e8m0 scales round-trip through dequant_fp4;
3. the kernels' integer decodes (_e2m1_x2 of the tl.dot kernel; the GEMV's fp32-bit decode x 2^(e-1), both mirrored here)
   give the e2m1 value of every code, the GEMV's exactly for every byte and e8m0 byte (subnormals included);
4. dispatch: gate off -> fp8 twin; on and rows <= MAX_M -> MXFP4; above -> fp8; above the fp8 cap -> stock;
   a drifted draft class is refused;
5. the release reads every key from the process env (no in-boot A/B registration) and the default launch is the
   bench's GB10 winner 32,512,8,1.
The Triton kernel itself is timed and checked on a GB10 by scripts/bench_draft_head_fp4.py.
"""
import importlib
import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "adapter"))

import torch  # noqa: E402

if importlib.util.find_spec("triton") is None:          # CPU hosts without Triton: enough for the module import
    tri = types.ModuleType("triton")
    tri.jit = lambda f=None, **kw: f if f is not None else (lambda g: g)
    tri.cdiv = lambda a, b: -(-a // b)
    tl = types.ModuleType("triton.language")
    tl.constexpr = int
    tri.language = tl
    sys.modules["triton"], sys.modules["triton.language"] = tri, tl

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FAILS = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name, flush=True)
    if not ok:
        FAILS.append(name)


def fresh(env):
    for k in [k for k in os.environ if k.startswith(("DSV41_DRAFT_HEAD", "DSV41_AB"))]:
        del os.environ[k]
    os.environ.update(env)
    for m in ("draft_head_fp8", "ab_variant"):
        sys.modules.pop(m, None)
    sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "_Finder"]
    return importlib.import_module("draft_head_fp8")


def ref_quant(w):
    """Independent reference: per 1 x 32 run, scale 2^(floor(log2 amax) - 2); each value -> the e2m1 grid point with the
    smallest |error|, a tie going to the even code; |v| > 6 saturates. Returns the dequantized fp32 tensor."""
    n, k = w.shape
    out = torch.empty(n, k)
    grid = torch.tensor(E2M1, dtype=torch.float64)
    wd = w.double()
    for i in range(n):
        for b in range(k // 32):
            run = wd[i, b * 32:(b + 1) * 32]
            amax = max(float(run.abs().max()), 2.0 ** -126)
            e = max(-127, min(127, int(torch.floor(torch.log2(torch.tensor(amax, dtype=torch.float64))).item()) - 2))
            s = 2.0 ** e
            for j in range(32):
                v = abs(float(run[j])) / s
                err = (grid - v).abs()
                best = int(torch.argmin(err))
                ties = [c for c in range(8) if abs(float(err[c]) - float(err[best])) == 0.0]
                code = next((c for c in ties if c % 2 == 0), ties[0]) if len(ties) > 1 else best
                out[i, b * 32 + j] = (-1.0 if float(run[j]) < 0 else 1.0) * E2M1[code] * s
    return out


def decode_x2(c):
    """Mirror of the kernel's _e2m1_x2 on torch int tensors."""
    m = c & 7
    sh = torch.clamp((m >> 1) - 1, min=0)
    v = torch.where(m < 2, m, (2 + (m & 1)) << sh)
    return torch.where(c >= 8, -v, v)


def test_quantizer():
    dh = fresh({"DSV41_DRAFT_HEAD_FP8": "1", "DSV41_DRAFT_HEAD_FP4": "1"})
    g = torch.Generator().manual_seed(7)
    w = (torch.randn(24, 128, generator=g) * 0.02)
    # edge runs: exact ties at every midpoint, a saturating run, an all-zero run, a tiny run, mixed signs
    w[0, :32] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 4.0] * 4) * 2.0 ** -6
    w[1, :32] = torch.tensor([7.9, -7.9, 6.5, 0.1] * 8) * 2.0 ** -4
    w[2, 32:64] = 0.0
    w[3, 64:96] = torch.randn(32, generator=g) * 1e-30
    w = w.to(torch.bfloat16)
    twin = dh.make_fp4_twin(w, rows=5)                     # odd chunking on purpose
    back = dh.dequant_fp4(twin)
    ref = ref_quant(w.float())
    check("quantizer: MXFP4 (floor scale, RNE, saturating) equals the independent reference",
          torch.equal(back.float(), ref.to(torch.bfloat16).float()))
    check("quantizer: the all-zero run stays zero", bool((back[2, 32:64] == 0).all()))
    p, s = twin
    check("layout: packed uint8 [N, K/2], e8m0 uint8 [N, K/32]",
          p.dtype == torch.uint8 and tuple(p.shape) == (24, 64) and s.dtype == torch.uint8 and tuple(s.shape) == (24, 4))
    # even k in the low nibble
    code = torch.stack((p & 15, p >> 4), -1).view(24, -1).long()
    mag = torch.tensor(E2M1)[code & 7] * torch.where(code >= 8, -1.0, 1.0)
    check("layout: low nibble = even k, high nibble = odd k",
          torch.equal((mag * torch.exp2(s.float() - 127).repeat_interleave(32, 1)).to(torch.bfloat16), back))
    rel = float((back.float() - w.float()).norm() / w.float().norm())
    check(f"quantizer: relative error of a random head in the MXFP4 band (rel {rel:.3f})", 0.05 < rel < 0.2)
    # the shipped size at the real shard shape
    n, k = 32320, 5120
    check("size: 82.7 + 5.2 MB per rank at 32320 x 5120", (n * k // 2 + n * k // 32) / 1e6 < 88.0)


def test_decode():
    c = torch.arange(16)
    want = torch.tensor([2 * v for v in E2M1] + [-2 * v for v in E2M1]).long()
    check("kernel decode: _e2m1_x2 gives 2 x value for all 16 codes", torch.equal(decode_x2(c), want))
    # the GEMV (_fp4_tile_rows): (b << 28) >> 6 & 0x81C00000 (low nibble), (b << 24) >> 6 & ... (high), as fp32 bits,
    # x 2^(min(e, 128) - 1), then x 2^(max(e, 128) - 128): the exact value for every byte and every e8m0 byte
    b = torch.arange(256, dtype=torch.int32)

    def bits(sh):
        v = ((b << sh) >> 6) & -2118123520          # int32 arithmetic shift, as in the kernel
        return v.view(torch.float32).double()
    e = torch.arange(255, dtype=torch.float64)[:, None]
    lo = bits(28)[None, :] * torch.exp2(torch.clamp(e, max=128) - 1) * torch.exp2(torch.clamp(e, min=128) - 128)
    hi = bits(24)[None, :] * torch.exp2(torch.clamp(e, max=128) - 1) * torch.exp2(torch.clamp(e, min=128) - 128)
    mag = torch.tensor(E2M1, dtype=torch.float64)
    val = lambda c: torch.where(c >= 8, -mag[c & 7], mag[c & 7])       # noqa: E731
    want_lo = val((b & 15).long())[None, :] * torch.exp2(e - 127)
    want_hi = val((b >> 4).long())[None, :] * torch.exp2(e - 127)
    check("kernel decode: the GEMV's fp32-bit decode is exact for all 256 bytes x 255 e8m0 bytes",
          torch.equal(lo, want_lo) and torch.equal(hi, want_hi))


class Norm:
    def __call__(self, x):
        return x


def make_module(n=64, k=64):
    calls = []

    class Head:
        weight = None

    class DSpark:
        _opt_markov_w2_tp_shard = True

        def attach_shared_modules(self, lm_head):
            self.lm_head = lm_head

        def _logits_from_x_post_hc(self, x):
            calls.append(("stock", x.shape[0]))
            return torch.zeros(x.shape[0], n)

    mod = types.SimpleNamespace(DeepseekV4ForCausalLMDSpark=DSpark, gather_and_crop_vocab=lambda local, head: local)
    head = Head()
    head.weight = (torch.randn(n, k) * 0.02).to(torch.bfloat16)
    return mod, head, calls


def run_dispatch(dh, rows_list):
    mod, head, calls = make_module()
    seen = []
    dh.head_fp8 = lambda x, twin: (seen.append(("fp8", x.shape[0])), torch.zeros(x.shape[0], 64))[1]
    dh.head_fp4 = lambda x, twin, tile=None: (seen.append(("fp4", x.shape[0])), torch.zeros(x.shape[0], 64))[1]
    dh.install(mod)
    torch.cuda.empty_cache = lambda: None
    m = mod.DeepseekV4ForCausalLMDSpark()
    m.stages = [types.SimpleNamespace(norm=Norm())]
    m.attach_shared_modules(head)
    for r in rows_list:
        m._logits_from_x_post_hc(torch.zeros(r, 64, dtype=torch.bfloat16))
    return seen + calls, head


def test_dispatch():
    dh = fresh({"DSV41_DRAFT_HEAD_FP8": "1"})
    got, _ = run_dispatch(dh, [5, 10])
    check("dispatch: gate off -> the fp8 twin, no MXFP4 copy built", got == [("fp8", 5), ("fp8", 10)] and not dh._TWIN4)
    dh = fresh({"DSV41_DRAFT_HEAD_FP8": "1", "DSV41_DRAFT_HEAD_FP4": "1"})
    got, head = run_dispatch(dh, [5, 6, 10, 80])
    check("dispatch: on -> MXFP4 at <= 5 rows, fp8 above, stock above the fp8 cap (64)",
          got == [("fp4", 5), ("fp8", 6), ("fp8", 10), ("stock", 80)] and head.weight.data_ptr() in dh._TWIN4)
    dh = fresh({"DSV41_DRAFT_HEAD_FP8": "1", "DSV41_DRAFT_HEAD_FP4": "1", "DSV41_DRAFT_HEAD_FP4_MAX_M": "10"})
    got, _ = run_dispatch(dh, [5, 10, 15])
    check("dispatch: MAX_M=10 -> MXFP4 up to 10 rows", got == [("fp4", 5), ("fp4", 10), ("fp8", 15)])
    for bad in ("0", "129", "x"):
        dh = fresh({"DSV41_DRAFT_HEAD_FP8": "1", "DSV41_DRAFT_HEAD_FP4": "1", "DSV41_DRAFT_HEAD_FP4_MAX_M": bad})
        try:
            dh.fp4_rows()
            check(f"dispatch: MAX_M={bad} refused", False)
        except RuntimeError:
            check(f"dispatch: MAX_M={bad} refused", True)
    dh = fresh({"DSV41_DRAFT_HEAD_FP8": "1", "DSV41_DRAFT_HEAD_FP4": "1"})
    check("tile: default 32,512,8,1; override parsed; junk refused",
          dh.parse_tile("") == (32, 512, 8, 1) and dh.parse_tile("64,128,8,4") == (64, 128, 8, 4))
    try:
        dh.parse_tile("33,128,4,3")
        check("tile: BN 33 refused", False)
    except RuntimeError:
        check("tile: BN 33 refused", True)
    try:
        dh.install(types.SimpleNamespace(DeepseekV4ForCausalLMDSpark=type("X", (), {})))
        check("install: drifted class refused", False)
    except RuntimeError:
        check("install: drifted class refused", True)


def test_process_env_only():
    dh = fresh({"DSV41_DRAFT_HEAD_FP8": "1"})
    import ab_variant as ab
    check("release: no in-boot A/B registration of the MXFP4 keys (process env only)",
          not {"DSV41_DRAFT_HEAD_FP4", "DSV41_DRAFT_HEAD_FP4_MAX_M", "DSV41_DRAFT_HEAD_FP4_TILE"} & set(ab.KNOWN)
          and not hasattr(dh, "_ab"))
    check("release: gate off -> no copy, row cap 0", not dh.fp4_installed() and dh.fp4_rows() == 0)
    dh = fresh({"DSV41_DRAFT_HEAD_FP8": "1", "DSV41_DRAFT_HEAD_FP4": "1"})
    check("release: DSV41_DRAFT_HEAD_FP4=1 alone -> copy built, row cap 5, launch 32,512,8,1",
          dh.fp4_installed() and dh.fp4_rows() == 5 and dh.parse_tile(dh.env("DSV41_DRAFT_HEAD_FP4_TILE", ""))
          == (32, 512, 8, 1))
    os.environ["DSV41_DRAFT_HEAD_FP4_MAX_M"] = "8"
    check("release: the row cap is read per call", dh.fp4_rows() == 8)
    os.environ["DSV41_DRAFT_HEAD_FP4"] = "0"
    check("release: the gate is read per call", dh.fp4_rows() == 0)
    fresh({})


if __name__ == "__main__":
    test_decode()
    test_quantizer()
    test_dispatch()
    test_process_env_only()
    if FAILS:
        raise SystemExit(f"test_draft_head_fp4: {len(FAILS)} FAILED: {FAILS}")
    print("test_draft_head_fp4: ALL PASS")
