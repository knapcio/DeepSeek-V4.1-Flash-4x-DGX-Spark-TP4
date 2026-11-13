"""The MXFP4 / fp8 draft-head Triton kernels in Triton's CPU interpreter (TRITON_INTERPRET=1, no GPU): the M <= 8 GEMV
over the launch shapes the bench sweeps (BN, BK = 32 x run lanes, warps, prefetch on / off; a row tail), the tl.dot
kernel at 9..16 rows and the two larger-row launches give the torch reference of their own weights (fp32 matmul of the
dequantized copy) within fp32 reordering, and the rows' argmax; the GEMV also on every code byte with extreme e8m0 bytes
(0, 1, 128, 129, 200: the subnormal and the > 128 scale paths) against a float64 reference. Run where Triton is installed:
  TRITON_INTERPRET=1 PYTHONPATH=adapter python3 tests/test_draft_head_fp4_interp.py
"""
import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ["DSV41_DRAFT_HEAD_FP8"] = "1"
os.environ["DSV41_DRAFT_HEAD_FP4"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "adapter"))
import torch  # noqa: E402

import draft_head_fp8 as dh  # noqa: E402


def _patch_interpreter_bf16_dot():
    """Triton's interpreter stores bf16 as uint16 and its create_dot multiplies those raw bits (3.8.0: only fp8 inputs are
    converted). Test-only fix: widen bf16 operands to fp32 (exact: bf16 is the top half of an fp32) before the matmul."""
    import numpy as np
    import triton.language as tl
    from triton.runtime import interpreter as itp
    orig = itp.InterpreterBuilder.create_dot

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        def widen(t):
            if t.dtype == tl.bfloat16:
                return itp.TensorHandle((t.data.astype(np.uint32) << 16).view(np.float32), tl.float32)
            return t
        return orig(self, widen(a), widen(b), d, input_precision, max_num_imprecise_acc)

    itp.InterpreterBuilder.create_dot = create_dot


if os.environ.get("TRITON_INTERPRET") == "1":
    _patch_interpreter_bf16_dot()


def main():
    torch.manual_seed(3)
    n, k = 100, 512                              # n not a multiple of BN: the tail mask is exercised
    w = (torch.randn(n, k) * 0.02).to(torch.bfloat16)
    w[3, :32] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -6.0] * 4) * 2.0 ** -6
    t4 = dh.make_fp4_twin(w)
    t8 = dh.make_twin(w)
    r4w = dh.dequant_fp4(t4).float()
    fails = 0
    gemv = ((32, 512, 8, 1), (32, 256, 4, 3), (32, 256, 4, 1), (16, 128, 2, 2), (64, 256, 8, 1), (16, 512, 4, 3), (128, 128, 4, 2),
            (32, 64, 1, 3))
    cases = [(m, tile) for m in (1, 5, 8) for tile in gemv] + [(5, None), (12, None), (16, None), (40, None), (80, None)]
    for m, tile in cases:
        x = torch.randn(m, k).to(torch.bfloat16)
        ref = x.float() @ r4w.t()
        y = dh.head_fp4(x, t4, tile=tile)
        err = float((y - ref).abs().max() / ref.abs().max())
        am = bool(torch.equal(y.argmax(-1), ref.argmax(-1)))
        ok = err < 1e-5 and am
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} mxfp4 M={m} tile={tile}: rel err {err:.1e} argmax {am}", flush=True)
    # every code byte, extreme e8m0 bytes: the exact-subnormal decode and the > 128 split (float64 reference)
    codes = torch.randint(0, 256, (16, 128), dtype=torch.uint8)
    codes[1, :] = torch.arange(128, dtype=torch.uint8)
    codes[2, :] = torch.arange(128, 256, dtype=torch.uint8)
    sc = torch.randint(100, 140, (16, 8), dtype=torch.uint8)
    sc[0] = torch.tensor([0, 1, 128, 129, 200, 127, 60, 2], dtype=torch.uint8)
    c = torch.stack((codes & 15, codes >> 4), -1).view(16, -1).long()
    mag = torch.tensor(dh.E2M1, dtype=torch.float64)[c & 7]
    val = torch.where(c >= 8, -mag, mag) * torch.exp2(sc.double() - 127).repeat_interleave(32, 1)
    for tile in ((16, 256, 4, 3), (32, 128, 2, 1)):
        x = torch.randn(5, 256).to(torch.bfloat16)
        ref = x.double() @ val.t()
        y = dh.head_fp4(x, (codes, sc), tile=tile).double()
        # per logit against the fp32 rounding scale of its own sum: |y - ref| / sum_k |x_k w_k| (fp32 eps = 1.2e-7)
        err = float(((y - ref).abs() / (x.double().abs() @ val.abs().t())).max())
        ok = err < 1e-5 and bool(torch.isfinite(y).all())
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} mxfp4 GEMV extreme scales tile={tile}: max |err| / sum|x w| {err:.1e}", flush=True)
    # the default launch 32,512,8,1 over several K tiles (production: K = 5120 = 10 tiles of 512; above, k = 512 is one tile)
    n2, k2 = 72, 2048
    w2 = (torch.randn(n2, k2) * 0.02).to(torch.bfloat16)
    t42 = dh.make_fp4_twin(w2)
    r42 = dh.dequant_fp4(t42).float()
    for m in (1, 5, 8):
        x = torch.randn(m, k2).to(torch.bfloat16)
        ref = x.float() @ r42.t()
        y = dh.head_fp4(x, t42)                    # tile None = the default launch
        err = float((y - ref).abs().max() / ref.abs().max())
        am = bool(torch.equal(y.argmax(-1), ref.argmax(-1)))
        ok = err < 1e-5 and am
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} mxfp4 default launch {dh.FP4_DEFAULT_TILE} M={m} K={k2} (4 K tiles): rel err {err:.1e} "
              f"argmax {am}", flush=True)
    x = torch.randn(5, k).to(torch.bfloat16)
    r8w = (t8[0].float().view(n, k // 32, 32) * torch.exp2(t8[1].float() - 127).repeat_interleave(32, 0)[:n].unsqueeze(-1)).view(n, k)
    y8 = dh.head_fp8(x, t8)
    ref8 = x.float() @ r8w.to(torch.bfloat16).float().t()
    err8 = float((y8 - ref8).abs().max() / ref8.abs().max())
    fails += not err8 < 1e-5
    print(f"{'PASS' if err8 < 1e-5 else 'FAIL'} fp8 twin M=5 (control): rel err {err8:.1e}", flush=True)
    if fails:
        raise SystemExit(f"test_draft_head_fp4_interp: {fails} FAILED")
    print("test_draft_head_fp4_interp: ALL PASS")


if __name__ == "__main__":
    main()
