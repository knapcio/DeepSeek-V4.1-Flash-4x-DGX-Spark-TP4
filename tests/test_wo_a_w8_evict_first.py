"""adapter/wo_a_w8.py DSV41_WO_A_W8_EVICT_FIRST: the twin loads' eviction policy is a cache hint only.

1. (CPU, TRITON_INTERPRET=1) both kernels produce bit-identical partials / outputs with EV="" and
   EV="evict_first" on the same inputs (the interpreter ignores the policy; this proves the kernel source
   still runs and the policy threads through the launchers, including the mid path's split-K configs);
2. (no GPU needed, needs the CUDA Triton backend + ptxas, i.e. the image) the PTX of the EV="evict_first"
   variant carries `createpolicy.fractional.L2::evict_first` and every W8 / S load is an
   `ld.global.L1::evict_first.L2::cache_hint`, while the activation (X) loads and the EV="" variant carry
   no hint. Skipped where triton.compile cannot target sm_121 (the Mac).
3. the per-call switch reads DSV41_WO_A_W8_EVICT_FIRST (through ab_variant.env when armed).
  CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 PYTHONPATH=adapter python3 tests/test_wo_a_w8_evict_first.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter"))
os.environ["DSV41_WO_A_W8"] = "1"
os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.pop("DSV41_WO_A_W8_EVICT_FIRST", None)

import torch  # noqa: E402

import wo_a_w8  # noqa: E402


def checkpoint_like(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(2, 1024, 4096, generator=g) * 60).clamp(-448, 448).to(torch.float8_e4m3fn)
    e = torch.randint(-14, -6, (2, 32, 128), generator=g).float()
    return (q.float().view(2, 32, 32, 128, 32) * torch.exp2(e)[:, :, None, :, None]).view(2, 1024, 4096).to(torch.bfloat16)


def test_switch():
    os.environ.pop("DSV41_WO_A_W8_EVICT_FIRST", None)
    assert wo_a_w8._ev() == ""
    for v in ("1", "on", "true"):
        os.environ["DSV41_WO_A_W8_EVICT_FIRST"] = v
        assert wo_a_w8._ev() == "evict_first"
    for v in ("0", "", "off"):
        os.environ["DSV41_WO_A_W8_EVICT_FIRST"] = v
        assert wo_a_w8._ev() == ""
    os.environ.pop("DSV41_WO_A_W8_EVICT_FIRST", None)


def test_interpreter_bitwise():
    if os.environ.get("TRITON_INTERPRET") != "1":
        print("SKIP interpreter (TRITON_INTERPRET != 1)")
        return
    twin = wo_a_w8.make_twin(checkpoint_like())
    assert twin is not None
    g = torch.Generator().manual_seed(1)
    outs = {}
    for ev in ("0", "1"):
        os.environ["DSV41_WO_A_W8_EVICT_FIRST"] = ev
        x = (torch.randn(6, 2, 4096, generator=g.manual_seed(1)) * 0.5).to(torch.bfloat16)
        outs[ev] = (wo_a_w8._partial_w8(x, twin, 6).clone(),
                    wo_a_w8.wo_a_mid((torch.randn(12, 2, 4096, generator=g.manual_seed(2)) * 0.5).to(torch.bfloat16),
                                     twin).clone(),
                    wo_a_w8.wo_a_mid((torch.randn(40, 2, 4096, generator=g.manual_seed(3)) * 0.5).to(torch.bfloat16),
                                     twin).clone())
    os.environ.pop("DSV41_WO_A_W8_EVICT_FIRST", None)
    for a, b in zip(outs["0"], outs["1"]):
        assert torch.equal(a, b)
    assert outs["0"][0].abs().sum() > 0


def _ptx(kernel, signature, constexprs):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    src = ASTSource(fn=kernel, signature=signature, constexprs=constexprs)
    return triton.compile(src, target=GPUTarget("cuda", 121, 32)).asm["ptx"]


def test_ptx_hint_placement():
    if os.environ.get("TRITON_INTERPRET") == "1":
        print("SKIP ptx (TRITON_INTERPRET=1; run with TRITON_INTERPRET=0 in the image)")
        return
    try:
        import triton  # noqa: F401
        from triton.backends.compiler import GPUTarget  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print("SKIP ptx (no CUDA triton backend):", exc)
        return
    sig = {"X": "*bf16", "W8": "*fp8e4nv", "S": "*u8", "P": "*fp32", "M": "constexpr", "SX": "constexpr",
           "QX": "constexpr", "EV": "constexpr"}
    for ev, want in (("", 0), ("evict_first", 1)):
        ptx = _ptx(wo_a_w8._wo_a_partial_w8, sig, {"M": 6, "SX": 8192, "QX": False, "EV": ev})
        hinted = [l for l in ptx.splitlines() if "ld.global" in l and "L2::cache_hint" in l]
        plain = [l for l in ptx.splitlines() if "ld.global" in l and "L2::cache_hint" not in l]
        policy = [l for l in ptx.splitlines() if "createpolicy.fractional.L2::evict_first" in l]
        if want:
            assert policy and hinted, (len(policy), len(hinted))
            assert all("L1::evict_first" in l for l in hinted)
            # the X tile is 16 rows x 128 bf16; the W8 tile 128 x 64 e4m3 and S 128 x 64 u8: the hinted loads
            # are the 8-bit ones, the un-hinted the 16-bit activation loads
            assert all((".b8" in l or ".u8" in l or ".v4.b8" in l or ".b32" in l or ".v2.b32" in l or ".v4.b32" in l)
                       for l in hinted), hinted[:3]
            assert plain, "the activation loads must stay un-hinted"
            assert all("evict" not in l for l in plain), [l for l in plain if "evict" in l][:3]
        else:
            assert not policy and not hinted and plain
        print(f"partial_w8 EV={ev!r}: {len(hinted)} hinted loads, {len(plain)} plain loads, {len(policy)} policies")
    sig_mid = {"X": "*bf16", "W8": "*fp8e4nv", "S": "*u8", "Y": "*bf16", "M": "i32", "SXM": "i32",
               "SPLIT_K": "constexpr", "BM": "constexpr", "BN": "constexpr", "BK": "constexpr",
               "QX": "constexpr", "EV": "constexpr"}
    ptx = _ptx(wo_a_w8._wo_a_mid_kernel, sig_mid,
               {"SPLIT_K": 1, "BM": 64, "BN": 32, "BK": 128, "QX": False, "EV": "evict_first"})
    assert "createpolicy.fractional.L2::evict_first" in ptx
    assert any("ld.global" in l and "L2::cache_hint" in l for l in ptx.splitlines())
    print("mid kernel EV='evict_first': hinted")


if __name__ == "__main__":
    test_switch()
    print("PASS test_switch")
    test_interpreter_bitwise()
    print("PASS test_interpreter_bitwise")
    test_ptx_hint_placement()
    print("PASS test_ptx_hint_placement")
