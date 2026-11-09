#!/usr/bin/env python3
"""DSV41_PREFILL_SP_FP8_MOE under the in-boot A/B harness (adapter/ab_variant.py), CPU only (torch, no GPU):

1. it is a union gate: a variant that enables it switches it on in the process env (prefill_sp.FP8_MOE and
   moe_b12x_next.PREQUANT_WARM read it at import), the pre-union base stays recorded;
2. prefill_sp.fp8_moe_on() follows the replayed variant per call (variant 0 off, variant 1 on), the process gate
   alone without the harness, and is off whenever FP8_MOE is off;
3. the MoE wrapper dispatches on fp8_moe_on() (not on the import-time constant), so A and A' run the bf16 gather;
4. prefill_sp and moe_b12x_next are among the hashed sources (two nodes with different copies refuse to run).
  python3 tests/test_prefill_sp_fp8_moe_ab.py
"""
import importlib.util
import inspect
import os
import sys

ADAPTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter")
sys.path.insert(0, ADAPTER)
for k in [k for k in os.environ if k.startswith(("DSV41_AB", "DSV41_PREFILL_SP"))]:
    del os.environ[k]

KEY = "DSV41_PREFILL_SP_FP8_MOE"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    fails = []
    ab = load("ab_variant", os.path.join(ADAPTER, "ab_variant.py"))
    sys.modules["ab_variant"] = ab
    if KEY not in ab.UNION_BOOL or KEY not in ab.KNOWN:
        fails.append("not a union gate")
    if not {"prefill_sp", "moe_b12x_next"} <= set(ab.HASHED_SOURCES):
        fails.append("prefill_sp / moe_b12x_next not hashed")
    env = {"DSV41_AB_VARIANTS": "4", "DSV41_AB_ALLOW_AA": "1", "DSV41_PREFILL_SP": "1", "DSV41_AB_MAX_BS": "0",
           "DSV41_AB_V0": f"{KEY}=0", "DSV41_AB_V1": f"{KEY}=1", "DSV41_AB_V2": f"{KEY}=0", "DSV41_AB_V3": f"{KEY}=0"}
    try:
        if not ab.configure(env):
            fails.append("configure did not arm")
        if env.get(KEY) != "1":
            fails.append(f"union did not switch the gate on: {env.get(KEY)!r}")
        if '"%s": null' % KEY not in env.get("DSV41_AB_BASE", ""):
            fails.append("the pre-union base does not record the unset gate")
        if [ab.env_for(i, KEY) for i in range(4)] != ["0", "1", "0", "0"]:
            fails.append(f"per-variant values {[ab.env_for(i, KEY) for i in range(4)]}")
        # prefill_sp imported AFTER the union, exactly as sitecustomize orders it
        os.environ.update({k: v for k, v in env.items() if k in (KEY, "DSV41_PREFILL_SP")})
        sp = load("prefill_sp", os.path.join(ADAPTER, "prefill_sp.py"))
        if not sp.FP8_MOE or sp.MODE != "shard":
            fails.append(f"prefill_sp FP8_MOE {sp.FP8_MOE} MODE {sp.MODE}")
        if sp._ab is not ab:
            fails.append("prefill_sp does not see the harness module")
        got = []
        for v in range(4):
            ab._runtime = v
            got.append(sp.fp8_moe_on())
        if got != [False, True, False, False]:
            fails.append(f"fp8_moe_on per variant {got}")
        ab._capture = 1          # a capture never reaches prefill SP, but the read follows current() like every gate
        if not sp.fp8_moe_on():
            fails.append("fp8_moe_on ignores the variant being captured")
        ab._capture = None
        ab._runtime = 0
        src = inspect.getsource(sp._make_moe_forward)
        if "if p.sharded and fp8_moe_on():" not in src or "and FP8_MOE:" in src:
            fails.append("the MoE wrapper does not dispatch per call")
        try:
            mbn = load("moe_b12x_next", os.path.join(ADAPTER, "moe_b12x_next.py"))
            if not mbn.PREQUANT_WARM:
                fails.append("moe_b12x_next.PREQUANT_WARM off under the union")
        except ImportError as exc:      # the image has every dependency; the Mac may not
            print(f"SKIP moe_b12x_next import ({exc!r})")
        # without the harness: the process gate alone
        ab.ACTIVE = False
        if not sp.fp8_moe_on():
            fails.append("fp8_moe_on off without the harness while FP8_MOE is on")
        ab.ACTIVE = True
        sp.FP8_MOE = False
        ab._runtime = 1
        if sp.fp8_moe_on():
            fails.append("fp8_moe_on on while FP8_MOE (the install gate) is off")
    finally:
        for k in (KEY, "DSV41_PREFILL_SP", "DSV41_AB_BASE"):
            os.environ.pop(k, None)
        sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "_Finder"]
    # default: no env -> off, and the harness inactive
    sp0 = load("prefill_sp_default", os.path.join(ADAPTER, "prefill_sp.py"))
    if sp0.FP8_MOE or sp0.fp8_moe_on() or sp0.MODE != "off":
        fails.append("not off by default")
    for f in fails:
        print("FAIL", f)
    print(f"test_prefill_sp_fp8_moe_ab: {'ALL 1 OK' if not fails else str(len(fails)) + ' FAIL'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
