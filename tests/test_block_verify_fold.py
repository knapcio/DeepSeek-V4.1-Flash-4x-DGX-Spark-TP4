"""CPU tests of adapter/block_verify_fold.py under the Triton interpreter (no GPU).

  docker run --rm --network none -v $PWD:/w -w /w -e TRITON_INTERPRET=1 -e PYTHONPATH=/w/adapter:/w/tests \
      ds-b3-cutecheck:1 python3 tests/test_block_verify_fold.py

Contract: the fold is BIT-IDENTICAL to production block_verify.block_accept_probs for the same inputs and draws.
1. The mirror prod_debug (tests/bv_fold_cases.py) is pinned to the production source and equals it on every case.
2. On every case of bv_fold_cases.cases(): fold tau and token == production, and the intermediates S, h, mass,
   residual and correction keys are bit-identical (int32 views) to production's.
   Cases: random production-shaped rows (V up to 129280, bs 1..4, V not a multiple of the chunk), live cutoff, dead
   positions, top-p truncated targets, p == q (zero mass), q(X) = 0, one-hot targets, eta exactly at h and one ulp
   either side, full accept, nothing accepted, every draft dead, equal / NaN / inf correction keys, near-zero mass.
3. Engine wiring: execute() with the fold draws the same random numbers and returns the same tau / token as the
   default path (same torch seed); per-call A/B switch (0 / 1 / check), receipts per path, check counters 0.
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "adapter"))
sys.path.insert(0, HERE)
os.environ.setdefault("DSV41_BLOCK_VERIFY", "1")
os.environ.setdefault("DSV41_BLOCK_VERIFY_FOLD", "1")
os.environ.setdefault("DSV41_BLOCK_VERIFY_FOLD_CPU", "1")
assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1 (CPU interpreter)"

import torch  # noqa: E402

import block_verify as bv  # noqa: E402
import block_verify_fold as bvf  # noqa: E402
import bv_fold_cases as cases_mod  # noqa: E402

FAILS = []
G = cases_mod.G


def check(ok, name, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def bits_equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def run_case(name, c):
    live_p = None if c["live"] is None else c["live"].to(torch.int64) - c["live_sub"]
    t_p, k_p = bv.block_accept_probs(c["X"].long(), c["tp"], c["dp"], eta=c["eta"], gumbel=c["gum"], live=live_p)
    t_m, k_m, dm = cases_mod.prod_debug(c["X"].long(), c["tp"], c["dp"], c["eta"], c["gum"], live=live_p)
    mirror_ok = torch.equal(t_p, t_m) and torch.equal(k_p, k_m)
    df = {}
    t_f, k_f = bvf.fold(c["X"], c["tp"], c["dp"], c["eta"], c["gum"], live=c["live"], live_sub=c["live_sub"], debug=df)
    out_ok = torch.equal(t_f, t_p) and torch.equal(k_f, k_p)
    inter = {k: bits_equal(dm[k], df[k]) for k in ("s", "mass", "res", "keys")}
    inter["h"] = bits_equal(dm["h"], df["h"])
    ok = mirror_ok and out_ok and all(inter.values())
    detail = f"tau {t_f.tolist()} tok {k_f.tolist()}" if ok else \
        f"mirror {mirror_ok} out {out_ok} (fold {t_f.tolist()} {k_f.tolist()} prod {t_p.tolist()} {k_p.tolist()}) {inter}"
    check(ok, name, detail)


# 1 ---------------------------------------------------------------------------------------------------------------
sha = cases_mod.prod_source_sha()
check(sha == cases_mod.MIRRORED_SHA, "prod_debug mirrors the current block_accept_probs source", sha[:16])

# 2 ---------------------------------------------------------------------------------------------------------------
for name, c in cases_mod.cases("cpu"):
    run_case(name, c)

# 3 ---------------------------------------------------------------------------------------------------------------
fake = types.SimpleNamespace()


class AcceptSampling:
    @staticmethod
    def execute(**kw):
        raise AssertionError("orig execute must not run without cutoff + vcap off")


class SoftmaxTemp:
    @staticmethod
    def execute(*, logits, temperatures, rows_per_request):
        t = temperatures.reshape(-1).float().repeat_interleave(rows_per_request)
        return torch.softmax(logits.float() / t[:, None], -1)


fake.AcceptSampling = AcceptSampling
fake._accept_sampling_core = lambda **kw: None
fake.SoftmaxTemp = SoftmaxTemp
for m in ("sglang", "sglang.srt", "sglang.srt.speculative"):
    sys.modules.setdefault(m, types.ModuleType(m))
du = types.ModuleType("sglang.srt.speculative.dflash_utils")
du.build_dflash_verify_target_probs = None
sys.modules["sglang.srt.speculative.dflash_utils"] = du
os.environ["DSV41_VERIFY_CAP"] = "conf:0.1"
bv.install(fake)
ex = AcceptSampling.execute


def engine_inputs(s):
    gen = torch.Generator().manual_seed(500 + s)
    bs, V = 2, 3000
    logits = torch.randn(bs * (G + 1), V, generator=gen) * 3
    dprobs = torch.softmax(torch.randn(bs, G, V, generator=gen) * 3, -1)
    cand = torch.cat([torch.randint(0, V, (bs, 1), generator=gen), cases_mod.sample_X(dprobs, gen)], 1).reshape(-1)
    si = types.SimpleNamespace(need_top_k_sampling=False, need_top_p_sampling=False,
                               temperatures=torch.full((bs, 1), 0.7))
    cut = torch.randint(2, G + 2, (bs,), generator=gen)
    return dict(candidates=cand.view(bs, G + 1), target_logits=logits, draft_probs=dprobs, sampling_info=si,
                draft_input=None, gamma=G, verify_num_draft_tokens=G + 1, cutoff_verify_lens=cut)


same = 0
for s in range(8):
    kw = engine_inputs(s)
    outs = []
    for use_fold in (True, False):
        bv._fold = bvf if use_fold else None
        torch.manual_seed(900 + s)
        outs.append(ex(**kw))
    same += int(torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1]))
check(same == 8, "engine wiring: fold path == default path for the same seed (same random stream)", f"{same}/8")

# in-boot A/B: the variant's DSV41_BLOCK_VERIFY_FOLD decides per call (0 / 1 / check), validated by ab_variant
import ab_variant as abv  # noqa: E402

fake_ab = types.SimpleNamespace(ACTIVE=True, mode="0", _fold_value=abv._fold_value)
fake_ab.env = lambda name, default=None: fake_ab.mode if name == "DSV41_BLOCK_VERIFY_FOLD" else default
bv._ab, bv._fold = fake_ab, None
kw = engine_inputs(0)
seen, rec = {}, {}
for mode in ("0", "1", "check"):
    fake_ab.mode = mode
    before = dict(bvf._R)
    torch.manual_seed(77)
    out = ex(**kw)
    seen[mode] = (out[0].tolist(), out[1].tolist())
    rec[mode] = {k: bvf._R[k] - before[k] for k in ("torch", "fold", "check")}
counts = bvf._C["counters"].tolist()
check(seen["0"] == seen["1"] == seen["check"] and rec["0"] == {"torch": 1, "fold": 0, "check": 0}
      and rec["1"] == {"torch": 0, "fold": 1, "check": 0} and rec["check"] == {"torch": 0, "fold": 0, "check": 1}
      and counts[1] == 2 and counts[2:] == [0, 0],
      "in-boot A/B: per-call switch, receipts per path, check counters (rows 2, mismatches 0)", f"{rec} {counts}")
bv._ab = None

print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
