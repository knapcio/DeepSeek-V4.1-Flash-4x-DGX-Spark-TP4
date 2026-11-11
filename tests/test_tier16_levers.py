"""CPU checks for the tier-16 verify threshold table (Fable round 7 #1). Runs with any CPU torch.

  PYTHONPATH=adapter python3 tests/test_tier16_levers.py

1. DSV41_VERIFY_CAP per-batch-size table (adapter/vcap_spec.py, verify_cap.py), rebased from perf/vcap-bs-on-v26-20261027:
   every single-threshold spec production or the earlier sweeps used parses to exactly the values of the old parser, and
   the live lengths equal the old rule's on adversarial confidences (NaN, < 0, > 1, products within ulps of the
   threshold) for every bs 1..64; a table gives each bs the entry with the largest min_bs <= bs; a repeated leading value
   closes a band (the upper bound: conf:0.1/4:0.25/8:0.1 = 0.25 at bs 4-7 only); malformed tables are refused.
2. eager_glue's fused vcapk kernel gets the old arguments without a table and the bs's own threshold with one; the
   table keeps production's mechanism: the update is captured into the draft graph (no forced eager path) and the
   live-length broadcast follows DSV41_SPEC_SYNC_FREE exactly as for one threshold.
3. (the in-boot A/B wiring and replicated_split per variant are test machinery of the tier-16 branch; not carried)
5. Exactness: block verification with a stopping rule whose threshold depends on the step's batch size reproduces the
   target's joint distribution (toy LM, chi-square), as with one threshold.
"""
import importlib
import itertools
import math
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(HERE, "..", "adapter")
sys.path.insert(0, ADAPTER)
for k in [k for k in os.environ if k.startswith(("DSV41_AB", "DSV41_VERIFY_CAP", "DSV41_REPLICATED_SPLIT",
                                                 "DSV41_SPEC_SYNC_FREE", "DSV41_EAGER_GLUE"))]:
    del os.environ[k]

import torch  # noqa: E402

LEGACY_SPECS = ["conf:0.1", "conf:0.05", "conf:0.2", "conf:0.25", "conf:0.35", "conf:0.15", "conf:0.07", "conf:0",
                "conf:1", "conf:0.25,0.25,0.03,0.06", "conf:0.2,0.2,0.06,0.06", "conf:2,2,2,2", "conf:0,0,2,2",
                "conf:0.1,0.2"]
VBS = "conf:0.1/4:0.25/8:0.1"           # the X_vbs arm (round 7 #1: 0.25 at bs 4-7)


# ---------------------------------------------------------------- the v2.7.4 parser and rule, verbatim
def legacy_parse(spec):
    vals = [float(v) for v in spec.split(":", 1)[1].split(",")]
    return vals[0] if len(vals) == 1 else [0.0] + vals


def legacy_live(confidence, thr, kmin, stride=6):
    """adapter/verify_cap.py set_live_from_confidence at 891f5e1, minus the buffer and broadcast."""
    cum = torch.cumprod(confidence.float().clamp(0, 1), dim=1)
    if isinstance(thr, list):
        t = (thr + [thr[-1]] * cum.shape[1])[:cum.shape[1]]
        thr = torch.tensor(t, dtype=torch.float32, device=cum.device)
    k = (cum >= thr).to(torch.int64).cumprod(dim=1).sum(dim=1).clamp(min=kmin, max=stride - 1)
    return k + 1


class FakeGroup:
    world_size = 1
    rank_in_group = 0


def load_vc(spec, kmin=None):
    os.environ["DSV41_VERIFY_CAP"] = spec
    if kmin is None:
        os.environ.pop("DSV41_VERIFY_CAP_MIN", None)
    else:
        os.environ["DSV41_VERIFY_CAP_MIN"] = str(kmin)
    for m in ("verify_cap", "eager_glue"):
        sys.modules.pop(m, None)
    vc = importlib.import_module("verify_cap")
    vc._state["tp_group"] = FakeGroup()
    return vc


def adversarial_conf(bs, thr, gen):
    """Rows whose running product lands on / within ulps of the threshold at a random position."""
    x = torch.rand(bs, 5, generator=gen) * 0.7 + 0.3
    t = thr if not isinstance(thr, list) else max(thr[1:] or [0.1])
    t = min(max(t, 1e-3), 0.99)
    j = int(torch.randint(0, 5, (1,), generator=gen))
    prod = torch.cumprod(x, 1)[:, j]
    x[:, 0] = (x[:, 0] * t / prod).clamp(max=1.0)
    x[:, 0] += torch.randint(-3, 4, (bs,), generator=gen) * 1e-8
    if bs > 1:
        x[0, 2] = float("nan")
        x[-1, 1] = -0.5
        x[bs // 2, 3] = 1.7
    return x.contiguous()


def new_live(vc, conf, bs):
    vc.live_buf("cpu").fill_(77)
    vc.set_live_from_confidence(conf, bs)
    return vc.live_buf()[:bs].clone()


# ---------------------------------------------------------------- 1. parser and live lengths
def test_legacy_specs_parse_identically():
    for spec in LEGACY_SPECS:
        vc = load_vc(spec)
        default, table = vc.parse_conf_spec(spec)
        ref = legacy_parse(spec)
        assert table == () and type(default) is type(ref) and default == ref, (spec, default, ref)
        assert vc._state["thr"] == ref and vc._state["thr_bs"] == () and vc._state["fixed"] is None
        assert all(vc.thresholds(bs) == ref for bs in range(1, 65))
    vc = load_vc("3")
    assert vc._state["fixed"] == 3 and vc._state["thr"] is None and vc._state["thr_bs"] == ()


def test_live_lengths_identical_without_table():
    gen = torch.Generator().manual_seed(7)
    cases = 0
    for spec in LEGACY_SPECS:
        for kmin in (None, 2):
            vc = load_vc(spec, kmin)
            thr = legacy_parse(spec)
            for bs in range(1, 65):
                for _ in range(2):
                    conf = adversarial_conf(bs, thr, gen)
                    got = new_live(vc, conf, bs)
                    ref = legacy_live(conf, thr, kmin or 1)
                    assert torch.equal(got, ref), (spec, kmin, bs, got.tolist(), ref.tolist())
                    cases += 1
    print(f"  {cases} live-length cases bit-identical to the 891f5e1 rule")


TABLE = "conf:0.1/4:0.2/8:0.15,0.15,0.1,0.1/16:0.3"


def expected_thr(bs):
    if bs >= 16:
        return 0.3
    if bs >= 8:
        return [0.0, 0.15, 0.15, 0.1, 0.1]
    if bs >= 4:
        return 0.2
    return 0.1


def test_table_selection_and_live_lengths():
    vc = load_vc(TABLE)
    assert vc._state["thr"] == 0.1
    assert vc._state["thr_bs"] == ((4, 0.2), (8, [0.0, 0.15, 0.15, 0.1, 0.1]), (16, 0.3))
    gen = torch.Generator().manual_seed(11)
    for bs in range(1, 65):
        assert vc.thresholds(bs) == expected_thr(bs), bs
        for _ in range(3):
            conf = adversarial_conf(bs, expected_thr(bs), gen)
            got = new_live(vc, conf, bs)
            assert torch.equal(got, legacy_live(conf, expected_thr(bs), 1)), bs
    vc = load_vc("conf:0.25,0.25,0.03,0.06/2:0.1")
    assert vc.thresholds(1) == [0.0, 0.25, 0.25, 0.03, 0.06] and vc.thresholds(2) == 0.1


def test_vbs_band_upper_bound():
    """The X_vbs spec: 0.25 exactly at bs 4-7, production's 0.1 at 1-3 and 8-64 (the bs upper bound)."""
    vc = load_vc(VBS)
    got = [vc.thresholds(bs) for bs in range(1, 65)]
    assert got == [0.1] * 3 + [0.25] * 4 + [0.1] * 57, got
    import vcap_spec
    assert vcap_spec.bands(vc.parse_conf_spec(VBS)) == "bs 1-3 0.1 | bs 4-7 0.25 | bs 8-64 0.1"
    assert vcap_spec.bands(vc.parse_conf_spec("conf:0.1")) == "bs 1-64 0.1"
    assert vcap_spec.per_bs("conf:0.1/8:0.1") == vcap_spec.per_bs("conf:0.1")
    gen = torch.Generator().manual_seed(5)
    for bs in (1, 3, 4, 7, 8, 16):
        conf = adversarial_conf(bs, 0.25 if 4 <= bs <= 7 else 0.1, gen)
        assert torch.equal(new_live(vc, conf, bs), legacy_live(conf, 0.25 if 4 <= bs <= 7 else 0.1, 1)), bs


def test_table_refusals():
    vc = load_vc("conf:0.1")
    for bad in ("conf:0.1/1:0.2", "conf:0.1/8:0.2/4:0.3", "conf:0.1/8:0.2/8:0.3", "conf:0.1/8",
                "conf:0.1/65:0.2", "conf:0.1/x:0.2", "conf:0.1/4:", "3", "conf:0.1/4:a"):
        try:
            vc.parse_conf_spec(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")
    try:
        load_vc("conf:0.1/8:0.2/4:0.3")
    except ValueError:
        pass
    else:
        raise AssertionError("module accepted a descending table")


# ---------------------------------------------------------------- 2. eager_glue + sync semantics
class Recorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kw):
            self.calls.append((args, kw))
        return launch


def legacy_kernel_args(thr, bs, conf, out, kmin=1):
    """eager_glue._launch_vcapk at 891f5e1: the values it passes (tensors compared by value)."""
    thr_t = None
    if isinstance(thr, list):
        thr_t = torch.tensor((thr + [thr[-1]] * 5)[:5], dtype=torch.float32)
    return ((conf, conf.stride(0), out, thr_t if thr_t is not None else out,
             0.0 if thr_t is not None else float(thr), bs, kmin, 5),
            dict(ORDER=0 if bs == 1 else 1, THR_T=thr_t is not None, BLOCK=max(16, 1 << (bs - 1).bit_length())))


def same_args(a, b):
    (aa, akw), (ba, bkw) = a, b
    if akw != bkw or len(aa) != len(ba):
        return False
    for x, y in zip(aa, ba):
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
                return False
            if x is y:
                continue
            if x.dtype != y.dtype or not torch.equal(x, y):
                return False
        elif x != y or type(x) is not type(y):
            return False
    return True


def kernel_args(eg, vc, bs):
    rec = Recorder()
    eg._vcap_kernel = rec
    eg.triton = types.SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length())
    conf = torch.rand(bs, 5)
    out = torch.empty(bs, dtype=torch.int64)
    eg._launch_vcapk(vc, conf, bs, out)
    return rec.calls[0], conf, out


def test_eager_glue_kernel_args():
    try:
        for spec in LEGACY_SPECS + [TABLE, VBS]:
            os.environ["DSV41_EAGER_GLUE"] = "vcap,vcapk"
            vc = load_vc(spec)
            vc.live_buf("cpu")                 # prebuilds the per-position tensors (outside capture)
            eg = importlib.import_module("eager_glue")
            for bs in (1, 2, 3, 4, 7, 8, 15, 16, 33, 64):
                if spec == TABLE:
                    thr = expected_thr(bs)
                elif spec == VBS:
                    thr = 0.25 if 4 <= bs <= 7 else 0.1
                else:
                    thr = legacy_parse(spec)
                got, conf, out = kernel_args(eg, vc, bs)
                ref = legacy_kernel_args(thr, bs, conf, out)
                assert same_args(got, ref), (spec, bs, got, ref)
    finally:
        os.environ.pop("DSV41_EAGER_GLUE", None)


class BcastGroup:
    world_size = 4
    rank_in_group = 0

    def __init__(self):
        self.calls = 0

    def broadcast(self, t, src=0):
        self.calls += 1


def test_table_keeps_production_sync_semantics():
    """A table is not a different mechanism: the live-length broadcast follows DSV41_SPEC_SYNC_FREE exactly as for one
    threshold (skipped under its vcap token, sent without it), and eager_glue still captures the update into the draft
    graph (no policy test routes a table to the eager path)."""
    import inspect
    for spec in ("conf:0.1", VBS):
        for sync, want in (("all", 0), ("", 1)):
            if sync:
                os.environ["DSV41_SPEC_SYNC_FREE"] = sync
            else:
                os.environ.pop("DSV41_SPEC_SYNC_FREE", None)
            sys.modules.pop("spec_sync_free", None)
            vc = load_vc(spec)
            g = BcastGroup()
            vc._state["tp_group"] = g
            sf = None
            if sync:
                sf = importlib.import_module("spec_sync_free")

                def tp_broadcast(group, t, site, _g=g):
                    if site == "vcap" and sf.SKIP_VCAP:
                        return
                    _g.broadcast(t, src=0)
                sf.tp_broadcast = tp_broadcast
            vc.share_live(torch.zeros(8, dtype=torch.int64), 4)
            assert g.calls == want, (spec, sync, g.calls)
    os.environ.pop("DSV41_SPEC_SYNC_FREE", None)
    import eager_glue
    src = inspect.getsource(eager_glue._install_vcap_sampler)
    assert "policy_needs_broadcast" not in src and "thr_bs" not in src


def test_exact_with_bs_dependent_threshold():
    import block_verify as bv
    V, G = 5, 3
    N = int(os.environ.get("VC_TEST_N", "40000"))
    gen = torch.Generator().manual_seed(2)
    tp, tq = {}, {}

    def p_of(pre):
        key = tuple(pre)
        if key not in tp:
            tp[key] = torch.softmax(torch.randn(V, generator=gen) * 1.5, -1)
        return tp[key]

    def q_of(pre):
        key = tuple(pre)
        if key not in tq:
            tq[key] = torch.softmax(torch.log(p_of(pre)) + 0.8 * torch.randn(V, generator=gen), -1)
        return tq[key]

    def thr_for(bs):           # the batch shrinks as sequences finish: three different thresholds
        return 0.6 if bs > N // 2 else (0.3 if bs > N // 8 else 0.05)

    out = torch.zeros(N, 3, dtype=torch.long)
    produced = torch.zeros(N, dtype=torch.long)
    seqs = [[] for _ in range(N)]
    used = set()
    while (produced < 3).any():
        act = [i for i in range(N) if produced[i] < 3]
        bs = len(act)
        X = torch.zeros(bs, G, dtype=torch.long)
        TP = torch.zeros(bs, G + 1, V)
        DP = torch.zeros(bs, G, V)
        for r, i in enumerate(act):
            pre = list(seqs[i])
            for j in range(G):
                DP[r, j], TP[r, j] = q_of(pre), p_of(pre)
                X[r, j] = int(torch.multinomial(DP[r, j], 1, generator=gen))
                pre.append(int(X[r, j]))
            TP[r, G] = p_of(pre)
        thr = thr_for(bs)
        used.add(thr)
        cum = torch.cumprod(DP.gather(-1, X.unsqueeze(-1)).squeeze(-1), 1)
        live = 1 + (cum[:, :G - 1] >= thr).long().cumprod(1).sum(1)
        tau, corr = bv.block_accept_probs(X, TP, DP, live=live)
        for r, i in enumerate(act):
            for t in X[r, :int(tau[r])].tolist() + [int(corr[r])]:
                seqs[i].append(t)
                if produced[i] < 3:
                    out[i, produced[i]] = t
                    produced[i] += 1
    counts = torch.zeros(V, V, V)
    for a, b, c in out.tolist():
        counts[a, b, c] += 1
    stat, dof = 0.0, 0
    for a, b, c in itertools.product(range(V), repeat=3):
        e = N * float(p_of([])[a] * p_of([a])[b] * p_of([a, b])[c])
        if e >= 5:
            stat += (counts[a, b, c] - e) ** 2 / e
            dof += 1
    dof -= 1
    z = (stat - dof) / math.sqrt(2 * dof)
    print(f"  bs-dependent stopping rule: thresholds used {sorted(used)}, chi-square z={z:+.2f}")
    assert len(used) >= 2, used
    assert abs(z) < 4.5, z


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name, flush=True)
    print("test_tier16_levers: ALL PASS")
