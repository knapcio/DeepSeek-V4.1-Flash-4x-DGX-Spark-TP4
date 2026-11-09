"""CPU tests of adapter/cert_head.py (no GPU, no Triton): tables, the bound, the tile mask, the argmax argument over
TP shards, full / non-finite modes, the engine routing with fake SGLang classes, the A/B harness registry, and
the X3b masked-mode row cap (DSV41_CERT_HEAD_MASK_MAX_M).

  uv run --no-project --with torch --with numpy python tests/test_cert_head.py
"""
import importlib
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "adapter"))
os.environ.setdefault("DSV41_CERT_HEAD", "1")
os.environ.setdefault("DSV41_CERT_HEAD_M", "6,12,18,24")

import torch  # noqa: E402

import cert_head as ch  # noqa: E402

FAILS = []


def check(ok, name, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILS.append(name)


torch.manual_seed(0)
GEN = torch.Generator().manual_seed(1234)
K = 5120          # DS V4.1 hidden size
V = 1024          # per-shard rows in the tests (the real shard is 32320)


def make_head(V=V, K=K, dup=()):
    w = (torch.randn(V, K, generator=GEN) * 0.02)
    w[:, torch.randint(0, K, (8,), generator=GEN)] *= 6.0        # a few heavy columns
    w = w.to(torch.bfloat16)
    for a, b in dup:                                               # exact duplicate rows -> exact logit ties
        w[b] = w[a]
    return w.contiguous()


def acts(kind, M, W, gamma):
    n = torch.randn(M, K, generator=GEN)
    if kind == "massive":
        n[:, torch.randint(0, K, (3,), generator=GEN)] *= 30.0
    n = n / n.pow(2).mean(-1, keepdim=True).sqrt()
    x = n * gamma[None, :]
    if kind in ("peaked", "neartie"):
        t = torch.randint(0, W.shape[0], (M,), generator=GEN)
        w = W[t].float()
        a = torch.empty(M, 1).uniform_(0.15, 0.45, generator=GEN)
        x = x + a * w / w.norm(dim=1, keepdim=True) * x.norm(dim=1, keepdim=True)
        if kind == "neartie":
            t2 = (t + 1) % W.shape[0]
            w2 = W[t2].float()
            x = x + a * w2 / w2.norm(dim=1, keepdim=True) * x.norm(dim=1, keepdim=True)
    return x.to(torch.bfloat16)


def stock_logits(W, x, order="seq"):
    """bf16(acc) with an fp32 accumulation in a chosen order (the stock kernel's exact order is unknown on CPU;
    the certificate must hold for every order)."""
    xf, wf = x.float(), W.float()
    if order == "f64":
        acc = (x.double() @ W.double().t()).float()
    elif order == "splitk8":
        parts = [(xf[:, i * K // 8:(i + 1) * K // 8] @ wf[:, i * K // 8:(i + 1) * K // 8].t()) for i in range(8)]
        acc = parts[0]
        for p in parts[1:]:
            acc = acc + p
    elif order == "k16chain":
        acc = torch.zeros(x.shape[0], W.shape[0])
        for k0 in range(0, K, 16):
            acc = acc + xf[:, k0:k0 + 16] @ wf[:, k0:k0 + 16].t()
    else:
        acc = xf @ wf.t()
    return acc.to(torch.bfloat16)


# ---------------------------------------------------------------------------------------------------
def test_tables():
    W = make_head()
    st = ch.build_tables(W)
    d_bf, raw = ch.deq_rows(st.q, st.e, 0, V)
    check(st.q.dtype == torch.int8 and st.e.dtype == torch.uint8 and st.e.shape == (V, K // 32), "tables: dtypes/shapes")
    check(int(st.q.abs().max()) <= 127, "tables: int8 range")
    check(st.n_unsafe == 0, "tables: dequantised copy is bf16-exact", f"unsafe {st.n_unsafe}")
    rel = float(((W.float() - d_bf).norm(dim=1) / W.float().norm(dim=1)).mean())
    check(rel < 0.012, "tables: MXINT8 relative error", f"eps {rel * 100:.3f} % (st.eps {st.eps * 100:.3f} %)")
    # residual group norms are upper bounds
    res = (W.double() - d_bf.double()).view(V, K // ch.GROUP, ch.GROUP).pow(2).sum(-1).sqrt()
    check(bool((st.rt.double().t() >= res).all()), "tables: residual norms rounded up")
    nrm = torch.maximum(W.double().norm(dim=1), d_bf.double().norm(dim=1))
    check(bool((st.nv.double() >= nrm).all()), "tables: row norms rounded up")
    bad = W.clone()
    bad[3, 5] = float("nan")
    try:
        ch.build_tables(bad)
        check(False, "tables: non-finite weights refused")
    except ValueError:
        check(True, "tables: non-finite weights refused")


def test_bound_and_argmax():
    """lo <= stock <= hi for every accumulation order; masked argmax == stock argmax over 4 shards, ties included."""
    shards = [make_head(dup=((5, 700), (11, 12))) for _ in range(4)]
    shards[2][100] = shards[0][100]                 # a cross-shard exact tie
    states = [ch.build_tables(w) for w in shards]
    gamma = (torch.rand(K, generator=GEN) * 2.0 + 0.2)
    Wfull = torch.cat(shards)
    worst_margin = float("inf")
    n_rows = n_bad = n_tie = 0
    tiles = []
    for kind in ("normlike", "massive", "peaked", "neartie"):
        for M in (6, 12, 24):
            x = acts(kind, M, Wfull, gamma)
            # force a few rows onto exact ties (duplicated rows 5 / 700 of shard 0; cross-shard row 100)
            x[0] = (Wfull[5].float() / Wfull[5].float().norm() * 40).to(torch.bfloat16)
            x[1] = (Wfull[100].float() / Wfull[100].float().norm() * 40).to(torch.bfloat16)
            for order in ("seq", "splitk8", "k16chain", "f64"):
                stock = torch.cat([stock_logits(w, x, order) for w in shards], dim=1).float()
                masked = []
                for w, st in zip(shards, states):
                    hi, T, bad = ch.reference_bounds(st, x)
                    ls = stock_logits(w, x, order).float()
                    # soundness of the interval (on the stock values of THIS order)
                    lo_ok = bool((ls <= hi).all())
                    hi_ok = bool((T[:, None] <= ls.max(dim=1, keepdim=True).values).all())
                    if not (lo_ok and hi_ok):
                        n_bad += 1
                    worst_margin = min(worst_margin, float((hi - ls).min()))
                    need = ch.reference_needed_tiles(hi, T, bad, full=False)
                    tiles.append(float(need.float().mean()))
                    masked.append(ch.reference_masked(stock_logits(w, x, order), need).float())
                masked = torch.cat(masked, dim=1)
                a, b = stock.argmax(dim=1), masked.argmax(dim=1)
                n_rows += M
                n_bad += int((a != b).sum())
                n_tie += int(((stock == stock.max(dim=1, keepdim=True).values).sum(dim=1) > 1).sum())
                if not torch.equal(stock.gather(1, a[:, None]), masked.gather(1, b[:, None])):
                    n_bad += 1
    check(n_bad == 0, "bound + argmax over 4 shards, 4 accumulation orders",
          f"{n_rows} rows, {n_tie} rows with exact ties, violations {n_bad}, min hi-l margin {worst_margin:.4f}")
    print(f"INFO needed tile fraction (16-row tiles, V={V} per shard): mean {sum(tiles) / len(tiles) * 100:.2f} %, "
          f"max {max(tiles) * 100:.2f} %")


def test_modes():
    W = make_head()
    st = ch.build_tables(W)
    gamma = torch.ones(K)
    x = acts("normlike", 6, W, gamma)
    hi, T, bad = ch.reference_bounds(st, x)
    nt = (V + ch.XBV - 1) // ch.XBV
    check(bool(ch.reference_needed_tiles(hi, T, bad, full=True).all()), "full mode computes every tile")
    x2 = x.clone()
    x2[3, 17] = float("inf")
    hi2, T2, bad2 = ch.reference_bounds(st, x2)
    check(bool(bad2[3]) and bool(ch.reference_needed_tiles(hi2, T2, bad2, full=False).all()),
          "non-finite row -> every tile computed")
    x3 = x.clone()
    x3[2] = 0
    hi3, T3, bad3 = ch.reference_bounds(st, x3)
    need3 = ch.reference_needed_tiles(hi3, T3, bad3, full=False)
    check(bool(need3.all()), "all-zero row (all logits tie) -> every tile computed", f"{int(need3.sum())}/{nt}")
    # a NaN screen value is a candidate and never raises T
    acc = x.float() @ ch.deq_rows(st.q, st.e, 0, V)[0].t()
    acc[1, 9] = float("nan")
    hi4, T4, _ = ch.reference_bounds(st, x, screen_acc=acc)
    check(hi4[1, 9] == float("inf") and torch.isfinite(T4).all(), "non-finite screen value: candidate, T unaffected")


def test_parse_and_hash():
    check(ch.parse_m("6,12-14") == frozenset({6, 12, 13, 14}), "parse_m")
    try:
        ch.parse_m("0")
        check(False, "parse_m refuses 0")
    except ValueError:
        check(True, "parse_m refuses 0")
    e1 = {"DSV41_CERT_HEAD": "1", "DSV41_CERT_HEAD_M": "6"}
    e2 = {"DSV41_CERT_HEAD": "1", "DSV41_CERT_HEAD_M": "6,12"}
    check(ch.config_hash(e1) != ch.config_hash(e2) and ch.config_hash(e1) == ch.config_hash(dict(e1)), "config hash")
    check(0 <= ch.config_hash(e1) < 2 ** 56, "config hash fits int64")


def test_masked_regime():
    """Peaked rows (the live regime: GLM measured ~10 candidates per rank and step): most tiles are -inf, the planted
    exact ties (duplicate rows, one of them across shards, one across a tile boundary) must survive."""
    shards = [make_head(dup=((5, 700), (15, 16))) for _ in range(4)]
    shards[3][200] = shards[1][200]
    Wfull = torch.cat(shards)
    states = [ch.build_tables(w) for w in shards]
    gamma = torch.rand(K, generator=GEN) * 2.0 + 0.2
    fr, n_bad, n_rows, n_masked_rows = [], 0, 0, 0
    for M in (6, 12, 24):
        for rep in range(4):
            n = torch.randn(M, K, generator=GEN)
            n = n / n.pow(2).mean(-1, keepdim=True).sqrt() * gamma[None, :]
            # one clear local winner per row on EVERY shard (a Gaussian bulk alone puts ~0.5 % of the vocab inside
            # the bound: real logits have a heavy right tail instead, GLM live: ~10 candidates per rank and step)
            xf = 0.05 * n
            for s_ in range(4):
                t = torch.randint(0, V, (M,), generator=GEN)
                if s_ == 0:
                    t[0], t[1] = 5, 15                  # tie pairs (5,700) and (15,16) (16 opens the next tile)
                if s_ == 1:
                    t[2] = 200                          # cross-shard tie with shard 3 row 200
                w = shards[s_][t].float()
                a = torch.empty(M, 1).uniform_(1.0, 3.0, generator=GEN) if s_ else torch.full((M, 1), 3.5)
                xf = xf + a * w / w.norm(dim=1, keepdim=True) * n.norm(dim=1, keepdim=True) / 4
            x = xf.to(torch.bfloat16)
            for order in ("seq", "splitk8"):
                stock = torch.cat([stock_logits(wi, x, order) for wi in shards], dim=1).float()
                masked = []
                for wi, st in zip(shards, states):
                    hi, T, bad = ch.reference_bounds(st, x)
                    need = ch.reference_needed_tiles(hi, T, bad, full=False)
                    fr.append(float(need.float().mean()))
                    masked.append(ch.reference_masked(stock_logits(wi, x, order), need).float())
                masked = torch.cat(masked, dim=1)
                a, b = stock.argmax(dim=1), masked.argmax(dim=1)
                n_bad += int((a != b).sum()) + int(not torch.equal(stock.gather(1, a[:, None]), masked.gather(1, b[:, None])))
                n_rows += M
                n_masked_rows += int((masked == float("-inf")).any(dim=1).sum())
    ties = [int(i) for i in (stock[0] == stock[0].max()).nonzero().flatten()]
    check(n_bad == 0 and max(fr) < 0.5, "masked regime: argmax identical with most tiles -inf",
          f"{n_rows} rows, tiles computed mean {sum(fr) / len(fr) * 100:.2f} % max {max(fr) * 100:.2f} %, "
          f"row-0 tie set {ties}")


# ---------------------------------------------------------------------------------------------------
# engine routing with fake SGLang classes
# ---------------------------------------------------------------------------------------------------
class FakeMode:
    def __init__(self, verify):
        self.verify = verify

    def is_target_verify(self):
        return self.verify


class FakeMeta:
    def __init__(self, verify):
        self.forward_mode = FakeMode(verify)


class FakeHead:
    def __init__(self, W):
        self.weight = W


def fake_lp_module():
    mod = types.ModuleType("fake_lp")

    class LogitsProcessor:
        def _get_logits(self, hidden_states, lm_head, logits_metadata, embedding_bias=None, use_logits_buffer=True):
            return self._compute_lm_head(hidden_states, lm_head, embedding_bias)

        def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None):
            return torch.matmul(hidden_states.to(lm_head.weight.dtype), lm_head.weight.T)

    mod.LogitsProcessor = LogitsProcessor
    return mod


def test_routing():
    importlib.reload(ch)
    W = make_head()
    st = ch.build_tables(W)
    head = FakeHead(W)
    st.lm_head = head
    mod = fake_lp_module()
    ch.install_logits(mod)
    lp_target, lp_draft = mod.LogitsProcessor(), mod.LogitsProcessor()
    ch._S.update(st=st, lp=lp_target, bufs=types.SimpleNamespace(flag_host=0, flag=None))
    calls = []

    def fake_cert(st_, bufs, x):
        calls.append(x.shape[0])
        return torch.full((x.shape[0], st_.V), -1.0, dtype=torch.bfloat16)

    ch.certified_logits = fake_cert
    gamma = torch.ones(K)
    x6, x7 = acts("normlike", 6, W, gamma), acts("normlike", 7, W, gamma)
    out = lp_target._get_logits(x6, head, FakeMeta(True))
    check(calls == [6] and bool((out == -1).all()), "routing: target verify, qualified M -> certified")
    lp_target._get_logits(x7, head, FakeMeta(True))
    lp_target._get_logits(x6, head, FakeMeta(False))
    lp_draft._get_logits(x6, head, FakeMeta(True))
    lp_target._get_logits(x6, FakeHead(W.clone()), FakeMeta(True))
    check(calls == [6], "routing: unqualified M / non-verify / draft processor / other head -> stock")
    os.environ["DSV41_CERT_HEAD"] = "0"
    lp_target._get_logits(x6, head, FakeMeta(True))
    check(calls == [6], "routing: mode 0 -> stock")
    os.environ["DSV41_CERT_HEAD"] = "check"
    got = {}
    ch._check = lambda bufs, stock, cert: got.update(stock=stock, cert=cert)
    out = lp_target._get_logits(x6, head, FakeMeta(True))
    ref = torch.matmul(x6, W.T)
    check(calls == [6, 6] and torch.equal(out, ref) and torch.equal(got["stock"], ref),
          "routing: check mode returns the stock logits and compares")
    os.environ["DSV41_CERT_HEAD"] = "1"


def test_step_armed_and_flag():
    importlib.reload(ch)

    class Pen:
        def __init__(self, prepared):
            self._p = prepared

        def is_prepared(self):
            return self._p

    class BatchedMinNewTokensPenalizer(Pen):
        pass

    class BatchedFrequencyPenalizer(Pen):
        pass

    def si(greedy=True, pens=(), **kw):
        orch = types.SimpleNamespace(is_required=any(p.is_prepared() for p in pens),
                                     penalizers={type(p): p for p in pens}) if pens else None
        d = dict(is_all_greedy=greedy, has_custom_logit_processor=False, logit_bias=None, grammar_mask=None,
                 penalizer_orchestrator=orch)
        d.update(kw)
        return types.SimpleNamespace(**d)

    tok = types.SimpleNamespace(additional_stop_token_ids={7}, eos_token_id=1)

    def req(stop=None):
        return types.SimpleNamespace(sampling_params=types.SimpleNamespace(stop_token_ids=stop), eos_token_ids={1, 2},
                                     tokenizer=tok)

    def b(**kw):
        d = dict(return_logprob=False, has_grammar=False, sampling_info=si(), reqs=[req()])
        d.update(kw)
        return types.SimpleNamespace(**d)

    ex = None
    check(ch.step_armed(ex, b()) == (True, ()), "armed: all greedy, nothing else")
    check(ch.step_armed(ex, b(sampling_info=None)) == (True, ()), "armed: no sampling info")
    check(not ch.step_armed(ex, b(sampling_info=si(greedy=False)))[0], "not armed: sampled rows")
    check(not ch.step_armed(ex, b(return_logprob=True))[0], "not armed: logprobs")
    check(not ch.step_armed(ex, b(has_grammar=True))[0], "not armed: grammar")
    check(not ch.step_armed(ex, b(sampling_info=si(logit_bias=object())))[0], "not armed: logit bias")
    check(not ch.step_armed(ex, b(sampling_info=si(has_custom_logit_processor=True)))[0], "not armed: custom processor")
    got = ch.step_armed(ex, b(sampling_info=si(pens=(BatchedMinNewTokensPenalizer(True),)), reqs=[req({99}), req()]))
    check(got == (True, (1, 2, 7, 99)), "armed: min_new_tokens only, stop ids = the penalizer's set", str(got))
    got = ch.step_armed(ex, b(sampling_info=si(pens=(BatchedMinNewTokensPenalizer(True), BatchedFrequencyPenalizer(True)))))
    check(not got[0], "not armed: another penalizer active")
    got = ch.step_armed(ex, b(sampling_info=si(pens=(BatchedMinNewTokensPenalizer(True), BatchedFrequencyPenalizer(False)))))
    check(got[0], "armed: an unprepared penalizer does not count")
    many = req(set(range(100, 200)))
    check(not ch.step_armed(ex, b(sampling_info=si(pens=(BatchedMinNewTokensPenalizer(True),)), reqs=[many]))[0],
          "not armed: more than NSTOP stop ids")
    writes = []

    class Flag:
        def fill_(self, v):
            writes.append(v)

    ch._S["bufs"] = types.SimpleNamespace(flag=Flag(), flag_host=0)
    for a in (False, True, True, True, False, False, True):
        ch.set_flag(a)
    check(writes == [1, 0, 1], "flag written only on change", str(writes))


def test_stop_rule():
    """min_new_tokens: stop ids excluded from T and always computed -> exact argmax with and without the -inf mask."""
    W = make_head()
    st = ch.build_tables(W)
    gamma = torch.ones(K)
    bad = 0
    for trial in range(40):
        x = acts("peaked", 6, W, gamma)
        ls = stock_logits(W, x).float()
        top = ls.argmax(dim=1)
        stops = sorted(set(int(t) for t in top[:3]) | {5, 900})   # the argmax of three rows is a stop id
        hi, T, bd = ch.reference_bounds(st, x, stop_ids=stops)
        need = ch.reference_needed_tiles(hi, T, bd, full=False)
        m = ch.reference_masked(stock_logits(W, x), need).float()
        for masked in (False, True):
            a, c = ls.clone(), m.clone()
            if masked:
                a[:, stops] = float("-inf")
                c[:, stops] = float("-inf")
            ia, ic = a.argmax(1), c.argmax(1)
            bad += int((ia != ic).sum()) + int((a.gather(1, ia[:, None]) != c.gather(1, ic[:, None])).sum())
    hi0, T0, b0 = ch.reference_bounds(st, x)
    need0 = ch.reference_needed_tiles(hi0, T0, b0, full=False)
    m0 = ch.reference_masked(stock_logits(W, x), need0).float()
    a = ls.clone()
    a[:, stops] = float("-inf")
    c = m0.clone()
    c[:, stops] = float("-inf")
    naive_bad = int((a.argmax(1) != c.argmax(1)).sum())
    check(bad == 0, "stop rule: argmax exact with and without the min_new_tokens mask", f"violations {bad}")
    check(naive_bad > 0, "stop rule is needed: without it the masked argmax breaks", f"{naive_bad} rows wrong")


def test_verify_hook():
    importlib.reload(ch)
    mod = types.ModuleType("fake_verify")
    seen = []

    class TargetVerifyExecutor:
        verify_epilogue = None

        def run_non_compact(self, *, batch, draft_input=None, verify_ids_2d=None, verify_window=None,
                            sampling_info=None):
            seen.append(("nc", ch._S["bufs"].flag_host))
            return "nc"

        def run_compact(self, **kw):
            seen.append(("c", ch._S["bufs"].flag_host))
            return "c"

        def run_idle_participation(self, **kw):
            seen.append(("idle", ch._S["bufs"].flag_host))

    mod.TargetVerifyExecutor = TargetVerifyExecutor

    class Flag:
        def fill_(self, v):
            pass

    ch._S["bufs"] = types.SimpleNamespace(flag=Flag(), flag_host=0)
    ch.install_verify(mod)

    ex = TargetVerifyExecutor()
    b = types.SimpleNamespace(return_logprob=False, has_grammar=False, sampling_info=None, reqs=[])
    ex.run_non_compact(batch=b, draft_input=None, verify_ids_2d=None, verify_window=None, sampling_info=None)
    ex.run_compact(batch=b)
    b.return_logprob = True
    ex.run_non_compact(batch=b, draft_input=None, verify_ids_2d=None, verify_window=None, sampling_info=None)
    check(seen == [("nc", 1), ("c", 0), ("nc", 0)], "verify hook sets the flag before each forward", str(seen))


def test_ab_registry():
    sys.path.insert(0, os.path.join(HERE, "..", "adapter"))
    import ab_variant as ab
    env = {"DSV41_AB_VARIANTS": "2", "DSV41_AB_V0": "DSV41_CERT_HEAD=0", "DSV41_AB_V1": "DSV41_CERT_HEAD=1"}
    ab.configure(env)
    check(env.get("DSV41_CERT_HEAD") == "1", "A/B: DSV41_CERT_HEAD unioned into the env")
    check(ab.env_for(0, "DSV41_CERT_HEAD") == "0" and ab.env_for(1, "DSV41_CERT_HEAD") == "1", "A/B: per-variant value")
    env = {"DSV41_AB_VARIANTS": "2", "DSV41_AB_V0": "DSV41_CERT_HEAD=1", "DSV41_AB_V1": "DSV41_CERT_HEAD=check"}
    try:
        ab.configure(env)
        check(True, "A/B: 1 vs check is not an A/A")
    except RuntimeError as e:
        check(False, "A/B: 1 vs check is not an A/A", repr(e))
    env = {"DSV41_AB_VARIANTS": "2", "DSV41_AB_V0": "DSV41_CERT_HEAD=1", "DSV41_AB_V1": "DSV41_CERT_HEAD=on"}
    try:
        ab.configure(env)
        check(False, "A/B: 1 vs on refused as A/A")
    except RuntimeError:
        check(True, "A/B: 1 vs on refused as A/A")
    ab.configure({})


def test_mask_cap_x3b():
    """X3b (DSV41_CERT_HEAD_MASK_MAX_M): unset = the production decisions; set, a greedy step above the cap takes full
    mode (the stock logits) and the greedy argmax / accept is unchanged at every M; the graphs are untouched."""
    os.environ.pop("DSV41_CERT_HEAD_MASK_MAX_M", None)
    importlib.reload(ch)
    greedy = types.SimpleNamespace(is_all_greedy=True, has_custom_logit_processor=False, logit_bias=None,
                                   grammar_mask=None, penalizer_orchestrator=None)
    sampled = types.SimpleNamespace(**dict(vars(greedy), is_all_greedy=False))

    def b(si, n=1):
        return types.SimpleNamespace(return_logprob=False, has_grammar=False, sampling_info=si,
                                     reqs=[object()] * n)

    # flag unset: no cap; every decision equals the stock (rows-less) one, the config hash is the pre-X3b one
    check(ch.mask_max_m() is None, "X3b off: no cap when unset")
    same = all(ch.step_armed(None, b(si), rows) == ch.step_armed(None, b(si))
               for si in (greedy, sampled) for rows in (None, 6, 30, 36, 48, 96, 128, 192))
    check(same, "X3b off: step_armed identical to the stock decision at every row count")
    import hashlib
    e = {"DSV41_CERT_HEAD": "1", "DSV41_CERT_HEAD_M": "6,12,18,24,30,36,42,48"}
    pre = "|".join(f"{k}={e.get(k, '')}" for k in ("DSV41_CERT_HEAD", "DSV41_CERT_HEAD_M", "DSV41_CERT_HEAD_GROUP",
                                                    "DSV41_CERT_HEAD_TILE", "DSV41_CERT_HEAD_EXACT"))
    pre_h = int.from_bytes(hashlib.sha256(pre.encode()).digest()[:7], "little")
    check(ch.config_hash(e) == pre_h and ch.config_hash(dict(e, DSV41_CERT_HEAD_MASK_MAX_M="")) == pre_h,
          "X3b off: config hash unchanged (rank handshake of production configs untouched)")
    check(ch.config_hash(dict(e, DSV41_CERT_HEAD_MASK_MAX_M="30")) != pre_h,
          "X3b on: the cap is in the rank handshake (ranks with different caps disable the head)")
    check(ch.qualified_m() is ch.QUALIFIED_M, "X3b off: qualified row list is the process one")

    # flag on: masked mode only up to the cap
    os.environ["DSV41_CERT_HEAD_MASK_MAX_M"] = "30"
    got = {M: ch.step_armed(None, b(greedy, M // 6), M) for M in (6, 12, 18, 24, 30, 36, 42, 48)}
    check(all(got[M] == (True, ()) for M in (6, 12, 18, 24, 30)) and all(got[M] == (False, ()) for M in (36, 42, 48)),
          "X3b on: greedy masked at M <= 30, full mode at M 36-48", str({M: g[0] for M, g in got.items()}))
    check(all(not ch.step_armed(None, b(sampled), M)[0] for M in (6, 30, 36, 48)), "X3b on: sampled unchanged (full)")
    check(ch.step_armed(None, b(greedy), None) == (True, ()), "X3b on: unknown row count keeps the stock decision")
    check(ch.qualified_m() == ch.QUALIFIED_M, "X3b on: the qualified row list (the graphs) is unchanged")
    os.environ["DSV41_CERT_HEAD_MASK_MAX_M"] = "0"
    check(not ch.step_armed(None, b(greedy), 6)[0], "X3b cap 0: never masked")
    os.environ["DSV41_CERT_HEAD_MASK_MAX_M"] = "thirty"
    check(ch.mask_max_m() is None and ch.step_armed(None, b(greedy, 8), 48)[0], "X3b malformed cap: no cap (stock)")
    for bad in ("-1", "x"):
        try:
            ch.parse_mask_cap(bad)
            check(False, f"parse_mask_cap refuses {bad!r}")
        except ValueError:
            check(True, f"parse_mask_cap refuses {bad!r}")
    check(ch.parse_mask_cap("500") == ch.MAX_M and ch.parse_mask_cap(" 30 ") == 30, "parse_mask_cap clamps / strips")

    # the verify hook: rows from verify_ids_2d (bs x stride), else len(reqs) x verify_num_draft_tokens; the same
    # decision on every rank (it reads only the batch and the env)
    os.environ["DSV41_CERT_HEAD_MASK_MAX_M"] = "30"
    flags_by_rank = []
    for r in range(4):
        importlib.reload(ch)
        ch._rank = lambda r=r: r
        mod = types.ModuleType("fake_verify")
        seen = []

        class TargetVerifyExecutor:
            verify_epilogue = None
            verify_num_draft_tokens = 6

            def run_non_compact(self, *, batch, draft_input=None, verify_ids_2d=None, verify_window=None,
                                sampling_info=None):
                seen.append(ch._S["bufs"].flag_host)

            def run_compact(self, **kw):
                pass

            def run_idle_participation(self, **kw):
                pass

        mod.TargetVerifyExecutor = TargetVerifyExecutor

        class Flag:
            def fill_(self, v):
                pass

        ch._S["bufs"] = types.SimpleNamespace(flag=Flag(), flag_host=0)
        ch.install_verify(mod)
        ex = TargetVerifyExecutor()
        for bs in (1, 5, 6, 8, 5):
            ex.run_non_compact(batch=b(greedy, bs), verify_ids_2d=torch.zeros(bs, 6, dtype=torch.int64))
        ex.run_non_compact(batch=b(greedy, 6))           # no verify_ids_2d: len(reqs) x stride = 36
        ex.run_non_compact(batch=b(greedy, 5))
        flags_by_rank.append(tuple(seen))
        capped = ch._S.get("capped_calls", 0)
    check(flags_by_rank[0] == (1, 1, 0, 0, 1, 0, 1) and len(set(flags_by_rank)) == 1 and capped == 3,
          "X3b hook: flag per step from the verify row count, identical on 4 ranks", str(flags_by_rank[0]))
    os.environ.pop("DSV41_CERT_HEAD_MASK_MAX_M", None)
    importlib.reload(ch)

    # identical outputs: at M 36-48 the full mode is the stock logits, and its argmax (the in-graph greedy accept)
    # equals the masked one over 4 TP shards, ties included
    shards = [make_head(dup=((5, 700),)) for _ in range(4)]
    shards[3][100] = shards[1][100]
    states = [ch.build_tables(w) for w in shards]
    Wfull = torch.cat(shards)
    gamma = (torch.rand(K, generator=GEN) * 2.0 + 0.2)
    n_bad = n_rows = 0
    full_bytes = True
    for kind in ("normlike", "peaked", "neartie"):
        for M in (36, 42, 48):
            x = acts(kind, M, Wfull, gamma)
            x[0] = (Wfull[5].float() / Wfull[5].float().norm() * 40).to(torch.bfloat16)
            stock = torch.cat([stock_logits(w, x) for w in shards], dim=1)
            full, masked = [], []
            for w, st in zip(shards, states):
                hi, T, bad = ch.reference_bounds(st, x)
                ex_l = stock_logits(w, x)
                full.append(ch.reference_masked(ex_l, ch.reference_needed_tiles(hi, T, bad, full=True)))
                masked.append(ch.reference_masked(ex_l, ch.reference_needed_tiles(hi, T, bad, full=False)))
            full, masked = torch.cat(full, dim=1), torch.cat(masked, dim=1)
            full_bytes &= torch.equal(full.view(torch.int16), stock.view(torch.int16))
            a, f, m = stock.float().argmax(1), full.float().argmax(1), masked.float().argmax(1)
            n_rows += M
            n_bad += int((a != f).sum()) + int((a != m).sum())
            n_bad += int((stock.float().gather(1, a[:, None]) != masked.float().gather(1, m[:, None])).sum())
    check(full_bytes, "X3b outputs: full mode at M 36-48 = the stock logits byte for byte")
    check(n_bad == 0, "X3b outputs: greedy argmax stock == full == masked at M 36-48 (4 shards)",
          f"{n_rows} rows, violations {n_bad}")


def test_ab_registry_x3b():
    sys.path.insert(0, os.path.join(HERE, "..", "adapter"))
    import ab_variant as ab
    os.environ.pop("DSV41_CERT_HEAD_MASK_MAX_M", None)
    importlib.reload(ch)
    check("DSV41_CERT_HEAD_MASK_MAX_M" in ab.RUNTIME and "DSV41_CERT_HEAD_MASK_MAX_M" not in ab.UNION_BOOL,
          "A/B X3b: the cap is a runtime (per step) key, never unioned")
    env = {"DSV41_AB_VARIANTS": "2", "DSV41_CERT_HEAD": "1", "DSV41_AB_V1": "DSV41_CERT_HEAD_MASK_MAX_M=30"}
    try:
        ab.configure(env)
        check("DSV41_CERT_HEAD_MASK_MAX_M" not in env, "A/B X3b: process env untouched by the cap")
        check(ab.env_for(0, "DSV41_CERT_HEAD_MASK_MAX_M") is None and ab.env_for(1, "DSV41_CERT_HEAD_MASK_MAX_M") == "30",
              "A/B X3b: per-variant value")
        ab.set_runtime(0)
        a0 = (ch.mask_max_m(), ch.step_armed(None, types.SimpleNamespace(return_logprob=False, has_grammar=False,
                                                                         sampling_info=None, reqs=[]), 48)[0])
        ab.set_runtime(1)
        a1 = (ch.mask_max_m(), ch.step_armed(None, types.SimpleNamespace(return_logprob=False, has_grammar=False,
                                                                         sampling_info=None, reqs=[]), 48)[0])
        check(a0 == (None, True) and a1 == (30, False), "A/B X3b: the replayed variant's cap decides the step",
              f"{a0} {a1}")
        ab.set_runtime(0)
    finally:
        ab.configure({})
    for v0, v1, aa in (("", "128", True), ("", "500", True), ("", "30", False), ("30", "36", False)):
        env = {"DSV41_AB_VARIANTS": "2", "DSV41_AB_V0": f"DSV41_CERT_HEAD_MASK_MAX_M={v0}",
               "DSV41_AB_V1": f"DSV41_CERT_HEAD_MASK_MAX_M={v1}"}
        try:
            ab.configure(env)
            check(not aa, f"A/B X3b: cap {v0 or 'unset'} vs {v1} is not an A/A")
        except RuntimeError:
            check(aa, f"A/B X3b: cap {v0 or 'unset'} vs {v1} refused as A/A")
        finally:
            ab.configure({})
    try:
        ab.configure({"DSV41_AB_VARIANTS": "2", "DSV41_AB_V1": "DSV41_CERT_HEAD_MASK_MAX_M=-3"})
        check(False, "A/B X3b: a negative cap refused at configure")
    except RuntimeError:
        check(True, "A/B X3b: a negative cap refused at configure")
    finally:
        ab.configure({})
    # X3 arm of the same boot: a variant's qualified rows are a subset of the process list
    env = {"DSV41_AB_VARIANTS": "2", "DSV41_CERT_HEAD": "1", "DSV41_CERT_HEAD_M": "6,12,18,24",
           "DSV41_AB_V1": "DSV41_CERT_HEAD_M=6,12,30"}
    try:
        ab.configure(env)
        ab.set_runtime(0)
        q0 = ch.qualified_m()
        with ab.capturing(1):
            q1 = ch.qualified_m()
        check(q0 == ch.QUALIFIED_M and q1 == frozenset({6, 12}),
              "A/B X3: per-capture qualified rows, intersected with the process list", f"{sorted(q1)}")
    finally:
        ab.configure({})
    check(ch.qualified_m() is ch.QUALIFIED_M, "A/B off again: the process list")


def test_sitecustomize_compiles():
    import py_compile
    p = os.path.join(HERE, "..", "adapter", "sitecustomize.py")
    py_compile.compile(p, doraise=True)
    s = open(p).read()
    check("'sglang.srt.layers.logits_processor'" in s and "install_cert_head_logits" in s
          and "install_cert_head_model" in s and "install_cert_head_verify" in s, "sitecustomize wires cert_head")


if __name__ == "__main__":
    test_tables()
    test_parse_and_hash()
    test_modes()
    test_bound_and_argmax()
    test_masked_regime()
    test_routing()
    test_step_armed_and_flag()
    test_stop_rule()
    test_verify_hook()
    test_ab_registry()
    test_mask_cap_x3b()
    test_ab_registry_x3b()
    test_sitecustomize_compiles()
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
