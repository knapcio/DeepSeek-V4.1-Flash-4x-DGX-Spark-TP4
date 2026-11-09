"""CPU checks for adapter/prefill_ced_skip.py (DSV41_PREFILL_CED_SKIP). No GPU.

1. plan(): a request qualifies only with >= 128 prompt rows after this chunk; unknown final lengths -> nothing;
2. the CED layout with nothing skipped == the engine's own late_layer_tail_layout (the pinned source, executed with
   CPU torch), single- and multi-request; skipped requests: their last row, floored at their own position;
3. the dependency rule on a simulated chunked prefill (the real chunk 8192 / window 128 and scaled-down ones):
   every late-layer SWA slot that a NON-skipped tail or a decode step reads was written by a non-skipped tail, so
   the skipped tails' rows are never read (what the exactness argument in the module docstring claims);
4. the engine hook on a fake backend module: gate off -> nothing wrapped; gate on -> stock passthrough without final
   lengths, the CED layout with them, composition with an earlier wrapper (extend_nosync's dispatcher), the in-boot
   A/B per-call switch (ab_variant RUNTIME key), drift -> "NOT armed" and the stock path.

  python3 tests/test_prefill_ced_skip.py           (torch needed for parts 2 and 4: uv run --no-project --with torch)
  DSV41_TEST_TAIL_LAYOUT_SRC=<path to dsv4_attn_metadata_kernels.py>  (default: the image's sglang, else the Mac
  canary tree of repos/deepseek/ds41-sip)
"""
import ast
import importlib.util
import os
import random
import sys
import textwrap
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(HERE, "..", "adapter")
sys.path.insert(0, ADAPTER)
for k in [k for k in os.environ if k.startswith(("DSV41_AB", "DSV41_PREFILL_CED"))]:
    del os.environ[k]

try:
    import torch
except ImportError:
    torch = None


def fresh(name, path="prefill_ced_skip.py"):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ADAPTER, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def engine_layout_source():
    cands = [os.environ.get("DSV41_TEST_TAIL_LAYOUT_SRC", "")]
    try:
        spec = importlib.util.find_spec("sglang")
        if spec is not None and spec.submodule_search_locations:
            cands.append(os.path.join(list(spec.submodule_search_locations)[0], "kernels", "ops", "attention",
                                      "dsv4_attn_metadata_kernels.py"))
    except (ImportError, ValueError):
        pass
    cands.append(os.path.join(HERE, "..", "..", "ds41-sip", "runtime", "sglang-canary", "python", "sglang", "kernels",
                              "ops", "attention", "dsv4_attn_metadata_kernels.py"))
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None


def engine_layout():
    """The engine's late_layer_tail_layout, extracted from its source and executed with CPU torch."""
    path = engine_layout_source()
    if path is None or torch is None:
        return None, path
    src = open(path).read()
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "late_layer_tail_layout"]
    assert len(fn) == 1, path
    code = ast.get_source_segment(src, fn[0])
    assert "max(0, s - tail_len)" in code and "(seq_lens - tail_len).clamp_min(0)" in code, \
        "the engine layout is not the #41303 one the adapter assumes"
    scope = {"torch": torch}
    exec(textwrap.dedent(code), scope)
    return scope["late_layer_tail_layout"], path


# ---------------------------------------------------------------- 1. plan
def test_plan():
    m = fresh("ced_plan")
    W = 128
    assert m.plan([8192], [8192], [8192 + 128], W) == [True]           # exactly one window after: skip
    assert m.plan([8192], [8192], [8192 + 127], W) is None             # a final chunk of < 128 rows reads this tail
    assert m.plan([8192], [16384], [16384], W) is None                 # the final chunk
    assert m.plan([8192], [8192], None, W) is None                     # unknown final lengths: stock
    assert m.plan([8192, 500], [8192, 9000], [40000, 9000], W) == [True, False]   # bypass prompt beside a chunk
    assert m.plan([1], [100], [10000], W) is None                      # one row: nothing to save
    assert m.plan([8192, 8], [8192], [9000, 9000], W) is None          # length mismatch: stock
    assert m.plan([8192], [8192], [None], W) is None


# ---------------------------------------------------------------- 2. layout vs the engine
def test_layout_matches_engine():
    m = fresh("ced_layout")
    W = 128
    # host lists without torch
    assert m.host_layout([10], [30], 4, [False]) == ([6, 7, 8, 9], [4], [26, 26, 26, 26])
    assert m.host_layout([10], [30], 4, [True]) == ([9], [1], [29])
    assert m.host_layout([10, 3], [30, 3], 4, [True, False]) == ([9, 10, 11, 12], [1, 3], [29, 0, 0, 0])
    eng, path = engine_layout()
    if eng is None:
        print(f"SKIP engine comparison (torch {'present' if torch else 'missing'}, engine source {path})")
        return
    rng = random.Random(7)
    cases = [([8192], [8192]), ([8192], [131072]), ([5], [5]), ([127], [1000]), ([200, 8192, 3], [200, 16384, 900]),
             ([1], [1])]
    for _ in range(200):
        bs = rng.randint(1, 5)
        ext = [rng.randint(1, 9000) for _ in range(bs)]
        cases.append((ext, [e + rng.randint(0, 50000) for e in ext]))
    for ext, seq in cases:
        ti, tl, fl = eng(extend_lens_cpu=ext, seq_lens_cpu=seq, tail_len=W, device="cpu")
        ci, ctl, cfl = m.ced_layout(ext, seq, W, "cpu", [False] * len(ext))
        assert ctl == tl and ci.dtype == ti.dtype and fl.dtype == cfl.dtype, (ext, seq)
        assert torch.equal(ci, ti) and torch.equal(cfl, fl), (ext, seq)
        idx, tails, floors = m.stock_host_layout(ext, seq, W)
        assert idx == ti.tolist() and tails == tl and floors == fl.tolist()
        # skipped requests: last row, own floor; the others byte-equal to the engine's rows
        skip = [rng.random() < 0.5 and e > 1 for e in ext]
        si, stl, sfl = m.ced_layout(ext, seq, W, "cpu", skip)
        start, pos_e, pos_c = 0, 0, 0
        for r, (e, s, k, t) in enumerate(zip(ext, seq, skip, tl)):
            if k:
                assert stl[r] == 1 and si[pos_c].item() == start + e - 1 and sfl[pos_c].item() == s - 1
                pos_c += 1
            else:
                assert torch.equal(si[pos_c:pos_c + t], ti[pos_e:pos_e + t])
                assert torch.equal(sfl[pos_c:pos_c + t], fl[pos_e:pos_e + t])
                pos_c += t
            pos_e += t
            start += e
        assert pos_c == si.shape[0] == sum(stl)
    print(f"engine layout: {len(cases)} cases equal with nothing skipped ({path})")


# ---------------------------------------------------------------- 3. the dependency rule
def simulate(prompt, chunk, W, ced, decode=300):
    """Late-layer SWA slots written (position -> writer) and every read, for one prompt prefilled in chunks then
    decoded. A tail row at position p with floor f reads slots [max(f, p - W + 1), p]; the row itself is written in
    the same forward (reads of the same forward's rows are fine). Decode at p reads [p - W + 1, p]."""
    m = fresh("ced_sim")
    written, bad, skipped_rows = {}, [], set()
    seq = 0
    k = 0
    while seq < prompt:
        n = min(chunk, prompt - seq)
        seq += n
        skip = m.plan([n], [seq], [prompt], W) if ced else None
        idx, tails, floors = m.host_layout([n], [seq], W, skip or [False])
        rows = [seq - n + i for i in idx]
        if skip:
            skipped_rows.update(range(seq - min(W, n), seq - 1))
        this = set(rows)
        for p, f in zip(rows, floors):
            for q in range(max(f, p - W + 1), p + 1):
                if q not in this and q not in written:
                    bad.append(("prefill", k, p, q))
        for p in rows:
            written[p] = k
        k += 1
    for p in range(prompt, prompt + decode):
        for q in range(p - W + 1, p):
            if q >= 0 and q not in written:
                bad.append(("decode", p, q))
        written[p] = "decode"
    return written, bad, skipped_rows


def test_dependency_rule():
    rng = random.Random(11)
    configs = [(8192, 128), (16, 4), (12, 4), (9, 4)]
    n_checked = 0
    for chunk, W in configs:
        lens = {1, W - 1, W, W + 1, chunk - 1, chunk, chunk + 1, chunk + W - 1, chunk + W, chunk + W + 1,
                2 * chunk + 1, 3 * chunk + W - 1, 5 * chunk + 2 * W}
        lens |= {rng.randint(1, 6 * chunk) for _ in range(60)}
        for L in sorted(x for x in lens if x > 0):
            w_s, bad_s, _ = simulate(L, chunk, W, ced=False)
            w_c, bad_c, skipped = simulate(L, chunk, W, ced=True)
            # stock reads only written rows (the engine's own invariant), and so does CED
            assert not bad_s, (chunk, W, L, bad_s[:3])
            assert not bad_c, (chunk, W, L, bad_c[:3])
            # every slot CED leaves unwritten was written by stock only in a skipped tail, and nothing reads it
            assert set(w_s) - set(w_c) == skipped, (chunk, W, L)
            assert all(q not in skipped for q in range(L - W + 1, L + 300)), (chunk, W, L)
            n_checked += 1
    print(f"dependency rule: {n_checked} prompt lengths, no read of a skipped tail's late-layer slot")


# ---------------------------------------------------------------- 4. the engine hook
BACKEND_SRC = '''
SWA_WINDOW = 128

def is_cp_active(forward_batch):
    return getattr(forward_batch, "attn_cp_metadata", None) is not None

class DeepseekV4AttnBackend:
    is_dspark_draft = False

    def _build_late_layer_tail_metadata(self, forward_batch):
        extend_lens_cpu = forward_batch.extend_seq_lens_cpu
        seq_lens_cpu = forward_batch.seq_lens_cpu
        token_indices, tail_lens_cpu, swa_replay_start = late_layer_tail_layout(
            extend_lens_cpu=extend_lens_cpu,
            seq_lens_cpu=seq_lens_cpu.tolist(),
            tail_len=SWA_WINDOW,
            device="cpu",
        )
        return dict(token_indices=token_indices.tolist(), num_tokens=sum(tail_lens_cpu),
                    extend_seq_lens_cpu=tail_lens_cpu, swa_replay_start=swa_replay_start.tolist())
'''


def fake_backend(name, eng):
    mod = types.ModuleType(name)
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"{name}.py")
    with open(path, "w") as f:
        f.write(BACKEND_SRC)
    mod.__file__ = path
    exec(compile(BACKEND_SRC, path, "exec"), mod.__dict__)
    mod.late_layer_tail_layout = eng
    return mod


def fb(ext, seq, finals=None):
    return types.SimpleNamespace(extend_seq_lens_cpu=list(ext), seq_lens_cpu=torch.tensor(seq),
                                 dsv41_final_lens=finals, attn_cp_metadata=None)


def test_engine_hook():
    eng, path = engine_layout()
    if eng is None:
        print("SKIP engine hook (needs torch + the engine layout source)")
        return
    # gate off: nothing wrapped
    m = fresh("ced_off")
    mod = fake_backend("fake_dsv4_off", eng)
    before = mod.DeepseekV4AttnBackend.__dict__["_build_late_layer_tail_metadata"]
    assert m.install_backend(mod, {}) is False
    assert mod.DeepseekV4AttnBackend.__dict__["_build_late_layer_tail_metadata"] is before and \
        mod.late_layer_tail_layout is eng
    # gate on, composed over an earlier wrapper (as extend_nosync's dispatcher sits there)
    m = fresh("ced_on")
    mod = fake_backend("fake_dsv4_on", eng)
    stock_meth = mod.DeepseekV4AttnBackend.__dict__["_build_late_layer_tail_metadata"]
    calls = []

    def dispatcher(*a, **kw):
        calls.append("nosync")
        return stock_meth(*a, **kw)

    dispatcher.__wrapped__ = stock_meth
    mod.DeepseekV4AttnBackend._build_late_layer_tail_metadata = dispatcher
    assert m.install_backend(mod, {"DSV41_PREFILL_CED_SKIP": "1"}) is True
    be = mod.DeepseekV4AttnBackend()
    stock = stock_meth(be, fb([8192], [8192]))
    got = be._build_late_layer_tail_metadata(fb([8192], [8192]))              # no final lengths: stock
    assert got == stock and calls == ["nosync"]
    got = be._build_late_layer_tail_metadata(fb([8192], [8192], [131072]))     # non-final chunk: one row
    assert got == dict(token_indices=[8191], num_tokens=1, extend_seq_lens_cpu=[1], swa_replay_start=[8191])
    got = be._build_late_layer_tail_metadata(fb([8192], [131072], [131072]))   # final chunk: stock
    assert got == stock_meth(be, fb([8192], [131072]))
    mixed = be._build_late_layer_tail_metadata(fb([8192, 300], [16384, 300], [40000, 300]))
    ref = stock_meth(be, fb([8192, 300], [16384, 300]))
    assert mixed["extend_seq_lens_cpu"] == [1, 128] and mixed["token_indices"] == [8191] + ref["token_indices"][128:]
    assert mixed["swa_replay_start"] == [16383] + ref["swa_replay_start"][128:]
    be.is_dspark_draft = True                                                   # never the draft's backend
    assert be._build_late_layer_tail_metadata(fb([8192], [8192], [131072])) == stock_meth(be, fb([8192], [8192]))
    be.is_dspark_draft = False
    cp = fb([8192], [8192], [131072])
    cp.attn_cp_metadata = object()                                              # context parallel: stock
    assert be._build_late_layer_tail_metadata(cp) == stock
    st = m.stats()
    assert st["forwards"] == 2 and st["skipped"] == 2 and st["rows_saved"] == 2 * 127 and st["unused"] == 0
    # drift: the builder no longer calls the layout by name -> NOT armed, nothing wrapped
    m2 = fresh("ced_drift")
    mod2 = fake_backend("fake_dsv4_drift", eng)
    src = BACKEND_SRC.replace("tail_len=SWA_WINDOW", "tail_len=128")
    exec(compile(src, mod2.__file__, "exec"), mod2.__dict__)
    mod2.late_layer_tail_layout = eng
    with open(mod2.__file__, "w") as f:
        f.write(src)
    assert m2.install_backend(mod2, {"DSV41_PREFILL_CED_SKIP": "1"}) is False and m2.stats()["why_not"]
    mod3 = fake_backend("fake_dsv4_w", eng)
    mod3.SWA_WINDOW = 256
    assert fresh("ced_w").install_backend(mod3, {"DSV41_PREFILL_CED_SKIP": "1"}) is False


def test_in_boot_ab_switch():
    eng, _ = engine_layout()
    if eng is None:
        print("SKIP A/B switch (needs torch + the engine layout source)")
        return
    ab = fresh("ab_variant", "ab_variant.py")
    sys.modules["ab_variant"] = ab
    try:
        env = {"DSV41_AB_VARIANTS": "3", "DSV41_AB_MAX_BS": "0", "DSV41_AB_ALLOW_AA": "1",
               "DSV41_AB_V0": "DSV41_PREFILL_CED_SKIP=0", "DSV41_AB_V1": "DSV41_PREFILL_CED_SKIP=1",
               "DSV41_AB_V2": "DSV41_PREFILL_CED_SKIP=0"}
        assert ab.configure(env) and "DSV41_PREFILL_CED_SKIP" not in env          # call-time key: never unioned
        m = fresh("ced_ab")
        assert m.wanted_at_install({}) is True                                      # an armed A/B naming it
        mod = fake_backend("fake_dsv4_ab", eng)
        stock_meth = mod.DeepseekV4AttnBackend.__dict__["_build_late_layer_tail_metadata"]
        assert m.install_backend(mod, {})
        be = mod.DeepseekV4AttnBackend()
        stock = stock_meth(be, fb([8192], [8192]))
        out = []
        for v in (0, 1, 2, 1):
            ab.set_runtime(v)
            out.append(be._build_late_layer_tail_metadata(fb([8192], [8192], [131072]))["num_tokens"])
        assert out == [128, 1, 128, 1] and stock["num_tokens"] == 128
        ab.set_runtime(0)
    finally:
        sys.modules.pop("ab_variant", None)
        sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "_Finder"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name, flush=True)
    print("ALL PASS")
