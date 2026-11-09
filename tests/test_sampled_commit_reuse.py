"""CPU checks for adapter/sampled_commit_reuse.py.

Part 1 (needs the SGLang tree of the image, skipped elsewhere): the adapter installs on the real engine (source pin
holds), and an eager sampled commit through the REAL TargetVerifyExecutor.commit_hidden / TargetHiddenKvInjector /
DsparkVerifyEpilogue._static_epilogue / DeepseekV4ForCausalLMDSpark.write_target_hidden_kv code writes the same
draft-KV cache bits as stock, without recomputing the projection. Only the model weights, the pool's store kernel
and the greedy accept kernel are CPU stand-ins.

Part 2 (stand-ins only): eligibility (padded replay, eager verify, compact, not captured), check-mode counters, a
corrupted buffer is caught in check mode and never stored, A/B variants per capture / per replay.

Run: PYTHONPATH=/opt/dsv41/adapter CUDA_VISIBLE_DEVICES= python3 tests/test_sampled_commit_reuse.py
"""
import importlib
import os
import sys
import types

os.environ["DSV41_SAMPLED_COMMIT_REUSE"] = "1"
os.environ.setdefault("DSV41_SAMPLED_COMMIT_REUSE_LOG_EVERY", "0")
import torch  # noqa: E402

import sampled_commit_reuse as scr  # noqa: E402

H, HK, D, S, STRIDE, MAX_BS = 64, 32, 16, 3, 6, 4
torch.manual_seed(0)


# ------------------------------------------------------------------------------------------------ stand-ins --
class Lin:
    def __init__(self, k, n):
        self.w = torch.randn(n, k, dtype=torch.float32).to(torch.bfloat16)
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return torch.nn.functional.linear(x, self.w), None


class Norm:
    def __init__(self, n):
        self.weight = types.SimpleNamespace(data=torch.randn(n).to(torch.bfloat16))

    def __call__(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.weight.data


def make_stages():
    stages = []
    for i in range(S):
        attn = types.SimpleNamespace(wkv=Lin(HK, D), layer_id=i, kv_norm=Norm(D), eps=1e-6,
                                     freqs_cis=torch.randn(256, D // 2))
        stages.append(types.SimpleNamespace(self_attn=attn, main_proj=Lin(H, HK), main_norm=Norm(HK)))
    return stages


class Pool:
    """The fused k-norm + RoPE + store as a deterministic CPU function of (kv, position); rows with loc < 0 skip."""
    uniform_fp8 = False

    def __init__(self, inplace=False):
        self.cache = torch.zeros(S, 512, D, dtype=torch.bfloat16)
        self.inplace = inplace
        self.full_to_swa_index_mapping = torch.arange(512)

    def translate_loc_from_full_to_swa(self, loc):
        return self.full_to_swa_index_mapping[loc]

    def set_swa_key_buffer_radix_fused_norm_rope(self, *, layer_id, swa_loc, kv, kv_weight, eps, freqs_cis,
                                                 positions):
        kf = kv.float()
        y = kf * torch.rsqrt(kf.pow(2).mean(-1, keepdim=True) + eps) * kv_weight.float()
        y = y + freqs_cis[positions].repeat(1, 2).float()
        if self.inplace:                     # the uniform-fp8 path norms + ropes kv in place
            kv.copy_(y.to(kv.dtype))
        for i in range(kv.shape[0]):
            if int(swa_loc[i]) >= 0:
                self.cache[layer_id, int(swa_loc[i])] = y[i].to(torch.bfloat16)


# -------------------------------------------------------------------------------------- part 1: real engine --
def _engine():
    try:
        dsp = importlib.import_module("sglang.srt.models.deepseek_v4_dspark")
        ver = importlib.import_module("sglang.srt.speculative.dspark_components.dspark_verify")
        inj = importlib.import_module("sglang.srt.speculative.dspark_components.dspark_kv_inject")
    except Exception as exc:  # noqa: BLE001
        print(f"test_sampled_commit_reuse: engine part skipped ({type(exc).__name__}: {exc})")
        return None
    return dsp, ver, inj


def engine_part():
    eng = _engine()
    if eng is None:
        return
    dsp, ver, inj = eng
    stock_write = dsp.DeepseekV4ForCausalLMDSpark.write_target_hidden_kv
    assert scr.source_sha(stock_write) == scr.PIN_WRITE_TARGET_HIDDEN_KV, "pin drifted: update the mirror + pin"

    ex_cls = ver.TargetVerifyExecutor
    calls = {"verify": 0}

    def fake_run_non_compact(self, **kw):           # the target forward + graph replay, stood in
        calls["verify"] += 1
        return ver.TargetVerifyResult(logits_output=self._fake_logits, can_run_cuda_graph=True)

    ex_cls.run_non_compact = fake_run_non_compact
    assert scr.install_dspark(dsp) and scr.install_verify(ver)

    def build(pool):
        model = dsp.DeepseekV4ForCausalLMDSpark.__new__(dsp.DeepseekV4ForCausalLMDSpark)
        torch.nn.Module.__init__(model)
        object.__setattr__(model, "stages", STAGES)
        runner = types.SimpleNamespace(bs=None, raw_bs=None)
        dmr = types.SimpleNamespace(token_to_kv_pool=pool)
        mr = types.SimpleNamespace(device="cpu", decode_cuda_graph_runner=runner)
        kvi = inj.TargetHiddenKvInjector(draft_model=model, draft_model_runner=dmr, model_runner=mr, device="cpu",
                                         verify_num_draft_tokens=STRIDE, block_pos_offsets=torch.arange(STRIDE))
        ex = ex_cls(target_worker=types.SimpleNamespace(model_runner=mr), gamma=STRIDE - 1,
                    verify_num_draft_tokens=STRIDE, model_runner=mr, kv_injector=kvi, tp_sync=None)
        epi = ver.DsparkVerifyEpilogue(max_bs=MAX_BS, verify_num_draft_tokens=STRIDE, device="cpu", tp_sync=None,
                                       commit_ctx=ver.CommitInjectCtx(draft_model=model,
                                                                      block_pos_offsets=torch.arange(STRIDE),
                                                                      resolve_pool=lambda: pool,
                                                                      resolve_req_to_token=lambda: None,
                                                                      kv_injector=kvi))
        return ex, epi, runner

    def step(ex, epi, runner, bs, hidden, commit_lens, capture, graph_bs=None):
        graph_bs = bs if graph_bs is None else graph_bs
        n = graph_bs * STRIDE
        loc = torch.arange(n) + 7 * bs
        pos = torch.arange(n) % 200 + 3
        g_hidden = torch.cat([hidden, torch.randn(n - hidden.shape[0], H).to(torch.bfloat16)]) if graph_bs > bs \
            else hidden
        fb = types.SimpleNamespace(batch_size=graph_bs, input_ids=torch.zeros(n, dtype=torch.int64),
                                   seq_lens=torch.full((graph_bs,), 50), out_cache_loc=loc, positions=pos,
                                   req_pool_indices=torch.zeros(graph_bs, dtype=torch.int64))
        epi._accept = lambda **kw: torch.full((graph_bs,), STRIDE, dtype=torch.int32)   # greedy accept, stood in
        epi.begin_static_step(graph_bs, armed=False)                                    # sampled: gate 0
        out = types.SimpleNamespace(next_token_logits=None, hidden_states=g_hidden)
        if capture:      # warm-up (eager), then the capture: no commit runs at capture time
            epi._static_epilogue(out, fb)
            old, scr._capturing = scr._capturing, (lambda: True)
            try:
                epi._static_epilogue(out, fb)
            finally:
                scr._capturing = old
            return
        epi._static_epilogue(out, fb)                     # the replay of the graph's epilogue
        runner.bs, runner.raw_bs = graph_bs, bs
        ex._fake_logits = types.SimpleNamespace(hidden_states=hidden)
        res = ex.run_non_compact(batch=None)
        vw = types.SimpleNamespace(verify_cache_loc=loc[:bs * STRIDE], verify_cache_loc_2d=loc[:bs * STRIDE].view(bs, STRIDE),
                                   positions_2d=pos[:bs * STRIDE].view(bs, STRIDE))
        ex.commit_hidden(batch=None, layout=None, hidden_strided=None, verify_window=vw,
                         logits_output=res.logits_output, commit_lens=commit_lens, bs=bs, run_compact=False)

    projections = lambda: sum(s.main_proj.calls for s in STAGES)  # noqa: E731
    for inplace in (False, True):
        for bs in (1, 3):
            hidden = torch.randn(bs * STRIDE, H).to(torch.bfloat16)
            commit = torch.randint(1, STRIDE + 1, (bs,), dtype=torch.int32)
            os.environ[scr.KEY] = "0"
            p0 = Pool(inplace)
            ex, epi, runner = build(p0)
            step(ex, epi, runner, bs, hidden, commit, capture=True)
            stock_proj = projections()
            step(ex, epi, runner, bs, hidden, commit, capture=False)
            stock_proj = projections() - stock_proj
            assert stock_proj == 2, stock_proj        # epilogue + eager commit each project once
            os.environ[scr.KEY] = "1"
            p1 = Pool(inplace)
            ex, epi, runner = build(p1)
            step(ex, epi, runner, bs, hidden, commit, capture=True)          # the capture records the shape
            before, reuse0 = projections(), scr.STATS["reuse"]
            step(ex, epi, runner, bs, hidden, commit, capture=False)        # a replay + eager sampled commit
            assert projections() - before == 1, "the eager commit recomputed the projection"
            assert scr.STATS["reuse"] == reuse0 + 1
            assert torch.equal(p0.cache.view(torch.int16), p1.cache.view(torch.int16)), "cache bits differ"
            assert int((p1.cache != 0).any(-1).sum()) == S * int(commit.sum())
            # padded replay (graph bs 4 for 3 real rows): the stock eager projection
            if bs == 3:
                p2 = Pool(inplace)
                ex, epi, runner = build(p2)
                step(ex, epi, runner, bs, hidden, commit, capture=True, graph_bs=4)
                stock0 = scr.REASONS.get("padded", 0)
                step(ex, epi, runner, bs, hidden, commit, capture=False, graph_bs=4)
                assert scr.REASONS["padded"] == stock0 + 1
                assert torch.equal(p0.cache.view(torch.int16), p2.cache.view(torch.int16))
            # check mode: stock store, 0 mismatches
            os.environ[scr.KEY] = "check"
            p3 = Pool(inplace)
            ex, epi, runner = build(p3)
            step(ex, epi, runner, bs, hidden, commit, capture=True)
            step(ex, epi, runner, bs, hidden, commit, capture=False)
            assert torch.equal(p0.cache.view(torch.int16), p3.cache.view(torch.int16))
            assert scr._C["counters"][2].item() == 0
            # corrupt the buffer between the replay and the commit: check counts it, the store stays stock
            mism0 = scr._C["counters"][2].item()
            orig_epi = ver.DsparkVerifyEpilogue._static_epilogue

            def corrupting(self, out, fb, _orig=orig_epi):
                _orig(self, out, fb)
                with torch.inference_mode():            # the buffer is born in the commit's inference mode
                    scr._S["buf"][1, 0, 0] += 1.0

            ver.DsparkVerifyEpilogue._static_epilogue = corrupting
            p4 = Pool(inplace)
            ex, epi, runner = build(p4)
            step(ex, epi, runner, bs, hidden, commit, capture=False)
            ver.DsparkVerifyEpilogue._static_epilogue = orig_epi
            assert scr._C["counters"][2].item() == mism0 + 1
            assert torch.equal(p0.cache.view(torch.int16), p4.cache.view(torch.int16))
            scr._C["counters"] = None
            os.environ[scr.KEY] = "1"
    print(f"test_sampled_commit_reuse: engine part ok (verify calls {calls['verify']}, stats {scr.STATS}, "
          f"reasons {scr.REASONS})")


# ----------------------------------------------------------------------------------- part 2: stand-ins only --
def standin_part():
    # eligibility on a minimal executor: every stock reason, then A/B per capture / per replay
    scr._S.update(buf=None, captured=set(), step=None, ctx=None)
    commits = []

    class Epi:
        max_bs, stride = MAX_BS, STRIDE

        def _static_epilogue(self, out, fb):
            commits.append(("epi", scr._S["ctx"]))

    class Ex:
        verify_num_draft_tokens = STRIDE

        def __init__(self):
            self.target_worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(
                decode_cuda_graph_runner=types.SimpleNamespace(bs=2, raw_bs=2)))
            self.can = True

        def run_non_compact(self, **kw):
            return types.SimpleNamespace(can_run_cuda_graph=self.can)

        def commit_hidden(self, **kw):
            commits.append(("commit", scr._S["ctx"]))

    scr.install_verify(types.SimpleNamespace(TargetVerifyExecutor=Ex, DsparkVerifyEpilogue=Epi))
    ex = Ex()
    os.environ[scr.KEY] = "1"
    kw = dict(bs=2, run_compact=False)

    def one(**over):
        commits.clear()
        ex.run_non_compact()
        ex.commit_hidden(**{**kw, **over})
        return commits[-1][1]

    assert one() is None and scr.REASONS["not_captured"] >= 1
    scr._S["captured"].add((None, 2 * STRIDE))
    ctx = one()
    assert ctx == ("reuse", None, 2 * STRIDE, "on"), ctx
    assert one(run_compact=True) is None
    ex.can = False
    assert one() is None and scr.REASONS["eager_verify"] >= 1
    ex.can = True
    ex.target_worker.model_runner.decode_cuda_graph_runner.bs = 4
    assert one() is None
    ex.target_worker.model_runner.decode_cuda_graph_runner.bs = 2
    commits.clear()
    ex.commit_hidden(**kw)                                  # no verify record this step: stock
    assert commits[-1][1] is None
    os.environ[scr.KEY] = "0"
    assert one() is None
    os.environ[scr.KEY] = "1"
    Epi()._static_epilogue(None, None)
    assert commits[-1] == ("epi", ("epilogue", MAX_BS * STRIDE))

    # A/B: v0 off, v1 on (ab_variant reads the gate per capture / per replay)
    import ab_variant as ab
    env = {"DSV41_AB_VARIANTS": "2", "DSV41_AB_V0": f"{scr.KEY}=0", "DSV41_AB_V1": f"{scr.KEY}=1",
           scr.KEY: "0"}
    assert ab.configure(env)
    assert env[scr.KEY] == "1", "union must install the hooks"
    try:
        with ab.capturing(0):
            assert scr.mode() == "off"
        with ab.capturing(1):
            assert scr.mode() == "on"
        scr._S["captured"] = {(1, 2 * STRIDE)}
        ab.set_runtime(0)
        assert one() is None
        ab.set_runtime(1)
        assert one() == ("reuse", 1, 2 * STRIDE, "on")
        ab._MAX_BS = 1                                       # shapes above 1 come from variant 0's set
        assert one() is None
        try:
            ab.configure({"DSV41_AB_VARIANTS": "2", "DSV41_AB_V0": f"{scr.KEY}=1", "DSV41_AB_V1": f"{scr.KEY}=on"})
            raise AssertionError("1 vs on is an A/A and must be refused")
        except RuntimeError:
            pass
        assert ab.configure({"DSV41_AB_VARIANTS": "2", "DSV41_AB_V0": f"{scr.KEY}=1",
                             "DSV41_AB_V1": f"{scr.KEY}=check"})
    finally:
        ab.configure({"DSV41_AB_VARIANTS": "1"})
    try:
        scr.parse_mode("maybe")
        raise AssertionError("bad value accepted")
    except RuntimeError:
        pass
    print("test_sampled_commit_reuse: stand-in part ok")


STAGES = make_stages()

if __name__ == "__main__":
    engine_part()
    standin_part()
    print("test_sampled_commit_reuse: ok")
