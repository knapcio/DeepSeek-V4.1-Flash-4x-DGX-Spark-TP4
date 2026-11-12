"""DSV41_GREEDY_NOGATHER x DSV41_EAGER_GLUE=vcap: the greedy draft flavour refreshes the verify_cap live length
(CPU, torch, no GPU, no engine):

  PYTHONPATH=adapter python3 tests/test_greedy_nogather_vcap.py
  docker run --rm --network none -v <ds41>:/ds41:ro -e PYTHONPATH=/ds41/adapter \
      --entrypoint python3 <image> /ds41/tests/test_greedy_nogather_vcap.py

eager_glue's vcap part captures verify_cap's live-length update into the draft graph at the padding-free sizes
(decode bs 1..8 here) and then skips the eager update on steps that replayed such a graph. The greedy flavour is a
second draft graph of the same size, so it must contain the update too: the sampler wrappers are installed through
adapter/sitecustomize.py's own dspark_draft_sampler branch, and for every bs 1..8 a run of greedy-flavour steps
with changing draft confidence must leave the live buffer at the lengths that step's confidence gives, with the
eager update skipped (with conf:0.1 and with the production table conf:0.1/4:0.25/8:0.1). The tier-12o window Q (2026-11-11) ran the flavour body outside eager_glue's wrapper: the live
buffer was never written on flavour steps and verify_cap kept a stale cap of 2 drafts.
"""
import importlib
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(HERE, "..", "adapter")
sys.path.insert(0, ADAPTER)

DECODE_BS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
GAMMA, QTN, MAX_BS = 5, 6, 64
SAMPLER_MODULE = "sglang.srt.speculative.dspark_components.dspark_draft_sampler"


def setup_env(spec="conf:0.1"):
    for k in [k for k in os.environ if k.startswith(("DSV41_AB", "DSV41_SPEC_SYNC_FREE", "DSV41_DRAFT_TAU",
                                                     "DSV41_SOURCE"))]:
        del os.environ[k]
    os.environ.update(DSV41_EAGER_GLUE="all", DSV41_VERIFY_CAP=spec, DSV41_GREEDY_NOGATHER="1")
    for m in ("sitecustomize", "eager_glue", "verify_cap", "vcap_spec", "greedy_nogather"):
        sys.modules.pop(m, None)
    rc = types.ModuleType("sglang.srt.runtime_context")
    rc.get_exec = lambda: types.SimpleNamespace(graph=types.SimpleNamespace(
        cuda_graph_config=types.SimpleNamespace(decode=types.SimpleNamespace(bs=DECODE_BS))))
    for name in ("sglang", "sglang.srt"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["sglang.srt.runtime_context"] = rc
    vc = importlib.import_module("verify_cap")
    vc._state["tp_group"] = types.SimpleNamespace(world_size=1, broadcast=lambda t, src=0: t)
    vc.live_buf(torch.device("cpu"))
    return vc


CONF = {"t": None}      # the draft confidence the next sampler call produces ([bs, 5])


class Head:
    supports_sharded_greedy = True
    greedy_calls = 0

    def sample_block_greedy_fused(self, base_logits, *, first_prev_tokens):
        Head.greedy_calls += 1
        return base_logits.argmax(-1)


class Model:
    def compute_base_logits(self, hidden):
        return torch.zeros(hidden.shape[0], 8), None


def make_sampler_module():
    stock = []

    class DsparkDraftSampler:
        def __init__(self):
            self.model, self.markov_head = Model(), Head()
            self.gamma, self.query_token_num, self.sample_from_anchor = GAMMA, QTN, False
            self.folded_sampling, self.exp_noise = True, None
            self.out = torch.zeros(MAX_BS * GAMMA, dtype=torch.int64)
            self.confidence_out = torch.zeros(MAX_BS, GAMMA)
            self.confidence_fn = lambda **kw: CONF["t"][: kw["draft_tokens"].shape[0]]

        def __call__(self, hidden_states, input_ids):     # the stock folded body
            stock.append(1)
            bs = hidden_states.shape[0] // self.query_token_num
            self.confidence_out[:bs].copy_(CONF["t"][:bs])

        def stage_sampling_params(self, *, bs, sampling_info):
            pass

    def select(h, bs, gamma):
        return h[: bs * gamma], h.view(bs, QTN, -1)[:, 1:]

    mod = types.ModuleType(SAMPLER_MODULE)
    mod.DsparkDraftSampler, mod.select_draft_hidden_without_anchor = DsparkDraftSampler, select
    return mod, stock


def install_via_sitecustomize(mod):
    """adapter/sitecustomize.py's own branch for the sampler module, in its own order."""
    site = importlib.import_module("sitecustomize")
    loader = site.EngramLoader(types.SimpleNamespace(exec_module=lambda m: None))
    loader.exec_module(mod)


def expected_live(conf, kmin=1, thr=0.1):
    cum = torch.cumprod(conf.float().clamp(0, 1), dim=1)
    return (cum >= thr).to(torch.int64).cumprod(dim=1).sum(dim=1).clamp(min=kmin, max=GAMMA) + 1


class capturing:
    """torch.cuda as seen inside a CUDA graph capture (CPU tensors: eager_glue takes the torch path)."""

    def __enter__(self):
        self.saved = torch.cuda.is_available, torch.cuda.is_current_stream_capturing
        torch.cuda.is_available, torch.cuda.is_current_stream_capturing = (lambda: True), (lambda: True)

    def __exit__(self, *exc):
        torch.cuda.is_available, torch.cuda.is_current_stream_capturing = self.saved
        return False


PROD_SPEC = "conf:0.1/4:0.25/8:0.1"        # the production profile's DSV41_VERIFY_CAP (v2.7.5)


def prod_thr(bs):
    return 0.25 if 4 <= bs <= 7 else 0.1


def test_greedy_flavour_refreshes_live_length(spec="conf:0.1", thr_of=lambda bs: 0.1):
    vc = setup_env(spec)
    mod, stock = make_sampler_module()
    install_via_sitecustomize(mod)
    eg, gn = sys.modules["eager_glue"], sys.modules["greedy_nogather"]
    assert eg._eligible_sizes() == frozenset(range(1, 9))

    class DraftBlockProposer:          # the draft runner: replays the stock graph or the greedy flavour
        def __init__(self, sampler):
            self.sampler = sampler

        def propose(self, *, bs, greedy):
            gn._state["flavour"] = greedy
            try:
                self.sampler(torch.zeros(bs * QTN, 4), torch.zeros(bs * QTN, dtype=torch.int64))
            finally:
                gn._state["flavour"] = False
            return types.SimpleNamespace(folded=True, confidence=self.sampler.confidence_out[:bs])

    eg.install_draft(types.SimpleNamespace(DraftBlockProposer=DraftBlockProposer))
    s = mod.DsparkDraftSampler()
    assert gn._state["eligible"] is True
    p = DraftBlockProposer(s)
    g = torch.Generator().manual_seed(7)
    for bs in range(1, 9):
        # graph capture at this size: the stock graph, then its greedy flavour (FullCudaGraphBackend order)
        CONF["t"] = torch.rand(MAX_BS, GAMMA, generator=g)
        with capturing():
            for greedy in (False, True):
                p.propose(bs=bs, greedy=greedy)
        assert (None, bs) in eg._vcap["captured"]
        # serving: the live buffer starts at a stale cap of 2 drafts; every all-greedy step's verify must see the
        # lengths its own confidence gives (high, low, mixed, random)
        vc.live_buf()[:].fill_(3)
        steps = [torch.full((MAX_BS, GAMMA), 0.9), torch.full((MAX_BS, GAMMA), 0.05),
                 torch.tensor([[0.9, 0.5, 0.2, 0.9, 0.9]] * MAX_BS)] + [
            torch.rand(MAX_BS, GAMMA, generator=g) ** 0.3 for _ in range(4)]
        for i, conf in enumerate(steps):
            CONF["t"] = conf
            greedy_before, stock_before, skip_before = Head.greedy_calls, len(stock), eg._stats["vcap_skip"]
            out = p.propose(bs=bs, greedy=True)
            assert Head.greedy_calls == greedy_before + 1 and len(stock) == stock_before, "flavour body not taken"
            vc.set_live_from_confidence(out.confidence, bs)          # the verify step's call
            assert eg._stats["vcap_skip"] == skip_before + 1, (bs, i, "eager update not skipped")
            got, want = vc.live_buf()[:bs], expected_live(conf[:bs], thr=thr_of(bs))
            assert torch.equal(got, want), (
                f"bs={bs} step {i}: verify_cap live length {got.tolist()} != {want.tolist()} "
                "(greedy flavour did not run the live-length update)")
        # a sampled step after the greedy run: the stock graph, same rule
        CONF["t"] = torch.full((MAX_BS, GAMMA), 0.9)
        vc.live_buf()[:].fill_(3)
        out = p.propose(bs=bs, greedy=False)
        vc.set_live_from_confidence(out.confidence, bs)
        assert torch.equal(vc.live_buf()[:bs], torch.full((bs,), 6, dtype=torch.int64)), bs


def test_greedy_flavour_production_thresholds():
    """The same with the production profile's per-batch-size table (0.25 at bs 4-7, 0.1 elsewhere): the flavour's
    captured update reads the captured size's threshold, as the stock graph's does."""
    test_greedy_flavour_refreshes_live_length(PROD_SPEC, prod_thr)


def test_wrapper_order():
    """eager_glue's vcap wrapper is outside the greedy flavour body (sitecustomize order)."""
    setup_env()
    mod, _ = make_sampler_module()
    install_via_sitecustomize(mod)
    call = mod.DsparkDraftSampler.__call__
    assert getattr(call, "__module__", "") == "eager_glue", call.__module__
    inner = [c.cell_contents for c in (call.__closure__ or ()) if callable(c.cell_contents)
             and getattr(c.cell_contents, "__module__", "") == "greedy_nogather"]
    assert inner, "eager_glue's vcap wrapper does not wrap greedy_nogather's flavour body"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("greedy_nogather x eager_glue vcap OK")
