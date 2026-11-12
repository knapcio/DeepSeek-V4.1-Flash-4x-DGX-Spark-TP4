"""Pure-Python checks for adapter/greedy_nogather.py (no torch, no GPU; runs on the Mac too):

1. off by default: nothing installs, the stock sampler / backend are untouched;
2. graph flavours: the draft runner captures every shape once more as FlavourKey(shape) with the flavour flag set
   (the warm-up reset in between), the target runner never; replay takes the flavour only on an all-greedy step of
   while the draft part is on; an ineligible head captures nothing extra;
3. the sampler: inside a flavour capture the sharded greedy body runs (out, confidence, the spec_sync_free noise
   counter advanced once), outside it (or with the part off) the stock call;
4. every wrapper keeps __wrapped__ (inspect.unwrap reaches the engine's function; tier-6 FIX-20261110) and a second
   install keeps the one wrapper.
  python3 tests/test_greedy_nogather.py
"""
import importlib.util
import os
import sys
import types

ADAPTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter")
sys.path.insert(0, ADAPTER)
for k in [k for k in os.environ if k.startswith(("DSV41_AB", "DSV41_GREEDY_NOGATHER"))]:
    del os.environ[k]


def fresh(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ADAPTER, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_backend_module():
    calls = []

    class FullCudaGraphBackend:
        def __init__(self, draft):
            self._graphs, self._outputs = {}, {}
            self._cuda_graph_runner = types.SimpleNamespace(model_runner=types.SimpleNamespace(is_draft_worker=draft))

        def capture_one(self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None):
            out = forward_fn()
            self._graphs[shape_key] = ("graph", out)
            self._outputs[shape_key] = out

        def replay(self, shape_key, static_forward_batch, **kwargs):
            calls.append(shape_key)
            return self._graphs[shape_key][1]

        def cleanup(self):
            self._graphs.clear()
            self._outputs.clear()

    return types.SimpleNamespace(FullCudaGraphBackend=FullCudaGraphBackend), calls


class Key:
    def __init__(self, size):
        self.size = size

    def __hash__(self):
        return hash(("key", self.size))

    def __eq__(self, other):
        return isinstance(other, Key) and other.size == self.size


def test_flavour_key():
    gn = fresh("gn_t0", "greedy_nogather.py")
    k = Key(4)
    f = gn.FlavourKey(k)
    assert f.size == 4 and f != k and k != f and f == gn.FlavourKey(Key(4)) and hash(f) == hash(gn.FlavourKey(Key(4)))
    d = {k: 1, f: 2}
    assert d[k] == 1 and d[gn.FlavourKey(Key(4))] == 2
    assert gn._parse("1") == {"draft", "target"} and gn._parse("draft") == {"draft"} and gn._parse("0") == frozenset()
    assert gn._parse("target,draft") == {"draft", "target"}
    try:
        gn._parse("logits")
        raise AssertionError("unknown part accepted")
    except RuntimeError:
        pass


def test_off_by_default():
    gn = fresh("gn_t1", "greedy_nogather.py")
    called = []
    cls = type("DsparkDraftSampler", (), {"__call__": lambda self, h, i: called.append("stock"),
                                          "__init__": lambda self: None})
    mod = types.SimpleNamespace(DsparkDraftSampler=cls, select_draft_hidden_without_anchor=None)
    orig = cls.__call__
    gn.install_sampler(mod)
    prop = types.SimpleNamespace(DraftBlockProposer=type("P", (), {"propose": lambda self, **kw: None}))
    gn.install_proposer(prop)
    assert cls.__call__ is orig and not getattr(prop.DraftBlockProposer, "_dsv41_greedy_nogather", False)


def test_backend_flavours():
    os.environ["DSV41_GREEDY_NOGATHER"] = "draft"
    try:
        gn = fresh("gn_t2", "greedy_nogather.py")
        mod, calls = fake_backend_module()
        gn.install_backend(mod)
        gn._state["eligible"] = True
        seen, resets = [], []

        def fwd():
            seen.append(gn._state["flavour"])
            return "flavour" if gn._state["flavour"] else "stock"

        draft, target = mod.FullCudaGraphBackend(True), mod.FullCudaGraphBackend(False)
        for size in (1, 2):
            draft.capture_one(Key(size), fwd, None, lambda: resets.append(1))
        target.capture_one(Key(1), fwd, None, lambda: resets.append(1))
        assert seen == [False, True, False, True, False], seen
        assert len(resets) == 2 and len(draft._graphs) == 4 and len(target._graphs) == 1
        # replay: flavour only on an all-greedy step with the part on
        assert draft.replay(Key(1), None) == "stock"
        gn._state["greedy"] = True
        assert draft.replay(Key(1), None) == "flavour"
        assert target.replay(Key(1), None) == "stock"
        os.environ["DSV41_GREEDY_NOGATHER"] = "0"     # the part read per replay
        assert draft.replay(Key(2), None) == "stock"
        os.environ["DSV41_GREEDY_NOGATHER"] = "draft"
        assert draft.replay(Key(2), None) == "flavour"
        gn._state["greedy"] = False
        assert gn.stats()[:3] == (2, 2, 2), gn.stats()
        # an ineligible head (no sharded greedy support): no extra capture
        gn._state["eligible"] = False
        other = mod.FullCudaGraphBackend(True)
        other.capture_one(Key(8), fwd, None, None)
        assert len(other._graphs) == 1
        other.cleanup()
        draft.cleanup()
        assert "_dsv41_greedy_keys" not in draft.__dict__ and not draft._graphs
    finally:
        del os.environ["DSV41_GREEDY_NOGATHER"]


class FakeTensor:
    """Just enough of a tensor for the sampler body: view / slicing / copy_ / numel / reshape / add_."""

    def __init__(self, data, shape=None):
        self.data = list(data)
        self.shape = tuple(shape) if shape is not None else (len(self.data),)

    def view(self, *shape):
        if -1 in shape:
            known = 1
            for d in shape:
                known *= d if d != -1 else 1
            shape = tuple(len(self.data) // known if d == -1 else d for d in shape)
        return FakeTensor(self.data, shape)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):            # [:, 0] on a 2-D view
            rows, cols = self.shape
            return FakeTensor([self.data[r * cols] for r in range(rows)])
        if isinstance(idx, slice):
            t = FakeTensor(self.data[idx])
            t.parent = self
            return t
        return self.data[idx]

    def numel(self):
        return len(self.data)

    def reshape(self, *shape):
        return FakeTensor(self.data, (len(self.data),))

    def copy_(self, other):
        self.copied = list(other.data)
        if getattr(self, "parent", None) is not None:
            self.parent.copied = self.copied
        return self

    def add_(self, v):
        self.data = [x + v for x in self.data]
        return self


def test_sampler_body():
    os.environ["DSV41_GREEDY_NOGATHER"] = "1"
    try:
        gn = fresh("gn_t3", "greedy_nogather.py")
        stock = []

        class Head:
            supports_sharded_greedy = True

            def __init__(self):
                self.calls = []

            def sample_block_greedy_fused(self, base_logits, *, first_prev_tokens):
                self.calls.append((base_logits.shape, list(first_prev_tokens.data)))
                return FakeTensor([7, 8, 9, 7, 8, 9], (2, 3))

        class Model:
            sample_from_anchor = False

            def compute_base_logits(self, hidden):
                return FakeTensor([0.0] * 2 * 3 * 5), "tap"

        class DsparkDraftSampler:
            def __init__(self):
                self.model, self.markov_head = Model(), Head()
                self.gamma, self.query_token_num, self.sample_from_anchor = 3, 4, False
                self.folded_sampling = True
                self.out = FakeTensor([0] * 6)
                self.exp_noise = types.SimpleNamespace(ctr=FakeTensor([5]))
                self.confidence_fn = lambda **kw: FakeTensor([0.5, 0.25])
                self.confidence_out = FakeTensor([0.0, 0.0])

            def __call__(self, hidden_states, input_ids):
                stock.append(1)

        def select(h, bs, gamma):
            return "model_hidden", "sample_hidden"

        mod = types.SimpleNamespace(DsparkDraftSampler=DsparkDraftSampler, select_draft_hidden_without_anchor=select)
        gn.install_sampler(mod)
        s = DsparkDraftSampler()
        assert gn._state["eligible"] is True
        hidden = FakeTensor([0] * 8, (8, 1))
        ids = FakeTensor([11, 1, 1, 1, 22, 2, 2, 2])
        s(hidden, ids)                                   # not inside a flavour capture: stock
        assert stock == [1]
        gn._state["flavour"] = True
        s(hidden, ids)
        assert stock == [1] and s.markov_head.calls == [((2, 3, 5), [11, 22])]
        assert s.out.copied == [7, 8, 9, 7, 8, 9] and s.confidence_out.copied == [0.5, 0.25]
        assert s.exp_noise.ctr.data == [6], s.exp_noise.ctr.data      # advanced once, as the stock graph does
        os.environ["DSV41_GREEDY_NOGATHER"] = "0"        # the part off at capture: the flavour captures stock
        s(hidden, ids)
        assert stock == [1, 1]
        gn._state["flavour"] = False
        # a head that refuses (None) falls back to the stock body inside the flavour
        os.environ["DSV41_GREEDY_NOGATHER"] = "1"
        s.markov_head.sample_block_greedy_fused = lambda *a, **k: None
        gn._state["flavour"] = True
        s(hidden, ids)
        assert stock == [1, 1, 1] and gn.stats()[3] == 1
        gn._state["flavour"] = False
    finally:
        os.environ.pop("DSV41_GREEDY_NOGATHER", None)


def test_target_part():
    """Target runner: a flavour per shape (when the logits hook is installed), replayed only on steps whose static
    epilogue is armed without logprobs; the flavour head hands pair ids to the accept and returns the unwritten
    buffer; outside the flavour (or with the part off) both hooks are the stock functions."""
    os.environ["DSV41_GREEDY_NOGATHER"] = "target"
    try:
        gn = fresh("gn_t6", "greedy_nogather.py")
        mod, _ = fake_backend_module()
        gn.install_backend(mod)
        seen = []

        def fwd():
            seen.append((gn._state["flavour"], gn._state["tflavour"]))
            return "tflavour" if gn._state["tflavour"] else "stock"

        tgt = mod.FullCudaGraphBackend(False)
        tgt.capture_one(Key(1), fwd, None, None)
        assert seen == [(False, False)] and len(tgt._graphs) == 1     # no logits hook yet: no flavour
        # the logits hook
        class Fm:
            def __init__(self, tv):
                self.tv = tv

            def is_target_verify(self):
                return self.tv

        class LogitsProcessor:
            vocab_size, logit_scale, final_logit_softcapping = 10, None, None
            do_tensor_parallel_all_gather, use_attn_tp_group, do_tensor_parallel_all_gather_dp_attn = True, False, False

            def _get_logits(self, hidden_states, lm_head, logits_metadata, embedding_bias=None, use_logits_buffer=True):
                return "stock-logits"

            def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None):
                return "local"

        gn.install_logits(types.SimpleNamespace(LogitsProcessor=LogitsProcessor))
        assert gn._state["target_ok"]
        got = []
        gn.pair_argmax = lambda local, **kw: got.append((local, kw["vocab_size"])) or FakeTensor([3, 4], (2,))
        sys.modules["sglang.srt.distributed"] = types.SimpleNamespace(
            get_tp_group=lambda: types.SimpleNamespace(rank_in_group=0, world_size=4, all_gather=None))
        buf = types.SimpleNamespace(shape=(2, 10))
        md = types.SimpleNamespace(forward_mode=Fm(True), next_token_logits_buffer=buf)
        lp = LogitsProcessor()
        hs = types.SimpleNamespace(shape=(2, 5))
        assert lp._get_logits(hs, None, md) == "stock-logits"           # not in a flavour capture
        gn._state["tflavour"] = True
        assert lp._get_logits(hs, None, md) is buf and got == [("local", 10)] and gn._state["ids"].data == [3, 4]
        # the accept takes the ids once, inside the flavour; the stock argmax otherwise
        acc = types.SimpleNamespace(_row_argmax=lambda logits: "stock-argmax")
        gn.install_accept(acc)
        assert acc._row_argmax(types.SimpleNamespace(shape=(2, 10))).data == [3, 4] and gn._state["ids"] is None
        assert acc._row_argmax(types.SimpleNamespace(shape=(2, 10))) == "stock-argmax"
        md.forward_mode = Fm(False)
        assert lp._get_logits(hs, None, md) == "stock-logits"           # not a verify forward
        md.forward_mode, md.next_token_logits_buffer = Fm(True), None
        assert lp._get_logits(hs, None, md) == "stock-logits" and gn.tstats()[3] == 1   # no buffer: stock head
        gn._state["tflavour"] = False
        # capture: now the target runner gets its flavour
        tgt2 = mod.FullCudaGraphBackend(False)
        tgt2.capture_one(Key(2), fwd, None, None)
        assert len(tgt2._graphs) == 2 and seen[-1] == (False, True)
        # selection: begin_static_step(armed) right before run_non_compact; logprob steps never
        calls = []

        class DsparkVerifyEpilogue:
            folds_commit = True

            def begin_static_step(self, bs, armed):
                calls.append(("begin", armed))

        class TargetVerifyExecutor:
            verify_epilogue = DsparkVerifyEpilogue()

            def run_non_compact(self, *, batch, **kw):
                return tgt2.replay(Key(2), None)

        gn.install_verify(types.SimpleNamespace(DsparkVerifyEpilogue=DsparkVerifyEpilogue,
                                                TargetVerifyExecutor=TargetVerifyExecutor))
        ep, ex = DsparkVerifyEpilogue(), TargetVerifyExecutor()
        ep.begin_static_step(2, True)
        assert ex.run_non_compact(batch=types.SimpleNamespace(return_logprob=False)) == "tflavour"
        ep.begin_static_step(2, False)
        assert ex.run_non_compact(batch=types.SimpleNamespace(return_logprob=False)) == "stock"
        ep.begin_static_step(2, True)
        assert ex.run_non_compact(batch=types.SimpleNamespace(return_logprob=True)) == "stock"
        assert ex.run_non_compact(batch=types.SimpleNamespace(return_logprob=False)) == "stock"   # no static step
        # draft capture armed without a folded commit: stock
        sys.modules["draft_capture"] = types.SimpleNamespace(_S={"armed": True})
        DsparkVerifyEpilogue.folds_commit = False
        ep.begin_static_step(2, True)
        assert ex.run_non_compact(batch=types.SimpleNamespace(return_logprob=False)) == "stock"
        DsparkVerifyEpilogue.folds_commit = True
        ep.begin_static_step(2, True)
        assert ex.run_non_compact(batch=types.SimpleNamespace(return_logprob=False)) == "tflavour"
        os.environ["DSV41_GREEDY_NOGATHER"] = "draft"                  # the draft part on, the target part off
        ep.begin_static_step(2, True)
        assert ex.run_non_compact(batch=types.SimpleNamespace(return_logprob=False)) == "stock"
        assert gn.tstats()[:3] == (5, 2, 1), gn.tstats()
    finally:
        os.environ.pop("DSV41_GREEDY_NOGATHER", None)
        sys.modules.pop("draft_capture", None)
        sys.modules.pop("sglang.srt.distributed", None)


def test_proposer_marks_greedy():
    os.environ["DSV41_GREEDY_NOGATHER"] = "1"
    try:
        gn = fresh("gn_t4", "greedy_nogather.py")
        mod, _ = fake_backend_module()
        gn._engine_backend = lambda: mod               # the real module needs SGLang
        seen = []

        class DraftBlockProposer:
            def propose(self, *, batch, sampling_info):
                seen.append(gn._state["greedy"])

        gn.install_proposer(types.SimpleNamespace(DraftBlockProposer=DraftBlockProposer))
        p = DraftBlockProposer()
        p.propose(batch=None, sampling_info=None)
        p.propose(batch=None, sampling_info=types.SimpleNamespace(is_all_greedy=False))
        p.propose(batch=None, sampling_info=types.SimpleNamespace(is_all_greedy=True))
        assert seen == [True, False, True] and gn._state["greedy"] is False
        assert getattr(mod.FullCudaGraphBackend, "_dsv41_greedy_nogather", False)
    finally:
        del os.environ["DSV41_GREEDY_NOGATHER"]


def test_wrappers_unwrap_to_engine():
    import inspect
    os.environ["DSV41_GREEDY_NOGATHER"] = "1"
    try:
        gn = fresh("gn_t8", "greedy_nogather.py")
        mod, _ = fake_backend_module()
        gn._engine_backend = lambda: mod

        def stock(name):
            def fn(*args, **kwargs):
                return name
            fn.__name__ = fn.__qualname__ = name
            return fn

        def cls(name, *methods):
            return type(name, (), {m: stock(f"{name}.{m}") for m in methods})

        sampler = types.SimpleNamespace(DsparkDraftSampler=cls("DsparkDraftSampler", "__init__", "__call__"),
                                        select_draft_hidden_without_anchor=None)
        proposer = types.SimpleNamespace(DraftBlockProposer=cls("DraftBlockProposer", "propose"))
        logits = types.SimpleNamespace(LogitsProcessor=cls("LogitsProcessor", "_get_logits"))
        accept = types.SimpleNamespace(_row_argmax=stock("_row_argmax"))
        verify = types.SimpleNamespace(DsparkVerifyEpilogue=cls("DsparkVerifyEpilogue", "begin_static_step"),
                                       TargetVerifyExecutor=cls("TargetVerifyExecutor", "run_non_compact"))
        fcgb = mod.FullCudaGraphBackend
        targets = [(sampler.DsparkDraftSampler, "__init__"), (sampler.DsparkDraftSampler, "__call__"),
                   (proposer.DraftBlockProposer, "propose"), (logits.LogitsProcessor, "_get_logits"),
                   (accept, "_row_argmax"), (verify.DsparkVerifyEpilogue, "begin_static_step"),
                   (verify.TargetVerifyExecutor, "run_non_compact"), (fcgb, "capture_one"), (fcgb, "replay"),
                   (fcgb, "cleanup")]
        before = [getattr(o, a) if not isinstance(o, type) else vars(o)[a] for o, a in targets]

        def install():
            gn.install_sampler(sampler)
            gn.install_proposer(proposer)
            gn.install_logits(logits)
            gn.install_accept(accept)
            gn.install_verify(verify)

        install()
        after = [getattr(o, a) if not isinstance(o, type) else vars(o)[a] for o, a in targets]
        install()                                            # idempotent: no wrapper of a wrapper
        assert after == [getattr(o, a) if not isinstance(o, type) else vars(o)[a] for o, a in targets]
        for (o, a), b0, w in zip(targets, before, after):
            assert w is not b0, a
            assert inspect.unwrap(w) is b0, f"{a} does not unwrap to the engine's function"
    finally:
        del os.environ["DSV41_GREEDY_NOGATHER"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
