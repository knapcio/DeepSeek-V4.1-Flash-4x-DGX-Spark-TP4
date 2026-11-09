"""The two install points, CPU with fakes: prefill_empty_cache.install with and without DSV41_PREFILL_KEEP_CACHE
(empty_cache only after the last chunk; byte-for-byte the old hook when off), ForwardBatch.init_new plumbing, and
mxfp8_b12x.install routing an eligible call through the row split.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=adapter python3 tests/test_prefill_hooks.py   (or pytest)
"""
import importlib
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter"))


class FakeCuda:
    """torch.cuda allocator calls the hooks make, with a scripted allocator."""

    def __init__(self):
        self.reserved = 0
        self.peak = 0
        self.allocs = 0
        self.empties = 0

    def memory_stats(self):
        return {"reserved_bytes.all.current": self.reserved, "allocated_bytes.all.peak": self.peak,
                "num_device_alloc": self.allocs, "num_device_free": 0}

    def empty_cache(self):
        self.empties += 1
        self.reserved = 0

    def reset_peak_memory_stats(self):
        self.peak = 0


def _install_hook(env):
    for k in ("DSV41_PREFILL_KEEP_CACHE", "DSV41_PREFILL_EMPTY_CACHE_TOKENS"):
        os.environ.pop(k, None)
    os.environ.update(env)
    import prefill_keep_cache
    import prefill_empty_cache
    importlib.reload(prefill_keep_cache)
    importlib.reload(prefill_empty_cache)
    fake = FakeCuda()
    cuda = torch.cuda
    saved = {n: getattr(cuda, n) for n in ("memory_stats", "empty_cache", "reset_peak_memory_stats")}
    for n in saved:
        setattr(cuda, n, getattr(fake, n))

    class ModelRunner:
        def forward(self, fb):
            # a chunk: first one after an empty cache mallocs 10 segments, later ones reuse them
            if fake.reserved == 0:
                fake.allocs += 10
                fake.reserved = 1000
            fake.peak = 990
            return "out"

    mod = types.SimpleNamespace(ModelRunner=ModelRunner)
    prefill_empty_cache.install(mod)
    return fake, ModelRunner(), saved


def _restore(saved):
    for n, f in saved.items():
        setattr(torch.cuda, n, f)


def _fb(seq, final, extend=True):
    return types.SimpleNamespace(
        forward_mode=types.SimpleNamespace(is_extend_without_speculative=lambda: extend),
        seq_lens_cpu=torch.tensor([seq]), dsv41_final_lens=None if final is None else [final])


def _prefill(runner, prompt, final=True):
    for seq in list(range(8192, prompt, 8192)) + [prompt]:
        assert runner.forward(_fb(seq, prompt if final else None)) == "out"


def test_old_hook_unchanged_when_off():
    fake, runner, saved = _install_hook({})
    try:
        _prefill(runner, 40000)
        assert fake.empties == 5                     # every long chunk, as before
        runner.forward(_fb(100, 100))                # short extend: nothing
        runner.forward(_fb(9000, 9000, extend=False))  # decode-like: nothing
        assert fake.empties == 5
    finally:
        _restore(saved)


def test_keep_cache_empties_only_at_the_end():
    fake, runner, saved = _install_hook({"DSV41_PREFILL_KEEP_CACHE": "1"})
    try:
        _prefill(runner, 40000)                      # first prefill after boot: emptied before chunk 1 (reference)
        first = fake.empties
        assert first == 2, first                     # before chunk 1 (unknown cache state), end after chunk 5
        allocs = fake.allocs
        _prefill(runner, 40000)
        assert fake.empties == first + 1             # only the end
        assert fake.allocs == allocs + 10            # chunk 1 refills the cache, chunks 2-5 malloc nothing
    finally:
        _restore(saved)


def test_keep_cache_without_final_lengths_is_the_old_hook():
    fake, runner, saved = _install_hook({"DSV41_PREFILL_KEEP_CACHE": "1"})
    try:
        _prefill(runner, 40000, final=False)
        assert fake.empties == 5
    finally:
        _restore(saved)


def test_pad_ok_false_when_bookkeeping_raises():
    """Re-review P2: an exception in the pre-forward bookkeeping must leave the forward at stock widths."""
    fake, runner, saved = _install_hook({"DSV41_PREFILL_KEEP_CACHE": "1"})
    try:
        import prefill_keep_cache as keep
        fb = _fb(8192, 40000)
        fb.dsv41_pad_ok = True
        boom = keep.read_stats

        def raising(torch_):
            raise RuntimeError("stats unavailable")

        keep.read_stats = raising
        try:
            assert runner.forward(fb) == "out"
        finally:
            keep.read_stats = boom
        assert fb.dsv41_pad_ok is False
        ok = _fb(8192, 40000)
        runner.forward(ok)
        assert ok.dsv41_pad_ok is True
        mixed = types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(is_extend_without_speculative=lambda: True),
            seq_lens_cpu=torch.tensor([40000, 4000]), dsv41_final_lens=[40000, 1 << 20])
        runner.forward(mixed)
        assert mixed.dsv41_pad_ok is False
    finally:
        _restore(saved)


def test_forward_batch_plumbing():
    import prefill_keep_cache as keep

    class FB:
        @classmethod
        def init_new(cls, batch, model_runner, **kw):
            fb = cls()
            fb.forward_mode = types.SimpleNamespace(is_extend_without_speculative=lambda: batch.extend)
            return fb

    req = types.SimpleNamespace(rid="r1", extend_range=types.SimpleNamespace(end=8192),
                                origin_input_ids=list(range(20024)))
    mod = types.SimpleNamespace(ForwardBatch=FB)
    assert keep.install_forward_batch(mod)
    fb = FB.init_new(types.SimpleNamespace(reqs=[req], extend=True), types.SimpleNamespace(is_draft_worker=False),
                     return_hidden_states_before_norm=False)
    assert fb.dsv41_final_lens == [20024] and fb.dsv41_rids == ["r1"]
    fb = FB.init_new(types.SimpleNamespace(reqs=[req], extend=False), types.SimpleNamespace(is_draft_worker=False))
    assert not hasattr(fb, "dsv41_final_lens")
    fb = FB.init_new(types.SimpleNamespace(reqs=[req], extend=True), types.SimpleNamespace(is_draft_worker=True))
    assert not hasattr(fb, "dsv41_final_lens")


def test_mxfp8_b12x_routes_eligible_calls_through_the_split():
    import prefill_gemm_split as pgs
    import test_prefill_gemm_split as t
    pgs.AGREE[0] = False
    pgs._DECIDED.clear()
    os.environ["DSV41_PREFILL_GEMM_ROW_SPLIT"] = "4096"
    os.environ["DSV41_PREFILL_GEMM_ROW_SPLIT_N"] = "all"
    try:
        calls = []
        mod, original = t.make_fp8_utils(calls=calls)
        fp8_utils = types.SimpleNamespace(
            flashinfer_mxfp8_blockscaled_linear=original, block_fp8_scale_to_mxfp8_e8m0=lambda *a: None,
            _raw_flashinfer_mm_mxfp8=mod._raw_flashinfer_mm_mxfp8,
            flashinfer_mxfp8_quantize=mod.flashinfer_mxfp8_quantize)
        import mxfp8_b12x
        importlib.reload(mxfp8_b12x)
        mxfp8_b12x.install(fp8_utils)
        linear = fp8_utils.flashinfer_mxfp8_blockscaled_linear
        assert linear is not original
        x, w, ws = t._inputs()
        ref = original(x, w, ws, None, None, torch.float32)
        assert torch.equal(linear(x, w, ws, None, None, torch.float32, backend="cutlass", pin_tactic=True), ref)
        calls.clear()
        assert torch.equal(linear(x, w, ws, None, None, torch.float32, backend="cutlass", pin_tactic=True), ref)
        assert calls == [4096, 4096], calls
        xs, wsm, wss = t._inputs(m=64)
        calls.clear()
        linear(xs, wsm, wss, None, None, torch.float32, backend="cutlass", pin_tactic=True)
        assert calls == [64], calls                  # decode-sized: the stock single launch
    finally:
        os.environ.pop("DSV41_PREFILL_GEMM_ROW_SPLIT")
        os.environ.pop("DSV41_PREFILL_GEMM_ROW_SPLIT_N", None)


class _Torch:
    def __init__(self, fake):
        self.cuda = fake


def _holding_policy(keep, fake):
    """keep.policy() holding a session for request 'a' (two kept chunks)."""
    p = keep.policy()
    p.emptied(keep.read_stats(_Torch(fake)))
    for seq in (8192, 16384):
        b = keep.Batch(["a"], [100000], [seq], [8192])
        st = keep.read_stats(_Torch(fake))
        p.begin(b, st)
        p.before(st)
        fake.reserved, fake.peak = 1000, 990
        assert p.after(b, keep.read_stats(_Torch(fake))) == (False, "keep")
    assert p.holding
    return p


def _scheduler_module():
    class Scheduler:
        def __init__(self):
            self.chunked_req = None
            self.next_batch = None

        def get_next_batch_to_run(self, running_batch=None, last_batch=None):
            return types.SimpleNamespace(batch_to_run=self.next_batch, running_batch=running_batch)

        def process_pending_chunked_abort(self):
            self.chunked_req = None

        def pause_generation(self, recv_req=None):
            self.chunked_req = None

        def on_idle(self):
            pass

    return types.SimpleNamespace(Scheduler=Scheduler)


def test_scheduler_releases_a_held_cache_when_its_prompt_cannot_continue():
    """P0-1 part 2: abort / pause / idle / an orphaned session (chunked request gone) empty the held cache."""
    import prefill_keep_cache as keep
    importlib.reload(keep)
    fake = FakeCuda()
    mod = _scheduler_module()
    assert keep.install_scheduler(mod, torch=_Torch(fake))
    sch = mod.Scheduler()
    req_a = types.SimpleNamespace(rid="a")
    for action in ("next-other", "abort", "pause", "idle"):
        importlib.reload(keep)
        keep.install_scheduler(mod := _scheduler_module(), torch=_Torch(fake))
        sch = mod.Scheduler()
        _holding_policy(keep, fake)
        fake.empties = 0
        # still owned: the chunked request is the session's, or the batch about to run carries it
        sch.chunked_req = req_a
        sch.get_next_batch_to_run()
        sch.chunked_req, sch.next_batch = None, types.SimpleNamespace(reqs=[req_a])
        sch.get_next_batch_to_run()
        assert fake.empties == 0 and keep.policy().holding
        if action == "next-other":
            sch.next_batch = types.SimpleNamespace(reqs=[types.SimpleNamespace(rid="b")])
            sch.get_next_batch_to_run()
        elif action == "abort":
            sch.chunked_req = req_a
            sch.process_pending_chunked_abort()
        elif action == "pause":
            sch.chunked_req = req_a
            sch.pause_generation()
        else:
            sch.on_idle()
        assert fake.empties == 1 and not keep.policy().holding, action
        sch.on_idle()
        assert fake.empties == 1                     # nothing held: no second empty


def test_on_idle_between_chunks_of_the_held_prompt_keeps_it():
    """Fleet run-20261109-011349: on_idle ran between the chunks of one prefill and released after every chunk
    (737 / 739 receipts released_by=idle). on_idle now releases only when the held prompt cannot continue."""
    import prefill_keep_cache as keep
    importlib.reload(keep)
    fake = FakeCuda()
    mod = _scheduler_module()
    keep.install_scheduler(mod, torch=_Torch(fake))
    sch = mod.Scheduler()
    _holding_policy(keep, fake)
    fake.empties = 0
    sch.chunked_req = types.SimpleNamespace(rid="a")     # the prefill in progress
    sch.on_idle()
    sch.on_idle()
    assert fake.empties == 0 and keep.policy().holding
    sch.chunked_req = types.SimpleNamespace(rid="other")
    sch.on_idle()
    assert fake.empties == 1 and not keep.policy().holding


def _decode_fb():
    return types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_extend_without_speculative=lambda: False),
                                 seq_lens_cpu=torch.tensor([9000]))


def _fb2(seq, final, prev):
    fb = _fb(seq, final)
    fb.dsv41_rids = ["p%d" % final]
    fb.extend_seq_lens_cpu = [seq - prev]
    return fb


def test_decode_between_chunks_stops_keeping_that_prompt():
    fake, runner, saved = _install_hook({"DSV41_PREFILL_KEEP_CACHE": "1"})
    try:
        import prefill_keep_cache as keep
        seqs = [8192, 16384, 24576, 32768, 40000]
        prev = 0
        for i, seq in enumerate(seqs[:2]):
            runner.forward(_fb2(seq, 40000, prev))
            prev = seq
        assert keep.policy().holding
        fake.empties = 0
        runner.forward(_decode_fb())                     # a decode step between chunks: release now
        assert fake.empties == 1 and not keep.policy().holding
        runner.forward(_decode_fb())                     # nothing held: no second empty
        assert fake.empties == 1
        pads = []
        for seq in seqs[2:]:
            fb = _fb2(seq, 40000, prev)
            runner.forward(fb)
            pads.append(fb.dsv41_pad_ok)
            prev = seq
            assert not keep.policy().holding             # the rest of the prompt runs as today
        assert fake.empties == 1 + 3 and pads == [False, False, False], (fake.empties, pads)
        # the next prompt (no decode between its chunks) keeps again
        prev = 0
        for seq in (8192, 16384):
            runner.forward(_fb2(seq, 30000, prev))
            prev = seq
        assert keep.policy().holding
        # a draft-model decode forward does not count
        runner.is_draft_worker = True
        runner.forward(_decode_fb())
        assert keep.policy().holding
    finally:
        _restore(saved)


def test_gate_off_releases_and_armed_ab_without_the_key_installs_nothing():
    import prefill_keep_cache as keep
    importlib.reload(keep)
    fake_ab = types.SimpleNamespace(ACTIVE=True, KNOWN=frozenset({"DSV41_PREFILL_KEEP_CACHE"}),
                                    _specs=[{}, {"DSV41_L2_PREFETCH_AHEAD": "1"}], env=lambda n, d=None: d)
    saved = keep._ab
    keep._ab = fake_ab
    try:
        os.environ.pop("DSV41_PREFILL_KEEP_CACHE", None)
        assert not keep.wanted_at_install()
        fake_ab._specs = [{"DSV41_PREFILL_KEEP_CACHE": "0"}, {"DSV41_PREFILL_KEEP_CACHE": "1"}]
        assert keep.wanted_at_install()
    finally:
        keep._ab = saved
    fake, runner, saved_cuda = _install_hook({"DSV41_PREFILL_KEEP_CACHE": "1"})
    try:
        import prefill_keep_cache as keep2
        _holding_policy(keep2, fake)
        fake.empties = 0
        os.environ["DSV41_PREFILL_KEEP_CACHE"] = "0"   # what an A/B switch to the off variant reads
        runner.forward(_fb(100, 100, extend=False))
        assert fake.empties == 1 and not keep2.policy().holding
    finally:
        os.environ.pop("DSV41_PREFILL_KEEP_CACHE", None)
        _restore(saved_cuda)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[OK] {name}", flush=True)
    print("all passed")
