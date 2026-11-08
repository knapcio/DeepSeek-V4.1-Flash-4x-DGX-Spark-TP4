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
        _prefill(runner, 40000)                      # first prefill after boot: chunk 1 has no reference
        first = fake.empties
        assert first == 2, first                     # no-reference after chunk 1, end after chunk 5
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


def test_forward_batch_plumbing():
    import prefill_keep_cache as keep

    class FB:
        @classmethod
        def init_new(cls, batch, model_runner, **kw):
            fb = cls()
            fb.forward_mode = types.SimpleNamespace(is_extend=lambda: batch.extend)
            return fb

    req = types.SimpleNamespace(extend_range=types.SimpleNamespace(end=8192), origin_input_ids=list(range(20024)))
    mod = types.SimpleNamespace(ForwardBatch=FB)
    assert keep.install_forward_batch(mod)
    fb = FB.init_new(types.SimpleNamespace(reqs=[req], extend=True), types.SimpleNamespace(is_draft_worker=False),
                     return_hidden_states_before_norm=False)
    assert fb.dsv41_final_lens == [20024]
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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[OK] {name}", flush=True)
    print("all passed")
