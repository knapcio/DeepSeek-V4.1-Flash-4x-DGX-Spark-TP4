"""DSV41_PREFILL_KEEP_CACHE review gap: a prefill that never reaches its last chunk (client cancel -> SGLang
process_pending_chunked_abort clears chunked_req; also pause_generation, retraction) leaves its cache kept, and the
next prefill runs on top of it.

The real hook (adapter/prefill_empty_cache.py install with the gate on and off) drives the allocator model and the
chunk allocation model of tests/test_prefill_keep_cache.py through torch.cuda fakes. Sequence: a reference prefill,
a 1M prompt cancelled after 40 chunks, a 40K prompt. Today (gate off) the 40K prompt's reserved high-water is its own
(~1.4 GiB of transients); with the gate on it starts on the cancelled prompt's kept ~3.9 GiB, because the policy only
ever empties AFTER a long extend forward and the cancelled prompt has no last chunk.

Expected to FAIL on d265c35. A fix makes the next long extend that is not the continuation of the kept prompt empty
the cache BEFORE it runs (and, separately, releases the cache when the chunked request is aborted or the scheduler
goes idle: not modelled here, see diagnostics/prefill-fixed-20261108/REVIEW.md).

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=adapter:tests python3 tests/test_prefill_keep_cache_abort.py   (or pytest)
"""
import importlib
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_prefill_keep_cache as model  # noqa: E402

MIB, GIB = model.MIB, model.GIB
RESIDENT = 60 * GIB


def _chunks(prompt, upto=None):
    seqs = list(range(model.CHUNK, prompt + 1, model.CHUNK))
    if seqs[-1] != prompt:
        seqs.append(prompt)
    return seqs if upto is None else seqs[:upto]


def _run(env, prompts):
    """prompts: [(label, seqs, final)] -> {label: (transient reserved high-water, transient reserved after)}"""
    os.environ.pop("DSV41_PREFILL_KEEP_CACHE", None)
    os.environ.update(env)
    import prefill_keep_cache
    import prefill_empty_cache
    importlib.reload(prefill_keep_cache)
    importlib.reload(prefill_empty_cache)
    alloc = model.Allocator()
    resident = alloc.malloc(RESIDENT)
    cuda = torch.cuda
    saved = {n: getattr(cuda, n) for n in ("memory_stats", "empty_cache", "reset_peak_memory_stats")}
    cuda.memory_stats = lambda: {"reserved_bytes.all.current": alloc.reserved, "allocated_bytes.all.peak": alloc.peak,
                                 "num_device_alloc": alloc.num_device_alloc, "num_device_free": alloc.num_device_free}
    cuda.empty_cache = alloc.empty_cache
    cuda.reset_peak_memory_stats = alloc.reset_peak
    pad = env.get("DSV41_PREFILL_KEEP_CACHE") == "1"
    high = [0]

    class ModelRunner:
        def forward(self, fb):
            high[0] = max(high[0], alloc.reserved - RESIDENT)      # what the forward starts on
            model.run_chunk(alloc, model.chunk_events(int(fb.seq_lens_cpu.max()), fb.dsv41_final_lens[0], pad))
            high[0] = max(high[0], alloc.reserved - RESIDENT)      # reserved only grows inside a forward
            return "out"

    prefill_empty_cache.install(types.SimpleNamespace(ModelRunner=ModelRunner))
    runner = ModelRunner()
    out = {}
    try:
        for label, seqs, final in prompts:
            high[0] = 0
            for seq in seqs:
                runner.forward(types.SimpleNamespace(
                    forward_mode=types.SimpleNamespace(is_extend_without_speculative=lambda: True),
                    seq_lens_cpu=torch.tensor([seq]), dsv41_final_lens=[final]))
            out[label] = (high[0], alloc.reserved - RESIDENT)
    finally:
        for n, f in saved.items():
            setattr(cuda, n, f)
        alloc.release(resident)
    return out


def test_cancelled_prefill_does_not_carry_its_cache_into_the_next_one():
    prompts = [("ref", _chunks(32768), 32768),
               ("cancelled", _chunks(1048576, 40), 1048576),     # 1M prompt, client cancels after 40 chunks
               ("next", _chunks(40000), 40000)]
    stock = _run({}, prompts)
    keep = _run({"DSV41_PREFILL_KEEP_CACHE": "1"}, prompts)
    report = {k: (round(stock[k][0] / GIB, 2), round(keep[k][0] / GIB, 2)) for k in stock}
    # the next prompt must not start on (or peak above) what today's hook would give it
    assert keep["next"][0] <= stock["next"][0] + 16 * MIB, ("high-water GiB (stock, keep) per prompt", report)


if __name__ == "__main__":
    test_cancelled_prefill_does_not_carry_its_cache_into_the_next_one()
    print("[OK] test_cancelled_prefill_does_not_carry_its_cache_into_the_next_one")
