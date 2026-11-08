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


def _run_steps(env, steps):
    """steps: [(label, ids, seqs, finals, extends)] one long extend forward each; the model chunk is the first
    request's (seq, final) with sum(extends) rows. -> {label: transient reserved high-water over its steps}"""
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
            high[0] = max(high[0], alloc.reserved - RESIDENT)
            chunk = model.CHUNK
            model.CHUNK = int(sum(fb.extend_seq_lens_cpu))
            try:
                model.run_chunk(alloc, model.chunk_events(int(fb.seq_lens_cpu[0]), fb.dsv41_final_lens[0],
                                                          pad and getattr(fb, "dsv41_pad_ok", True)))
            finally:
                model.CHUNK = chunk
            high[0] = max(high[0], alloc.reserved - RESIDENT)
            return "out"

    prefill_empty_cache.install(types.SimpleNamespace(ModelRunner=ModelRunner))
    runner = ModelRunner()
    out = {}
    try:
        for label, ids, seqs, finals, extends in steps:
            high[0] = 0
            runner.forward(types.SimpleNamespace(
                forward_mode=types.SimpleNamespace(is_extend_without_speculative=lambda: True),
                seq_lens_cpu=torch.tensor(seqs), dsv41_final_lens=list(finals), dsv41_rids=list(ids),
                extend_seq_lens_cpu=list(extends)))
            out[label] = max(out.get(label, 0), high[0])
    finally:
        for n, f in saved.items():
            setattr(cuda, n, f)
        alloc.release(resident)
    return out


def _prompt_steps(label, rid, final, upto=None, start=0):
    steps, prev = [], start
    for seq in _chunks(final, upto):
        if seq <= start:
            continue
        steps.append((label, [rid], [seq], [final], [seq - prev]))
        prev = seq
    return steps


def test_mixed_batch_a_last_b_first_does_not_carry_a_into_b():
    a_final, b_final = 1048576 - 8192 + 3000, 40000
    steps = _prompt_steps("a", "a", a_final, upto=None)[:-1]
    last_a = (a_final // model.CHUNK) * model.CHUNK
    b_first = model.CHUNK - (a_final - last_a)
    steps.append(("mixed", ["a", "b"], [a_final, b_first], [a_final, b_final], [a_final - last_a, b_first]))
    steps += _prompt_steps("b", "b", b_final, start=b_first)
    stock = _run_steps({}, steps)
    keep = _run_steps({"DSV41_PREFILL_KEEP_CACHE": "1"}, steps)
    assert keep["b"] <= stock["b"] + 16 * MIB, (stock["b"] / GIB, keep["b"] / GIB)


def test_more_rows_mid_prompt_empties_before_the_wider_chunk():
    """A chunk with more rows than DSV41_PREFILL_KEEP_CACHE_PAD_MAX_ROWS (bypass admitted without solo) runs from an
    emptied cache with stock widths and is emptied after: it costs what stock costs at that chunk, and the following
    chunks do not carry its blocks. Before the fix: +1.5..2.7 GiB above stock's whole-prompt peak."""
    final = 136072
    for extra in (256, 8192):
        steps, prev = [], 0
        for i, seq in enumerate(_chunks(final)):
            seq = prev + model.CHUNK + extra if i == 4 else max(seq, prev + 1)
            seq = min(seq, final)
            steps.append(("wide" if i == 4 else "rest", ["a"], [seq], [final], [seq - prev]))
            prev = seq
            if seq == final:
                break
        stock = _run_steps({}, steps)
        keep = _run_steps({"DSV41_PREFILL_KEEP_CACHE": "1"}, steps)
        bound = stock["wide"] + 16 * MIB         # emptied before, stock widths (pad_ok False), emptied after
        assert keep["wide"] <= bound, (extra, stock["wide"] / GIB, keep["wide"] / GIB, max(stock.values()) / GIB)
        assert keep["rest"] <= max(stock.values()) + 16 * MIB, (extra, keep["rest"] / GIB, max(stock.values()) / GIB)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[OK] {name}", flush=True)
