"""DSV41_PREFILL_CED_SKIP on the engine's own call path (CPU; needs the image's sglang, skips cleanly elsewhere).

Regression for window C (2026-11-09 run-20261109-163256-1665083): the gate was armed on every node and never skipped.
The eager runner's load_batch hands the model cuda_graph_buffer_registry.extract_buffer(...) = dataclasses.replace() of
the batch ForwardBatch.init_new built, which keeps dataclass fields (rids) and drops extra attributes
(dsv41_final_lens), so the backend saw no final prompt lengths. This test drives, chunk by chunk, a 131,072-token
prompt in 8192-row chunks (one request) and a bypass prompt beside a chunk (two requests) through:
  1. ForwardBatch.init_new, wrapped by the keep-cache plumbing AND prefill_ced_skip.install_forward_batch, on a real
     ForwardBatch instance (the stand-in init_new builds the dataclass the way the engine's does, with rids);
  2. the engine's own CudaGraphBufferRegistry.extract_buffer (the eager runner's load_batch copy);
  3. the engine's own DeepseekV4AttnBackend._build_late_layer_tail_metadata with extend_nosync installed (production)
     and the CED hook on top, on a minimal backend object (init_forward_metadata_prefill / the SWA loc translation
     stubbed: they only receive the layout's outputs);
and checks: the copy drops dsv41_final_lens and keeps rids; with the gate on, chunks 0-14 get a one-row late-layer
tail floored at their own position and chunk 15 (the prompt's last) the stock 128 rows; the bypass prompt keeps its
stock tail beside a skipped chunk; with the gate off every chunk is stock (the per-variant switch: test_prefill_ced_skip.py); the CED
forward receipts are printed.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=adapter python3 tests/test_prefill_ced_skip_engine.py
"""
import dataclasses
import importlib
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "adapter"))
for k in [k for k in os.environ if k.startswith(("DSV41_AB", "DSV41_PREFILL_CED", "DSV41_PREFILL_KEEP"))]:
    del os.environ[k]

try:
    import torch
    fbi = importlib.import_module("sglang.srt.model_executor.forward_batch_info")
    bk = importlib.import_module("sglang.srt.layers.attention.deepseek_v4_backend")
    reg_mod = importlib.import_module("sglang.srt.model_executor.cuda_graph_buffer_registry")
except Exception as exc:  # noqa: BLE001
    print(f"SKIP: the engine is not importable here ({exc!r})")
    sys.exit(0)

CHUNK, PROMPT, W = 8192, 131072, 128


class Req:
    def __init__(self, rid, prompt, start, end):
        self.rid, self.origin_input_ids = rid, [0] * prompt
        self.extend_range = types.SimpleNamespace(start=start, end=end, length=end - start)


def shim_module():
    """A forward_batch_info stand-in whose ForwardBatch is the engine's dataclass and whose init_new builds it from
    a ScheduleBatch-like object (reqs, prefix lens) as the engine does for an extend."""
    class FB(fbi.ForwardBatch):
        pass

    def init_new(cls, batch, model_runner):
        ext = [r.extend_range.length for r in batch.reqs]
        seq = [r.extend_range.end for r in batch.reqs]
        n = sum(ext)
        pos = torch.cat([torch.arange(r.extend_range.start, r.extend_range.end) for r in batch.reqs])
        return fbi.ForwardBatch(
            forward_mode=fbi.ForwardMode.EXTEND, batch_size=len(batch.reqs), input_ids=torch.zeros(n, dtype=torch.int64),
            req_pool_indices=torch.arange(len(batch.reqs)), seq_lens=torch.tensor(seq), out_cache_loc=torch.arange(n),
            seq_lens_sum=sum(seq), seq_lens_cpu=torch.tensor(seq), extend_seq_lens=torch.tensor(ext),
            extend_seq_lens_cpu=ext, extend_prefix_lens_cpu=[r.extend_range.start for r in batch.reqs],
            positions=pos, rids=[r.rid for r in batch.reqs])

    FB.init_new = classmethod(init_new)
    return types.SimpleNamespace(ForwardBatch=FB, __name__="sglang.srt.model_executor.forward_batch_info")


class StubBackend:
    """Only what _build_late_layer_tail_metadata touches besides the layout."""
    is_dspark_draft = False

    def __init__(self):
        self.token_to_kv_pool = types.SimpleNamespace(request_window=None,
                                                      translate_loc_from_full_to_swa=lambda loc: loc)

    def init_forward_metadata_prefill(self, **kw):
        self.last_kw = kw
        return types.SimpleNamespace(core_attn_metadata=types.SimpleNamespace())

    def _late_layer_tail_cp_layout(self, *a, **k):
        raise AssertionError("CP layout on a non-CP batch")


def main():
    import extend_nosync
    import prefill_keep_cache as keep
    os.environ["DSV41_EXTEND_NOSYNC"] = "1"
    os.environ["DSV41_PREFILL_CED_SKIP"] = "1"
    import prefill_ced_skip as ced
    assert extend_nosync.install_backend(bk)
    stock_build = bk.DeepseekV4AttnBackend._build_late_layer_tail_metadata
    assert ced.install_backend(bk), ced.stats()["why_not"]
    shim = shim_module()
    assert keep.install_forward_batch(shim)            # production's plumbing: sets dsv41_final_lens on the batch
    assert ced.install_forward_batch(shim)             # the fix: records them by rid
    registry = reg_mod.CudaGraphBufferRegistry(device=torch.device("cpu"), max_bs=8, max_num_tokens=16384)
    mr = types.SimpleNamespace(is_draft_worker=False)
    be = StubBackend()

    def run(reqs, gate="1"):
        os.environ["DSV41_PREFILL_CED_SKIP"] = gate
        ced._STATE["static"] = gate == "1"
        outer = shim.ForwardBatch.init_new(types.SimpleNamespace(reqs=reqs), mr)
        inner = registry.extract_buffer(padded_bs=outer.batch_size, padded_num_tokens=outer.input_ids.shape[0],
                                        forward_batch_template=outer)
        md = bk.DeepseekV4AttnBackend._build_late_layer_tail_metadata(be, inner)
        return outer, inner, md.late_layer_tail

    # 1. the copy the backend sees: extra attributes dropped, rids kept (the window C failure mode)
    outer, inner, _ = run([Req("r0", PROMPT, 0, CHUNK)])
    assert getattr(outer, "dsv41_final_lens", None) == [PROMPT], getattr(outer, "dsv41_final_lens", None)
    assert not hasattr(inner, "dsv41_final_lens") and inner.rids == ["r0"]
    print("extract_buffer: dsv41_final_lens dropped, rids kept (why window C never skipped)")
    ced._STATE.update(forwards=0, skipped=0, rows_saved=0, no_finals=0)
    # 2. a cold 131K prompt, 16 chunks
    tails = []
    for k in range(PROMPT // CHUNK):
        _, inner, tail = run([Req("r1", PROMPT, k * CHUNK, (k + 1) * CHUNK)])
        tails.append(list(tail.extend_seq_lens_cpu))
        floor = be.last_kw["swa_replay_start"]
        if k < 15:
            assert tails[-1] == [1] and tail.contiguous_start == CHUNK - 1, (k, tails[-1], tail.contiguous_start)
            assert tail.positions.tolist() == [(k + 1) * CHUNK - 1], (k, tail.positions)
            assert floor.tolist() == [(k + 1) * CHUNK - 1], (k, floor)
        else:
            assert tails[-1] == [W] and floor.tolist() == [PROMPT - W] * W, (k, tails[-1])
    st = ced.stats()
    assert st["forwards"] == 15 and st["skipped"] == 15 and st["rows_saved"] == 15 * (W - 1) and st["no_finals"] == 0, st
    print(f"131K prompt: tails {tails[0]} x15 then {tails[-1]}; {st['forwards']} skipping forwards, rows saved "
          f"{st['rows_saved']}")
    # 3. a bypass prompt beside a chunk: the chunk skips, the bypass prompt keeps its stock tail
    _, inner, tail = run([Req("r2", PROMPT, CHUNK, 2 * CHUNK), Req("b1", 500, 0, 500)])
    assert list(tail.extend_seq_lens_cpu) == [1, W], tail.extend_seq_lens_cpu
    assert tail.token_indices.tolist() == [CHUNK - 1] + list(range(CHUNK + 500 - W, CHUNK + 500))
    # 4. gate off: stock everywhere; equal to the stock builder's own output
    for k in (0, 7, 15):
        _, inner, tail = run([Req("r3", PROMPT, k * CHUNK, (k + 1) * CHUNK)], gate="0")
        assert list(tail.extend_seq_lens_cpu) == [W], (k, tail.extend_seq_lens_cpu)
        floor = be.last_kw["swa_replay_start"].tolist()
        ref = stock_build(be, inner).late_layer_tail
        assert tail.contiguous_start == ref.contiguous_start == CHUNK - W and list(ref.extend_seq_lens_cpu) == [W]
        assert tail.positions.tolist() == ref.positions.tolist() and floor == be.last_kw["swa_replay_start"].tolist()
    # 5. unknown request: stock tail and one warning
    outer = shim.ForwardBatch.init_new(types.SimpleNamespace(reqs=[Req("r4", PROMPT, 0, CHUNK)]), mr)
    inner = dataclasses.replace(outer, rids=["never-recorded"])
    os.environ["DSV41_PREFILL_CED_SKIP"] = "1"
    ced._STATE["static"] = True
    tail = bk.DeepseekV4AttnBackend._build_late_layer_tail_metadata(be, inner).late_layer_tail
    assert list(tail.extend_seq_lens_cpu) == [W] and ced.stats()["no_finals"] == 1
    print("ALL PASS")


if __name__ == "__main__":
    main()
