"""CPU checks for adapter/sampled_draft_probs_early.py (streams are no-ops in the CPU test mode).

With the image's SGLang tree: the REAL dspark_verify.accept_draft_tokens, run on a sampled batch after the
proposal hook, receives draft probabilities bit-identical to stock and does not recompute them; check mode counts 0
mismatches and catches a corrupted early tensor; greedy batches, other SoftmaxTemp calls (other tensors), a second
call in the same step, bs above MAX_BS and the off mode all take the stock call. Skipped where SGLang is missing.

Run: PYTHONPATH=/opt/dsv41/adapter CUDA_VISIBLE_DEVICES= python3 tests/test_sampled_draft_probs_early.py
"""
import importlib
import os
import types

os.environ["DSV41_SAMPLED_DRAFT_PROBS_EARLY"] = "1"
os.environ["DSV41_SAMPLED_DRAFT_PROBS_EARLY_CPU"] = "1"
os.environ.setdefault("DSV41_SAMPLED_DRAFT_PROBS_EARLY_LOG_EVERY", "0")
import torch  # noqa: E402

import sampled_draft_probs_early as dpe  # noqa: E402

G, V = 5, 1000
torch.manual_seed(0)


def main():
    try:
        ver = importlib.import_module("sglang.srt.speculative.dspark_components.dspark_verify")
        dra = importlib.import_module("sglang.srt.speculative.dspark_components.dspark_draft")
    except Exception as exc:  # noqa: BLE001
        print(f"test_sampled_draft_probs_early: skipped ({type(exc).__name__}: {exc})")
        return
    stock_softmax = ver.SoftmaxTemp
    calls = {"n": 0}
    orig_exec = stock_softmax.execute

    def counting(*a, **kw):
        calls["n"] += 1
        return orig_exec(*a, **kw)

    stock_softmax.execute = staticmethod(counting)
    seen = {}

    class AcceptSampling:                                # the eager sampled accept, stood in: records q
        @staticmethod
        def execute(*, draft_probs, **kw):
            seen["q"] = draft_probs
            bs = draft_probs.shape[0]
            z = torch.zeros(bs, dtype=torch.int32)
            return z, z.long(), z

    ver.AcceptSampling = AcceptSampling
    assert dpe.install_verify(ver)
    assert ver.SoftmaxTemp is not stock_softmax and getattr(ver.SoftmaxTemp, "_dsv41_draft_probs_early", False)

    class Proposer:                                      # DraftBlockProposer.propose, stood in
        def propose(self, *, sampling_info, bs, **kw):
            return types.SimpleNamespace(draft_block=self.block, folded=True)

    assert dpe.install_draft(types.SimpleNamespace(DraftBlockProposer=Proposer))
    prop = Proposer()
    corrected_out = torch.randn(4 * G, V).to(torch.bfloat16)          # the draft graph's persistent buffer
    temps_buf = torch.tensor([0.7, 0.49, 0.7, 0.7], dtype=torch.float32)

    def block(bs):
        return dra.DraftBlockResult(draft_tokens=torch.zeros(bs, G, dtype=torch.int64),
                                    corrected_logits=corrected_out[: bs * G].view(bs, G, -1),
                                    greedy_mask=torch.zeros(bs, dtype=torch.bool), temperatures=temps_buf[:bs])

    sampled = types.SimpleNamespace(is_all_greedy=False, is_any_greedy=False)
    greedy = types.SimpleNamespace(is_all_greedy=True, is_any_greedy=True)

    def step(bs, si):
        prop.block = block(bs)
        before = calls["n"]
        prop.propose(sampling_info=si, bs=bs)
        early = calls["n"] - before
        cand = torch.zeros(bs, G + 1, dtype=torch.int64)
        if si.is_all_greedy:
            return early, None, 0
        before = calls["n"]
        ver.accept_draft_tokens(candidates=cand, target_logits=torch.zeros(bs * (G + 1), V),
                                draft_block=prop.block, sampling_info=si, draft_input=None, gamma=G,
                                verify_num_draft_tokens=G + 1)
        return early, seen["q"], calls["n"] - before

    def stock_q(bs):
        return orig_exec(logits=corrected_out[: bs * G].view(bs, G, -1).reshape(bs * G, V),
                         temperatures=temps_buf[:bs], rows_per_request=G).view(bs, G, V)

    for bs in (1, 2):
        early, q, late = step(bs, sampled)
        assert early == 1 and late == 0, (early, late)     # computed once, at the proposal
        assert torch.equal(q.view(torch.int32), stock_q(bs).view(torch.int32))
    used = dpe.STATS["used"]
    # bs above MAX_BS: stock at the accept
    early, q, late = step(3, sampled)
    assert early == 0 and late == 1 and dpe.REASONS["bs"] == 1
    assert torch.equal(q.view(torch.int32), stock_q(3).view(torch.int32))
    # greedy batch: nothing launched, nothing pending
    assert step(1, greedy)[0] == 0 and dpe._S["pending"] is None
    # a SoftmaxTemp call on other tensors (e.g. target probabilities) is not served from the early tensor
    prop.block = block(1)
    prop.propose(sampling_info=sampled, bs=1)
    other = torch.randn(G + 1, V)
    before = calls["n"]
    ver.SoftmaxTemp.execute(logits=other, temperatures=temps_buf[:1], rows_per_request=G + 1)
    assert calls["n"] - before == 1 and dpe._S["pending"] is not None
    # the next proposal replaces an unconsumed tensor (counted), the step's accept uses the new one
    early, q, late = step(1, sampled)
    assert late == 0 and dpe.STATS["unused"] == 1 and dpe.STATS["used"] == used + 1
    # a second identical call in the same step recomputes (one-shot)
    before = calls["n"]
    ver.SoftmaxTemp.execute(logits=corrected_out[:G], temperatures=temps_buf[:1], rows_per_request=G)
    assert calls["n"] - before == 1
    # check mode: stock tensor returned, 0 mismatches; a corrupted early tensor is counted
    os.environ[dpe.KEY] = "check"
    early, q, late = step(2, sampled)
    assert early == 1 and late == 1 and dpe._C["counters"][2].item() == 0
    assert torch.equal(q.view(torch.int32), stock_q(2).view(torch.int32))
    prop.block = block(1)
    prop.propose(sampling_info=sampled, bs=1)
    dpe._S["pending"][1].view(-1)[7] += 1.0
    ver.accept_draft_tokens(candidates=torch.zeros(1, G + 1, dtype=torch.int64), target_logits=torch.zeros(G + 1, V),
                            draft_block=prop.block, sampling_info=sampled, draft_input=None, gamma=G,
                            verify_num_draft_tokens=G + 1)
    assert dpe._C["counters"][2].item() == 1
    assert torch.equal(seen["q"].view(torch.int32), stock_q(1).view(torch.int32))
    # off: stock everywhere
    os.environ[dpe.KEY] = "0"
    early, q, late = step(1, sampled)
    assert early == 0 and late == 1
    try:
        dpe.parse_mode("maybe")
        raise AssertionError("bad value accepted")
    except RuntimeError:
        pass
    print(f"test_sampled_draft_probs_early: ok (stats {dpe.STATS}, skipped {dpe.REASONS})")


if __name__ == "__main__":
    main()
