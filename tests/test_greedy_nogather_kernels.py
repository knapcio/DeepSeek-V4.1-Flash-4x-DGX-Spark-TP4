"""Kernel-level exactness of adapter/greedy_nogather.py against the engine's own kernels (CPU, TRITON_INTERPRET=1; in the
image: no GPU needed, every TP rank simulated by its vocab shard, the all-gather by stacking the ranks' pairs):

target  pair_partial on each rank's shard + pair_finish == the stock accept's argmax of the gathered, vocab-cut fp32
        logits (sglang row_argmax, the <= 64-row path, and torch.argmax above it), on random rows, exact ties across
        ranks and inside a block, -inf-masked tiles (the certified head's masked mode), a padded last shard, all rows
        equal; bitwise, no exception allowed.
draft   upstream's sharded greedy step (sharded_greedy partial / finish kernels) == the stock folded sampler's greedy
        rows (sample_step_tokens_triton, greedy_mask on) on random rows and exact ties. On distinct values closer than
        the exp-key resolution the two may pick different draft tokens; the test counts those cases on adversarial
        rows and prints them (draft-only: greedy verification keeps the emitted tokens exact).
  TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=adapter python3 tests/test_greedy_nogather_kernels.py
"""
import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ["DSV41_GREEDY_NOGATHER"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter"))

import torch  # noqa: E402

import greedy_nogather as gn  # noqa: E402
from sglang.kernels.ops.speculative.row_argmax import row_argmax  # noqa: E402
from sglang.kernels.ops.speculative.dspark.sharded_greedy import (  # noqa: E402
    _sharded_greedy_finish_kernel, _sharded_greedy_partial_kernel)
from sglang.kernels.ops.speculative.dspark.dspark_draft_model import sample_step_tokens_triton  # noqa: E402

WORLD = 4


def stock_target(full, vocab):
    """What the stock accept computes: _row_argmax of the fp32 buffer the gathered logits were copied into."""
    x = full[:, :vocab].float().contiguous()
    return row_argmax(x) if x.shape[0] <= 64 else torch.argmax(x, dim=-1)


def ours_target(full, vocab, world=WORLD):
    width = full.shape[1] // world
    parts = []
    for r in range(world):
        p = gn.pair_partial(full[:, r * width:(r + 1) * width].contiguous(), rank=r, vocab_size=vocab)
        parts.append(p.view(full.shape[0], -1))
    gathered = torch.cat(parts, dim=0).contiguous()
    return gn.pair_finish(gathered, rows=full.shape[0], parts=parts[0].shape[1] // 2, world=world)


def check_target(name, full, vocab):
    a, b = stock_target(full, vocab), ours_target(full, vocab)
    bad = int((a != b).sum())
    print(f"target {name}: rows {full.shape[0]} width {full.shape[1]} vocab {vocab}: mismatches {bad}")
    assert bad == 0, (name, a.tolist(), b.tolist())


def test_target():
    g = torch.Generator().manual_seed(1)
    V, W = 4 * 8192, 8192                      # small shards (2 blocks per rank) keep the interpreter quick
    full = (torch.randn(12, V, generator=g) * 4).to(torch.bfloat16)
    check_target("random bf16", full, V)
    t = full.float().clone()
    m = t.max()
    t[0, 3 * W + 7] = m + 1; t[0, 1 * W + 9] = m + 1             # tie across ranks -> lower id
    t[1, 2 * W + 4096 + 5] = m + 2; t[1, 2 * W + 4096 + 3] = m + 2  # tie inside a block
    t[2, :] = -float("inf"); t[2, W + 100] = 0.5; t[2, 3 * W + 1] = 0.25   # masked tiles (certified head)
    t[3, :] = 1.0                                                  # all equal -> 0
    t[4, 4095] = m + 3; t[4, 4096] = m + 3                         # tie across a block boundary
    check_target("ties / masked / constant", t, V)
    # a padded last shard: the gathered tensor is cut at vocab like _copy_logits_to_buffer
    tp = t.clone()
    tp[5, V - 10] = m + 50                                         # in the padding: must be ignored
    check_target("padded vocab", tp, V - 100)
    # more than 64 rows: the stock path is torch.argmax
    check_target("80 rows", (torch.randn(80, V, generator=g) * 3).to(torch.bfloat16).float(), V)


def stock_draft(step_full):
    bs = step_full.shape[0]
    return sample_step_tokens_triton(step_logits=step_full, temperatures=torch.ones(bs),
                                     greedy_mask=torch.ones(bs, dtype=torch.bool), exp_noise=torch.ones_like(step_full))


def ours_draft(bias_full, base_full, world=WORLD):
    rows, V = base_full.shape
    width = V // world
    block, parts = 4096, (width + 4095) // 4096
    stacked = []
    for r in range(world):
        b = bias_full[:, r * width:(r + 1) * width].contiguous()
        x = base_full[:, r * width:(r + 1) * width].contiguous()
        p = torch.empty((rows, parts, 2), dtype=torch.float32)
        _sharded_greedy_partial_kernel[(rows, parts)](b, x, p, b.stride(0), x.stride(0), width, r * width, parts, block,
                                                       num_warps=4)
        stacked.append(p.view(rows, parts * 2))
    gathered = torch.cat(stacked, dim=0).contiguous()
    out = torch.empty(rows, dtype=torch.int64)
    _sharded_greedy_finish_kernel[(rows,)](gathered, out, rows, parts, world, 1 << (parts * world - 1).bit_length(),
                                           num_warps=4)
    return out


def test_draft():
    g = torch.Generator().manual_seed(2)
    V = 4 * 8192
    base = (torch.randn(10, V, generator=g) * 4).to(torch.bfloat16)
    bias = (torch.randn(10, V, generator=g) * 0.5).to(torch.bfloat16)
    step = base.float() + bias.float()                              # BuildStepLocal: fp32 add, gathered side by side
    a, b = stock_draft(step), ours_draft(bias, base)
    assert torch.equal(a, b), (a.tolist(), b.tolist())
    print(f"draft random: rows 10 vocab {V}: mismatches 0")
    # exact ties: both pick the lowest id
    base2, bias2 = base.clone(), torch.zeros_like(bias)
    m = float(base2.float().max()) + 1
    base2[0, 5 * 4096 + 1] = m; base2[0, 2 * 4096 + 7] = m
    a, b = stock_draft(base2.float() + bias2.float()), ours_draft(bias2, base2)
    assert int(a[0]) == int(b[0]) == 2 * 4096 + 7
    # adversarial near-ties (values 1 fp32 ulp apart): counted, not asserted
    near = torch.full((4, V), -8.0)
    near[:, 100] = 1.0
    near[:, 9000] = float(torch.nextafter(torch.tensor(1.0), torch.tensor(2.0)))
    near[1, 100] = 0.001
    near[1, 9000] = float(torch.nextafter(torch.tensor(0.001), torch.tensor(1.0)))
    a, b = stock_draft(near.clone()), ours_draft(torch.zeros(4, V), near.clone())
    print(f"draft near-ties (1 ulp): stock {a.tolist()} sharded {b.tolist()} -> {int((a != b).sum())} of 4 differ "
          "(draft token only; the sharded pick is the exact fp32 argmax)")
    assert b.tolist() == [9000] * 4


if __name__ == "__main__":
    test_target()
    print("PASS test_target")
    test_draft()
    print("PASS test_draft")
