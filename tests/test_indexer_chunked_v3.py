"""CPU equivalence test (v2, candidate_metadata variant): chunked dense indexer vs the stock
function lifted verbatim from the branch's deepseek_v4_backend.py, both driven by the
same deterministic fake kernels.  No GPU, no sglang import.

Run:  python3 test_indexer_chunked.py /path/to/backend-ours.py
"""
import re
import sys
import types
from types import SimpleNamespace

import torch

BACKEND_SRC = sys.argv[1] if len(sys.argv) > 1 else "backend-f80c91a4b.py"
TOPK = 4
HEADS = 2


# ----- deterministic fake kernels (row-invariant, so chunking cannot change values) -----
def ceil_align(x, a):
    return (x + a - 1) // a * a


def _as_int_list(x):
    return [int(v) for v in x] if x is not None else None


WIDTHS = []  # (max_seqlen_k, kv rows) of every scorer call: the PAD runs must really be wider


def _dense_fp4_mqa_logits(q_fp4, kv_fp4, weights, ks, ke, max_seqlen_k):
    WIDTHS.append((max_seqlen_k, kv_fp4[0].shape[0]))
    q, _ = q_fp4
    rows = q.shape[0]
    # score depends only on the row's own token id (q[:,0,0]) and the column: row-invariant
    tok = q[:, 0, 0].to(torch.float32)
    j = torch.arange(max_seqlen_k, dtype=torch.float32)
    logits = ((tok[:, None] * 31 + j[None, :] * 7) % 97) + 0.01 * j[None, :]
    # garbage beyond ke - ks, as the real kernel leaves it
    valid = j[None, :] < (ke - ks).to(torch.float32)[:, None]
    return torch.where(valid, logits, torch.full_like(logits, 12345.0))


def topk_transform_ragged_v2(logits, lens, out_offsets, out_indices):
    rows, width = logits.shape
    k = out_indices.shape[1]
    for r in range(rows):
        n = int(lens[r])
        sc = logits[r, :n]
        if n == 0:
            out_indices[r].fill_(-1)
            continue
        order = torch.argsort(sc, descending=True, stable=True)[:k]
        out = torch.full((k,), -1, dtype=torch.int32)
        out[: order.shape[0]] = (order + int(out_offsets[r])).to(torch.int32)
        out_indices[r] = out


def _mask_topk_scores(scores, indices, offsets=None):
    columns = indices.to(torch.int64)
    if offsets is not None:
        columns = columns - offsets[:, None]
    selected_scores = scores.gather(1, columns.clamp(0, scores.shape[1] - 1))
    valid = (columns >= 0) & (columns < scores.shape[1]) & (selected_scores > -torch.inf)
    return indices.masked_fill(~valid, -1)


def select_candidate_blocks(scores, lens, topk_blocks, block_size):
    rows, lc = scores.shape
    nb = (lc + block_size - 1) // block_size
    mask = torch.zeros((rows, lc), dtype=torch.bool)
    for r in range(rows):
        bs = torch.full((nb,), -torch.inf)
        for b in range(nb):
            seg = scores[r, b * block_size : min(lc, (b + 1) * block_size)]
            seg = seg[seg > -torch.inf]
            if seg.numel():
                bs[b] = seg.max()
        keep = torch.argsort(bs, descending=True, stable=True)[:topk_blocks]
        for b in keep.tolist():
            if bs[b] > -torch.inf:
                mask[r, b * block_size : min(lc, (b + 1) * block_size)] = True
    return mask


class CandidateMasks:
    def __init__(self, request_masks):
        self.request_masks = request_masks


def published_masks(candidate):
    assert isinstance(candidate, CandidateMasks), "candidate masks missing"
    return candidate


mask_topk_scores = _mask_topk_scores


def quantize_fp4_indexer_tensor(q, rne=True):
    # q: [T*H, 128] -> ([T*H, 64] int8, [T*H] int32); keep the token id in column 0
    return q[:, :64].to(torch.int8), torch.zeros(q.shape[0], dtype=torch.int32)


fake_fp4 = types.ModuleType("sglang.kernels.ops.attention.dsv4.fp4_indexer")
fake_fp4.quantize_fp4_indexer_tensor = quantize_fp4_indexer_tensor
for name in ("sglang", "sglang.kernels", "sglang.kernels.ops", "sglang.kernels.ops.attention",
             "sglang.kernels.ops.attention.dsv4"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["sglang.kernels.ops.attention.dsv4.fp4_indexer"] = fake_fp4

fake_module = types.ModuleType("deepseek_v4_backend_fake")
fake_module.get_parallel = lambda: None
for name, obj in dict(
    _dense_fp4_mqa_logits=_dense_fp4_mqa_logits, topk_transform_ragged_v2=topk_transform_ragged_v2,
    _mask_topk_scores=_mask_topk_scores, select_candidate_blocks=select_candidate_blocks,
    ceil_align=ceil_align, _as_int_list=_as_int_list, torch=torch,
    _TORCH_INDEXER_SCORE_BUDGET_BYTES=1 << 30, CandidateMasks=CandidateMasks, published_masks=published_masks, mask_topk_scores=mask_topk_scores,
).items():
    setattr(fake_module, name, obj)


# ----- lift the stock methods verbatim from the image's source -----
src = open(BACKEND_SRC).read()


def lift(name):
    m = re.search(r"\n    def " + name + r"\(.*?(?=\n    def |\Z)", src, re.S)
    assert m, name
    body = "\n".join(line[4:] if line.startswith("    ") else line for line in m.group(0).splitlines())
    ns = dict(vars(fake_module))
    exec(body, ns)
    return ns[name]


stock_dense = lift("_low_ratio_index_topk_dense")
stock_publish = lift("_publish_or_consume_candidates")
sys.path.insert(0, ".")
import indexer_chunked_v3 as indexer_chunked  # noqa: E402
import spark_prefill_dense  # noqa: E402


class StockBackend:
    _low_ratio_index_topk_dense = stock_dense
    _publish_or_consume_candidates = stock_publish


# ----- scenario builder -----
def build(seq_lens, q_lens, ratio, source, consume_masks=None, tail_lens=None, topk=TOPK):
    T = sum(q_lens)
    max_slots = max(seq_lens) + 8
    req_to_token = torch.stack([torch.arange(max_slots) + 1000 * r for r in range(len(seq_lens))])

    class Pool:
        def get_low_ratio_index_k_fp4(self, layer_id, k_slots):
            return k_slots.to(torch.int8), torch.zeros(k_slots.shape[0], dtype=torch.int32)

    core = SimpleNamespace(
        _page=torch.full((T + 3, topk), -7, dtype=torch.int32),
        _raw=torch.full((T + 3, topk), -7, dtype=torch.int32),
    )
    core.sparse_page_indices = lambda r: core._page
    core.sparse_raw_indices = lambda r: core._raw
    indexer = SimpleNamespace(
        is_candidate_source=source,
        uses_candidates=source or consume_masks is not None,
        index_topk=topk, candidate_topk_blocks=2, candidate_block_size=3,
        queries=lambda q_lora, freqs: q_lora,
        head_weights=lambda x: torch.ones(x.shape[0], HEADS),
    )
    layer = SimpleNamespace(compress_ratio=ratio, indexer=indexer, freqs_cis=torch.zeros(4096, 4), layer_id=1)
    pos = torch.cat([torch.arange(s - q, s) for s, q in zip(seq_lens, q_lens)])
    q_lora = torch.zeros(T, HEADS, 128)
    q_lora[:, :, 0] = torch.arange(T)[:, None].to(q_lora.dtype)  # token id in column 0
    x = torch.zeros(T, 8)
    fb = SimpleNamespace(seq_lens_cpu=torch.tensor(seq_lens), req_pool_indices=torch.arange(len(seq_lens)), forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True))
    # DSV41_PREFILL_KEEP_CACHE_PAD: the requests' final prompt lengths (ForwardBatch.init_new attaches them)
    fb.dsv41_final_lens = None if FINAL_EXTRA[0] is None else [s + FINAL_EXTRA[0] * (b + 1) for b, s in enumerate(seq_lens)]
    tail = None
    if tail_lens is not None:
        tail = SimpleNamespace(late_layer_tail=SimpleNamespace(cp_metadata=None, extend_seq_lens_cpu=list(tail_lens)))
    self = SimpleNamespace(
        token_to_kv_pool=Pool(), forward_metadata=SimpleNamespace(core_metadata=core, candidate_metadata=(CandidateMasks(consume_masks) if consume_masks is not None else None)),
        req_to_token=req_to_token, candidate_masks=consume_masks, tail_forward_metadata=tail,
    )
    self._dsv41_native_prefill_tp = SPLIT[0]
    return self, layer, x, q_lora, pos, fb, torch.tensor(q_lens), list(q_lens)


def run(fn, budget, **kw):
    self, layer, x, q_lora, pos, fb, q_lens, q_lens_cpu = build(**kw)
    if fn is None:
        self._publish_or_consume_candidates = types.MethodType(stock_publish, self)
        StockBackend._low_ratio_index_topk_dense(self, layer, x, q_lora, pos, fb, q_lens, q_lens_cpu)
    else:
        f = indexer_chunked.make_index_topk_dense(fake_module, quantize_fp4_indexer_tensor, budget)
        f(self, layer, x, q_lora, pos, fb, q_lens, q_lens_cpu)
    cm = self.forward_metadata.candidate_metadata
    return self.forward_metadata.core_metadata._page.clone(), self.forward_metadata.core_metadata._raw.clone(), (cm.request_masks if isinstance(cm, CandidateMasks) else None)


SPLIT = [None]
FINAL_EXTRA = [None]    # None = no final lengths on the batch (padding off)


class FakeGroup:
    """all_gather that recomputes every rank's local selections from the recorded call."""
    def __init__(self, world):
        self.world_size = world; self.rank_in_group = 0; self.last = None
    def all_gather(self, local, dim=0):
        parts = []
        for r in range(self.world_size):
            a = dict(self.last); a["rank"] = r
            parts.append(REAL_SPLIT_LOCAL(**a))
        return torch.cat(parts, dim=0)


def make_split(world):
    g = FakeGroup(world)
    tp = spark_prefill_dense.NativePrefillTPSplit(g, 0, world, True, minimum_context=1, minimum_rows=1)
    def recording(indexer,q_fp4,q_sf,k_fp4,k_sf,weights,ks,lens,lengths,q_lengths,world_,rank,**kw):
        g.last = dict(indexer=indexer,q_fp4=q_fp4,q_sf=q_sf,k_fp4=k_fp4,k_sf=k_sf,weights=weights,ks=ks,lens=lens,lengths=lengths,q_lengths=q_lengths,world=world_,rank=rank,**kw)
        return REAL_SPLIT_LOCAL(indexer,q_fp4,q_sf,k_fp4,k_sf,weights,ks,lens,lengths,q_lengths,world_,rank,**kw)
    indexer_chunked._split_local = recording
    return tp


REAL_SPLIT_LOCAL = indexer_chunked._split_local


def check(name, budgets, **kw):
    ref_page, ref_raw, ref_masks = run(None, None, **kw)
    tail_lens = kw.get("tail_lens")
    for budget in budgets:
        page, raw, masks = run(True, budget, **kw)
        assert torch.equal(page, ref_page), f"{name} budget={budget}: page_indices differ"
        assert torch.equal(raw, ref_raw), f"{name} budget={budget}: raw_indices differ"
        if kw["source"]:
            assert len(masks) == len(ref_masks)
            for b, (m, rm) in enumerate(zip(masks, ref_masks)):
                if tail_lens is not None and rm.numel():
                    rm = rm[rm.shape[0] - tail_lens[b]:]  # what enter_late_layer_tail keeps
                assert m.shape == rm.shape and torch.equal(m, rm), f"{name} budget={budget}: mask {b} differs {tuple(m.shape)} vs {tuple(rm.shape)}"
    print(f"[OK]   {name} (budgets {budgets})")


small = [16, 64, 160, 1 << 31]  # 16 B => 1 row per chunk at lc<=4 ... up to one chunk
check("plain, single request", small, seq_lens=[40], q_lens=[12], ratio=2, source=False)
check("plain, three requests ragged", small, seq_lens=[40, 9, 61], q_lens=[12, 9, 5], ratio=1, source=False)
check("plain, request with lc=0 and t_len=0", small, seq_lens=[1, 30, 20], q_lens=[1, 0, 6], ratio=2, source=False)
check("publish full masks (no tail metadata)", small, seq_lens=[40, 25], q_lens=[10, 7], ratio=1, source=True)
check("publish tail-only masks", small, seq_lens=[40, 25, 33], q_lens=[10, 7, 9], ratio=1, source=True, tail_lens=[3, 7, 1])
masks = [torch.rand(10, 40) > 0.4, torch.rand(7, 25) > 0.4]
check("consume candidate masks", small, seq_lens=[40, 25], q_lens=[10, 7], ratio=1, source=False, consume_masks=masks)
torch.cuda.is_current_stream_capturing = lambda: False
for world in (1, 2, 4):
    SPLIT[0] = make_split(world)
    check(f"SPLIT world={world}: plain ragged", [64, 1 << 31], seq_lens=[300, 129, 700], q_lens=[140, 129, 260], ratio=1, source=False)
    check(f"SPLIT world={world}: publish masks", [64, 1 << 31], seq_lens=[300, 260], q_lens=[140, 130], ratio=1, source=True)
    check(f"SPLIT world={world}: publish tail-only", [64, 1 << 31], seq_lens=[300, 260, 500], q_lens=[140, 130, 200], ratio=1, source=True, tail_lens=[3, 130, 7])
    masks = [torch.rand(140, 300) > 0.4, torch.rand(130, 260) > 0.4]
    check(f"SPLIT world={world}: consume masks", [64, 1 << 31], seq_lens=[300, 260], q_lens=[140, 130], ratio=1, source=False, consume_masks=masks)
SPLIT[0] = None

# ----- DSV41_PREFILL_KEEP_CACHE_PAD: rows padded to the prompt's final width give the same selections and masks -----
import os  # noqa: E402
os.environ["DSV41_PREFILL_KEEP_CACHE"] = "1"
WIDTHS.clear()
for extra in (1, 5, 257):
    FINAL_EXTRA[0] = extra
    check(f"PAD +{extra}: plain ragged", small, seq_lens=[40, 9, 61], q_lens=[12, 9, 5], ratio=1, source=False)
    check(f"PAD +{extra}: publish full masks", small, seq_lens=[40, 25], q_lens=[10, 7], ratio=1, source=True)
    check(f"PAD +{extra}: publish tail-only", small, seq_lens=[40, 25, 33], q_lens=[10, 7, 9], ratio=1, source=True, tail_lens=[3, 7, 1])
    masks = [torch.rand(10, 40) > 0.4, torch.rand(7, 25) > 0.4]
    check(f"PAD +{extra}: consume masks", small, seq_lens=[40, 25], q_lens=[10, 7], ratio=1, source=False, consume_masks=masks)
    check(f"PAD +{extra}: ratio 4", small, seq_lens=[400, 131], q_lens=[40, 31], ratio=4, source=True)
    for world in (1, 2, 4):
        SPLIT[0] = make_split(world)
        check(f"PAD +{extra} SPLIT world={world}: plain ragged", [64, 1 << 31], seq_lens=[300, 129, 700], q_lens=[140, 129, 260], ratio=1, source=False)
        check(f"PAD +{extra} SPLIT world={world}: publish masks", [64, 1 << 31], seq_lens=[300, 260], q_lens=[140, 130], ratio=1, source=True)
        check(f"PAD +{extra} SPLIT world={world}: publish tail-only", [64, 1 << 31], seq_lens=[300, 260, 500], q_lens=[140, 130, 200], ratio=1, source=True, tail_lens=[3, 130, 7])
    # split enabled but this chunk below its context threshold: the unsplit path with the last-unsplit-width cap
    tp = make_split(4)
    tp.minimum_context = 280
    SPLIT[0] = tp
    check(f"PAD +{extra} unsplit under an enabled split: publish masks", small, seq_lens=[200, 150], q_lens=[40, 30], ratio=1, source=True)
    SPLIT[0] = None
assert any(w > kv + 3 for w, kv in WIDTHS), "PAD runs never widened a logits row"
os.environ.pop("DSV41_PREFILL_KEEP_CACHE")
FINAL_EXTRA[0] = None
print("all passed")
