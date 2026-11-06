"""v3 = v2 (sglang#39187 verbatim, candidate_metadata backends) + rhys101's SG18 native prefill
TP split: when a prefill chunk is eligible (>= SPARK_PREFILL_TP_MIN_ROWS rows and
>= SPARK_PREFILL_TP_MIN_CONTEXT context, SPARK_PREFILL_TP_SPLIT=1), each TP rank scores only
its slice of the query rows and the top-k / candidate block ids are all-gathered as ints.
Otherwise the chunked #39187 path runs unchanged.

Original v2 header: Bound the dense prefill indexer transient on the upstream dsv4.1 branch: verbatim
backport of sgl-project/sglang#39187 (head cb8dd033, kpham-sgl) for backends that keep
candidate masks in ``forward_metadata.candidate_metadata`` (branch f80c91a4b and later).
The v1 adapter (indexer_chunked.py) covers image e087e662, which uses self.candidate_masks.

Gate: ``DSV41_INDEXER_CHUNKED=1``.  Budget: ``DSV41_INDEXER_LOGITS_BUDGET_BYTES``
(default 2 GiB of fp32 logits).  Off = stock path untouched.

``DSV41_INDEXER_NOSYNC=1`` (default off; needs DSV41_INDEXER_CHUNKED=1): the eager prefill indexer of every
index-source layer builds its compressed-slot ids without host synchronizations. The stock per-request loop indexes
``req_to_token`` with a 0-dim CUDA tensor (``req_pool_indices[r]``), which PyTorch turns into ``.item()``, a full
stream sync per request and layer (8 per c1 prefill, 128 per bs16 prefill), and copies the request starts with
``torch.tensor(..., device=cuda)``, a pageable copy that syncs the stream again. With the knob the slot ids come from
one gather over device indices (``repeat_interleave`` with a host ``output_size``) and both host lists reach the GPU
as pinned non-blocking copies. Integer ops only, same elements in the same order: k_slots, ks, the selected slots,
page/raw indices and the candidate masks are bit-identical (tests/test_indexer_nosync.py). Read per call through
adapter/ab_variant.py when the in-boot A/B is armed (a RUNTIME key), else once.
"""
import inspect
import logging
import os

import torch

logger = logging.getLogger(__name__)
DEFAULT_BUDGET_BYTES = 1 << 31


def _enabled(name, default="0"):
    return os.environ.get(name, default).strip() not in ("0", "off", "false", "")


def budget_bytes():
    raw = os.environ.get("DSV41_INDEXER_LOGITS_BUDGET_BYTES", "").strip()
    if not raw:
        return DEFAULT_BUDGET_BYTES
    # Override site: a set DSV41_INDEXER_LOGITS_BUDGET_BYTES that is not an int, or is <= 0, was
    # silently replaced by DEFAULT_BUDGET_BYTES. A value the user set must take effect, so a bad
    # value now stops the boot instead of being replaced.
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"DSV41_INDEXER_LOGITS_BUDGET_BYTES={raw!r}: not an int; the default "
                           f"{DEFAULT_BUDGET_BYTES} would replace it; refusing to boot") from None
    if value <= 0:
        raise RuntimeError(f"DSV41_INDEXER_LOGITS_BUDGET_BYTES={raw!r}: must be > 0; the default "
                           f"{DEFAULT_BUDGET_BYTES} would replace it; refusing to boot")
    return value


try:  # DSV41_AB_VARIANTS in-boot A/B (test only)
    import ab_variant as _ab
except ImportError:
    _ab = None

_OFF = ("0", "off", "false", "", "no")


def _nosync_reader():
    """() -> bool for DSV41_INDEXER_NOSYNC: per call under the armed in-boot A/B, else fixed at build time."""
    if _ab is not None and getattr(_ab, "ACTIVE", False) and "DSV41_INDEXER_NOSYNC" in getattr(_ab, "KNOWN", ()):
        return lambda: nosync_value(_ab.env("DSV41_INDEXER_NOSYNC", "0"))
    fixed = nosync_value(os.environ.get("DSV41_INDEXER_NOSYNC", "0"))
    return lambda: fixed


def nosync_value(raw):
    """One parser for both readers: '', 0, off, false, no (any case) = off."""
    return str(raw or "0").strip().lower() not in _OFF


def h2d(values, dtype, device):
    """torch.tensor(values, dtype=dtype, device=device) without the stream sync of a pageable host copy: a pinned
    staging tensor and a non-blocking copy (the caching host allocator keeps the block until the copy has run)."""
    host = torch.tensor(values, dtype=dtype)
    if device.type != "cuda":
        return host.to(device)
    return host.pin_memory().to(device, non_blocking=True)


def slot_ids_nosync(req_to_token, req_pool_indices, lc_per_req, ratio, device):
    """(k_slots or None, starts) of the stock per-request loop

        for r, lc in enumerate(lc_per_req):
            starts.append(start)
            if lc: chunks.append(req_to_token[req_pool_indices[r], arange(lc) * ratio].to(int64) // ratio); start += lc
        k_slots = cat(chunks)

    with no host read of device data: request r's lc rows come from one gather at (req_pool_indices[r],
    (j - starts[r]) * ratio) for j in [starts[r], starts[r] + lc). Requests with lc == 0 contribute no rows."""
    starts, start = [], 0
    for lc in lc_per_req:
        starts.append(start)
        start += lc
    if start == 0:
        return None, starts
    n = len(lc_per_req)
    meta = h2d(list(lc_per_req) + starts, torch.int64, device)       # one copy: counts, then starts
    counts, first = meta[:n], meta[n:]
    rows = torch.repeat_interleave(req_pool_indices[:n], counts, output_size=start)
    cols = torch.arange(start, device=device) - torch.repeat_interleave(first, counts, output_size=start)
    return req_to_token[rows, cols * ratio].to(torch.int64) // ratio, starts


def _split_local(indexer, q_fp4, q_sf, k_fp4, k_sf, weights, ks, lens, lengths, q_lengths,
                 world, rank, *, budget, scorer, topk_fn, mask_topk, consume, ceil_align):
    """This rank's slice of the query rows, scored in row chunks of <= budget bytes of fp32
    logits (the #39187 bound applied inside the SG18 partition). Returns the packed int32
    [extent, topk + blocks] tensor the all-gather concatenates: top-k ids, then candidate
    block ids for a source layer."""
    from spark_prefill_dense import partition, candidate_block_ids

    rows = q_fp4.shape[0]
    start, end, extent = partition(rows, world, rank)
    source = indexer.is_candidate_source
    width_max = max(lengths)
    blocks = (min(indexer.candidate_topk_blocks,
                  (width_max + indexer.candidate_block_size - 1) // indexer.candidate_block_size)
              if source else 0)
    packed = torch.full((extent, indexer.index_topk + blocks), -1, dtype=torch.int32, device=q_fp4.device)
    if start >= end:
        return packed
    lc_aligned = ceil_align(width_max, 4)
    rows_per_chunk = max(1, budget // (lc_aligned * 4))
    # request boundaries in global row space
    bounds = []
    offset = 0
    for width, count in zip(lengths, q_lengths):
        bounds.append((offset, offset + count, width))
        offset += count
    assert offset == rows
    for c0 in range(start, end, rows_per_chunk):
        c1 = min(c0 + rows_per_chunk, end)
        rows_sl = slice(c0, c1)
        local_lens = lens[rows_sl]
        local_ks = ks[rows_sl]
        logits = scorer((q_fp4[rows_sl], q_sf[rows_sl]), (k_fp4, k_sf), weights[rows_sl],
                        local_ks, local_ks + local_lens, lc_aligned)
        for request, (lo_r, hi_r, width) in enumerate(bounds):
            lo = max(c0, lo_r)
            hi = min(c1, hi_r)
            if not width or lo >= hi:
                continue
            take = slice(lo - c0, hi - c0)
            scores = logits[take, :width]
            if source:
                row_lens = local_lens[take, None]
                scores.masked_fill_(torch.arange(width, device=logits.device)[None, :] >= row_lens, -torch.inf)
                nblocks = min(indexer.candidate_topk_blocks,
                              (width + indexer.candidate_block_size - 1) // indexer.candidate_block_size)
                ids = candidate_block_ids(scores, row_lens, indexer.candidate_topk_blocks,
                                          indexer.candidate_block_size)
                packed[lo - start:hi - start, indexer.index_topk:indexer.index_topk + nblocks] = ids
            elif consume is not None:
                scores.masked_fill_(~consume[request][lo - lo_r:hi - lo_r], -torch.inf)
        selected = torch.empty((c1 - c0, indexer.index_topk), dtype=torch.int32, device=logits.device)
        topk_fn(logits, local_lens, out_offsets=local_ks, out_indices=selected)
        if consume is not None:
            selected = mask_topk(logits, selected, local_ks)
        packed[c0 - start:c1 - start, :indexer.index_topk] = selected
        del logits
    return packed


def _split_select(tp, indexer, q_fp4, q_sf, k_fp4, k_sf, weights, ks, lens, lengths, q_lengths,
                  *, budget, scorer, topk_fn, mask_topk, consume, publish_rows, layer_id, ratio,
                  ceil_align):
    """Score this rank's partition, all-gather the packed ids, rebuild the (tail-only)
    candidate masks. Mirrors NativePrefillTPSplit.select with the chunk bound and the
    #39187 tail-only publishing."""
    from spark_prefill_dense import partition, candidate_mask

    local = _split_local(indexer, q_fp4, q_sf, k_fp4, k_sf, weights, ks, lens, lengths, q_lengths,
                         tp.world, tp.rank, budget=budget, scorer=scorer, topk_fn=topk_fn,
                         mask_topk=mask_topk, consume=consume, ceil_align=ceil_align)
    gathered = tp.group.all_gather(local, dim=0)[: q_fp4.shape[0]]
    masks = None
    if indexer.is_candidate_source:
        masks = []
        offset = 0
        for b, (width, count) in enumerate(zip(lengths, q_lengths)):
            if not width or not count:
                masks.append(torch.zeros((0, 0), dtype=torch.bool, device=gathered.device))
            else:
                nblocks = min(indexer.candidate_topk_blocks,
                              (width + indexer.candidate_block_size - 1) // indexer.candidate_block_size)
                pub = count if publish_rows is None else min(int(publish_rows[b]), count)
                ids = gathered[offset + count - pub: offset + count,
                               indexer.index_topk: indexer.index_topk + nblocks]
                masks.append(candidate_mask(ids, width, indexer.candidate_block_size))
            offset += count
    key = (ratio, bool(indexer.is_candidate_source), bool(indexer.uses_candidates))
    if key not in tp._logged:
        tp._logged.add(key)
        start, end, extent = partition(q_fp4.shape[0], tp.world, tp.rank)
        logger.warning("DSV41 prefill TP split (v3) rank=%s layer=%s ratio=%s source=%s consumer=%s rows=%s width=%s start=%s end=%s budget=%d MiB",
                       tp.rank, layer_id, ratio, indexer.is_candidate_source, indexer.uses_candidates,
                       q_fp4.shape[0], max(lengths), start, end, budget >> 20)
    return gathered[:, : indexer.index_topk], masks


def make_index_topk_dense(module, quantize_fp4_indexer_tensor, budget):
    """The PR's function, bound to the module's own helpers (names resolved once)."""
    _dense_fp4_mqa_logits = module._dense_fp4_mqa_logits
    topk_transform_ragged_v2 = module.topk_transform_ragged_v2
    mask_topk_scores = module.mask_topk_scores
    select_candidate_blocks = module.select_candidate_blocks
    ceil_align = module.ceil_align
    _as_int_list = module._as_int_list
    CandidateMasks = module.CandidateMasks
    published_masks = module.published_masks
    _DENSE_INDEXER_LOGITS_BUDGET_BYTES = budget
    _nosync = _nosync_reader()

    # ---- begin verbatim from sglang#39187 (deepseek_v4_backend.py @ cb8dd033) ----

    def _low_ratio_index_topk_dense(
        self, layer, x, q_lora, pos, forward_batch, q_lens, q_lens_cpu
    ) -> None:
        """Dense fp4 indexer over rows laid out request after request, `q_lens[b]` each."""
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            quantize_fp4_indexer_tensor,
        )

        pool = self.token_to_kv_pool
        core = self.forward_metadata.core_metadata
        ratio = layer.compress_ratio
        indexer = layer.indexer
        page_indices = core.sparse_page_indices(ratio)
        raw_indices = core.sparse_raw_indices(ratio)
        page_indices.fill_(-1)
        if raw_indices is not None:
            raw_indices.fill_(-1)

        seq_lens_cpu = _as_int_list(forward_batch.seq_lens_cpu)
        assert seq_lens_cpu is not None
        device = pos.device
        # Visible compressed positions per request at its newest token; the
        # per-token count (pos + 1) // ratio bounds each row below.
        lc_per_req = [s // ratio for s in seq_lens_cpu]
        req_pool_indices = forward_batch.req_pool_indices.to(torch.int64)
        nosync = _nosync()  # ds41 DSV41_INDEXER_NOSYNC (off = the verbatim loop below)
        if nosync:
            _k_slots, starts = slot_ids_nosync(
                self.req_to_token, req_pool_indices, lc_per_req, ratio, device
            )
            slot_chunks = [] if _k_slots is None else [_k_slots]
        else:
            slot_chunks, starts, start = [], [], 0
            for r, lc in enumerate(lc_per_req):
                starts.append(start)
                if lc == 0:
                    continue
                j = torch.arange(lc, device=device)
                slot_chunks.append(
                    self.req_to_token[req_pool_indices[r], j * ratio].to(torch.int64)
                    // ratio
                )
                start += lc
        empty_mask = torch.zeros(0, 0, dtype=torch.bool, device=device)
        num_tokens = pos.shape[0]
        # TODO: move this to candidate indexer
        if not slot_chunks or num_tokens == 0:
            if indexer.is_candidate_source:
                self.forward_metadata.candidate_metadata = CandidateMasks(
                    request_masks=[empty_mask for _ in lc_per_req]
                )
            return
        k_slots = slot_chunks[0] if nosync else torch.cat(slot_chunks)
        k_fp4, k_sf = pool.get_low_ratio_index_k_fp4(layer.layer_id, k_slots)

        q = indexer.queries(q_lora, layer.freqs_cis[pos])  # [T, H, 128] fp4 grid
        num_heads = q.shape[1]
        q_fp4, q_sf = quantize_fp4_indexer_tensor(q.flatten(0, 1), rne=True)
        q_fp4 = q_fp4.view(num_tokens, num_heads, 64)
        q_sf = q_sf.view(num_tokens, num_heads)
        weights = indexer.head_weights(x).float()
        compress_lens = ((pos + 1) // ratio).to(torch.int32)
        ks = torch.repeat_interleave(
            h2d(starts, torch.int32, device) if nosync else
            torch.tensor(starts, dtype=torch.int32, device=device),
            q_lens.to(torch.int64),
            output_size=num_tokens,
        )
        topk = indexer.index_topk
        selected = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
        _tp = getattr(self, "_dsv41_native_prefill_tp", None)
        if _tp is not None and _tp.eligible(
            num_tokens, max(lc_per_req),
            prefill=forward_batch.forward_mode.is_extend_without_speculative(),
            device=device,
        ):
            _consume = (
                published_masks(self.forward_metadata.candidate_metadata).request_masks
                if indexer.uses_candidates and not indexer.is_candidate_source else None
            )
            _tail = self.tail_forward_metadata
            _pub_rows = None
            if (indexer.is_candidate_source and _tail is not None
                    and _tail.late_layer_tail is not None
                    and getattr(_tail.late_layer_tail, "cp_metadata", None) is None):
                _pub_rows = list(_tail.late_layer_tail.extend_seq_lens_cpu)
            selected, _masks = _split_select(
                _tp, indexer, q_fp4, q_sf, k_fp4, k_sf, weights, ks, compress_lens,
                lc_per_req, q_lens_cpu, budget=_DENSE_INDEXER_LOGITS_BUDGET_BYTES,
                scorer=_dense_fp4_mqa_logits, topk_fn=topk_transform_ragged_v2,
                mask_topk=mask_topk_scores, consume=_consume, publish_rows=_pub_rows,
                layer_id=layer.layer_id, ratio=ratio, ceil_align=ceil_align,
            )
            if indexer.is_candidate_source:
                self.forward_metadata.candidate_metadata = CandidateMasks(request_masks=_masks)
            unselected = torch.iinfo(torch.int32).max
            selected = selected.masked_fill(selected < 0, unselected).sort(dim=-1).values
            chosen = selected != unselected
            page_indices[:num_tokens, :topk] = torch.where(
                chosen, k_slots[selected.clamp_max(k_slots.shape[0] - 1)], -1
            ).to(torch.int32)
            if raw_indices is not None:
                raw_indices[:num_tokens, :topk] = torch.where(
                    chosen, selected - ks[:, None], -1
                )
            return

        publish = [] if indexer.is_candidate_source else None
        consume = (
            published_masks(self.forward_metadata.candidate_metadata).request_masks
            if indexer.uses_candidates and publish is None
            else None
        )
        publish_rows_per_req = None
        tail_metadata = self.tail_forward_metadata
        if (
            publish is not None
            and tail_metadata is not None
            and tail_metadata.late_layer_tail is not None
            and getattr(tail_metadata.late_layer_tail, "cp_metadata", None) is None
        ):
            publish_rows_per_req = tail_metadata.late_layer_tail.extend_seq_lens_cpu
            assert len(publish_rows_per_req) == len(q_lens_cpu)

        tok_start = 0
        for b, (lc, t_len) in enumerate(zip(lc_per_req, q_lens_cpu)):
            rows_b = slice(tok_start, tok_start + t_len)
            tok_start += t_len
            if lc == 0 or t_len == 0:
                if publish is not None:
                    publish.append(empty_mask)
                if t_len:
                    selected[rows_b].fill_(-1)
                continue
            lc_aligned = ceil_align(lc, 4)
            rows_per_chunk = max(
                1, _DENSE_INDEXER_LOGITS_BUDGET_BYTES // (lc_aligned * 4)
            )
            mask_b = None
            first_pub = 0
            if publish is not None:
                pub_rows = (
                    t_len
                    if publish_rows_per_req is None
                    else min(int(publish_rows_per_req[b]), t_len)
                )
                first_pub = t_len - pub_rows
                mask_b = torch.empty((pub_rows, lc), dtype=torch.bool, device=device)
            j = torch.arange(lc, device=device)
            for c0 in range(0, t_len, rows_per_chunk):
                c1 = min(c0 + rows_per_chunk, t_len)
                rows = slice(rows_b.start + c0, rows_b.start + c1)
                lens = compress_lens[rows]
                logits = _dense_fp4_mqa_logits(
                    (q_fp4[rows], q_sf[rows]),
                    (k_fp4, k_sf),
                    weights[rows],
                    ks[rows],
                    ks[rows] + lens,
                    lc_aligned,
                )
                scores = logits[:, :lc]
                if publish is not None:
                    scores.masked_fill_(j[None, :] >= lens[:, None], -torch.inf)
                    p0 = max(c0, first_pub)
                    if p0 < c1:
                        mask_b[p0 - first_pub : c1 - first_pub] = (
                            select_candidate_blocks(
                                scores[p0 - c0 :],
                                lens[p0 - c0 :, None],
                                topk_blocks=indexer.candidate_topk_blocks,
                                block_size=indexer.candidate_block_size,
                            )
                        )
                elif consume is not None:
                    scores.masked_fill_(~consume[b][c0:c1], -torch.inf)
                topk_transform_ragged_v2(
                    logits, lens, out_offsets=ks[rows], out_indices=selected[rows]
                )
                if consume is not None:
                    selected[rows] = mask_topk_scores(logits, selected[rows], ks[rows])
                del logits, scores
            if publish is not None:
                publish.append(mask_b)
        if publish is not None:
            self.forward_metadata.candidate_metadata = CandidateMasks(
                request_masks=publish
            )
        unselected = torch.iinfo(torch.int32).max
        selected = selected.masked_fill(selected < 0, unselected).sort(dim=-1).values
        chosen = selected != unselected
        page_indices[:num_tokens, :topk] = torch.where(
            chosen, k_slots[selected.clamp_max(k_slots.shape[0] - 1)], -1
        ).to(torch.int32)
        if raw_indices is not None:
            raw_indices[:num_tokens, :topk] = torch.where(
                chosen, selected - ks[:, None], -1
            )
    # ---- end verbatim ----
    return _low_ratio_index_topk_dense


NEEDED = ("_dense_fp4_mqa_logits", "topk_transform_ragged_v2", "mask_topk_scores",
          "select_candidate_blocks", "ceil_align", "_as_int_list", "CandidateMasks",
          "published_masks", "get_parallel")


def install(module):
    if not _enabled("DSV41_INDEXER_CHUNKED"):
        return
    cls = getattr(module, "DeepseekV4AttnBackend", None)
    missing = [n for n in NEEDED if not hasattr(module, n)]
    if cls is None or missing or not hasattr(cls, "_low_ratio_index_topk_dense"):
        raise RuntimeError(f"DSV41 indexer chunked v3: backend drifted (missing {missing or 'class/method'}); refusing to boot")
    stock_src = inspect.getsource(cls._low_ratio_index_topk_dense)
    if "candidate_metadata" not in stock_src or "self.candidate_masks" in stock_src:
        raise RuntimeError("DSV41 indexer chunked v2: stock indexer is not the candidate_metadata variant; use indexer_chunked (v1) -- refusing to boot")
    # Override site: when the stock indexer already carries sglang#39187, the adapter used to log
    # "leaving it alone" and return, so DSV41_INDEXER_CHUNKED=1, DSV41_INDEXER_LOGITS_BUDGET_BYTES and
    # SPARK_PREFILL_TP_SPLIT took no effect. A gate the user set must take effect, so this now stops
    # the boot.
    if "_DENSE_INDEXER_LOGITS_BUDGET_BYTES" in stock_src:
        raise RuntimeError(
            f"DSV41_INDEXER_CHUNKED={os.environ.get('DSV41_INDEXER_CHUNKED', '')!r}: the stock indexer already "
            f"carries sglang#39187, so the v3 replacement would not be installed and "
            f"DSV41_INDEXER_LOGITS_BUDGET_BYTES / SPARK_PREFILL_TP_SPLIT="
            f"{os.environ.get('SPARK_PREFILL_TP_SPLIT', '0')!r} would be ignored; unset DSV41_INDEXER_CHUNKED "
            f"-- refusing to boot")
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import quantize_fp4_indexer_tensor
    budget = budget_bytes()
    cls._low_ratio_index_topk_dense = make_index_topk_dense(module, quantize_fp4_indexer_tensor, budget)
    split_on = _enabled("SPARK_PREFILL_TP_SPLIT")
    if split_on:
        from spark_prefill_dense import NativePrefillTPSplit
        get_parallel = module.get_parallel
        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._dsv41_native_prefill_tp = NativePrefillTPSplit.from_parallel(
                get_parallel(), is_draft=self.is_draft_runner
            )

        cls.__init__ = __init__
    logger.warning("DSV41 indexer chunked v3 (sglang#39187 verbatim + SG18 prefill TP split=%s) ARMED: <= %d MiB fp32 logits per row chunk, tail-only candidate masks; DSV41_INDEXER_CHUNKED=0 disables", split_on, budget >> 20)
    if nosync_value(os.environ.get("DSV41_INDEXER_NOSYNC", "0")) or (_ab is not None and getattr(_ab, "ACTIVE", False)):
        print("DSV41 indexer nosync: DSV41_INDEXER_NOSYNC=%s (in-boot A/B %s): index-source slot ids without host "
              "syncs when on" % (os.environ.get("DSV41_INDEXER_NOSYNC", "0"),
                                 "armed, read per call" if _ab is not None and getattr(_ab, "ACTIVE", False) else "off"),
              flush=True)
