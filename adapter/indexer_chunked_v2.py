"""Bound the dense prefill indexer transient on the upstream dsv4.1 branch: verbatim
backport of sgl-project/sglang#39187 (head cb8dd033, kpham-sgl) for backends that keep
candidate masks in ``forward_metadata.candidate_metadata`` (branch f80c91a4b and later).
The v1 adapter (indexer_chunked.py) covers image e087e662, which uses self.candidate_masks.

Gate: ``DSV41_INDEXER_CHUNKED=1``.  Budget: ``DSV41_INDEXER_LOGITS_BUDGET_BYTES``
(default 2 GiB of fp32 logits).  Off = stock path untouched.
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
        k_slots = torch.cat(slot_chunks)
        k_fp4, k_sf = pool.get_low_ratio_index_k_fp4(layer.layer_id, k_slots)

        q = indexer.queries(q_lora, layer.freqs_cis[pos])  # [T, H, 128] fp4 grid
        num_heads = q.shape[1]
        q_fp4, q_sf = quantize_fp4_indexer_tensor(q.flatten(0, 1), rne=True)
        q_fp4 = q_fp4.view(num_tokens, num_heads, 64)
        q_sf = q_sf.view(num_tokens, num_heads)
        weights = indexer.head_weights(x).float()
        compress_lens = ((pos + 1) // ratio).to(torch.int32)
        ks = torch.repeat_interleave(
            torch.tensor(starts, dtype=torch.int32, device=device),
            q_lens.to(torch.int64),
            output_size=num_tokens,
        )
        topk = indexer.index_topk
        selected = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)

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
          "published_masks")


def install(module):
    if not _enabled("DSV41_INDEXER_CHUNKED"):
        return
    cls = getattr(module, "DeepseekV4AttnBackend", None)
    missing = [n for n in NEEDED if not hasattr(module, n)]
    if cls is None or missing or not hasattr(cls, "_low_ratio_index_topk_dense"):
        raise RuntimeError(f"DSV41 indexer chunked v2: backend drifted (missing {missing or 'class/method'}); refusing to boot")
    stock_src = inspect.getsource(cls._low_ratio_index_topk_dense)
    if "candidate_metadata" not in stock_src or "self.candidate_masks" in stock_src:
        raise RuntimeError("DSV41 indexer chunked v2: stock indexer is not the candidate_metadata variant; use indexer_chunked (v1) -- refusing to boot")
    # Override site: when the stock indexer already carries sglang#39187, the adapter used to log
    # "leaving it alone" and return, so DSV41_INDEXER_CHUNKED=1 and DSV41_INDEXER_LOGITS_BUDGET_BYTES
    # took no effect. A gate the user set must take effect, so this now stops the boot.
    if "_DENSE_INDEXER_LOGITS_BUDGET_BYTES" in stock_src:
        raise RuntimeError(
            f"DSV41_INDEXER_CHUNKED={os.environ.get('DSV41_INDEXER_CHUNKED', '')!r}: the stock indexer already "
            f"carries sglang#39187, so the v2 replacement would not be installed and "
            f"DSV41_INDEXER_LOGITS_BUDGET_BYTES would be ignored; unset DSV41_INDEXER_CHUNKED -- refusing to boot")
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import quantize_fp4_indexer_tensor
    budget = budget_bytes()
    cls._low_ratio_index_topk_dense = make_index_topk_dense(module, quantize_fp4_indexer_tensor, budget)
    logger.warning("DSV41 indexer chunked v2 (sglang#39187 verbatim) ARMED: <= %d MiB fp32 logits per row chunk, tail-only candidate masks; DSV41_INDEXER_CHUNKED=0 disables", budget >> 20)
