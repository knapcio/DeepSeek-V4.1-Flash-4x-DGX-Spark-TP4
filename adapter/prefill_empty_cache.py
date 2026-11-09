"""Give a long prefill's transient memory back after every chunk.

Each 2048-token chunk of a DeepSeek-V4 prefill scores its queries against the whole
prefix so far: the indexer logits are a [chunk, prefix] fp32 buffer per low-ratio
source layer (deepseek_v4_backend: _dense_fp4_mqa_logits with max_seqlen_k = the
longest compressed length), plus candidate masks and top-k scratch of the same shape.
They are freed after the layer, but PyTorch's native caching allocator cannot serve
the next chunk's slightly larger request from the smaller cached block, so it
cudaMallocs a new one every chunk and reserved memory grows with the SQUARE of the
prompt (logs/profile-2026-09-10/REPORT.md sections 15 and 17). On a Spark that
reserved memory is host memory: a 34k-token prompt kept 1-1.5 GB, 64k exhausted the
head. expandable_segments would coalesce it but produces NaN logits on this stack.

This hook calls torch.cuda.empty_cache() after every extend forward whose longest
sequence is at least DSV41_PREFILL_EMPTY_CACHE_TOKENS (default 8192): the unused
cached blocks go back to the driver, the next chunk allocates fresh, and the peak is
one chunk's live set instead of the sum over chunks. Decode CUDA graphs live in
private pools and are untouched. Cost: one device sync and a few cudaMalloc/cudaFree
per long-prefill chunk (milliseconds against a 300-800 ms chunk). 0 disables it.

DSV41_PREFILL_KEEP_CACHE=1 (default 0, adapter/prefill_keep_cache.py): between the chunks of one prefill the cache is
kept (the indexer's growing tensors are padded to the prompt's final width so the next chunk reuses the same blocks)
and empty_cache() runs only after the prefill's last chunk, on an unknown final length, or when the memory guard
trips. Allocation only. Off = the hook below, byte for byte the same behaviour.
"""
import logging
import os

import torch

logger = logging.getLogger(__name__)


def install(module):
    threshold = int(os.environ.get('DSV41_PREFILL_EMPTY_CACHE_TOKENS', '8192'))
    if threshold <= 0:
        return
    cls = module.ModelRunner
    original = cls.forward
    keep = None
    try:
        import prefill_keep_cache as _keep
        if _keep.wanted_at_install():
            keep = _keep
    except ImportError:
        keep = None

    def _long_extend(forward_batch):
        if not forward_batch.forward_mode.is_extend_without_speculative():
            return False
        lens = forward_batch.seq_lens_cpu
        longest = int(lens.max()) if lens is not None and lens.numel() else 0
        return longest >= threshold

    def forward(self, forward_batch, *args, **kwargs):
        if keep is not None:
            return _forward_keep(self, forward_batch, args, kwargs)
        output = original(self, forward_batch, *args, **kwargs)
        try:
            if _long_extend(forward_batch):
                torch.cuda.empty_cache()
        except Exception as exc:  # bookkeeping must never break a forward
            logger.warning('DSV41 prefill empty_cache hook: %s', exc)
        return output

    state = {'warned': False}

    def _forward_keep(self, forward_batch, args, kwargs):
        try:
            gate = keep.enabled()
            on = gate and _long_extend(forward_batch)
        except Exception as exc:  # noqa: BLE001
            logger.warning('DSV41 prefill keep-cache hook: %s', exc)
            gate, on = False, False
        if not on:
            if not gate:
                keep.release(torch, 'gate-off')   # in-boot A/B switched it off: nothing may stay held
            else:
                try:
                    mode = forward_batch.forward_mode
                    if not getattr(self, 'is_draft_worker', False) and not mode.is_extend_without_speculative():
                        # decode / verify between the chunks of a kept prompt: stop keeping it (stall-11 receipt:
                        # the held peak plus the decode steps' allocations cost 0.45 GiB of head low-water)
                        keep.release_for_decode(torch)
                except Exception as exc:  # noqa: BLE001
                    logger.warning('DSV41 prefill keep-cache decode release: %s', exc)
            output = original(self, forward_batch, *args, **kwargs)
            try:
                if _long_extend(forward_batch):
                    torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                logger.warning('DSV41 prefill empty_cache hook: %s', exc)
            return output
        policy = keep.policy()
        forward_batch.dsv41_pad_ok = False    # stock widths unless the bookkeeping below completes
        try:
            batch = keep.batch_of(forward_batch)
            if batch.finals is None and not state['warned']:
                state['warned'] = True
                logger.warning('DSV41_PREFILL_KEEP_CACHE: no final prompt lengths on the batch; emptying every chunk')
            empty_first, why = policy.begin(batch, keep.read_stats(torch))
            if empty_first:
                # the held cache is not this prompt's (another prompt, a mixed batch, more rows, a gap), or a new
                # prompt must start from an empty cache for its own guard reference
                torch.cuda.empty_cache()
                policy.emptied(keep.read_stats(torch))
            forward_batch.dsv41_pad_ok = policy.pad_ok()
            torch.cuda.reset_peak_memory_stats()
            policy.before(keep.read_stats(torch))
        except Exception as exc:  # noqa: BLE001 - fall back to today's behaviour for this forward
            logger.warning('DSV41 prefill keep-cache stats: %s', exc)
            forward_batch.dsv41_pad_ok = False
            batch = None
        output = original(self, forward_batch, *args, **kwargs)
        try:
            if batch is None:
                raise RuntimeError('no batch descriptor')
            empty, reason = policy.after(batch, keep.read_stats(torch))
            if empty:
                torch.cuda.empty_cache()
                policy.emptied(keep.read_stats(torch))
                if reason.startswith('trip'):
                    logger.warning('DSV41_PREFILL_KEEP_CACHE guard %s: cache emptied (slack above frag_ref + %d MiB)',
                                   reason, keep.slack_bytes() >> 20)
            while policy.sessions:
                s = policy.sessions.pop()
                if s['chunks'] > 1 or any(k not in ('end', 'unknown-final') for k in s['empty']):
                    print(keep.receipt(s), flush=True)
        except Exception as exc:  # bookkeeping must never break a forward; fall back to emptying
            logger.warning('DSV41 prefill keep-cache hook: %s', exc)
            try:
                torch.cuda.empty_cache()
                policy.reset()
            except Exception:  # noqa: BLE001
                pass
        return output

    cls.forward = forward
    logger.warning('DSV41 prefill empty_cache hook installed (extend forwards with a '
                   'sequence >= %d tokens)%s', threshold,
                   '; DSV41_PREFILL_KEEP_CACHE: kept between the chunks of one prefill' if keep is not None else '')
