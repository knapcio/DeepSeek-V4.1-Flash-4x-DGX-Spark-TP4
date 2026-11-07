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

DSV41_PREFILL_TRIM=1 (off by default; unset = the per-chunk empty_cache above, unchanged): the
working-set-preserving trim of Eric Jinho Jang (dealign.ai), jjang-ai/dealignai-deepseek-v4.1-flash-1017-4x-dgx-spark,
commits c175335 / a9f87fe, adapter/prefill_empty_cache.py (AGPL-3.0-or-later; a modification of this file). Their text:

  The upstream hook called torch.cuda.empty_cache() after every long extend forward: the peak stays one chunk's live
  set, but a 1M prompt is ~128 multi-GB cudaMalloc/cudaFree cycles of unified memory. This hook keeps the scratch
  cached between chunks and trims instead: after every forward (two counter reads), if cached-but-free memory exceeds
  the working set (the last long forward's transient peak) + DSV41_PREFILL_TRIM_GIB (1), the largest free blocks
  covering the working set are held for a moment, empty_cache() returns everything else, and the held blocks go back
  to the cache. The next chunk finds its scratch cached, and the waste that mixed batches leave behind (vision, chat
  and long prompts in one batch) is returned. At most every DSV41_PREFILL_TRIM_EVERY_S (10 s).
  Measured on 4x GB10, 4.26M-token KV pool: 30-min mixed soak (8 clients) with reserved memory flat at ~92.5 GiB and
  every node >= 2 GiB available; cold 1M prefill unchanged. Do not add max_split_size_mb to PYTORCH_CUDA_ALLOC_CONF:
  oversize blocks are then not reused by the growing indexer scratch, and reserved memory ran away to 110 GiB within
  minutes.

Port notes: the logic is theirs; here it is a class over an injectable `cuda` / clock so the decisions are CPU-tested
(tests/test_prefill_trim.py), the hold loop stops at the first block whose allocation grew the reserve (theirs too),
and a trim's failure never fails the forward (theirs too). Allocation only: no kernel or value changes. Memory: up to
the working set + DSV41_PREFILL_TRIM_GIB stays cached per rank after a long prefill (by design), which is host
memory on a Spark; the per-chunk path returns it every chunk. DSV41_PREFILL_MEM_PROBE=1 logs allocator vs host
memory every 8 long forwards (either mode).
"""
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)
_GIB = 1 << 30
_MIN_HOLD = 64 << 20          # only blocks this large are worth holding through a trim
_ON = ('1', 'on', 'true', 'yes')


def trim_enabled(env=None):
    env = os.environ if env is None else env
    return str(env.get('DSV41_PREFILL_TRIM', '0')).strip().lower() in _ON


class Trim:
    """Per-process state (one scheduler process per TP rank)."""

    def __init__(self, cuda=None, clock=None, trim_bytes=_GIB, every_s=10.0, min_hold=_MIN_HOLD,
                 hold=None):
        self.cuda = torch.cuda if cuda is None else cuda
        self.hold = hold or (lambda size: torch.empty(size, dtype=torch.uint8, device='cuda'))
        self.clock = time.monotonic if clock is None else clock
        self.trim_bytes = int(trim_bytes)
        self.every_s = float(every_s)
        self.min_hold = int(min_hold)
        self.last = None          # monotonic time of the last trim (None: never)
        self.trims = 0
        self.working = 0          # bytes: the last long forward's transient peak
        self.base = 0

    def before_long(self):
        self.cuda.reset_peak_memory_stats()
        self.base = self.cuda.memory_allocated()

    def after_long(self):
        self.working = max(0, self.cuda.max_memory_allocated() - self.base)

    def _free_blocks(self):
        # only free blocks this allocation can reuse: default pool (not CUDA-graph or other private pools), the
        # current stream
        stream = self.cuda.current_stream().cuda_stream
        return sorted((b['size'] for seg in self.cuda.memory_snapshot()
                       if tuple(seg.get('segment_pool_id', (0, 0))) == (0, 0) and seg.get('stream') == stream
                       for b in seg['blocks'] if b['state'] == 'inactive' and b['size'] >= self.min_hold),
                      reverse=True)

    def after(self):
        """Returns the bytes kept through a trim, or None when no trim ran."""
        reserved, allocated = self.cuda.memory_reserved(), self.cuda.memory_allocated()
        if reserved - allocated <= self.working + self.trim_bytes:
            return None
        now = self.clock()
        if self.last is not None and now - self.last < self.every_s:
            return None
        self.last = now
        self.trims += 1
        held, kept = [], 0
        if self.working:
            for size in self._free_blocks():
                if kept >= self.working:
                    break
                before = self.cuda.memory_reserved()
                held.append(self.hold(size))
                if self.cuda.memory_reserved() > before:   # the allocator had to cudaMalloc: stop
                    break
                kept += size
        self.cuda.empty_cache()
        del held
        if self.trims <= 3 or self.trims % 50 == 0:
            logger.warning('DSV41 prefill trim %d: %.2f GiB cached but free, kept %.2f GiB working set -> '
                           'reserved %.2f GiB', self.trims, (reserved - allocated) / _GIB, kept / _GIB,
                           self.cuda.memory_reserved() / _GIB)
        return kept


def _meminfo_gib(key):
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith(key + ':'):
                return int(line.split()[1]) / 1048576
    return -1.0


_probe_n = [0]


def _probe(longest):
    _probe_n[0] += 1
    if _probe_n[0] % 8:
        return
    st = torch.cuda.memory_stats()
    logger.warning('DSV41_MEM_PROBE longest=%d alloc=%.2f reserved=%.2f peak=%.2f memavail=%.2f '
                   'segments=%d inactive_split=%.2f',
                   longest, torch.cuda.memory_allocated() / _GIB, torch.cuda.memory_reserved() / _GIB,
                   torch.cuda.max_memory_allocated() / _GIB, _meminfo_gib('MemAvailable'),
                   st.get('segment.all.current', 0), st.get('inactive_split_bytes.all.current', 0) / _GIB)


def _longest_extend(forward_batch, threshold):
    if not forward_batch.forward_mode.is_extend_without_speculative():
        return 0
    lens = forward_batch.seq_lens_cpu
    longest = int(lens.max()) if lens is not None and lens.numel() else 0
    return longest if longest >= threshold else 0


def install(module, env=None):
    env = os.environ if env is None else env
    threshold = int(env.get('DSV41_PREFILL_EMPTY_CACHE_TOKENS', '8192'))
    if threshold <= 0:
        return
    cls = module.ModelRunner
    original = cls.forward
    probe = str(env.get('DSV41_PREFILL_MEM_PROBE', '0')).strip() not in ('0', '')

    if not trim_enabled(env):
        def forward(self, forward_batch, *args, **kwargs):
            output = original(self, forward_batch, *args, **kwargs)
            try:
                longest = _longest_extend(forward_batch, threshold)
                if longest:
                    torch.cuda.empty_cache()
                    if probe:
                        _probe(longest)
            except Exception as exc:  # bookkeeping must never break a forward
                logger.warning('DSV41 prefill empty_cache hook: %s', exc)
            return output

        cls.forward = forward
        logger.warning('DSV41 prefill empty_cache hook installed (extend forwards with a '
                       'sequence >= %d tokens)', threshold)
        return

    trim = Trim(trim_bytes=float(env.get('DSV41_PREFILL_TRIM_GIB', '1')) * _GIB,
                every_s=float(env.get('DSV41_PREFILL_TRIM_EVERY_S', '10')))

    def forward(self, forward_batch, *args, **kwargs):
        longest = 0
        try:
            longest = _longest_extend(forward_batch, threshold)
            if longest:
                trim.before_long()
        except Exception as exc:  # bookkeeping must never break a forward
            logger.warning('DSV41 prefill trim hook (before): %s', exc)
        output = original(self, forward_batch, *args, **kwargs)
        try:
            if longest:
                trim.after_long()
                if probe:
                    _probe(longest)
            trim.after()
        except Exception as exc:
            logger.warning('DSV41 prefill trim hook (after): %s', exc)
        return output

    cls.forward = forward
    logger.warning('DSV41 prefill trim hook installed (DSV41_PREFILL_TRIM=1: long prefill >= %d tokens; trim above '
                   'working set + %.2f GiB cached-free, every >= %.0f s)', threshold, trim.trim_bytes / _GIB,
                   trim.every_s)
