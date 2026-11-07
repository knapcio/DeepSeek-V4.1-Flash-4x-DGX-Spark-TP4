"""Route small TP all-gathers through the RoCEnante one-shot kernel (DSV41_ROCE_GATHER=<max bytes per rank>).

The SG17 overlay builds the runtime with max_gather_bytes=0, so every all-gather (draft logits /
markov gathers, a few per step) still goes to NCCL. The runtime's slots are max(max_size, gather)
wide, so any shard up to max_size fits without re-sizing; the gather launcher and the padded
scratch are prepared here, before CUDA graph capture. Byte copy: outputs are bit-identical.
"""
import logging
import os

logger = logging.getLogger(__name__)
_WARNED = []


def install(module):
    cls = module.PyNcclCommunicator
    limit = int(os.environ.get('DSV41_ROCE_GATHER', '0') or 0)
    orig_init, orig_gather = cls.__init__, cls.all_gather

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        roce = getattr(self, 'roce', None)
        # Fall-through site: a communicator without a RoCE runtime keeps NCCL for every all-gather,
        # so DSV41_ROCE_GATHER takes no effect on it. Kept (groups other than TP have no RoCE runtime);
        # one WARNING line per process.
        if roce is None and limit > 0 and not _WARNED:
            _WARNED.append(True)
            logger.warning('WARNING DSV41_ROCE_GATHER=%d: communicator rank=%s has no RoCE runtime; '
                           'kernel chosen: NCCL all-gather on such communicators (logged once)',
                           limit, getattr(self, 'rank', '?'))
        if roce is None or limit <= 0:
            return
        # Override site: a DSV41_ROCE_GATHER above the runtime's slot size was silently clamped to the
        # slot size. A value the user set must take effect, so it now stops the boot.
        if limit > roce._slot_bytes:
            raise RuntimeError(f'DSV41_ROCE_GATHER={limit}: above the RoCE slot size {roce._slot_bytes} bytes; '
                               f'it would be clamped to {roce._slot_bytes}; refusing to boot')
        roce.max_gather_bytes = min(limit, roce._slot_bytes)
        roce.prepare((), padded_gather=True)
        self._roce_gather_logged = False
        logger.info('DSV41_ROCE_GATHER ready: max %d bytes per rank', roce.max_gather_bytes)

    def all_gather(self, output_tensor, input_tensor, sizes=None):
        roce = getattr(self, 'roce', None)
        if (roce is not None and not self.disabled and sizes is None and roce.max_gather_bytes > 0
                and output_tensor.is_contiguous()
                and output_tensor.numel() == input_tensor.numel() * self.world_size
                and roce.should_all_gather(input_tensor, 0)):
            out = output_tensor.view(self.world_size * input_tensor.shape[0], *input_tensor.shape[1:]) \
                if input_tensor.dim() > 0 else output_tensor
            roce.all_gather(input_tensor, dim=0, out=out, stream=self._resolve_stream())
            if not self._roce_gather_logged:
                logger.info('DSV41_ROCE_GATHER route rank=%d dtype=%s bytes=%d', self.rank,
                            input_tensor.dtype, input_tensor.numel() * input_tensor.element_size())
                self._roce_gather_logged = True
            return
        return orig_gather(self, output_tensor, input_tensor, sizes)

    cls.__init__, cls.all_gather = __init__, all_gather
