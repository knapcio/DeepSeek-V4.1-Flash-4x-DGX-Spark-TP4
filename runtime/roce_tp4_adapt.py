"""Adapt the SG17 RoCEnante overlay (TP8-only) to TP4 after `patch -p1` applied it.

DSV41_ROCE_RING=1 loads b12x.comm.roce_ring (per-peer paths for a switchless ring, see
docs/switchless-ring.md) instead of b12x.comm.roce; unset, the overlay is unchanged."""
p = "/sgl-workspace/sglang/python/sglang/srt/distributed/device_communicators/pynccl.py"
s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a
    s = s.replace(a, b)
rep('if roce_enabled and os.environ.get("SGLANG8_ROCE_ALLREDUCE", "0") == "1":\n            if self.world_size != 8 or isinstance(group, StatelessProcessGroup):\n                raise RuntimeError("SG17 RoCEnante requires the TP8 Gloo exchange group")',
    'if roce_enabled and (os.environ.get("SGLANG_ROCE_ALLREDUCE", "0") == "1"\n                             or os.environ.get("SGLANG8_ROCE_ALLREDUCE", "0") == "1"):\n            if self.world_size not in (4, 8) or isinstance(group, StatelessProcessGroup):\n                raise RuntimeError("RoCEnante route needs the TP4/TP8 Gloo exchange group")')
rep('if len(self.roce.hca_names) != 2:\n                raise RuntimeError("SG17 RoCEnante requires two active RoCE interfaces")',
    'if len(self.roce.hca_names) < 1:\n                raise RuntimeError("RoCEnante found no active RoCE interface")\n            if len(self.roce.hca_names) == 1:\n                logger.warning("ROCE single-rail: %s", self.roce.hca_names)')
rep('            from b12x.comm import roce\n',
    '            import importlib\n'
    '            roce = importlib.import_module(\n'
    '                "b12x.comm.roce_ring" if os.environ.get("DSV41_ROCE_RING", "0") == "1" else "b12x.comm.roce")\n')
rep('max_size=512 * 1024, max_gather_bytes=0,', 'max_size=int(os.environ.get("SGLANG_ROCE_MAX_SIZE", str(512 * 1024))), max_gather_bytes=0,')
open(p, "w").write(s)
import ast; ast.parse(s); print("pynccl.py adapted to TP4")
