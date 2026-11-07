"""CPU checks of DSV41_KV_LAYOUT=v41c step 3: SGLang's DeepSeek-V4 pools hold the ratio-1/2 caches as V4.1 FP4 records
(adapter/v41c_pool.py, wired by adapter/sitecustomize.py).

The SGLang code under test is the real source of the staged v2.6 tree (runtime/sglang-canary/python, f744d83; or the
image's /sgl-workspace/sglang/python; or SGLANG_CANARY_PY): the functions the patch wraps or relies on are cut out of
their files with ast and executed against small stubs of the runtime context, so the stock sizing / allocation code
runs as written. The patch's source pins are checked against the same files.

- flag: unset / v4 off, v41c on, anything else raises (== dsv41_v41c.enabled()).
- pins: every PINS entry equals the staged tree; a one-character edit refuses; KVLayout values are the record ABI.
- layout resolution (real resolve_compressed_kv_layout / compressed_kv_layout / is_valid_kv_layout_pair): off -> V4
  for every ratio; on -> V41_FP4 for ratios 1 / 2 with a V4 main cache, stock for 4 / 128 and the V41 main cache; a
  forced SGLANG_DSV4_COMPRESSED_KV_LAYOUT and a non-FlashInfer SM120 backend refuse.
- sizing (real DSV4PoolConfigurator.__init__ / calculate_pool_sizes on the production profile: TP4, DSpark, paged
  SWA cap mode, 1M context, 16 requests): stock bytes_per_full_token == 1670.75 (the value every v2.6 boot logs),
  v41c == 912.25, the SWA terms unchanged, the same byte budget -> 1.8315x tokens.
- allocation (torch; real DeepSeekV4SingleKVPool + _init_compressed_pools): the ratio-1 / ratio-2 pools are
  V41_FP4 with 73,728 / 36,864-byte pages (256 / 128 tokens), full_size // ratio tokens, and their bytes are
  288 / 584 x ... of the V4 pools; off -> exactly the stock V4 pools. The receipt line names them.
- default off: sitecustomize.py minus the v41c lines is byte-identical to f744d83's; with the flag unset the finder
  list is f744d83's and v41c_pool is never imported; set to v41c the two pool modules are hooked.

usage: python3 tests/test_v41c_pool.py        (torch for the allocation cases: ds-b3-cutecheck:1 / the image)
"""
import ast
import contextlib
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADAPTER = ROOT / "adapter"
sys.path.insert(0, str(ADAPTER))
import v41c_pool as P  # noqa: E402

# the base sitecustomize.py: v2.6.4 + the issue #66 fix (main @ 540bf07: the history_repair hook) on this branch
SITE_F744D83_SHA = "ebc976542bf989bbc6f5a77b5032447f639cf72dfa2101de2e2100576d6f8011"

try:
    import torch
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False


def sglang_root():
    for c in (os.environ.get("SGLANG_CANARY_PY"), ROOT / "runtime" / "sglang-canary" / "python",
              ROOT.parent / "ds41-release" / "runtime" / "sglang-canary" / "python", "/sgl-workspace/sglang/python"):
        if c and (Path(c) / "sglang" / "srt" / "mem_cache" / "deepseek_v4_memory_pool.py").exists():
            return Path(c)
    return None


SG = sglang_root()
FILES = {} if SG is None else {
    P.POOL_MODULE: SG / "sglang/srt/mem_cache/deepseek_v4_memory_pool.py",
    P.CONFIG_MODULE: SG / "sglang/srt/model_executor/pool_configurator.py",
    P.BACKEND_MODULE: SG / "sglang/srt/layers/attention/deepseek_v4_backend.py",
}
KV_LAYOUT_PY = None if SG is None else SG / "sglang/kernels/ops/attention/dsv4/kv_layout.py"
NEED_SG = unittest.skipIf(SG is None, "no staged SGLang tree (SGLANG_CANARY_PY)")

# DeepSeek-V4.1-Flash text_config (40 layers + 3 MTP; the engine's stage covers the 40)
RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
SOURCES = [2, 8, 14, 20]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cut(path, names):
    """Source of the named top-level defs / classes / assignments of a file, in file order."""
    text = Path(path).read_text()
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    out, want = [], set(names)
    for n in tree.body:
        nm = n.name if isinstance(n, (ast.FunctionDef, ast.ClassDef)) else (
            n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else None)
        if nm in want:
            first = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
            out.append("".join(lines[first - 1:n.end_lineno]))
            want.discard(nm)
    assert not want, f"not found in {path}: {want}"
    return "\n\n".join(out)


class Env:
    """envs.X.get() stub."""
    def __init__(self, **kv):
        self.kv = {"SGLANG_SM120_FLASHMLA_BACKEND": "flashinfer", "SGLANG_DSV4_COMPRESS_STATE_DTYPE": "float32",
                   "SGLANG_OPT_USE_ONLINE_COMPRESS": False, "SGLANG_EXPERIMENTAL_ONLINE_C128_MTP": False,
                   "SGLANG_SWA_EVICTION_INTERVAL": 128, **kv}

    def __getattr__(self, k):
        if k == "kv":
            raise AttributeError(k)
        return types.SimpleNamespace(get=lambda: self.kv[k])


@contextlib.contextmanager
def env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def kv_layout_module():
    return _load("kv_layout_under_test", KV_LAYOUT_PY)


def pool_module(backend="flashinfer"):
    """The pool module's layout / allocation code (real source) on stubs; __file__ = the real file (for the pins)."""
    kl = kv_layout_module()
    src = cut(FILES[P.POOL_MODULE], ["resolve_compressed_kv_layout", "DeepSeekV4SingleKVPool", "DeepSeekV4TokenToKVPool",
                                     "_CompressedPoolConfig"])
    mod = types.ModuleType("v41c_pool_stub_" + P.POOL_MODULE.replace(".", "_"))
    ns = mod.__dict__

    class KVCache:
        def __init__(self, size, page_size, dtype, layer_num, device, enable_memory_saver, start_layer=None,
                     end_layer=None):
            self.size, self.page_size, self.dtype, self.layer_num, self.device = size, page_size, dtype, layer_num, device
            self.store_dtype = torch.uint8
            self.start_layer = start_layer or 0
            self.memory_saver_adapter = types.SimpleNamespace(region=lambda *_: contextlib.nullcontext())
            self.custom_mem_pool = None

    ns.update(dict(KVLayout=kl.KVLayout, is_valid_kv_layout_pair=kl.is_valid_kv_layout_pair, KVCache=KVCache,
                   BaseSWAKVPool=object, envs=Env(SGLANG_SM120_FLASHMLA_BACKEND=backend), logger=P.logger,
                   torch=torch if HAVE_TORCH else None, nullcontext=contextlib.nullcontext, ceil_div=lambda a, b: -(-a // b),
                   GPU_MEMORY_TYPE_KV_CACHE="kv_cache", NamedTuple=__import__("typing").NamedTuple,
                   DSV41_INDEX_PAGE_SIZE=64))
    exec(compile("from __future__ import annotations\nfrom typing import *\n" + src, str(FILES[P.POOL_MODULE]), "exec"), ns)
    mod.__file__ = str(FILES[P.POOL_MODULE])
    mod.__name__ = P.POOL_MODULE
    return mod, kl


# ------------------------------------------------------------------------------------------------ flag
class Flag(unittest.TestCase):
    def test_values(self):
        for val, want in ((None, False), ("", False), ("v4", False), ("v41c", True), (" v41c ", True)):
            with env(DSV41_KV_LAYOUT=val):
                self.assertIs(P.enabled(), want)
        for val in ("v41", "V41C", "1", "fp4"):
            with env(DSV41_KV_LAYOUT=val), self.assertRaises(ValueError):
                P.enabled()

    def test_same_rule_as_the_readers(self):
        src = next((c for c in (ROOT / "runtime/flashinfer_v41c/v41c.py",
                                None if SG is None else SG / "sglang/kernels/ops/attention/dsv41_v41c.py")
                    if c is not None and c.exists()), None)
        if src is None:
            self.skipTest("no v41c.py / dsv41_v41c.py")
        tree = ast.parse(src.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "enabled")
        txt = ast.unparse(fn)
        self.assertIn("('', 'v4')", txt)
        self.assertIn("== 'v41c'", txt)
        self.assertEqual(P.ENV, "DSV41_KV_LAYOUT")


# ------------------------------------------------------------------------------------------------ pins / layout
@NEED_SG
class Pins(unittest.TestCase):
    def test_pins_equal_the_staged_tree(self):
        for mod, pins in P.PINS.items():
            text = FILES[mod].read_text()
            for q, want in pins.items():
                self.assertEqual(P.source_pin(text, q), want, f"{mod} {q}")

    def test_edit_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            text = FILES[P.CONFIG_MODULE].read_text()
            i = text.index("self.kv_bytes = self.qk_nope_head_dim + self.qk_rope_head_dim * 2 + 8")
            bad = Path(d) / "pool_configurator.py"
            bad.write_text(text[:i] + text[i:].replace("* 2 + 8", "* 2 + 9", 1))
            m = types.SimpleNamespace(__name__=P.CONFIG_MODULE, __file__=str(bad))
            with self.assertRaisesRegex(RuntimeError, "not the pinned SGLang source"):
                P.check_pins(m)
            P.check_pins(types.SimpleNamespace(__name__=P.CONFIG_MODULE, __file__=str(FILES[P.CONFIG_MODULE])))
            with self.assertRaisesRegex(RuntimeError, "no source file"):
                P.check_pins(types.SimpleNamespace(__name__=P.CONFIG_MODULE, __file__=None))

    def test_layout_values(self):
        kl = kv_layout_module()
        P.check_layout(kl.KVLayout)
        fake = types.SimpleNamespace(
            V41_FP4=types.SimpleNamespace(bytes_per_token=288, data_bytes=256, scale_bytes=32,
                                          page_bytes=lambda n: n * 288 + 256), V4=kl.KVLayout.V4)
        with self.assertRaisesRegex(RuntimeError, "record ABI"):
            P.check_layout(fake)

    def test_backend_pins_name_the_call_sites(self):
        text = FILES[P.BACKEND_MODULE].read_text()
        tree = ast.parse(text)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4AttnBackend")
        src = {f.name: ast.get_source_segment(text, f) for f in cls.body if isinstance(f, ast.FunctionDef)}
        self.assertIn("kv_layout = pool.get_extra_key_layout(layer_id)", src["_low_ratio_compress_fused"])
        self.assertIn("layout=kv_layout", src["_low_ratio_compress_fused"])
        self.assertIn("is KVLayout.V41_FP4", src["_low_ratio_write_group"])
        self.assertIn("freqs_cis=freqs", src["_low_ratio_write_group"])
        self.assertIn("get_extra_key_bytes_per_token", src["_forward_attention"])
        self.assertIn("not get_platform().is_sm120", src["_forward_attention"])   # no sparse-prefill dequant on SM120


# ------------------------------------------------------------------------------------------------ layouts
@NEED_SG
class Resolution(unittest.TestCase):
    def _pool(self, mod, kv_layout, option=None):
        p = object.__new__(mod.DeepSeekV4TokenToKVPool)
        p.kv_layout, p.compressed_kv_layout_option = kv_layout, option
        return p

    def test_off_is_stock(self):
        mod, kl = pool_module()
        V4, V41, FP4 = kl.KVLayout.V4, kl.KVLayout.V41, kl.KVLayout.V41_FP4
        for r in (1, 2, 4, 128):
            self.assertIs(self._pool(mod, V4).compressed_kv_layout(r), V4)
        self.assertIs(self._pool(mod, V41).compressed_kv_layout(1), FP4)
        self.assertIs(self._pool(mod, V41).compressed_kv_layout(4), V41)
        self.assertFalse(mod.is_valid_kv_layout_pair(V4, FP4))

    def test_on(self):
        mod, kl = pool_module()
        P.install_pool(mod)
        P.install_pool(mod)          # idempotent
        V4, V41, FP4 = kl.KVLayout.V4, kl.KVLayout.V41, kl.KVLayout.V41_FP4
        self.assertIs(self._pool(mod, V4).compressed_kv_layout(1), FP4)
        self.assertIs(self._pool(mod, V4).compressed_kv_layout(2), FP4)
        self.assertIs(self._pool(mod, V4, "auto").compressed_kv_layout(2), FP4)
        self.assertIs(self._pool(mod, V4).compressed_kv_layout(4), V4)
        self.assertIs(self._pool(mod, V4).compressed_kv_layout(128), V4)
        self.assertIs(self._pool(mod, V41).compressed_kv_layout(1), FP4)      # V4.1 main: stock
        self.assertIs(self._pool(mod, V41).compressed_kv_layout(128), V41)
        self.assertTrue(mod.is_valid_kv_layout_pair(V4, FP4))
        self.assertFalse(mod.is_valid_kv_layout_pair(V4, V41))
        self.assertTrue(mod.is_valid_kv_layout_pair(V41, FP4))
        for opt in ("fp8", "fp4"):
            with self.assertRaisesRegex(ValueError, "conflicts"):
                self._pool(mod, V4, opt).compressed_kv_layout(1)

    def test_backend_must_be_flashinfer(self):
        mod, _ = pool_module(backend="triton")
        with self.assertRaisesRegex(RuntimeError, "FLASHMLA_BACKEND=flashinfer"):
            P.install_pool(mod)


# ------------------------------------------------------------------------------------------------ sizing
def config_module():
    """pool_configurator's DSV4 sizing (real source) on stubs of the runtime context."""
    src = cut(FILES[P.CONFIG_MODULE], ["MemoryPoolConfig", "_get_dsv4_compress_state_dtype_sizes", "MemoryPoolConfigurator",
                                       "compute_swa_request_cap", "DSV4_DEFAULT_SWA_FULL_TOKENS_RATIO",
                                       "_operator_swa_full_tokens_ratio", "_DSV4PoolSizes", "DSV4PoolConfigurator"])
    psrc = cut(FILES[P.POOL_MODULE], ["get_dsv4_indexer_bytes_per_token", "get_compress_state_ring_size",
                                      "get_compress_state_write_pad", "get_swa_ring_size"])
    mod = types.ModuleType("v41c_cfg_stub")
    sys.modules[mod.__name__] = STUBBED[mod.__name__] = mod      # dataclasses look their module up
    ns = mod.__dict__
    ns.update(dict(envs=Env(), logger=P.logger, _is_hip=False, ONLINE_C128=False, ceil_div=lambda a, b: -(-a // b),
                   ceil_align=lambda a, b: -(-a // b) * b, get_alloc_len_per_decode=lambda: 6,
                   spec_decode_alloc_len_per_request=lambda **_: 6, max_speculative_num_draft_tokens=lambda: 6,
                   get_spec=lambda: SPEC, get_schedule=lambda: SCHED, get_disagg=lambda: DISAGG,
                   get_memory=lambda: MEMORY, get_exec=lambda: EXEC, TYPE_CHECKING=False))
    # names the stock functions import inside their bodies
    for name, attrs in {"sglang": {}, "sglang.kernels": {}, "sglang.kernels.ops": {}, "sglang.kernels.ops.attention": {},
                        "sglang.kernels.ops.attention.dsv4": {}, "sglang.kernels.ops.attention.dsv4.unified_kv_kernels": {},
                        "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate":
                            {"is_unified_kv_fp8": lambda: False, "is_unified_kv_triton": lambda: False},
                        "sglang.srt": {}, "sglang.srt.mem_cache": {},
                        "sglang.srt.mem_cache.deepseek_v4_memory_pool": {"dsv4_unified_row_bytes": lambda *a: 1024},
                        "sglang.srt.runtime_context": {"get_exec": lambda: EXEC}}.items():
        m = sys.modules.get(name) if name in STUBBED else None
        if m is None:
            m = types.ModuleType(name)
            m.__path__ = []
            STUBBED[name] = m
            sys.modules[name] = m
        m.__dict__.update(attrs)
    exec(compile("from __future__ import annotations\nfrom dataclasses import dataclass\nfrom typing import *\n"
                 + psrc + "\n\n" + src, str(FILES[P.CONFIG_MODULE]), "exec"), ns)
    mod.__file__ = str(FILES[P.CONFIG_MODULE])
    mod.__name__ = P.CONFIG_MODULE
    return mod


STUBBED = {}
SPEC = types.SimpleNamespace(speculative_algorithm="DSPARK", speculative_num_draft_tokens=6, speculative_num_steps=1,
                             speculative_eagle_topk=1)
SCHED = types.SimpleNamespace(max_running_requests=16, chunked_prefill_size=8192, page_size=256,
                              disable_overlap_schedule=False, swa_prefix_tails=None, swa_full_tokens_ratio=0.1,
                              _swa_full_tokens_ratio_explicitly_set=False)
DISAGG = types.SimpleNamespace(disaggregation_mode="null", disaggregation_decode_extra_slots=0)
MEMORY = types.SimpleNamespace(enable_hisparse=False, disable_radix_cache=False)
EXEC = types.SimpleNamespace(kernel=types.SimpleNamespace(enable_deepseek_v4_fp4_indexer=True),
                             features=types.SimpleNamespace(enable_encoder_swa_bounded_replay=False))


def production_kvc(start=0, end=40):
    """v2.6's TP4 profile as the configurator sees it: 40 target layers, DSpark, 1M context, 16 requests."""
    hf = types.SimpleNamespace(kv_source_layer_ids=SOURCES)
    mc = types.SimpleNamespace(qk_nope_head_dim=448, qk_rope_head_dim=64, index_head_dim=128, compress_ratios=RATIOS[:40],
                               context_len=1048576, window_size=128, hf_config=hf)
    return types.SimpleNamespace(kv_cache_dtype_str="fp8_e4m3", model_config=mc,
                                 layer_info=types.SimpleNamespace(start_layer=start, end_layer=end),
                                 ps=types.SimpleNamespace(pp_size=1, attn_dp_size=1), sliding_window_size=128,
                                 page_size=256, spec_algorithm=types.SimpleNamespace(is_dspark=lambda: True,
                                                                                     is_eagle=lambda: False,
                                                                                     is_none=lambda: False),
                                 spec_aux_config=types.SimpleNamespace(dflash_draft_num_layers=1))


@NEED_SG
class Sizing(unittest.TestCase):
    def tearDown(self):
        for n in list(STUBBED):
            sys.modules.pop(n, None)
            STUBBED.pop(n)

    def test_production_profile(self):
        stock = config_module().DSV4PoolConfigurator(production_kvc())
        self.assertAlmostEqual(stock.bytes_per_full_token, 1670.75, places=6)      # every v2.6 boot logs this
        mod = config_module()
        P.install_configurator(mod)
        P.install_configurator(mod)
        v = mod.DSV4PoolConfigurator(production_kvc())
        self.assertAlmostEqual(v.low_ratio_bytes_per_full_token, 2.5 * (288 + 68), places=6)
        self.assertAlmostEqual(v.bytes_per_full_token, 912.25, places=6)
        self.assertEqual(v.bytes_per_swa_token, stock.bytes_per_swa_token)          # SWA stays V4
        self.assertEqual(v.swa_cap_tokens, stock.swa_cap_tokens)
        self.assertEqual(v._get_swa_fixed_bytes(), stock._get_swa_fixed_bytes())
        budget = int(9.25 * 2**30)        # a v2.6 boot's available_bytes (release gate 2026-10-26: 9.25 GB)
        a, b = stock.calculate_pool_sizes(budget, 256), v.calculate_pool_sizes(budget, 256)
        free = budget - stock._get_swa_fixed_bytes()
        self.assertEqual(a.max_total_num_tokens, int(free / 1670.75) // 256 * 256)
        self.assertEqual(b.max_total_num_tokens, int(free / 912.25) // 256 * 256)
        self.assertAlmostEqual(b.max_total_num_tokens / a.max_total_num_tokens, 1670.75 / 912.25, places=3)
        self.assertEqual(a.swa_max_total_num_tokens, b.swa_max_total_num_tokens)
        # a pinned pool: the same token count costs 912.25 / 1670.75 of the bytes
        self.assertEqual(v.calculate_pool_sizes_from_max_tokens(4194304, 256).max_total_num_tokens, 4194304)

    def test_pp_stage_and_draft(self):
        mod = config_module()
        P.install_configurator(mod)
        v = mod.DSV4PoolConfigurator(production_kvc(start=0, end=16))      # sources 2, 8, 14 (ratio 2)
        self.assertAlmostEqual(v.low_ratio_bytes_per_full_token, 1.5 * (288 + 68), places=6)
        kvc = production_kvc()
        kvc.model_config.hf_config = types.SimpleNamespace(kv_source_layer_ids=[])
        kvc.model_config.compress_ratios = [0] * 3
        kvc.layer_info = types.SimpleNamespace(start_layer=0, end_layer=3)
        d = mod.DSV4PoolConfigurator(kvc)                                       # no low-ratio layers: unchanged
        self.assertEqual(d.low_ratio_bytes_per_full_token, 0)
        self.assertIn("low_ratio_layers=0", P.resize(d, kvc, mod.get_dsv4_indexer_bytes_per_token))

    def test_formula_drift_refuses(self):
        mod = config_module()
        conf = object.__new__(mod.DSV4PoolConfigurator)
        conf._unified, conf.kv_bytes, conf.indexer_head_dim = False, 584, 128
        conf.low_ratio_bytes_per_full_token = 1.0
        with self.assertRaisesRegex(RuntimeError, "not the formula"):
            P.resize(conf, production_kvc(), mod.get_dsv4_indexer_bytes_per_token)
        conf._unified = True
        with self.assertRaisesRegex(RuntimeError, "unified_kv"):
            P.resize(conf, production_kvc(), mod.get_dsv4_indexer_bytes_per_token)


# ------------------------------------------------------------------------------------------------ allocation
@NEED_SG
@unittest.skipUnless(HAVE_TORCH, "torch")
class Allocation(unittest.TestCase):
    FULL = 1 << 16          # full-pool tokens of the test pool (a multiple of the 256-token page)

    def _pools(self, on):
        mod, kl = pool_module()
        if on:
            P.install_pool(mod)
        p = object.__new__(mod.DeepSeekV4TokenToKVPool)
        p.kv_layout, p.compressed_kv_layout_option = kl.KVLayout.V4, None
        p.compressed_pool_configs, p.full_size, p._unified_kv, p.uniform_fp8 = {}, self.FULL, False, False
        p.sources_by_ratio = {2: [2, 8, 14], 1: [20]}
        p.qk_nope_head_dim, p.qk_rope_head_dim, p.indexer_head_dim = 448, 64, 128
        p._make_indexer_pool = lambda *a, **k: "indexer"
        p._init_compressed_pools(stage_ratios=RATIOS[:40], page_size=256, dtype=torch.float8_e4m3fn, device="cpu",
                                 enable_memory_saver=False, enable_hisparse=False,
                                 kv_pool_cls=mod.DeepSeekV4SingleKVPool)
        return p, kl, mod

    def test_on(self):
        p, kl, _ = self._pools(True)
        for ratio, page, layers in ((1, 256, 1), (2, 128, 3)):
            q = p.kv_pools[ratio]
            self.assertIs(q.kv_layout, kl.KVLayout.V41_FP4)
            self.assertEqual(q.page_size, page)
            self.assertEqual(len(q.kv_buffer), layers)
            self.assertEqual(q.size, self.FULL // ratio)
            for b in q.kv_buffer:
                self.assertEqual(tuple(b.shape), ((q.size + page + 1) // page, page * 288))
                self.assertEqual(b.dtype, torch.uint8)
        line = P.check_pool(p, kl.KVLayout)
        self.assertRegex(line, r"^DSV41_V41C_POOL swa=v4 r1=v41_fp4:page=256:page_bytes=73728:layers=1:pages=\d+:bytes=\d+ "
                               r"r2=v41_fp4:page=128:page_bytes=36864:layers=3:pages=\d+:bytes=\d+ full_size=65536$")

    def test_off_is_v4_and_bytes_ratio(self):
        off, kl, _ = self._pools(False)
        on, _, _ = self._pools(True)
        for ratio in (1, 2):
            a, b = off.kv_pools[ratio], on.kv_pools[ratio]
            self.assertIs(a.kv_layout, kl.KVLayout.V4)
            self.assertEqual(a.kv_buffer[0].shape[0], b.kv_buffer[0].shape[0])        # same pages
            v4 = sum(x.numel() for x in a.kv_buffer)
            fp4 = sum(x.numel() for x in b.kv_buffer)
            self.assertAlmostEqual(fp4 / v4, 288 / (kl.KVLayout.V4.page_bytes(a.page_size) / a.page_size), places=9)
        with self.assertRaisesRegex(RuntimeError, "ratio-1 pool is"):
            P.check_pool(off, kl.KVLayout)

    def test_check_pool_refuses(self):
        p, kl, _ = self._pools(True)
        p.kv_layout = kl.KVLayout.V41
        with self.assertRaisesRegex(RuntimeError, "keeps the SWA cache V4"):
            P.check_pool(p, kl.KVLayout)
        p.kv_layout, p._unified_kv = kl.KVLayout.V4, True
        with self.assertRaisesRegex(RuntimeError, "unified_kv"):
            P.check_pool(p, kl.KVLayout)
        draft = types.SimpleNamespace(kv_layout=kl.KVLayout.V4, kv_pools={}, _unified_kv=False)
        self.assertIn("low_ratio_pools=0", P.check_pool(draft, kl.KVLayout))

    def test_wrapped_init_prints_the_receipt(self):
        mod, kl = pool_module()
        cls = mod.DeepSeekV4TokenToKVPool
        stock = cls.__init__
        made = {}

        def fake_init(self, *a, **k):
            made["args"] = (a, k)
            self.kv_layout, self._unified_kv, self.kv_pools, self.full_size = kl.KVLayout.V4, False, {}, 0
        cls.__init__ = fake_init
        try:
            P.install_pool(mod)
            self.assertIs(cls.__init__.__wrapped__, fake_init)
            out = subprocess_capture(lambda: cls(1, x=2))
            self.assertEqual(made["args"], ((1,), {"x": 2}))
            self.assertIn("DSV41_V41C_POOL swa=v4 low_ratio_pools=0", out)
        finally:
            cls.__init__ = stock


def subprocess_capture(fn):
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


# ------------------------------------------------------------------------------------------------ default off
V41C_SITE_BLOCKS = (
    re.compile(r"            # DSV41_KV_LAYOUT=v41c \(adapter/v41c_pool\.py\).*?_v41c_check_backend\(module\)\n", re.S),
    re.compile(r"        elif module\.__name__ == 'sglang\.srt\.mem_cache\.deepseek_v4_memory_pool':.*?"
               r"(?=        elif module\.__name__ == 'sglang\.srt\.entrypoints\.openai\.encoding_dsv41':)", re.S),
    re.compile(r"# DSV41_KV_LAYOUT unset or v4 \(the default\).*?\n\n(?=class EngramFinder)", re.S),
    re.compile(r" \+ _V41C_MODULES(?=:\n            return None)"),
)
# the tokenizer heap trim's block (perf/v41n-trim-v264-20261106: heap_trim2 on v2.6.4n; off by default, launch_server
# only, its own tests in tests/test_tokenizer_heap_trim2.py): stripped too, so the rest is still the base's text
TRIM2_SITE_BLOCK = re.compile(r"# DSV41_TOKENIZER_HEAP=trim2 \(off by default\).*?"
                              r"print\(f'DSV41_TOKENIZER_HEAP=trim2: not installed \(\{exc!r\}\)', flush=True\)\n\n", re.S)


class DefaultOff(unittest.TestCase):
    SITE = ADAPTER / "sitecustomize.py"

    @unittest.skip("v2.7: sitecustomize routes DSV41_KV_LAYOUT through _kv_layout_adapter (v41 / v41c); "
                   "tests/test_v41_pool.py DefaultOff pins the text")
    def test_text_minus_v41c_is_f744d83(self):
        s = self.SITE.read_text()
        for rx in V41C_SITE_BLOCKS:
            s, n = rx.subn("", s)
            self.assertEqual(n, 1, rx.pattern[:60])
        s, n = TRIM2_SITE_BLOCK.subn("", s)
        self.assertEqual(n, 1, "trim2 block")
        self.assertEqual(hashlib.sha256(s.encode()).hexdigest(), SITE_F744D83_SHA)

    def _finder_names(self, value):
        code = textwrap.dedent(f"""
            import importlib.machinery, json, sys
            sys.path.insert(0, {str(ADAPTER)!r})
            import sitecustomize as s
            seen = []
            importlib.machinery.PathFinder.find_spec = classmethod(lambda cls, n, p=None, t=None: seen.append(n))
            f = s.EngramFinder()
            names = ['sglang.srt.mem_cache.deepseek_v4_memory_pool', 'sglang.srt.model_executor.pool_configurator',
                     'sglang.srt.layers.attention.deepseek_v4_backend', 'sglang.srt.models.deepseek_v4']
            for n in names:
                f.find_spec(n)
            print(json.dumps({{"hooked": seen, "v41c_imported": "v41c_pool" in sys.modules}}))
        """)
        e = {k: v for k, v in os.environ.items() if k not in ("DSV41_KV_LAYOUT", "DSV41_SOURCE", "DSV41_AB_VARIANTS")}
        if value is not None:
            e["DSV41_KV_LAYOUT"] = value
        r = subprocess.run([sys.executable, "-S", "-c", code], env=e, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        import json
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_finder(self):
        stock = ["sglang.srt.layers.attention.deepseek_v4_backend", "sglang.srt.models.deepseek_v4"]
        for val in (None, "", "v4"):
            got = self._finder_names(val)
            self.assertEqual(got["hooked"], stock, val)
            self.assertFalse(got["v41c_imported"])
        got = self._finder_names("v41c")
        self.assertEqual(got["hooked"], ["sglang.srt.mem_cache.deepseek_v4_memory_pool",
                                         "sglang.srt.model_executor.pool_configurator"] + stock)
        self.assertFalse(got["v41c_imported"])            # imported only when the hooked module executes

    def test_hook_branches(self):
        s = self.SITE.read_text()
        tree = ast.parse(s)
        self.assertIn("_kv.install_pool(module)", s)          # v2.7: v41c_pool or v41_pool via _kv_layout_adapter
        self.assertIn("_kv.install_configurator(module)", s)
        # every hooked name has its own branch before the final else (whose body assumes dsv4.metadata)
        for name in P.MODULES:
            self.assertIn(f"elif module.__name__ == '{name}':", s)
        self.assertTrue(tree)

    @unittest.skipUnless((ROOT / "runtime/flashinfer_v41c/v41c.py").exists(), "not a repo checkout (in an image)")
    def test_dockerfiles_run_the_pool_test(self):
        for df in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            self.assertIn("python3 /opt/dsv41/tests/test_v41c_pool.py", (ROOT / df).read_text(), df)


V41N_KEYS = ["DSV41_KV_LAYOUT=v41c", "DSV41_V41C_ARM=1", "DSV41_V41C_READER=native"]


@unittest.skipUnless((ROOT / ".env.tp4.example").exists(), "not a repo checkout (in an image)")
@unittest.skip("v2.7: the v41n keys left the production line (DSV41_KV_LAYOUT=v41); tests/test_v41_pool.py Profile")
class Profile(unittest.TestCase):
    """v2.6.5: the production line of .env.tp4.example ends with the three v41n keys (after v2.6.4's cone keys) and
    v2.6.5's tokenizer heap trim, no other profile line sets the layout or a reader, no diagnostic v41c / v41n knob is
    in any profile, every v41n key is volatile for DSV41_AUTOTUNE_KEEP, the module stays off when the layout is unset,
    and the production settings pin the v41n pool with one #MAX_TOTAL_TOKENS line (the memory ladder's SAFE pin)."""

    def _lines(self):
        return (ROOT / ".env.tp4.example").read_text().splitlines()

    def _production_line(self):
        lines = self._lines()
        i = next(n for n, line in enumerate(lines) if line.startswith("# Production settings, together with the line"))
        self.assertTrue(lines[i - 1].startswith('#EXTRA_CONTAINER_ENV="'), lines[i - 1][:60])
        return lines[i - 1]

    def test_production_line_ends_with_v41n_once(self):
        items = self._production_line().split('"')[1].split()
        self.assertEqual(items[-7:], ["DSV41_SWA_THIN=1", "DSV41_SWA_CONE_REPLAY=1", "DSV41_SWA_CONE_TOKENS=8192"] + V41N_KEYS
                         + ["DSV41_TOKENIZER_HEAP=trim2"])
        for k in V41N_KEYS:
            self.assertEqual(sum(x.split("=")[0] == k.split("=")[0] for x in items), 1, k)

    def test_production_pool_pin(self):
        # the v41n pool pin is ONE parameter of the production block (the memory ladder's SAFE pin; it may move to a larger
        # measured value together with the memory fraction): exactly one #MAX_TOTAL_TOKENS= line there, a multiple of the
        # 256-token KV page, between 4M and 10M tokens; exactly one #MEM_FRACTION_STATIC= and #HEAD_MEM_FRACTION_STATIC=
        # line, equal to each other
        lines = self._lines()
        i = next(n for n, line in enumerate(lines) if line.startswith("# Production settings, together with the line"))
        j = next(n for n in range(i, len(lines)) if not lines[n].startswith("#"))
        pins = [line for line in lines[i:j] if line.startswith("#MAX_TOTAL_TOKENS=")]
        self.assertEqual(len(pins), 1, pins)
        pin = int(pins[0].split("=", 1)[1])
        self.assertEqual(pin % 256, 0)
        self.assertTrue(4_000_000 <= pin <= 10_000_000, pin)
        fr = [line.split("=", 1)[1] for line in lines[i:j] if line.startswith("#MEM_FRACTION_STATIC=")]
        hfr = [line.split("=", 1)[1] for line in lines[i:j] if line.startswith("#HEAD_MEM_FRACTION_STATIC=")]
        self.assertEqual(len(fr), 1)
        self.assertEqual(fr, hfr)
        # the active default stays v2.6.4's unpinned 8,000,000 (the pin lives only in the commented production block)
        self.assertEqual([line for line in lines if line.startswith("MAX_TOTAL_TOKENS=")], ["MAX_TOTAL_TOKENS=8000000"])

    def test_only_the_production_line_sets_the_layout(self):
        on = [line for line in self._lines() if "DSV41_KV_LAYOUT" in line and not line.startswith("# ")]
        self.assertEqual(on, [self._production_line()])
        knobs = ("DSV41_V41N_", "DSV41_V41C_DEDUP", "DSV41_V41C_READER=io", "DSV41_V41C_READER=group")
        bad = [line for line in self._lines() if not line.startswith("# ") and any(k in line for k in knobs)]
        self.assertEqual(bad, [], "diagnostic / alternative reader knobs are in no profile")

    def test_volatile_in_autotune_keep(self):
        tree = ast.parse((ADAPTER / "autotune_keep.py").read_text())
        vol = next(ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_VOLATILE")
        for k in V41N_KEYS + ["DSV41_V41C_DEDUP_ROWS", "DSV41_V41C_DEDUP_MIN_ROWS", "DSV41_V41N_TACTICS",
                              "DSV41_V41N_DIAG", "DSV41_V41N_PREFILL", "DSV41_V41N_PAGE_MIN_ROWS", "DSV41_V41N_PAGE_CAP_MB"]:
            self.assertIn(k.split("=")[0], vol)

    def test_default_off_in_module(self):
        with env(DSV41_KV_LAYOUT=None):
            self.assertFalse(P.enabled())
        with env(DSV41_KV_LAYOUT="v4"):
            self.assertFalse(P.enabled())
        with env(DSV41_KV_LAYOUT="v41c"):
            self.assertTrue(P.enabled())


if __name__ == "__main__":
    unittest.main(verbosity=1)
