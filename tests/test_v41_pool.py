"""CPU checks of DSV41_KV_LAYOUT=v41 (v2.7, adapter/v41_pool.py, wired by adapter/sitecustomize.py): the V4.1 KV layouts
on SM12x with FlashInfer >= 0.7.2.

The SGLang code under test is the real source of the staged v2.6.5 tree (runtime/sglang-canary/python, or the image's
/sgl-workspace/sglang/python, or SGLANG_CANARY_PY), cut out with ast and run against small stubs (the helpers of
tests/test_v41c_pool.py).

- flag: only "v41" enables it; sitecustomize routes v41 -> v41_pool, v41c -> v41c_pool, unset / v4 -> nothing.
- pins: every PINS entry equals the staged tree; a one-character edit refuses; KVLayout values are the V4.1 ABI.
- selection (real select_dsv4_kv_layout): SM12x -> (V41, None); SM10 -> the stock function; other GPUs refuse;
  SGLANG_DSV4_KV_LAYOUT=v4, a forced compressed layout, a non-FlashInfer SM120 backend and FlashInfer < 0.7.2 refuse;
  the V41 main cache pairs with V41_FP4 for ratios 1 / 2 (stock resolve_compressed_kv_layout).
- sizing (real DSV4PoolConfigurator on the production profile: TP4, DSpark, paged SWA cap mode, 1M context, 16
  requests; and the encoder request-window mode): bytes_per_full_token 1670.75 -> 912.25, bytes_per_swa_token and the
  SWA fixed bytes x 528 / 584, the request window re-priced by its own formula; a drifted formula refuses.
- allocation (torch): a V41 main pool is 528 B per token with 512-byte TMA-padded pages, the ratio-1/2 pools V41_FP4;
  the receipt names them; a V4 main cache refuses.
- default off: sitecustomize minus the KV-layout blocks is the base text; the finder hooks kv_cache_configurator only
  for v41; kv_cache_configurator's by-name import is re-pointed.
- profile: the production line sets DSV41_KV_LAYOUT=v41 and SGLANG_SM120_V41_PRECISION=bf16 once and none of the
  FlashInfer-0.6.18-only knobs; the pool pin is one multiple-of-256 line.

usage: python3 tests/test_v41_pool.py        (torch for the allocation cases)
"""
import ast
import contextlib
import hashlib
import os
import re
import subprocess
import sys
import textwrap
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "adapter"))
import test_v41c_pool as T  # noqa: E402  (helpers: staged tree, cut, stubs, production profile)
import v41_pool as V  # noqa: E402

NEED_SG = T.NEED_SG
# a real torch (other suites in the same process may stub sys.modules['torch'] when it is absent)
HAVE_TORCH = T.HAVE_TORCH and hasattr(getattr(T, 'torch', None), 'float8_e4m3fn')
FILES = {} if T.SG is None else {
    V.POOL_MODULE: T.SG / "sglang/srt/mem_cache/deepseek_v4_memory_pool.py",
    V.CONFIG_MODULE: T.SG / "sglang/srt/model_executor/pool_configurator.py",
    V.BACKEND_MODULE: T.SG / "sglang/srt/layers/attention/deepseek_v4_backend.py",
}


class FakeTorch:
    """torch stub for the stock selector: no CUDA (so the stock function answers from its env alone)."""
    cuda = types.SimpleNamespace(is_available=lambda: False, get_device_capability=lambda: (12, 1))
    version = types.SimpleNamespace(cuda="13.0")
    uint8 = "uint8"


def select_module(**envkv):
    """The pool module's selector + layout code (real source) on stubs."""
    kl = T.kv_layout_module()
    names = ["resolve_compressed_kv_layout", "flashmla_supports_v41_kv_layouts", "select_dsv4_kv_layout",
             "DeepSeekV4SingleKVPool", "DeepSeekV4TokenToKVPool", "_CompressedPoolConfig"]
    src = T.cut(FILES[V.POOL_MODULE], names)
    mod = types.ModuleType("v41_pool_stub")
    ns = mod.__dict__
    e = T.Env(**{"SGLANG_DSV4_KV_LAYOUT": "v4", "SGLANG_DSV4_COMPRESSED_KV_LAYOUT": "auto", **envkv})

    class KVCache:
        def __init__(self, size, page_size, dtype, layer_num, device, enable_memory_saver, start_layer=None,
                     end_layer=None):
            self.size, self.page_size, self.dtype, self.layer_num, self.device = size, page_size, dtype, layer_num, device
            self.store_dtype = T.torch.uint8 if HAVE_TORCH else "uint8"
            self.start_layer = start_layer or 0
            self.memory_saver_adapter = types.SimpleNamespace(region=lambda *_: contextlib.nullcontext())
            self.custom_mem_pool = None

    ns.update(dict(KVLayout=kl.KVLayout, is_valid_kv_layout_pair=kl.is_valid_kv_layout_pair, KVCache=KVCache,
                   BaseSWAKVPool=object, envs=e, logger=V.logger, torch=T.torch if HAVE_TORCH else FakeTorch,
                   nullcontext=contextlib.nullcontext, ceil_div=lambda a, b: -(-a // b),
                   GPU_MEMORY_TYPE_KV_CACHE="kv_cache", NamedTuple=__import__("typing").NamedTuple,
                   DSV41_INDEX_PAGE_SIZE=64))
    exec(compile("from __future__ import annotations\nfrom typing import *\n" + src, str(FILES[V.POOL_MODULE]), "exec"), ns)
    mod.__file__ = str(FILES[V.POOL_MODULE])
    mod.__name__ = V.POOL_MODULE
    if HAVE_TORCH:
        # the stock selector looks at the real CUDA state; the tests choose the capability through the patch, so the
        # module sees real torch with a CUDA-less torch.cuda
        class _Torch:
            cuda = FakeTorch.cuda

            def __getattr__(self, name):
                return getattr(T.torch, name)

        ns["torch"] = _Torch()
    return mod, kl


class Flag(unittest.TestCase):
    def test_values(self):
        for val, on in ((None, False), ("", False), ("v4", False), ("v41c", False), ("v41", True), (" v41 ", True)):
            with T.env(DSV41_KV_LAYOUT=val):
                self.assertEqual(V.enabled(), on, val)

    def test_sitecustomize_routes(self):
        code = textwrap.dedent(f"""
            import json, sys
            sys.path.insert(0, {str(ROOT / 'adapter')!r})
            import sitecustomize as s
            m = s._kv_layout_adapter()
            print(json.dumps(None if m is None else m.__name__))
        """)
        for val, want in ((None, None), ("v4", None), ("v41c", "v41c_pool"), ("v41", "v41_pool")):
            e = {k: v for k, v in os.environ.items() if k not in ("DSV41_KV_LAYOUT", "DSV41_SOURCE")}
            if val is not None:
                e["DSV41_KV_LAYOUT"] = val
            r = subprocess.run([sys.executable, "-S", "-c", code], env=e, capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip().splitlines()[-1], "null" if want is None else f'"{want}"', val)
        e = dict(os.environ, DSV41_KV_LAYOUT="v41x")
        r = subprocess.run([sys.executable, "-S", "-c", code], env=e, capture_output=True, text=True, timeout=120)
        self.assertNotEqual(r.returncode, 0)          # a typo stops (v41c_pool.enabled raises)


@NEED_SG
class Pins(unittest.TestCase):
    def test_pins_equal_the_staged_tree(self):
        for mod, path in FILES.items():
            text = path.read_text()
            for q, want in V.PINS[mod].items():
                self.assertEqual(V.source_pin(text, q), want, f"{mod}:{q}")

    def test_edit_refuses(self):
        mod, _ = select_module()
        with tempfile_copy(FILES[V.POOL_MODULE], "is_sm100 = (", "is_sm100  = (") as p:
            mod.__file__ = str(p)
            with self.assertRaisesRegex(RuntimeError, "not the pinned SGLang source"):
                V.install_pool(mod, capability=lambda: 12, fi_version="0.7.2")

    def test_layout_values(self):
        V.check_layout(T.kv_layout_module().KVLayout)


@contextlib.contextmanager
def tempfile_copy(path, old, new):
    import tempfile
    text = Path(path).read_text()
    assert text.count(old) >= 1
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / Path(path).name
        p.write_text(text.replace(old, new, 1))
        yield p


@NEED_SG
class Selection(unittest.TestCase):
    def _install(self, major=12, fi="0.7.2", **envkv):
        mod, kl = select_module(**envkv)
        V.install_pool(mod, capability=lambda: major, fi_version=fi)
        return mod, kl

    def test_sm12x_selects_v41(self):
        with T.env(SGLANG_DSV4_KV_LAYOUT=None):
            mod, kl = self._install()
            V.install_pool(mod, capability=lambda: 12, fi_version="0.7.2")      # idempotent
            self.assertEqual(mod.select_dsv4_kv_layout(), (kl.KVLayout.V41, None))
        with T.env(SGLANG_DSV4_KV_LAYOUT="v41"):
            self.assertEqual(self._install()[0].select_dsv4_kv_layout()[0], self._install()[1].KVLayout.V41)

    def test_stock_without_the_patch_refuses_v41_on_sm12x(self):
        mod, kl = select_module(SGLANG_DSV4_KV_LAYOUT="v41")
        with self.assertRaises(AssertionError):
            mod.select_dsv4_kv_layout()            # stock asserts SM100 (cuda unavailable in the stub)

    def test_sm10_is_stock_and_others_refuse(self):
        with T.env(SGLANG_DSV4_KV_LAYOUT=None):
            mod, kl = self._install(major=10)
            self.assertEqual(mod.select_dsv4_kv_layout(), (kl.KVLayout.V4, None))     # stock: env default "v4"
            mod, _ = self._install(major=9)
            with self.assertRaisesRegex(RuntimeError, "SM12x"):
                mod.select_dsv4_kv_layout()

    def test_conflicts_refuse(self):
        with T.env(SGLANG_DSV4_KV_LAYOUT="v4"):
            with self.assertRaisesRegex(ValueError, "conflicts"):
                self._install()
        with T.env(SGLANG_DSV4_KV_LAYOUT=None):
            with self.assertRaisesRegex(ValueError, "conflicts"):
                self._install(SGLANG_DSV4_COMPRESSED_KV_LAYOUT="fp8")
            with self.assertRaisesRegex(RuntimeError, "FLASHMLA_BACKEND=flashinfer"):
                self._install(SGLANG_SM120_FLASHMLA_BACKEND="triton")
            for fi in ("0.6.18", "0.7.1", "garbage"):
                with self.assertRaisesRegex(RuntimeError, "FlashInfer >= 0.7.2"):
                    self._install(fi=fi)
            for fi in ("0.7.2", "0.7.2.post1", "0.7.3", "0.8.0"):
                self._install(fi=fi)

    def test_v41_pairs_with_v41_fp4(self):
        mod, kl = select_module()
        p = object.__new__(mod.DeepSeekV4TokenToKVPool)
        p.kv_layout, p.compressed_kv_layout_option = kl.KVLayout.V41, None
        self.assertIs(p.compressed_kv_layout(1), kl.KVLayout.V41_FP4)
        self.assertIs(p.compressed_kv_layout(2), kl.KVLayout.V41_FP4)
        self.assertTrue(mod.is_valid_kv_layout_pair(kl.KVLayout.V41, kl.KVLayout.V41_FP4))

    def test_kv_configurator_rebind(self):
        with T.env(SGLANG_DSV4_KV_LAYOUT=None):
            mod, _ = self._install()
        saved = sys.modules.get(V.POOL_MODULE)
        sys.modules[V.POOL_MODULE] = mod
        try:
            kvc = types.ModuleType(V.KVC_MODULE)
            kvc.select_dsv4_kv_layout = mod.select_dsv4_kv_layout.__wrapped__
            V.install_kv_configurator(kvc)
            self.assertIs(kvc.select_dsv4_kv_layout, mod.select_dsv4_kv_layout)
            V.install_kv_configurator(kvc)
            kvc.select_dsv4_kv_layout = lambda: None
            with self.assertRaisesRegex(RuntimeError, "neither the stock"):
                V.install_kv_configurator(kvc)
        finally:
            if saved is None:
                sys.modules.pop(V.POOL_MODULE, None)
            else:
                sys.modules[V.POOL_MODULE] = saved


@NEED_SG
class Sizing(unittest.TestCase):
    def tearDown(self):
        for n in list(T.STUBBED):
            sys.modules.pop(n, None)
            T.STUBBED.pop(n)
        T.EXEC.features.enable_encoder_swa_bounded_replay = False

    def test_production_profile(self):
        stock = T.config_module().DSV4PoolConfigurator(T.production_kvc())
        self.assertAlmostEqual(stock.bytes_per_full_token, 1670.75, places=6)
        mod = T.config_module()
        V.install_configurator(mod)
        V.install_configurator(mod)
        v = mod.DSV4PoolConfigurator(T.production_kvc())
        self.assertEqual(v.kv_bytes, 528)
        self.assertAlmostEqual(v.low_ratio_bytes_per_full_token, 2.5 * (288 + 68), places=6)
        self.assertAlmostEqual(v.bytes_per_full_token, 912.25, places=6)           # = v2.6.5's v41n
        self.assertAlmostEqual(v.bytes_per_swa_token, stock.bytes_per_swa_token * 528 / 584, places=6)
        self.assertEqual(v.swa_cap_tokens, stock.swa_cap_tokens)
        self.assertAlmostEqual(v._get_swa_fixed_bytes(), stock._get_swa_fixed_bytes() * 528 / 584, delta=1.0)
        budget = int(17.15e9)                                     # v2.6.5's head log: available_bytes=17.15 GB
        a, b = stock.calculate_pool_sizes(budget, 256), v.calculate_pool_sizes(budget, 256)
        self.assertGreater(b.max_total_num_tokens, a.max_total_num_tokens)
        self.assertEqual(v.calculate_pool_sizes_from_max_tokens(8541952, 256).max_total_num_tokens, 8541952)

    def test_encoder_request_window(self):
        T.EXEC.features.enable_encoder_swa_bounded_replay = True
        stock = T.config_module().DSV4PoolConfigurator(T.production_kvc())
        self.assertGreater(stock.request_window_bytes, 0)
        mod = T.config_module()
        V.install_configurator(mod)
        v = mod.DSV4PoolConfigurator(T.production_kvc())
        self.assertGreater(v.request_window_bytes, 0)
        self.assertLess(v.request_window_bytes, stock.request_window_bytes)
        want = V.request_window_bytes(stock, 528, T.SCHED.chunked_prefill_size, lambda a, b: -(-a // b) * b)
        self.assertEqual(v.request_window_bytes, want)
        self.assertAlmostEqual(v.bytes_per_swa_token, 528 * stock.paged_draft_layers, places=6)

    def test_formula_drift_refuses(self):
        mod = T.config_module()
        conf = mod.DSV4PoolConfigurator(T.production_kvc())
        conf.bytes_per_swa_token += 1.0
        with self.assertRaisesRegex(RuntimeError, "not the formula"):
            V.resize(conf, T.production_kvc(), mod)
        conf = mod.DSV4PoolConfigurator(T.production_kvc())
        conf.kv_bytes = 528
        with self.assertRaisesRegex(RuntimeError, "not the V4"):
            V.resize(conf, T.production_kvc(), mod)
        conf._unified = True
        with self.assertRaisesRegex(RuntimeError, "unified_kv"):
            V.resize(conf, T.production_kvc(), mod)

    def test_receipt_line(self):
        mod = T.config_module()
        conf = mod.DSV4PoolConfigurator(T.production_kvc())
        line = V.resize(conf, T.production_kvc(), mod)
        self.assertIn("DSV41_V41_SIZING slot_bytes=584->528 record_bytes=288", line)
        self.assertIn("bytes_per_full_token=1670.75->912.25", line)


@NEED_SG
@unittest.skipUnless(HAVE_TORCH, "torch")
class Allocation(unittest.TestCase):
    FULL = 1 << 16

    def _pool(self, main="V41"):
        mod, kl = select_module()
        p = object.__new__(mod.DeepSeekV4TokenToKVPool)
        p.kv_layout, p.compressed_kv_layout_option = getattr(kl.KVLayout, main), None
        p.compressed_pool_configs, p.full_size, p._unified_kv, p.uniform_fp8 = {}, self.FULL, False, False
        p.sources_by_ratio = {2: [2, 8, 14], 1: [20]}
        p.qk_nope_head_dim, p.qk_rope_head_dim, p.indexer_head_dim = 448, 64, 128
        p._make_indexer_pool = lambda *a, **k: "indexer"
        p._init_compressed_pools(stage_ratios=T.RATIOS[:40], page_size=256, dtype=T.torch.float8_e4m3fn,
                                 device="cpu", enable_memory_saver=False, enable_hisparse=False,
                                 kv_pool_cls=mod.DeepSeekV4SingleKVPool)
        p.swa_kv_pool = mod.DeepSeekV4SingleKVPool(size=4096, page_size=256, dtype=T.torch.float8_e4m3fn,
                                                   qk_nope_head_dim=448, qk_rope_head_dim=64, layer_num=2,
                                                   device="cpu", enable_memory_saver=False,
                                                   kv_layout=p.kv_layout)
        p.request_window = None
        p.get_swa_key_bytes_per_token = lambda: p.swa_kv_pool.kv_cache_total_dim
        return p, kl

    def test_v41_pools_and_receipt(self):
        p, kl = self._pool()
        swa = p.swa_kv_pool
        self.assertEqual(swa.kv_cache_total_dim, 528)
        self.assertEqual(swa.bytes_per_page_padded, 135168)            # 256 x 528, a multiple of the 512-byte stride
        self.assertEqual(tuple(swa.kv_buffer[0].shape), ((4096 + 256 + 1) // 256, 135168))
        for ratio in (1, 2):
            self.assertIs(p.kv_pools[ratio].kv_layout, kl.KVLayout.V41_FP4)
        line = V.check_pool(p, kl.KVLayout)
        self.assertTrue(line.startswith("DSV41_V41_POOL main=v41 swa=v41:page=256:layers=2"), line)
        self.assertIn("r1=v41_fp4:page=256", line)
        self.assertIn("r2=v41_fp4:page=128", line)

    def test_v4_main_refuses(self):
        p, kl = self._pool(main="V4")
        with self.assertRaisesRegex(RuntimeError, "not V41"):
            V.check_pool(p, kl.KVLayout)


# the v2.7 KV-layout blocks of sitecustomize.py; without them (and the trim2 block) the text is the base's
V41_SITE_BLOCKS = (
    re.compile(r"            # DSV41_KV_LAYOUT=v41c \(adapter/v41c_pool\.py\) / v41.*?_kv\.check_backend\(module\)\n", re.S),
    re.compile(r"        elif module\.__name__ == 'sglang\.srt\.mem_cache\.deepseek_v4_memory_pool':.*?"
               r"(?=        elif module\.__name__ == 'sglang\.srt\.entrypoints\.openai\.encoding_dsv41':)", re.S),
    re.compile(r"# DSV41_KV_LAYOUT unset or v4 \(the default\).*?\n\n(?=class EngramFinder)", re.S),
    re.compile(r" \+ _V41C_MODULES(?=:\n            return None)"),
)


# v2.7.1's sitecustomize hooks (extend nosync, keep-cache, sampled commit reuse / early draft probabilities, the extend
# sync audit; every one gated and off by default, their own tests in tests/test_*.py): stripped too
V271_SITE_BLOCKS = tuple(re.compile(p, re.S) for p in (
    r"            # Gated on DSV41_EXTEND_NOSYNC=1 \(off by default\), FIRST:.*?install_extend_nosync_engram\(module\)\n",
    r"            # Gated on DSV41_EXTEND_NOSYNC=1 \(off by default\): the extend's.*?install_extend_nosync_backend\(module\)\n",
    r"            # Gated on DSV41_PREFILL_KEEP_CACHE=1 .*?install_keep_cache_forward_batch\(module\)\n",
    r"            # Gated on DSV41_SAMPLED_COMMIT_REUSE: sampled steps.*?install_sampled_commit_reuse_dspark\(module\)\n",
    r"            # Gated on DSV41_SAMPLED_COMMIT_REUSE: the epilogue copy.*?install_sampled_draft_probs_early_verify\(module\)\n",
    r"            # Gated on DSV41_SAMPLED_DRAFT_PROBS_EARLY: sampled draft probabilities.*?"
    r"install_sampled_draft_probs_early_draft\(module\)\n",
    r"            # Gated on DSV41_SAMPLED_COMMIT_REUSE: terminal reuse receipts.*?"
    r"install_sampled_draft_probs_early_scheduler\(module\)\n",
    r"            # DSV41_PREFILL_KEEP_CACHE=1 \(or named by an armed in-boot A/B\).*?install_extend_sync_audit\(module\)\n",
))
# v2.7.2's CED skip hooks (gated, off by default; tests/test_prefill_ced_skip*.py): the backend wrap after extend nosync and
# the ForwardBatch final-length record (the keep-cache plumbing it shares is stripped with the keep-cache block above)
V272_SITE_BLOCKS = tuple(re.compile(p, re.S) for p in (
    r"            # Gated on DSV41_PREFILL_CED_SKIP=1 \(or an armed in-boot A/B naming it.*?install_ced_backend\(module\)\n",
    r"            # DSV41_PREFILL_CED_SKIP: final prompt lengths by rid.*?install_ced_forward_batch\(module\)\n",
))


class DefaultOff(unittest.TestCase):
    SITE = ROOT / "adapter" / "sitecustomize.py"

    def test_text_minus_kv_layout_blocks_is_the_base(self):
        s = self.SITE.read_text()
        for rx in V41_SITE_BLOCKS:
            s, n = rx.subn("", s)
            self.assertEqual(n, 1, rx.pattern[:60])
        s, n = T.TRIM2_SITE_BLOCK.subn("", s)
        self.assertEqual(n, 1, "trim2 block")
        s, n = T.PROMPT_TAIL_SITE_BLOCK.subn("", s)        # v2.6.6's prompt-tail hook
        self.assertEqual(n, 1, "prompt-tail block")
        for rx in V271_SITE_BLOCKS + V272_SITE_BLOCKS:      # v2.7.1's and v2.7.2's hooks
            s, n = rx.subn("", s)
            self.assertEqual(n, 1, rx.pattern[:60])
        self.assertEqual(hashlib.sha256(s.encode()).hexdigest(), T.SITE_F744D83_SHA)

    def test_finder(self):
        base = ["sglang.srt.layers.attention.deepseek_v4_backend", "sglang.srt.models.deepseek_v4"]
        pool = ["sglang.srt.mem_cache.deepseek_v4_memory_pool", "sglang.srt.model_executor.pool_configurator"]
        code = textwrap.dedent(f"""
            import importlib.machinery, json, sys
            sys.path.insert(0, {str(ROOT / 'adapter')!r})
            import sitecustomize as s
            seen = []
            importlib.machinery.PathFinder.find_spec = classmethod(lambda cls, n, p=None, t=None: seen.append(n))
            f = s.EngramFinder()
            for n in {pool + ['sglang.srt.mem_cache.kv_cache_configurator'] + base!r}:
                f.find_spec(n)
            print(json.dumps({{"hooked": seen, "imported": [m for m in ("v41_pool", "v41c_pool") if m in sys.modules]}}))
        """)
        import json
        for val, want in ((None, base), ("v41c", pool + base), ("v41", pool + ["sglang.srt.mem_cache.kv_cache_configurator"] + base)):
            e = {k: v for k, v in os.environ.items() if k not in ("DSV41_KV_LAYOUT", "DSV41_SOURCE", "DSV41_AB_VARIANTS")}
            if val is not None:
                e["DSV41_KV_LAYOUT"] = val
            r = subprocess.run([sys.executable, "-S", "-c", code], env=e, capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            got = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertEqual(got["hooked"], want, val)
            self.assertEqual(got["imported"], [], val)     # imported only when a hooked module executes


@unittest.skipUnless((ROOT / ".env.tp4.example").exists(), "not a repo checkout (in an image)")
class Profile(unittest.TestCase):
    def _lines(self):
        return (ROOT / ".env.tp4.example").read_text().splitlines()

    def _production_line(self):
        lines = self._lines()
        i = next(n for n, line in enumerate(lines) if line.startswith("# Production settings, together with the line"))
        self.assertTrue(lines[i - 1].startswith('#EXTRA_CONTAINER_ENV="'), lines[i - 1][:60])
        return lines[i - 1]

    def test_production_line(self):
        items = self._production_line().split('"')[1].split()
        # v2.6.6's SWA prompt tail follows the v2.7 keys and trim2; v2.7.1's seven new keys follow them, v2.7.2's CED
        # skip comes last
        self.assertEqual(items[-12:-8], ["DSV41_KV_LAYOUT=v41", "SGLANG_SM120_V41_PRECISION=bf16",
                                         "DSV41_TOKENIZER_HEAP=trim2", "DSV41_SWA_PROMPT_TAIL_KEEP=1024"])
        self.assertEqual(items[-1], "DSV41_PREFILL_CED_SKIP=1")
        self.assertEqual(items[-8:-1], ["DSV41_PREFILL_GEMM_ROW_SPLIT=4096", "DSV41_PREFILL_GEMM_ROW_SPLIT_N=1792,1152",
                                        "DSV41_L2_PREFETCH_WOA_MB=0", "DSV41_L2_PREFETCH_WOB_MB=6", "DSV41_EXTEND_NOSYNC=1",
                                        "DSV41_SAMPLED_COMMIT_REUSE=1", "DSV41_SAMPLED_DRAFT_PROBS_EARLY=1"])
        for k in ("DSV41_KV_LAYOUT", "SGLANG_SM120_V41_PRECISION"):
            self.assertEqual(sum(x.split("=")[0] == k for x in items), 1, k)
        retired = ("DSV41_V41C_", "DSV41_SWA_NATIVE_DECODE", "DSV41_PREFILL_NATIVE_EXTRA", "DSV41_V41N_")
        self.assertEqual([x for x in items if x.startswith(retired)], [])

    def test_boot_guard_accepts_the_production_line(self):
        sys.path.insert(0, str(ROOT / "tests"))
        import test_v27_fi07_guard as G
        boot = G._load_boot()
        env = dict(x.split("=", 1) for x in self._production_line().split('"')[1].split())
        self.assertEqual(boot.retired_fi0618_knobs(env, "0.7.2"), [])

    def test_only_the_production_line_sets_the_layout(self):
        on = [line for line in self._lines() if "DSV41_KV_LAYOUT" in line and not line.startswith("# ")]
        self.assertEqual(on, [self._production_line()])

    def test_production_pool_pin(self):
        lines = self._lines()
        i = next(n for n, line in enumerate(lines) if line.startswith("# Production settings, together with the line"))
        j = next(n for n in range(i, len(lines)) if not lines[n].startswith("#"))
        pins = [int(line.split("=", 1)[1]) for line in lines[i:j] if line.startswith("#MAX_TOTAL_TOKENS=")]
        self.assertEqual(len(pins), 1, pins)
        self.assertEqual(pins[0] % 256, 0)
        self.assertTrue(4_000_000 <= pins[0] <= 10_000_000, pins)


if __name__ == "__main__":
    unittest.main(verbosity=1)
