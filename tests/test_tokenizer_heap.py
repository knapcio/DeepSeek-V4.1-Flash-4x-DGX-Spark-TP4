"""CPU checks of DSV41_TOKENIZER_HEAP (adapter/mem_reclaim.py, the tokenizer-heap switch of perf/mem-reclaim-20261029
@ 13e68a5 taken alone). The TokenizerHeap cases are 13e68a5's tests/test_mem_reclaim.py cases byte for byte; the helpers
they use (_interp, the process argv fixtures) are copied from the same file. SourcePin checks that the module's switch
section is 13e68a5's byte for byte (sha256 pinned below).

Run: PYTHONPATH=adapter python3 tests/test_tokenizer_heap.py   (the glibc cases run in the Linux image)"""
import hashlib
import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ADAPTER = HERE.parent / 'adapter'
sys.path.insert(0, str(ADAPTER))
import mem_reclaim as mr  # noqa: E402

# 13e68a5: adapter/mem_reclaim.py from "# ---...\n# DSV41_TOKENIZER_HEAP" through install_tokenizer_heap's "return mode"
SECTION_SHA256 = '4c3cf1dde99f275ceb88e7e1ff2187eec1d0c71c7ba3e2f650e92bc2234a8915'


def _interp(code_args, env, cwd=None):
    """Run python3 with the adapter's sitecustomize; returns its stdout lines."""
    full = dict(os.environ)
    for k in list(full):
        if k.startswith('DSV41_') or k in ('TP_SIZE', 'TP'):
            del full[k]
    full.update(env)
    full['PYTHONPATH'] = str(ADAPTER)
    r = subprocess.run([sys.executable] + code_args, env=full, capture_output=True, text=True, timeout=300, cwd=cwd)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.strip().splitlines()


class LeanAux:
    """13e68a5's process argv fixtures (the LeanAux test class's constants)."""
    BOOT = (['/opt/dsv41/boot.py', 'run'], ['/opt/sglang/bin/python3', '-u', '/opt/dsv41/boot.py', 'run'])
    TRACKER = (['-c'], ['/opt/sglang/bin/python3', '-c', 'from multiprocessing.resource_tracker import main;main(53)'])
    SERVER = (['/sgl-workspace/sglang/python/sglang/launch_server.py', '--model-path', '/models/x'],
              ['/opt/sglang/bin/python3', '-m', 'sglang.launch_server', '--model-path', '/models/x'])
    SPAWN = (['-c'], ['/opt/sglang/bin/python3', '-c', 'from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=9)'])


class SourcePin(unittest.TestCase):
    def test_switch_section_is_13e68a5s(self):
        src = (ADAPTER / 'mem_reclaim.py').read_text()
        i = src.index('# ' + '-' * 117 + '\n# DSV41_TOKENIZER_HEAP')
        j = src.index('    return mode\n', src.index('def install_tokenizer_heap')) + len('    return mode\n')
        self.assertEqual(hashlib.sha256(src[i:j].encode()).hexdigest(), SECTION_SHA256)

    def test_only_the_tokenizer_heap_switch(self):
        src = (ADAPTER / 'mem_reclaim.py').read_text()
        for other in ('DSV41_LEAN_AUX', 'DSV41_MEMCAP_MEMINFO', 'DSV41_KV_RESERVE_GIB', 'DSV41_MEM_RECEIPTS', 'install_kv_configurator'):
            self.assertNotIn(other, src)
        site = (ADAPTER / 'sitecustomize.py').read_text()
        self.assertEqual(site.count('mem_reclaim'), 2)      # the comment and the one import: no other mem_reclaim switch
        self.assertIn("from mem_reclaim import install_tokenizer_heap as _install_tokenizer_heap", site)


import platform  # noqa: E402
IS_GLIBC = sys.platform.startswith('linux') and platform.libc_ver()[0] == 'glibc'
TOKENIZER = os.environ.get('DSV41_TEST_TOKENIZER', '')
HAVE_TOKENIZERS = importlib.util.find_spec('tokenizers') is not None

# a fresh interpreter: RssAnon of the tokenizer-manager workload, stock vs DSV41_TOKENIZER_HEAP=trim
_TOK_WORKLOAD = r"""
import json, random, sys, collections, gc
sys.path.insert(0, %r)
import mem_reclaim as mr
from tokenizers import Tokenizer
mode, path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
tok = Tokenizer.from_file(path)
words = 'river stone lamp orbit cedar violet market engine harbor quiet copper meadow signal'.split()
rnd = random.Random(16)
docs = {s: ' '.join(rnd.choice(words) for _ in range(int(s / 1.05))) for s in (200000, 400000, 800000, 950000)}
start = mr.rss_anon_kb()
live = collections.deque(maxlen=4)
for i in range(n):
    text = docs[rnd.choice(list(docs))] + ' ' + ' '.join(rnd.choice(words) for _ in range(rnd.randint(512, 4096)))
    chunks = [text[k:k + 65536] for k in range(0, len(text), 65536)]      # the #41259 helper: encode_batch of pieces
    ids = [i for e in tok.encode_batch(chunks, add_special_tokens=False) for i in e.ids]
    live.append((text, ids))
    if mode == 'trim' and mr.trim_due(mr.heap_free_bytes(), 256 << 20):
        mr.trim_now()
live.clear(); del text, ids, chunks
gc.collect()
if mode == 'trim' and mr.trim_due(mr.heap_free_bytes(), 256 << 20):
    mr.trim_now()
idle = mr.rss_anon_kb()
print(json.dumps({'start_mib': start >> 10, 'idle_mib': idle >> 10}))
"""


class TokenizerHeap(unittest.TestCase):
    """DSV41_TOKENIZER_HEAP (2026-10-30, C1: launch_server RssAnon 1.15 -> 3.91 GiB, all glibc [heap])."""
    SERVER = LeanAux.SERVER
    OTHERS = (LeanAux.SPAWN, LeanAux.BOOT, LeanAux.TRACKER, ([], []))

    def test_off_by_default_and_only_launch_server(self):
        self.assertEqual(mr.heap_mode({}, *self.SERVER), '')
        for v in ('0', 'off', 'bogus', ''):
            self.assertEqual(mr.heap_mode({'DSV41_TOKENIZER_HEAP': v}, *self.SERVER), '', v)
        for m in mr.HEAP_MODES:
            self.assertEqual(mr.heap_mode({'DSV41_TOKENIZER_HEAP': m}, *self.SERVER), m)
            for proc in self.OTHERS:            # scheduler / detokenizer (spawn children), boot.py, tracker
                self.assertEqual(mr.heap_mode({'DSV41_TOKENIZER_HEAP': m}, *proc), '', (m, proc))
        self.assertTrue(mr.server_role(['x'], ['/usr/bin/python3', '-m', 'sglang.launch_server', '--port', '1']))

    def test_install_is_a_no_op_when_off(self):
        saved = dict(mr._HEAP)
        try:
            mr._HEAP.clear()
            self.assertEqual(mr.install_tokenizer_heap({}, *self.SERVER), '')
            self.assertEqual(mr.install_tokenizer_heap({'DSV41_TOKENIZER_HEAP': 'trim'}, *LeanAux.SPAWN), '')
            self.assertNotIn('installed', mr._HEAP)
        finally:
            mr._HEAP.clear(); mr._HEAP.update(saved)

    def test_trim_threshold(self):
        self.assertFalse(mr.trim_due(255 << 20, 256 << 20))
        self.assertTrue(mr.trim_due(256 << 20, 256 << 20))

    def test_sitecustomize_gate(self):
        src = (ADAPTER / 'sitecustomize.py').read_text()
        self.assertIn("os.environ.get('DSV41_TOKENIZER_HEAP', '')", src)
        self.assertIn('install_tokenizer_heap', src)

    @unittest.skipUnless(IS_GLIBC, 'glibc only (Linux image)')
    def test_malloc_trim_returns_a_fragmented_heap(self):
        code = (f"import sys; sys.path.insert(0, {str(ADAPTER)!r}); import ctypes, mem_reclaim as mr\n"
                "lib = mr._libc(); lib.malloc.restype = ctypes.c_void_p; lib.free.argtypes = [ctypes.c_void_p]\n"
                "a = mr.rss_anon_kb(); ps = [lib.malloc(2000) for _ in range(300000)]\n"
                "[lib.free(p) for i, p in enumerate(ps) if i % 64]\n"     # 1 block in 64 stays: a fragmented heap
                "b = mr.rss_anon_kb(); free = mr.heap_free_bytes(); dt, _, c = mr.trim_now()\n"
                "print(a >> 10, b >> 10, c >> 10, free >> 20, round(dt * 1e3, 1))")
        a, b, c, free, ms = (float(x) for x in _interp(['-c', code], {})[-1].split())
        self.assertGreater(b - a, 400)          # ~570 MiB allocated, 98 % freed, still resident (stock glibc)
        self.assertGreater(free, 400)
        self.assertLess(c - a, 0.25 * (b - a))  # malloc_trim(0) returns free pages from the middle of the heap

    @unittest.skipUnless(IS_GLIBC, 'glibc only (Linux image)')
    def test_installed_thread_trims_in_launch_server(self):
        code = (f"import sys, time; sys.path.insert(0, {str(ADAPTER)!r}); import ctypes, mem_reclaim as mr\n"
                "env = {'DSV41_TOKENIZER_HEAP': 'trim', 'DSV41_TOKENIZER_TRIM_S': '0.1', 'DSV41_TOKENIZER_TRIM_MIN_MIB': '64'}\n"
                "mode = mr.install_tokenizer_heap(env, ['/x/sglang/launch_server.py'], ['python3', '-m', 'sglang.launch_server'])\n"
                "lib = mr._libc(); lib.malloc.restype = ctypes.c_void_p; lib.free.argtypes = [ctypes.c_void_p]\n"
                "a = mr.rss_anon_kb(); ps = [lib.malloc(2000) for _ in range(300000)]\n"
                "[lib.free(p) for i, p in enumerate(ps) if i % 64]; b = mr.rss_anon_kb(); time.sleep(1.0)\n"
                "print(mode, a >> 10, b >> 10, mr.rss_anon_kb() >> 10, mr._HEAP.get('trims', 0))")
        mode, a, b, c, trims = _interp(['-c', code], {})[-1].split()
        a, b, c = float(a), float(b), float(c)
        self.assertEqual(mode, 'trim')
        self.assertGreaterEqual(int(trims), 1)
        self.assertLess(c - a, 0.25 * (b - a))

    @unittest.skipUnless(IS_GLIBC and HAVE_TOKENIZERS and TOKENIZER and os.path.exists(TOKENIZER),
                         'needs glibc, tokenizers and DSV41_TEST_TOKENIZER=<DeepSeek-V4.1 tokenizer.json>')
    def test_rss_returns_after_large_tokenization(self):
        code = _TOK_WORKLOAD % str(ADAPTER)
        import json as _json
        stock = _json.loads(_interp(['-c', code, 'stock', TOKENIZER, '24'], {})[-1])
        trim = _json.loads(_interp(['-c', code, 'trim', TOKENIZER, '24'], {})[-1])
        kept, left = stock['idle_mib'] - stock['start_mib'], trim['idle_mib'] - trim['start_mib']
        self.assertGreater(kept, 150, stock)        # stock glibc keeps the high-water (2026-10-30 image run: +249 MiB)
        self.assertLess(left, 128, trim)            # trim gives the free heap back (+83 MiB: tokenizer caches, arenas)
        self.assertLess(left, 0.4 * kept, (stock, trim))


if __name__ == '__main__':
    unittest.main(verbosity=2)
