"""CPU check for adapter/autotune_keep.py: per-rank caches with disjoint shape entries digest alike
when every rank's sidecar matches its launch, and differently (-> dropped) when one does not."""
import contextlib
import json
import os
import tempfile
import types
from pathlib import Path

os.environ["DSV41_AUTOTUNE_KEEP"] = "1"
import autotune_keep as ak  # noqa: E402


def fake_module(paths):
    mod = types.SimpleNamespace()
    mod._autotune_cache_digest = lambda p, env: "stock"

    @contextlib.contextmanager
    def ctx(model_runner, **kw):
        yield

    mod.flashinfer_autotune_context = ctx
    mod.flashinfer_autotune_cache_path = lambda mr: paths[mr]
    return mod


def main():
    d = Path(tempfile.mkdtemp())
    meta = {"flashinfer": "0.6.18", "arch": "sm121"}
    paths = {}
    for rank, shapes in ((0, ["(192, 2304, 320)"]), (1, ["(64, 2304, 320)"])):
        p = d / f"rank_tp{rank}_pp0_dp0.json"
        p.write_text(json.dumps({"_metadata": meta, **{s: [16, 36] for s in shapes}}))
        paths[rank] = p
    mod = fake_module(paths)
    ak.install(mod)
    env = {"cuda": "13.0"}
    assert mod._autotune_cache_digest(paths[0], env) == "", "no sidecar yet: must not be kept"
    assert not paths[0].exists(), "a cache without a matching sidecar must be deleted (forces a re-tune)"
    for rank, shapes in ((0, ["(192, 2304, 320)"]), (1, ["(64, 2304, 320)"])):
        with mod.flashinfer_autotune_context(rank):       # a tuning boot rewrites cache and sidecar
            paths[rank].write_text(json.dumps({"_metadata": meta, **{s: [16, 36] for s in shapes}}))
    d0, d1 = mod._autotune_cache_digest(paths[0], env), mod._autotune_cache_digest(paths[1], env)
    assert d0 and d0 == d1, "disjoint EP shape sets with matching launches must digest alike"
    ak.sidecar(paths[1]).write_text("other-launch\n")
    assert mod._autotune_cache_digest(paths[1], env) == "", "a changed launch must drop the cache"
    assert not paths[1].exists(), "and delete it, so no rank can load it"
    fp_before = ak.launch_fingerprint()
    os.environ["SGLANG_RUN_ID"] = "sglang-run-1.0-1"   # per boot, set by the engine
    assert ak.launch_fingerprint() == fp_before
    os.environ["DSV41_REPLICATED_SPLIT"] = "wqkv_a"      # decode-side switch: volatile
    assert ak.launch_fingerprint() == fp_before
    os.environ["DSV41_SWA_NATIVE_DECODE"] = "1"          # swan: stock tactics, never tunes: volatile
    assert ak.launch_fingerprint() == fp_before
    os.environ["DSV41_DECODE_YIELD"] = "33"              # scheduler-side chunk timing: volatile
    os.environ["DSV41_DECODE_YIELD_RID_ARMS"] = "1"
    assert ak.launch_fingerprint() == fp_before
    os.environ["DSV41_SWA_THIN"] = "1"                   # prefix-cache LRU order / cone replay: after boot,
    os.environ["DSV41_SWA_CONE_REPLAY"] = "1"            # never during the tuning pass: volatile
    os.environ["DSV41_SWA_CONE_TOKENS"] = "8192"
    os.environ["DSV41_SWA_CONE_MIN_SAVE"] = "8192"
    os.environ["DSV41_SWA_CONE_TEST"] = "1"
    assert ak.launch_fingerprint() == fp_before
    os.environ["DSV41_SWA_PROMPT_TAIL_KEEP"] = "1024"    # decode-time SWA free timing: volatile
    os.environ["DSV41_SWA_PROMPT_TAIL_HOLD"] = "8192"
    assert ak.launch_fingerprint() == fp_before
    os.environ["DSV41_VERIFY_CAP"] = "conf:0.1"          # A/B-neutral switch: same fingerprint
    assert ak.launch_fingerprint() == ak.launch_fingerprint()
    fp_a = ak.launch_fingerprint()
    os.environ["DSV41_VERIFY_CAP"] = "3"
    assert ak.launch_fingerprint() == fp_a
    os.environ["SGLANG_SOMETHING_SHAPED"] = "1"
    assert ak.launch_fingerprint() != fp_a
    print("test_autotune_keep: ok")


def test_v41c_and_pool_size_are_volatile():
    """The v41c / v41n switches and --max-total-tokens leave the launch fingerprint unchanged (a capacity A/B's X and
    CAP boots reuse the controls' tactics); a real tuned-shape switch still changes it."""
    import sys
    saved_argv, saved_env = list(sys.argv), dict(os.environ)
    try:
        sys.argv = ["launch_server", "--tp", "4", "--max-total-tokens", "4194304", "--port", "18888"]
        base = ak.launch_fingerprint()
        os.environ.update(DSV41_KV_LAYOUT="v41c", DSV41_V41C_ARM="1", DSV41_V41C_READER="native",
                          DSV41_V41N_PREFILL="auto")
        sys.argv = ["launch_server", "--tp", "4", "--max-total-tokens=7680000", "--port", "18888"]
        assert ak.launch_fingerprint() == base, "v41c keys / pool size changed the fingerprint"
        sys.argv = ["launch_server", "--tp", "4", "--port", "18888"]
        assert ak.launch_fingerprint() == base, "dropping --max-total-tokens changed the fingerprint"
        os.environ["SGLANG_V41C_TEST_SHAPED"] = "1"
        assert ak.launch_fingerprint() != base, "a tuned-shape switch must change the fingerprint"
        os.environ.pop("SGLANG_V41C_TEST_SHAPED")
        sys.argv = ["launch_server", "--tp", "8", "--port", "18888"]
        assert ak.launch_fingerprint() != base, "another argv must change the fingerprint"
    finally:
        sys.argv = saved_argv
        os.environ.clear()
        os.environ.update(saved_env)
    print("test_autotune_keep: v41c switches and --max-total-tokens are volatile: ok")


def _legacy_fingerprint(volatile):
    """The launch fingerprint of the adapters before _VOLATILE_ARGS (3305c29 / v2.6.2 / v2.6.3b): the raw argv. A/B
    controls boot those images, so an X boot must recognise the .launch they write and must not rewrite it."""
    import sys
    env = {k: v for k, v in os.environ.items()
           if k.startswith(ak._PREFIXES) and k not in volatile and not k.startswith("DSV41_AB_")}
    return __import__("hashlib").sha256(json.dumps({"argv": sys.argv, "env": env}, sort_keys=True).encode()).hexdigest()


def test_cross_adapter_reuse():
    """Capacity A/B order A (older adapter) -> X (this one, v41c keys) -> A' (older) -> CAP (this one, other pool size):
    every boot after A keeps A's cache, A's .launch is never rewritten, and a re-tune by an older adapter unbinds the
    .launch-pool written before it. 2026-11-03: without this every boot of the capacity A/B re-tuned (X wrote its
    pool-free fingerprint over A's raw-argv one, A' then dropped the cache)."""
    import sys
    old_volatile = tuple(k for k in ak._VOLATILE if not k.startswith(("DSV41_KV_LAYOUT", "DSV41_V41C_", "DSV41_V41N_")))
    saved_argv, saved_env = list(sys.argv), dict(os.environ)
    d = Path(tempfile.mkdtemp())
    cache = d / "rank_tp0_pp0_dp0.json"
    meta = {"_metadata": {"flashinfer": "0.6.18"}}
    try:
        for k in [k for k in os.environ if k.startswith(ak._PREFIXES)]:
            os.environ.pop(k)
        os.environ.update(DSV41_AUTOTUNE_KEEP="1", DSV41_BLOCK_VERIFY_FOLD="1", SGLANG_RUN_ID="run-a")
        ctl_argv = ["launch_server", "--tp", "4", "--max-total-tokens", "4194304", "--port", "18888"]

        def legacy_boot():                      # the older adapter: compare .launch verbatim, re-tune and rewrite
            fp_old = _legacy_fingerprint(old_volatile)
            kept = cache.is_file() and ak.sidecar(cache).is_file() and ak.sidecar(cache).read_text().strip() == fp_old
            if not kept:
                cache.write_text(json.dumps(meta))
                ak.sidecar(cache).write_text(fp_old + "\n")
            return kept

        def boot():                             # this adapter, one engine process
            mod = fake_module({0: cache})
            ak.install(mod)
            with mod.flashinfer_autotune_context(0):
                if not cache.is_file():
                    cache.write_text(json.dumps(meta))     # the stock save after a tuning run
            return mod._autotune_cache_digest(cache, {}) != "" and cache.is_file()

        def x_env():
            os.environ.update(DSV41_KV_LAYOUT="v41c", DSV41_V41C_ARM="1", DSV41_V41C_READER="native")

        def a_env():
            for k in ("DSV41_KV_LAYOUT", "DSV41_V41C_ARM", "DSV41_V41C_READER"):
                os.environ.pop(k, None)

        sys.argv = list(ctl_argv)
        assert not legacy_boot(), "A tunes (empty cache dir)"
        launch_a = ak.sidecar(cache).read_text()
        x_env(); os.environ["SGLANG_RUN_ID"] = "run-x"
        mod = fake_module({0: cache}); ak.install(mod)
        assert mod._autotune_cache_digest(cache, {}) != "", "X must keep the cache A's older adapter tuned"
        with mod.flashinfer_autotune_context(0):
            pass
        assert ak.sidecar(cache).read_text() == launch_a, "X must not rewrite A's .launch"
        a_env(); os.environ["SGLANG_RUN_ID"] = "run-a2"
        assert legacy_boot(), "A' (older adapter) must keep A's cache after X"
        x_env(); os.environ["SGLANG_RUN_ID"] = "run-cap"
        sys.argv = ["launch_server", "--tp", "4", "--max-total-tokens", "33554432", "--port", "18888"]
        assert boot(), "CAP (other pool size) must keep the cache through X's .launch-pool"
        assert ak.sidecar(cache).read_text() == launch_a, "CAP must not rewrite A's .launch"
        # an older adapter re-tunes under another launch: the .launch-pool written before is unbound
        a_env(); sys.argv = list(ctl_argv); os.environ["DSV41_SOMETHING_SHAPED"] = "1"
        assert not legacy_boot()
        x_env(); os.environ.pop("DSV41_SOMETHING_SHAPED")
        sys.argv = ["launch_server", "--tp", "4", "--max-total-tokens", "33554432", "--port", "18888"]
        mod = fake_module({0: cache}); ak.install(mod)
        assert mod._autotune_cache_digest(cache, {}) == "", "a stale .launch-pool must not keep a re-tuned cache"
        # this adapter tunes first (empty dir): it writes the raw-argv .launch an older adapter can keep
        cache.unlink(missing_ok=True); ak.sidecar(cache).unlink(missing_ok=True); ak.pool_sidecar(cache).unlink(missing_ok=True)
        sys.argv = list(ctl_argv)
        boot()
        assert ak.sidecar(cache).read_text().strip() == ak.launch_fingerprint(raw_argv=True), "a tune writes the raw-argv .launch"
        a_env()
        assert legacy_boot(), "an older adapter with the same raw launch keeps a cache this adapter tuned"
        # a tuned-shape switch still re-tunes this adapter
        x_env(); os.environ["SGLANG_V41C_TEST_SHAPED"] = "1"
        mod = fake_module({0: cache}); ak.install(mod)
        assert mod._autotune_cache_digest(cache, {}) == "", "a tuned-shape switch must drop the cache"
    finally:
        sys.argv = saved_argv
        os.environ.clear()
        os.environ.update(saved_env)
    print("test_autotune_keep: older-adapter controls and pool-size boots share one cache: ok")


if __name__ == "__main__":
    main()
    test_v41c_and_pool_size_are_volatile()
    test_cross_adapter_reuse()
