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
    os.environ["DSV41_VERIFY_CAP"] = "conf:0.1"          # A/B-neutral switch: same fingerprint
    assert ak.launch_fingerprint() == ak.launch_fingerprint()
    fp_a = ak.launch_fingerprint()
    os.environ["DSV41_VERIFY_CAP"] = "3"
    assert ak.launch_fingerprint() == fp_a
    os.environ["SGLANG_SOMETHING_SHAPED"] = "1"
    assert ak.launch_fingerprint() != fp_a
    print("test_autotune_keep: ok")


if __name__ == "__main__":
    main()
