"""Keep the FlashInfer autotune cache across boots under expert parallelism. Default OFF.

DSV41_AUTOTUNE_KEEP=1. The engine's entry gate (_drop_diverged_autotune_cache) digests each rank's
whole cache file and drops every cache when the digests differ. Under EP each rank tunes its own
MoE shapes, so the files can never agree: the cache is deleted on every boot and the fused-MoE
tactics are re-drawn from timing noise (sglang#40320, fix proposed in sglang#40420).

Here the digest covers only the load decision, as in #40420, plus one guard of our own: a sidecar
fingerprint of the launch (argv and the SGLANG_/DSV41_/SPARK_/B12X_/NCCL_ environment) written next
to the cache after tuning. A cache is kept only when its sidecar matches the current launch, so a
config change cannot leave one rank with a partial hit (a hit skips a profile, and a skipped
profile on one rank desynchronises the timing reduction).

Two fingerprints, two sidecars. `<rank>.launch` holds the fingerprint over the raw argv, the form every
adapter before this one wrote and compares verbatim (an A/B control booting an older release image reads
it). `<rank>.launch-pool` holds this adapter's fingerprint with the pool-size argv dropped, bound to the
`.launch` text it was written next to. A cache is kept when `.launch` equals either fingerprint of the
current launch, or `.launch-pool` names the current pool-free fingerprint AND the `.launch` that is on disk
now. A kept cache's `.launch` is never rewritten, so an older adapter that tuned it still reuses it on its
next boot; a re-tune rewrites `.launch`, which unbinds every `.launch-pool` written before it.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ENABLED = os.environ.get("DSV41_AUTOTUNE_KEEP", "0").strip() not in ("0", "", "off", "false")
_PREFIXES = ("SGLANG_", "DSV41_", "SPARK_", "B12X_", "NCCL_")
# Switches that change neither the GEMM shapes nor the kernels tuned: A/B arms over them share tactics.
_VOLATILE = ("DSV41_DRAFT_CAPTURE", "DSV41_DRAFT_CAPTURE_OUT", "DSV41_DRAFT_CAPTURE_TRIGGER",
             "DSV41_DRAFT_CAPTURE_MAX_GIB", "DSV41_VERIFY_CAP", "DSV41_VERIFY_CAP_MIN", "DSV41_DRAFT_TAU",
             "DSV41_BLOCK_VERIFY", "DSV41_FOLDED_FENCE", "DSV41_AUTOTUNE_KEEP",
             # set by the engine itself, a new timestamp every boot (found by MiaAI-Lab)
             "SGLANG_RUN_ID",
             # switches that change neither the tuned MoE shapes nor their kernels: toggling one must
             # reuse the tactics, or every A/B boot re-draws them (which moves the numerics)
             "DSV41_DRAFT_HEAD_FP8", "DSV41_ENGRAM_PREFETCH", "DSV41_ENGRAM_PREFETCH_CHECK",
             "DSV41_REPLICATED_SPLIT", "DSV41_REPLICATED_SPLIT_MAX_M", "DSV41_DRAFT_MAIN_PROJ_SPLIT",
             "DSV41_DRAFT_MAIN_PROJ_MAX_M", "DSV41_ROCE_GATHER", "DSV41_ROUTER_LIVE", "DSV41_DYN_SHARED",
             "DSV41_DYN_SHARED_RATIO", "DSV41_DYN_SHARED_FIXED", "DSV41_DYN_SHARED_MAX_M",
             "DSV41_DYN_SHARED_SERIAL", "DSV41_VERIFY_CAP_LOG", "DSV41_VERIFY_CAP_LOG_FEATURES",
             "DSV41_ROUTE_STATS", "DSV41_ROUTE_RING", "DSV41_DRAFT_TAU_POS", "DSV41_ENGRAM_DRM_NODE",
             "DSV41_ENGRAM_DRM_MIB", "DSV41_ENGRAM_DRM_LAYER",
             # the swan decode never tunes (tuning mode takes the stock split) and runs the tactic the stock
             # call resolves for the same shape: the tuned shapes and kernels are the same with it on or off
             "DSV41_SWA_NATIVE_DECODE",
             # the decode yield only decides WHEN the next prefill chunk runs (scheduler side, after boot):
             # no GEMM shape, kernel or tuning pass depends on it
             "DSV41_DECODE_YIELD", "DSV41_DECODE_YIELD_RID_ARMS",
             # SWA thin / cone replay act on the prefix-cache tree and on replays of a cached prefix (scheduler
             # planning, a page copy and a write mask in the prefill metadata), after boot; the autotune pass has
             # no cached prefix, so no replay runs while tuning, and no tuned shape or kernel depends on them
             "DSV41_SWA_THIN", "DSV41_SWA_CONE_REPLAY", "DSV41_SWA_CONE_TOKENS", "DSV41_SWA_CONE_MIN_SAVE",
             "DSV41_SWA_CONE_TEST",
             # the SWA prompt-tail floor (in the production profile since v2.6.6) only changes when request-owned SWA
             # slots are freed (scheduler side, after boot): no GEMM shape, kernel or tuning pass depends on it, so
             # the v2.6.5 tactic table stays valid
             "DSV41_SWA_PROMPT_TAIL_KEEP", "DSV41_SWA_PROMPT_TAIL_HOLD",
             # the opt-in V4.1 FP4 compressed-KV layout and its readers (DSV41_KV_LAYOUT=v41c): they change the KV pool
             # and the attention readers, never a tuned GEMM / MoE shape or kernel; the v41n reader never tunes and
             # resolves the stock decode tactics from the kept cache, so its A/B boots must reuse the controls' cache
             "DSV41_KV_LAYOUT", "DSV41_V41C_ARM", "DSV41_V41C_READER", "DSV41_V41C_DEDUP_ROWS",
             "DSV41_V41C_DEDUP_MIN_ROWS", "DSV41_V41N_TACTICS", "DSV41_V41N_DIAG", "DSV41_V41N_PREFILL",
             "DSV41_V41N_PAGE_MIN_ROWS", "DSV41_V41N_PAGE_CAP_MB",
             # the tokenizer heap trim runs in launch_server only (the HTTP / tokenizer process): no scheduler, GEMM
             # shape or kernel sees it
             "DSV41_TOKENIZER_HEAP", "DSV41_TOKENIZER_TRIM2_IDLE_S", "DSV41_TOKENIZER_TRIM2_POLL_S",
             "DSV41_TOKENIZER_TRIM2_MIN_MIB", "DSV41_TOKENIZER_TRIM2_MIN_INTERVAL_S", "DSV41_TOKENIZER_TRIM2_MAX_BUSY_S",
             "DSV41_TOKENIZER_TRIM2_INFO_EVERY_S",
             "DSV41_TOKENIZER_TRIM2_BUDGET_MS", "DSV41_TOKENIZER_TRIM2_MAX_DEFER_S", "DSV41_TOKENIZER_TRIM2_LOG_MIN_S",
             "DSV41_TOKENIZER_TRIM2_SUMMARY_S", "DSV41_TOKENIZER_TRIM2_MS_PER_GIB")
# argv options that size the KV pool only (no tuned shape depends on them): a capacity boot reuses the tactics
_VOLATILE_ARGS = ("--max-total-tokens",)


def _argv_key(argv):
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in _VOLATILE_ARGS:
            skip = True
            continue
        if any(a.startswith(v + "=") for v in _VOLATILE_ARGS):
            continue
        out.append(a)
    return out


def launch_fingerprint(raw_argv: bool = False) -> str:
    """raw_argv=False: argv without the pool-size options (this adapter's own key). raw_argv=True: the whole argv,
    i.e. the fingerprint an adapter without _VOLATILE_ARGS computes for the same launch (env minus the volatile keys
    either way: a key volatile here and not there is absent from that adapter's boots)."""
    env = {k: v for k, v in os.environ.items()
           if k.startswith(_PREFIXES) and k not in _VOLATILE and not k.startswith("DSV41_AB_")}
    # In-boot A/B (adapter/ab_variant.py, test only) unions some gates into os.environ; fingerprint
    # the base config it recorded instead, so an A/B boot keys like the boot of its base env.
    for k, v in json.loads(os.environ.get("DSV41_AB_BASE") or "{}").items():
        if k in _VOLATILE:
            continue
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    payload = {"argv": list(sys.argv) if raw_argv else _argv_key(sys.argv), "env": env}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def sidecar(cache_path: Path) -> Path:
    return cache_path.with_suffix(".launch")


def pool_sidecar(cache_path: Path) -> Path:
    return cache_path.with_suffix(".launch-pool")


def same_launch(cache_path: Path, fp: str, fp_raw: str) -> bool:
    """The cache at cache_path was tuned by this launch, up to volatile keys and the pool size (module docstring)."""
    try:
        launch = sidecar(cache_path).read_text().strip()
    except OSError:
        return False
    if launch and launch in (fp, fp_raw):
        return True
    try:
        pool = pool_sidecar(cache_path).read_text().split()
    except OSError:
        return False
    return len(pool) == 2 and pool[0] == fp and pool[1] == launch and bool(launch)


def install(mod):
    """sglang.srt.model_executor.runner.flashinfer_autotune"""
    if not ENABLED:
        return
    for name in ("_autotune_cache_digest", "flashinfer_autotune_context"):
        if not hasattr(mod, name):
            raise RuntimeError(f"DSV41_AUTOTUNE_KEEP: {name} is gone; engine drifted")
    if getattr(mod, "_dsv41_autotune_keep", False):
        return
    mod._dsv41_autotune_keep = True
    orig_ctx = mod.flashinfer_autotune_context
    fp, fp_raw = launch_fingerprint(), launch_fingerprint(raw_argv=True)

    def _autotune_cache_digest(cache_path, env):
        cache_path = Path(cache_path)
        if not cache_path.is_file():
            return ""
        try:
            configs = json.loads(cache_path.read_text())
            same = same_launch(cache_path, fp, fp_raw)
        except (OSError, ValueError):
            configs, same = None, False
        if not isinstance(configs, dict) or not same:
            # a different launch re-tunes: an empty digest alone would still let the stock code
            # load the old file when every rank reports "" (they agree)
            cache_path.unlink(missing_ok=True)
            pool_sidecar(cache_path).unlink(missing_ok=True)
            return ""
        stamp = configs.get("_metadata")
        if not isinstance(stamp, dict):
            stamp = None
        # fp itself differs between ranks (node rank, per-node env); only "same launch as when
        # this rank saved" enters the cross-rank digest.
        payload = {"loadable": True, "stamp": stamp, "env": env}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    import contextlib

    @contextlib.contextmanager
    def flashinfer_autotune_context(model_runner, *args, **kwargs):
        cache_path = Path(mod.flashinfer_autotune_cache_path(model_runner))
        before = _autotune_cache_digest(cache_path, {}) != ""
        with orig_ctx(model_runner, *args, **kwargs):
            # the stock entry gate has run: a cache another rank's digest made it drop is gone by now
            kept = before and cache_path.is_file()
            yield
        try:
            if cache_path.is_file():
                if kept:
                    # never rewrite a kept cache's .launch: the adapter that tuned it compares it verbatim
                    launch = sidecar(cache_path).read_text().strip()
                else:
                    launch = fp_raw
                    sidecar(cache_path).write_text(fp_raw + "\n")
                pool_sidecar(cache_path).write_text(f"{fp} {launch}\n")
        except OSError:
            pass
        print(f"[autotune_keep] {'reused' if kept else 'tuned and saved'} {cache_path.parent.name}/{cache_path.name}",
              flush=True)

    mod._autotune_cache_digest = _autotune_cache_digest
    mod.flashinfer_autotune_context = flashinfer_autotune_context
    print("[autotune_keep] armed", flush=True)
