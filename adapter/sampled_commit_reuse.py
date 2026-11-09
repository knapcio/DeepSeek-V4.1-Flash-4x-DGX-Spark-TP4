"""Sampled DSpark steps reuse the draft-KV commit projection the verify graph already computed. Default OFF.

DSV41_SAMPLED_COMMIT_REUSE=1 | check | 0.

Why. In static verify mode the target-verify CUDA graph ends with DsparkVerifyEpilogue._static_epilogue: the
in-graph greedy accept, then the draft-KV commit (draft model write_target_hidden_kv): main_proj of the six
target hidden rows (+ the DSV41_DRAFT_MAIN_PROJ_SPLIT all-gather), main_norm, the stacked wkv MXFP8 GEMM of the
three draft stages, then three fused k-norm + RoPE + FlashMLA stores whose rows are gated by commit_lens *
inject_gate. On a sampled step (any non-greedy row, a grammar, a logit bias...) fold_eligible is False:
inject_gate is 0, so the graph computes the whole projection and stores nothing, and the worker then runs
TargetVerifyExecutor.commit_hidden EAGERLY, which recomputes the same projection from the same hidden rows and
stores it with the sampled commit_lens. Trace 2026-11-08 (Tier 2 boot 1, rank 0, prose, T 1.0 / top-p 0.95 vs
T 0): the eager copy is gemv_rowscale 135 us + NCCL all-gather 104 us + RMSNorm / MXFP8 quant / wkv GEMM 45 us +
~8 launch gaps, about 0.3 ms of the +0.65 ms eager tail a sampled step pays (diagnostics/c3-sampled-accept-
20261109/DESIGN.md).

What. The epilogue's projection output (the three stage KV rows, before norm / RoPE) is copied in the graph into a
persistent buffer (one _foreach_copy_, ~2 us). On an eligible eager commit the projection is not recomputed: the
three stores run on the buffered rows with the eager path's own swa_loc and positions. Nothing else changes.

Exactness (bitwise, Class A). The buffered rows are the output of the same kernels on the same input: the hidden
rows the eager commit reads (logits_output.hidden_states) are the graph's own output rows, the projection is
row-independent only up to the GEMM's M, so reuse requires the replayed graph's batch to equal the real batch
(runner.bs == runner.raw_bs == bs, i.e. no padding rows: M is the same) and the graph to be one this process
captured with the buffer copy (per A/B variant). The all-gather is a copy (RoCE in the graph, NCCL eagerly: same
bytes). No random numbers are drawn on this path, so the RNG stream is untouched. check mode recomputes the
projection eagerly as stock does, stores from it (production behaviour unchanged) and counts on device every row
whose buffered bits differ (contract: 0), logged every DSV41_SAMPLED_COMMIT_REUSE_LOG_EVERY commits and at each
busy -> idle transition.

Rank invariance. Reusing skips the eager all-gather, a collective: every TP rank must decide alike. Every input of
the decision is rank-invariant: the mode (env, or the A/B variant that all ranks switch on the same step), the
captured set (every rank captures the same shapes in the same order), can_run_cuda_graph, runner bs / raw_bs and
the real bs. The buffer is allocated at warm-up; a failed allocation raises (fails the boot) instead of leaving
one rank on the stock path.

Memory: max_bs * verify_tokens * stages * kv_dim bf16 = 16 * 6 * 3 * 512 * 2 B ~ 0.3 MB per rank.

Engine pin: the stock write_target_hidden_kv body is mirrored here (projection / stores split apart); on a source
drift the adapter does not install (stock behaviour) and says so.
"""
import hashlib
import inspect
import os

import torch

KEY = "DSV41_SAMPLED_COMMIT_REUSE"
_OFF = ("", "0", "off", "false", "no")
LOG_EVERY = int(os.environ.get("DSV41_SAMPLED_COMMIT_REUSE_LOG_EVERY", "2000") or 0)

# sha256 of inspect.getsource(inspect.unwrap(DeepseekV4ForCausalLMDSpark.write_target_hidden_kv)), SGLang tree of
# release/v2.7-rc 1303d2f (image rg27dry-overlay:v27)
PIN_WRITE_TARGET_HIDDEN_KV = "65394827989745e5e3e21f42ef5b54d2840cf1964e384679896014609d9cb59e"

try:  # DSV41_AB_VARIANTS in-boot A/B (test only): the buffer copy per capture, the reuse per replayed variant
    import ab_variant as _ab
except ImportError:  # pragma: no cover
    _ab = None


def parse_mode(value):
    v = (value or "0").strip().lower()
    if v in _OFF:
        return "off"
    if v == "check":
        return "check"
    if v in ("1", "on", "true", "yes"):
        return "on"
    raise RuntimeError(f"{KEY}={value!r}: expected 0 / 1 / check")


def installed():
    """The install-time gate (os.environ holds the union of the variants under the A/B harness)."""
    return parse_mode(os.environ.get(KEY, "0")) != "off"


def mode():
    """off / on / check for the variant being captured or replayed (plain env when not under A/B)."""
    if _ab is not None and getattr(_ab, "ACTIVE", False):
        return parse_mode(_ab.env(KEY, "0"))
    return parse_mode(os.environ.get(KEY, "0"))


def _capture_tag():
    return _ab.current() if _ab is not None and getattr(_ab, "ACTIVE", False) else None


def _graph_variant(bs):
    """Which A/B graph set a replay at this size comes from (ab_variant.install_graph_backend)."""
    if _ab is None or not getattr(_ab, "ACTIVE", False):
        return None
    limit = getattr(_ab, "_MAX_BS", None)
    return _ab.current() if limit is None or bs <= limit else 0


def _capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


# state: the buffer, the (variant, rows) shapes whose graphs copy into it, the per-step verify record, the context
_S = {"buf": None, "captured": set(), "step": None, "ctx": None, "engine": None}
STATS = {"reuse": 0, "check": 0, "stock": 0, "commits": 0}
REASONS = {}
_C = {"counters": None}          # check mode, on device: [checked commits, rows, mismatched rows]
_LOGGED = {"commits": 0}


def _rank():
    try:
        import torch.distributed as dist
        return dist.get_rank() if dist.is_initialized() else 0
    except Exception:  # noqa: BLE001
        return 0


def _stock(reason):
    STATS["stock"] += 1
    REASONS[reason] = REASONS.get(reason, 0) + 1


def _receipt_line(at=None):
    tail = "" if at is None else f" at={at}"
    return (f"[sampled_commit_reuse] receipt rank={_rank()} commits={STATS['commits']} reuse={STATS['reuse']} "
            f"check={STATS['check']} stock={STATS['stock']} reasons={dict(sorted(REASONS.items()))}{tail}")


def _check_line(at=None):
    c = _C["counters"]
    if c is None:
        return None
    v = c.tolist()                                  # host sync: log lines only (check mode)
    tail = "" if at is None else f" at={at}"
    return (f"[sampled_commit_reuse] check: rank={_rank()} commits {v[0]} rows {v[1]} "
            f"mismatched rows {v[2]}{tail}")


def _tick():
    STATS["commits"] += 1
    if LOG_EVERY and STATS["commits"] % LOG_EVERY == 0:
        print(_receipt_line(), flush=True)
        line = _check_line()
        if line:
            print(line, flush=True)


def flush(at="idle"):
    """Terminal receipt at the scheduler's busy -> idle transition (every rank, same step)."""
    if STATS["commits"] == _LOGGED["commits"]:
        return
    _LOGGED["commits"] = STATS["commits"]
    try:
        print(_receipt_line(at), flush=True)
        line = _check_line(at)
        if line:
            print(line, flush=True)
    except Exception as exc:  # noqa: BLE001 - a receipt never stops the scheduler
        print(f"[sampled_commit_reuse] terminal receipt failed: {exc!r}", flush=True)


# ------------------------------------------------------------------------------- the mirrored stock body --
def _project(model, main_hidden, pool, eng):
    """write_target_hidden_kv up to the stores: the per-stage KV rows (stock order and arguments)."""
    main_x = model.project_target_hidden(main_hidden)
    return eng.CommitKvProj.execute(
        main_x=main_x,
        wkv_linears=[stage.self_attn.wkv for stage in model.stages],
        allow_strided_output=(
            eng.get_platform().is_blackwell
            and not eng.is_unified_kv_triton()
            and not pool.uniform_fp8
        ),
    )


def _store(model, kvs, swa_loc, positions, pool, eng):
    """write_target_hidden_kv's stores (stock order and arguments)."""
    swa_loc = swa_loc.to(torch.int32)
    store_kv = (
        pool.set_unified_key_buffer_radix_fused_norm_rope
        if eng.is_unified_kv_triton()
        else pool.set_swa_key_buffer_radix_fused_norm_rope
    )
    for stage, kv in zip(model.stages, kvs):
        attn = stage.self_attn
        store_kv(
            layer_id=attn.layer_id,
            swa_loc=swa_loc,
            kv=kv,
            kv_weight=attn.kv_norm.weight.data,
            eps=attn.eps,
            freqs_cis=attn.freqs_cis,
            positions=positions,
        )


def _ensure_buf(kvs, rows_max):
    buf = _S["buf"]
    s, d, dt, dev = len(kvs), kvs[0].shape[-1], kvs[0].dtype, kvs[0].device
    if buf is not None and buf.shape[0] == s and buf.shape[2] == d and buf.dtype == dt and buf.device == dev \
            and buf.shape[1] >= rows_max:
        return buf
    if _capturing():
        return None                                  # never allocate inside a capture: this shape stays stock
    # a failed allocation raises: a rank left on the stock path would hang the others' skipped all-gather
    _S["buf"] = buf = torch.empty((s, rows_max, d), dtype=dt, device=dev)
    _S["captured"].clear()                          # graphs captured before point at the old buffer
    print(f"[sampled_commit_reuse] rank {_rank()}: commit buffer {tuple(buf.shape)} {dt} "
          f"({buf.numel() * buf.element_size() / 2**20:.2f} MiB)", flush=True)
    return buf


def _epilogue_write(model, main_hidden, swa_loc, positions, pool, eng, rows_max):
    kvs = _project(model, main_hidden, pool, eng)
    n = main_hidden.shape[0]
    if mode() != "off" and n <= rows_max and all(kv.shape[0] == n for kv in kvs):
        buf = _ensure_buf(kvs, rows_max)
        if buf is not None:
            torch._foreach_copy_([buf[i, :n] for i in range(len(kvs))], list(kvs))
            if _capturing():
                _S["captured"].add((_capture_tag(), n))
    _store(model, kvs, swa_loc, positions, pool, eng)


def _reuse_write(model, main_hidden, swa_loc, positions, pool, eng, ctx):
    _tag, n, m = ctx
    buf = _S["buf"]
    if main_hidden.shape[0] != n or positions.shape[0] != n or buf is None:
        _stock("rows")
        _store(model, _project(model, main_hidden, pool, eng), swa_loc, positions, pool, eng)
        return
    held = [buf[i, :n] for i in range(buf.shape[0])]
    if m == "check":
        STATS["check"] += 1
        kvs = _project(model, main_hidden, pool, eng)
        c = _C["counters"]
        if c is None or c.device != kvs[0].device:
            c = _C["counters"] = torch.zeros(3, dtype=torch.int64, device=kvs[0].device)
        bad = torch.zeros((n,), dtype=torch.bool, device=kvs[0].device)
        for h, kv in zip(held, kvs):
            bad |= (h.view(torch.int16) != kv.view(torch.int16)).any(-1)
        one = torch.ones((), dtype=torch.int64, device=c.device)
        c.add_(torch.stack([one, torch.full_like(one, n), bad.sum()]))
        _store(model, kvs, swa_loc, positions, pool, eng)
        return
    STATS["reuse"] += 1
    _store(model, held, swa_loc, positions, pool, eng)


# ------------------------------------------------------------------------------------------------- install --
def source_sha(fn):
    return hashlib.sha256(inspect.getsource(inspect.unwrap(fn)).encode()).hexdigest()


def install_dspark(module):
    """sglang.srt.models.deepseek_v4_dspark: DeepseekV4ForCausalLMDSpark.write_target_hidden_kv."""
    if not installed():
        return False
    cls = getattr(module, "DeepseekV4ForCausalLMDSpark", None)
    fn = getattr(cls, "write_target_hidden_kv", None) if cls is not None else None
    missing = [n for n in ("CommitKvProj", "get_platform", "is_unified_kv_triton") if not hasattr(module, n)]
    if fn is None or missing:
        print(f"[sampled_commit_reuse] NOT installed: write_target_hidden_kv / {missing} gone (engine drift)",
              flush=True)
        return False
    if getattr(cls, "_dsv41_sampled_commit_reuse", False):
        return True
    got = source_sha(fn)
    if got != PIN_WRITE_TARGET_HIDDEN_KV:
        print(f"[sampled_commit_reuse] NOT installed: write_target_hidden_kv source drifted ({got[:16]} != "
              f"{PIN_WRITE_TARGET_HIDDEN_KV[:16]}); stock behaviour", flush=True)
        return False
    cls._dsv41_sampled_commit_reuse = True
    _S["engine"] = module
    orig = cls.write_target_hidden_kv

    def write_target_hidden_kv(self, *args, **kw):
        ctx = _S["ctx"]
        if ctx is None or args or set(kw) != {"main_hidden", "swa_loc", "positions", "pool"}:
            return orig(self, *args, **kw)
        if ctx[0] == "epilogue":
            return _epilogue_write(self, kw["main_hidden"], kw["swa_loc"], kw["positions"], kw["pool"], module,
                                   ctx[1])
        return _reuse_write(self, kw["main_hidden"], kw["swa_loc"], kw["positions"], kw["pool"], module, ctx[1:])

    write_target_hidden_kv.__wrapped__ = orig
    cls.write_target_hidden_kv = write_target_hidden_kv
    print(f"[sampled_commit_reuse] armed on write_target_hidden_kv (mode {mode()})", flush=True)
    return True


def _runner_shape(executor):
    runner = getattr(getattr(getattr(executor, "target_worker", None), "model_runner", None),
                     "decode_cuda_graph_runner", None)
    return getattr(runner, "bs", None), getattr(runner, "raw_bs", None)


def install_verify(module):
    """sglang.srt.speculative.dspark_components.dspark_verify: the epilogue context, the per-step verify record and
    the eager commit."""
    if not installed():
        return False
    ex = getattr(module, "TargetVerifyExecutor", None)
    ep = getattr(module, "DsparkVerifyEpilogue", None)
    need = [(ex, "run_non_compact"), (ex, "commit_hidden"), (ep, "_static_epilogue")]
    if ex is None or ep is None or any(not hasattr(c, n) for c, n in need):
        print("[sampled_commit_reuse] NOT installed: TargetVerifyExecutor / DsparkVerifyEpilogue drifted", flush=True)
        return False
    if getattr(ex, "_dsv41_sampled_commit_reuse", False):
        return True
    ex._dsv41_sampled_commit_reuse = True
    orig_epi, orig_run, orig_commit = ep._static_epilogue, ex.run_non_compact, ex.commit_hidden

    def _static_epilogue(self, out, forward_batch):
        prev, _S["ctx"] = _S["ctx"], ("epilogue", self.max_bs * self.stride)
        try:
            return orig_epi(self, out, forward_batch)
        finally:
            _S["ctx"] = prev

    def run_non_compact(self, *args, **kw):
        _S["step"] = None
        result = orig_run(self, *args, **kw)
        g_bs, raw_bs = _runner_shape(self)
        _S["step"] = (bool(getattr(result, "can_run_cuda_graph", False)), g_bs, raw_bs)
        return result

    def commit_hidden(self, *args, **kw):
        step, _S["step"] = _S["step"], None          # one verify, one commit
        m = mode()
        if m == "off" or args:
            return orig_commit(self, *args, **kw)
        _tick()
        bs = kw.get("bs")
        if kw.get("run_compact", True):
            _stock("compact")
        elif step is None or not step[0]:
            _stock("eager_verify")
        elif not (isinstance(bs, int) and step[1] == bs and step[2] == bs):
            _stock("padded")
        elif (_graph_variant(bs), bs * self.verify_num_draft_tokens) not in _S["captured"]:
            _stock("not_captured")
        else:
            prev, _S["ctx"] = _S["ctx"], ("reuse", _graph_variant(bs), bs * self.verify_num_draft_tokens, m)
            try:
                return orig_commit(self, *args, **kw)
            finally:
                _S["ctx"] = prev
        return orig_commit(self, *args, **kw)

    ep._static_epilogue, ex.run_non_compact, ex.commit_hidden = _static_epilogue, run_non_compact, commit_hidden
    print("[sampled_commit_reuse] armed on the static verify epilogue + eager commit_hidden", flush=True)
    return True


def install_scheduler(module):
    """sglang.srt.managers.scheduler: terminal receipts on the busy -> idle transition."""
    if not installed():
        return
    cls = module.Scheduler
    if getattr(cls, "_dsv41_scr_receipt", False):
        return
    name = next((n for n in ("on_idle", "self_check_during_idle") if hasattr(cls, n)), None)
    if name is None:
        print("[sampled_commit_reuse] Scheduler has no on_idle hook: no terminal receipts", flush=True)
        return
    original = getattr(cls, name)

    def idle_hook(self, *args, **kwargs):
        flush("idle")
        return original(self, *args, **kwargs)

    idle_hook.__wrapped__ = original
    setattr(cls, name, idle_hook)
    cls._dsv41_scr_receipt = True
