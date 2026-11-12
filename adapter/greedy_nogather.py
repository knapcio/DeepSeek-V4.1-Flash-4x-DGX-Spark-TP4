"""Greedy steps without the full-vocabulary logits gathers (DSV41_GREEDY_NOGATHER, default off; production profile 1 since
v2.7.6).

Production forces folded sampling (SGLANG_DSPARK_FOLDED_SAMPLING=2): the DSpark draft graph then takes
markov_head.sample_block for every batch, which all-gathers the full [bs * gamma, V] fp32 step logits at each of
the gamma Markov steps (MarkovHead._apply_step_logits_sharded -> self._vocab_gather(step_local)) and copies the
corrected [bs * gamma, V] logits into corrected_out. Upstream's own greedy fast path
(MarkovHead.sample_block_greedy_fused -> sharded_greedy_step: per-rank (max, argmax) over the vocab shard, one
compact stacked gather of [rows, 2 * cdiv(shard, 4096)] fp32, a finish kernel) is skipped by the
`if not self.folded_sampling` guard of DsparkDraftSampler.__call__. Measured cost of the gathers (10-24 traces):
c16 5 x 392.7 us = 1.96 ms/step, c1 7 x 48.8 us.

The target verify head does the same at every verify step: the rank's [M, V/4] logits are all-gathered in full
(c16: one NCCL AllGather_RING_LL of 96 x 129280 fp32, 935 us, above the 2 MiB RoCE limit) and copied into the fp32
logits buffer, only for the in-graph greedy accept (DsparkVerifyEpilogue -> accept_greedy_triton -> _row_argmax).

This adapter gives every draft (and target verify) CUDA graph shape a second, "greedy" flavour and replays it on
steps that are all greedy. Any other step replays the stock graph, so sampled rows are untouched.

  draft   the DSpark draft graph's flavour runs the sharded greedy Markov body (upstream's
          sample_block_greedy_fused) on steps whose batch is all greedy (sampling_info.is_all_greedy: per-batch
          scheduler state, identical on every TP rank)
  target  the target verify graph's flavour computes the rank's logits (the certified head's where qualified),
          reduces them to one (max, lowest argmax) pair per 4096 columns, all-gathers the pairs ([M, 16] fp32 per
          rank) and hands the argmax ids to the accept; the fp32 logits buffer is returned unwritten. Replayed only
          on steps whose accept runs in the graph: the static epilogue armed (begin_static_step(bs,
          fold_eligible): all greedy, no logit adjustments / penalizers, no grammar), no return_logprob, and
          adapter/draft_capture.py not reading the logits (tap off, or the commit folded). Every other consumer of
          the verify logits exists only on steps that never take the flavour.

Exactness, target: the accept's target argmax is the same function of the same fp32 values (lowest index on
ties, the stock row_argmax / torch.argmax rule; the columns cut at vocab_size exactly as _copy_logits_to_buffer
cuts the gathered tensor); bitwise for every row of finite logits, so the emitted tokens are the stock ones.

Exactness, draft. The draft token of a greedy row is the argmax of the same fp32 values (bf16 base logits + bf16 Markov
bias, added in fp32) in both bodies; ties go to the lowest vocab id in both. The stock folded kernel ranks the
values through exp(s - tile_max) keys, which can merge two distinct values closer than ~1e-7 (exp rounding to 1.0)
and then picks the lower id; the sharded kernel compares the fp32 values themselves. On such a near-tie the draft
token can differ. The emitted tokens cannot: greedy verification accepts exactly the target's argmax, so a draft
token changes only how many tokens a step accepts (and, through that, which rows a later verify batches together,
the same class of effect as any draft-only change; gate with the T=0 bracket). With DSV41_SPEC_SYNC_FREE's
rank-invariant draft noise the greedy flavour still advances the noise counter once per replay (as the stock graph
does), so later sampled steps draw exactly the stock noise; without it, the torch generator does not advance on
greedy steps (sampled outputs were never replay-identical across boots without it).

Not in this adapter: the sampled tiers of decode.md idea #3 (b: sharded exponential race, c: sharded block
verification); see diagnostics/decode-batch-20261109/DESIGN.md.

Wiring (sitecustomize.py, gate checked before the import):
  sglang.srt.speculative.dspark_components.dspark_draft_sampler -> install_sampler (after DSV41_SPEC_SYNC_FREE,
      which checks the engine's own __call__ source at install, and BEFORE DSV41_EAGER_GLUE: its vcap wrapper must
      stay outside the flavour body so the verify_cap live-length update is captured into the flavour graph too;
      tests/test_greedy_nogather_vcap.py)
  sglang.srt.speculative.dspark_components.dspark_draft         -> install_proposer (also wraps
      FullCudaGraphBackend, imported there)
  sglang.srt.layers.logits_processor                            -> install_logits (target; BEFORE cert_head)
  sglang.kernels.ops.speculative.dspark.dspark_accept           -> install_accept (target)
  sglang.srt.speculative.dspark_components.dspark_verify        -> install_verify (target)
Memory: the flavours are one more capture of every draft / target-verify shape (production: 12 draft and 24 target
shapes) in the shared graph pool.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

PARTS = ("draft", "target")
_OFF = ("", "0", "off", "false", "no")
LOG_EVERY = int(os.environ.get("DSV41_GREEDY_NOGATHER_LOG_EVERY", "2000") or 0)

_state = {"greedy": False, "flavour": False, "installed": set(), "logged": set(),
          "captured": 0, "replays": [0, 0], "fallback": 0,
          # target part: flavour capture flag, the step's selection, the pair-argmax ids handed to the accept
          "tflavour": False, "tgreedy": False, "static_armed": None, "ids": None, "tcaptured": 0,
          "treplays": [0, 0], "tfallback": 0, "target_ok": False, "eligible": False}


def _parse(value):
    raw = (value or "").strip().lower()
    if raw in _OFF:
        return frozenset()
    if raw in ("1", "on", "true", "all"):
        return frozenset(PARTS)
    parts = frozenset(p.strip() for p in raw.split(",") if p.strip())
    bad = parts - set(PARTS)
    if bad:
        raise RuntimeError(f"DSV41_GREEDY_NOGATHER: unknown part(s) {sorted(bad)}; known {PARTS}")
    return parts


def installed(part="draft") -> bool:
    """Install gate: the process env."""
    return part in _parse(os.environ.get("DSV41_GREEDY_NOGATHER", ""))


def parts() -> frozenset:
    """Parts enabled at capture / replay (the process env)."""
    return _parse(os.environ.get("DSV41_GREEDY_NOGATHER", ""))


class FlavourKey:
    """The greedy flavour's graph key: never equal to an engine ShapeKey; `size` (receipts, dumps) and every other
    attribute come from the base key."""
    __slots__ = ("base",)

    def __init__(self, base):
        object.__setattr__(self, "base", base)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "base"), name)

    def __hash__(self):
        return hash(("dsv41-greedy-nogather", self.base))

    def __eq__(self, other):
        return isinstance(other, FlavourKey) and other.base == self.base

    def __repr__(self):
        return f"FlavourKey({self.base!r})"


def _is_draft(backend) -> bool:
    runner = getattr(backend, "_cuda_graph_runner", None)
    mr = getattr(runner, "model_runner", None)
    return bool(getattr(mr, "is_draft_worker", False))


def _log_once(key, msg, *args):
    if key not in _state["logged"]:
        _state["logged"].add(key)
        logger.warning("[greedy_nogather] " + msg, *args)


def install_backend(module):
    """FullCudaGraphBackend: the draft runner captures each shape once more as FlavourKey(shape) with the greedy
    body, and replays that graph on all-greedy steps while the draft part is on."""
    cls = module.FullCudaGraphBackend
    if getattr(cls, "_dsv41_greedy_nogather", False):
        return
    cls._dsv41_greedy_nogather = True
    orig_capture, orig_replay = cls.capture_one, cls.replay

    def capture_one(self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None):
        orig_capture(self, shape_key, forward_fn, capture_inputs, post_warmup_hook)
        if isinstance(shape_key, FlavourKey):
            return
        draft = _is_draft(self)
        flag = "flavour" if draft else "tflavour"
        if not (_state["eligible"] if draft else (_state["target_ok"] and installed("target"))):
            return
        if post_warmup_hook is not None:
            # the stock capture left the attention backend holding metadata built inside that graph; the warm-up
            # reset restores the pre-capture metadata
            post_warmup_hook()
        _state[flag] = True
        try:
            orig_capture(self, FlavourKey(shape_key), forward_fn, capture_inputs, post_warmup_hook)
        finally:
            _state[flag] = False
            _state["ids"] = None
        keys = self.__dict__.setdefault("_dsv41_greedy_keys", set())
        keys.add(shape_key)
        _state["captured" if draft else "tcaptured"] += 1
        _log_once(("capture", draft, getattr(shape_key, "size", None)),
                  "%s graph size=%s: greedy flavour captured (parts at capture: %s)", "draft" if draft else "target",
                  getattr(shape_key, "size", "?"), ",".join(sorted(parts())) or "none")

    def replay(self, shape_key, static_forward_batch, **kwargs):
        keys = self.__dict__.get("_dsv41_greedy_keys")
        if keys and shape_key in keys and not isinstance(shape_key, FlavourKey):
            draft = _is_draft(self)
            n = _state["replays"] if draft else _state["treplays"]
            if LOG_EVERY and (n[0] + n[1] + 1) % LOG_EVERY == 0:
                _log_counts()
            if _state["greedy" if draft else "tgreedy"] and ("draft" if draft else "target") in parts():
                n[1] += 1
                return orig_replay(self, FlavourKey(shape_key), static_forward_batch, **kwargs)
            n[0] += 1
        return orig_replay(self, shape_key, static_forward_batch, **kwargs)

    def cleanup(self, _orig=cls.cleanup):
        self.__dict__.pop("_dsv41_greedy_keys", None)
        return _orig(self)

    # every wrapper keeps __wrapped__: inspect.unwrap() reaches the engine's function, so an adapter that hash-pins it
    # through unwrap sees the stock source (tier-6 FIX-20261110)
    capture_one.__wrapped__, replay.__wrapped__, cleanup.__wrapped__ = orig_capture, orig_replay, cls.cleanup
    cls.capture_one, cls.replay, cls.cleanup = capture_one, replay, cleanup
    logger.warning("[greedy_nogather] FullCudaGraphBackend: draft / target shapes get a greedy flavour "
                   "(DSV41_GREEDY_NOGATHER)")


def _log_counts():
    """Receipt line (every DSV41_GREEDY_NOGATHER_LOG_EVERY replays of a flavoured shape, per part)."""
    try:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:  # noqa: BLE001
        rank = 0
    logger.warning("[greedy_nogather] rank %d replays: draft stock %d greedy %d (flavours %d, fallbacks %d); "
                   "target stock %d greedy %d (flavours %d, fallbacks %d)", rank, *stats(), *tstats())


def install_sampler(module):
    """DsparkDraftSampler.__call__: inside a greedy-flavour capture (draft part on), the proposal takes the sharded
    greedy body; everything else is the stock call."""
    if not installed("draft"):
        return
    cls = module.DsparkDraftSampler
    if getattr(cls, "_dsv41_greedy_nogather", False):
        return
    cls._dsv41_greedy_nogather = True
    orig_call, orig_init = cls.__call__, cls.__init__
    select = module.select_draft_hidden_without_anchor

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        ok = bool(self.folded_sampling and getattr(self.markov_head, "supports_sharded_greedy", False))
        _state["eligible"] = ok
        if not ok:
            _log_once("ineligible", "not armed: folded_sampling=%s supports_sharded_greedy=%s (stock draft graphs only)",
                      self.folded_sampling, getattr(self.markov_head, "supports_sharded_greedy", False))
        else:
            _log_once("armed", "armed: folded draft graphs get a sharded-greedy flavour for all-greedy steps")

    def __call__(self, hidden_states, input_ids):
        if not (_state["flavour"] and "draft" in parts()):
            return orig_call(self, hidden_states, input_ids)
        bs = hidden_states.shape[0] // self.query_token_num
        if self.sample_from_anchor:
            model_hidden = hidden_states
            sample_hidden = hidden_states.view(bs, self.gamma, -1)
        else:
            model_hidden, sample_hidden = select(hidden_states, bs=bs, gamma=self.gamma)
        base_logits, confidence_tap = self.model.compute_base_logits(model_hidden)
        base_logits = base_logits.view(bs, self.gamma, -1)
        anchor = input_ids.view(bs, self.query_token_num)[:, 0]
        draft_tokens = self.markov_head.sample_block_greedy_fused(base_logits, first_prev_tokens=anchor)
        if draft_tokens is None:
            # not eligible after all (the head refused): the stock body, so the flavour is the stock graph
            _state["fallback"] += 1
            _log_once("fallback", "sample_block_greedy_fused returned None: the greedy flavour runs the stock body")
            return orig_call(self, hidden_states, input_ids)
        ctr = getattr(self.exp_noise, "ctr", None) if self.exp_noise is not None else None
        if ctr is not None and getattr(ctr, "add_", None) is not None:
            ctr.add_(1)   # DSV41_SPEC_SYNC_FREE: the stock graph advances its noise counter once per replay
        self.out[: draft_tokens.numel()].copy_(draft_tokens.reshape(-1))
        if self.confidence_out is not None:
            confidence = self.confidence_fn(
                draft_hidden=sample_hidden,
                anchor_tokens=anchor,
                draft_tokens=draft_tokens,
                confidence_tap=confidence_tap,
            )
            self.confidence_out[:bs].copy_(confidence)

    __init__.__wrapped__, __call__.__wrapped__ = orig_init, orig_call
    cls.__init__, cls.__call__ = __init__, __call__


def _engine_backend():
    import importlib
    return importlib.import_module("sglang.srt.model_executor.runner_backend.full_cuda_graph_backend")


def install_proposer(module):
    """DraftBlockProposer.propose: marks the step all-greedy (or not) for the replay above; imports and wraps the
    graph backend."""
    if not (installed("draft") or installed("target")):
        return
    install_backend(_engine_backend())      # both parts capture their flavours through it
    cls = module.DraftBlockProposer
    if not installed("draft") or getattr(cls, "_dsv41_greedy_nogather", False):
        return
    cls._dsv41_greedy_nogather = True
    orig = cls.propose

    def propose(self, *args, **kwargs):
        si = kwargs.get("sampling_info")
        prev = _state["greedy"]
        _state["greedy"] = si is None or bool(getattr(si, "is_all_greedy", False))
        try:
            return orig(self, *args, **kwargs)
        finally:
            _state["greedy"] = prev

    propose.__wrapped__ = orig
    cls.propose = propose


def stats():
    """(stock replays, greedy-flavour replays, flavours captured, fallbacks) of this process (draft part)."""
    return (_state["replays"][0], _state["replays"][1], _state["captured"], _state["fallback"])


def tstats():
    """The same for the target part."""
    return (_state["treplays"][0], _state["treplays"][1], _state["tcaptured"], _state["tfallback"])


# -- target part ------------------------------------------------------------------------------------------------

_KERNELS = {}


def _kernels():
    """The per-rank (max, lowest argmax) partial over 4096-wide blocks of the rank's logits: upstream's
    _sharded_greedy_partial_kernel without the Markov bias operand (same NaN / tie rules), and upstream's own
    finish kernel over the stacked [world * rows, parts * 2] gather."""
    if not _KERNELS:
        import triton
        import triton.language as tl
        from sglang.kernels.ops.speculative.dspark.sharded_greedy import _sharded_greedy_finish_kernel

        @triton.jit
        def _pair_partial_kernel(X, P, XS: tl.constexpr, WIDTH, OFFSET, PARTS: tl.constexpr, BLOCK: tl.constexpr):
            row, part = tl.program_id(0), tl.program_id(1)
            i = part * BLOCK + tl.arange(0, BLOCK)
            valid = i < WIDTH
            value = tl.load(X + row * XS + i, valid, 0).to(tl.float32)
            nan = valid & (value != value)
            has_nan = tl.sum(nan.to(tl.int32), 0) > 0
            maximum = tl.max(tl.where(valid & ~nan, value, -float("inf")), 0)
            wins = tl.where(has_nan, nan, valid & (value == maximum))
            idx = tl.min(tl.where(wins, i + OFFSET, 2147483647), 0)
            maximum = tl.where(has_nan, float("nan"), maximum)
            tl.store(P + (row * PARTS + part) * 2, maximum)
            tl.store(P + (row * PARTS + part) * 2 + 1, idx.to(tl.float32, bitcast=True))

        _KERNELS.update(partial=_pair_partial_kernel, finish=_sharded_greedy_finish_kernel, triton=triton)
    return _KERNELS


def pair_argmax(local, *, rank, world, vocab_size, gather):
    """argmax over the vocab of the rank-ordered gather of `local` ([rows, local_width] per rank, side by side),
    cut at vocab_size like LogitsProcessor._copy_logits_to_buffer, without moving the logits: one
    (value, index-bits) pair per 4096 columns per row crosses the wire. Lowest index on ties, as row_argmax /
    torch.argmax resolve them; identical to them on every row of finite logits."""
    partial = pair_partial(local, rank=rank, vocab_size=vocab_size)
    rows, parts_ = partial.shape[0], partial.shape[1]
    gathered = gather(partial.view(rows, parts_ * 2)) if world > 1 else partial
    return pair_finish(gathered, rows=rows, parts=parts_, world=world)


_BLOCK = 4096


def pair_partial(local, *, rank, vocab_size):
    """[rows, local_width] -> [rows, parts, 2] fp32: per 4096 columns (max, lowest global argmax as int32 bits)."""
    import torch
    k = _kernels()
    rows, width = local.shape
    assert local.stride(1) == 1
    offset = rank * width
    real = max(0, min(width, vocab_size - offset))
    parts_ = k["triton"].cdiv(width, _BLOCK)
    partial = torch.empty((rows, parts_, 2), device=local.device, dtype=torch.float32)
    k["partial"][(rows, parts_)](local, partial, local.stride(0), real, offset, parts_, _BLOCK, num_warps=4)
    return partial


def pair_finish(gathered, *, rows, parts, world):
    """The rank-stacked pairs [world * rows, parts * 2] -> [rows] int64 (upstream's finish kernel)."""
    import torch
    k = _kernels()
    out = torch.empty(rows, device=gathered.device, dtype=torch.int64)
    k["finish"][(rows,)](gathered, out, rows, parts, world, k["triton"].next_power_of_2(parts * world), num_warps=4)
    return out


def install_logits(lp_module):
    """LogitsProcessor._get_logits: inside a target greedy-flavour capture (target part on), the
    verify head computes the rank's logits, reduces them to (max, argmax) pairs, gathers the pairs, and hands the
    argmax ids to the epilogue's accept; the logits buffer is returned unwritten (its consumers on an all-greedy
    folded step are the accept alone). Install BEFORE cert_head.install_logits (its _get_logits wrapper must stay
    outside, so the rank's logits are the certified ones where the head is qualified)."""
    if not installed("target"):
        return
    cls = lp_module.LogitsProcessor
    if getattr(cls, "_dsv41_greedy_nogather", False):
        return
    cls._dsv41_greedy_nogather = True
    orig = cls._get_logits
    _state["target_ok"] = True

    def _get_logits(self, hidden_states, lm_head, logits_metadata, embedding_bias=None, use_logits_buffer=True):
        fm = getattr(logits_metadata, "forward_mode", None)
        if not (_state["tflavour"] and "target" in parts() and fm is not None and fm.is_target_verify()):
            return orig(self, hidden_states, lm_head, logits_metadata, embedding_bias, use_logits_buffer)
        buf = getattr(logits_metadata, "next_token_logits_buffer", None)
        rows = int(hidden_states.shape[0])
        why = None
        if not use_logits_buffer or buf is None or tuple(buf.shape) != (rows, self.vocab_size):
            why = "no logits buffer of the verify shape"
        elif (not self.do_tensor_parallel_all_gather or getattr(self, "use_attn_tp_group", False)
              or getattr(self, "do_tensor_parallel_all_gather_dp_attn", False)):
            why = "not a plain TP vocab gather"
        elif self.logit_scale is not None or self.final_logit_softcapping:
            why = "logit scale / softcapping"
        if why is not None:
            _state["tfallback"] += 1
            _log_once(("tfallback", why), "target flavour runs the stock head: %s", why)
            return orig(self, hidden_states, lm_head, logits_metadata, embedding_bias, use_logits_buffer)
        from sglang.srt.distributed import get_tp_group
        group = get_tp_group()
        local = self._compute_lm_head(hidden_states, lm_head, embedding_bias)
        _state["ids"] = pair_argmax(local, rank=group.rank_in_group, world=group.world_size,
                                    vocab_size=self.vocab_size, gather=lambda t: group.all_gather(t, dim=0))
        return buf

    _get_logits.__wrapped__ = orig
    cls._get_logits = _get_logits
    logger.warning("[greedy_nogather] LogitsProcessor: target verify greedy flavour hands pair-gathered argmax ids "
                   "to the accept")


def install_accept(accept_module):
    """dspark_accept._row_argmax: the accept captured into a target greedy flavour takes the pair-argmax ids of the
    same forward instead of reading the (unwritten) logits buffer."""
    if not installed("target"):
        return
    orig = accept_module._row_argmax
    if getattr(orig, "_dsv41_greedy_nogather", False):
        return

    def _row_argmax(logits):
        ids = _state["ids"]
        if _state["tflavour"] and ids is not None and int(logits.shape[0]) == int(ids.shape[0]):
            _state["ids"] = None
            return ids
        return orig(logits)

    _row_argmax.__wrapped__ = orig
    _row_argmax._dsv41_greedy_nogather = True
    accept_module._row_argmax = _row_argmax


def _capture_safe(executor):
    """adapter/draft_capture.py reads the verify logits in commit_hidden, which a folded step skips only when the
    epilogue folds the commit; with the tap armed and no folded commit the step keeps the stock graph."""
    import sys
    dc = sys.modules.get("draft_capture")
    st = getattr(dc, "_S", None) if dc is not None else None
    if not (isinstance(st, dict) and st.get("armed")):
        return True
    ep = getattr(executor, "verify_epilogue", None)
    try:
        return bool(ep is not None and ep.folds_commit)
    except Exception:  # noqa: BLE001
        return False


def install_verify(verify_module):
    """The target flavour is replayed exactly on the steps whose accept runs in the verify graph: the static
    epilogue armed (DsparkVerifyEpilogue.begin_static_step(bs, fold_eligible), called right before
    run_non_compact; fold_eligible = all greedy, no logit adjustments, no grammar, no simulated acceptance), and no
    return_logprob (compute_spec_logprobs reads the logits). Compact / ragged verify keeps the stock graph."""
    if not installed("target"):
        return
    ep_cls, ex = verify_module.DsparkVerifyEpilogue, verify_module.TargetVerifyExecutor
    if getattr(ex, "_dsv41_greedy_nogather", False):
        return
    ex._dsv41_greedy_nogather = True
    orig_begin, orig_nc = ep_cls.begin_static_step, ex.run_non_compact

    def begin_static_step(self, bs, armed):
        _state["static_armed"] = bool(armed)
        return orig_begin(self, bs, armed)

    def run_non_compact(self, *, batch, **kw):
        armed = _state["static_armed"]
        _state["static_armed"] = None
        _state["tgreedy"] = bool(armed) and not getattr(batch, "return_logprob", False) and _capture_safe(self)
        try:
            return orig_nc(self, batch=batch, **kw)
        finally:
            _state["tgreedy"] = False

    begin_static_step.__wrapped__, run_non_compact.__wrapped__ = orig_begin, orig_nc
    ep_cls.begin_static_step = begin_static_step
    ex.run_non_compact = run_non_compact
