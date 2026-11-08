"""Block verification for DSpark sampled decoding (Sun et al., "Block Verification Accelerates
Speculative Decoding", ICLR 2025). Default OFF. Lossless: the output distribution is the target's.

DSV41_BLOCK_VERIFY=1 replaces the chain (token-by-token) rejection sampler used for sampled rows.
Token verification accepts draft i with min(1, p_i/q_i) independently; block verification decides
the accepted length jointly over the block and is provably optimal for a single draft chain:
    w_0 = 1,  w_i = min(w_{i-1} * p_i(X_i) / q_i(X_i), 1)
    h_i = S_i / (S_i + 1 - w_i),  S_i = sum_x max(w_i * p_{i+1}(x) - q_{i+1}(x), 0)   (i < gamma)
    h_gamma = w_gamma
    tau = max{ i : eta_i <= h_i }  (0 if none),  eta_i ~ U(0,1)
    correction token ~ normalise(max(w_tau * p_{tau+1} - q_{tau+1}, 0))   (tau < gamma)
                     ~ p_{gamma+1}                                          (tau = gamma)
p_i = target distribution after temperature / top-k / top-p (the same one the chain sampler uses),
q_i = the distribution the draft token was actually sampled from.
Measured offline on engine-exact p and q (capture schema v4, coding-with-thinking traffic):
2.927 -> 2.999 tokens per step (+2.5 %).

Random draws come from torch's device generator; this function consumes the same shapes in the same
order as before. Equal draw consumption does not make rank-local values equal: the TP ranks agree on
the published decision because SGLang synchronizes the eager sampled accept (DSPARK_ACCEPT_SAMPLE;
spec_sync_free only skips graph-accept sites). Greedy rows: an all-greedy batch never calls this; in a
mixed batch it runs on every row and SelectMixedAccept keeps the greedy rows' own AcceptGreedy result.
"""
import os

import torch

ENABLED = os.environ.get("DSV41_BLOCK_VERIFY", "0").strip() not in ("0", "", "off", "false")


def block_accept_probs(X, target_probs, draft_probs, eta=None, gumbel=None, live=None):
    """X [bs, g] drafted tokens, target_probs [bs, g+1, V], draft_probs [bs, g, V].
    live [bs] (optional): drafts actually verified per request (<= g); positions beyond it are
    treated as never drafted (q = 0, p(X) = 0), which is the block rule for that shorter block.
    Returns (tau [bs] int64, correction token [bs] int64)."""
    bs, g = X.shape
    dev = X.device
    p_tok = target_probs[:, :g].gather(-1, X.unsqueeze(-1)).squeeze(-1).float()
    q_tok = draft_probs.gather(-1, X.unsqueeze(-1)).squeeze(-1).float()
    ratio = p_tok / q_tok.clamp_min(1e-30)
    if live is not None:
        alive = torch.arange(g, device=dev).view(1, -1) < live.view(-1, 1)
        ratio = torch.where(alive, ratio, torch.zeros_like(ratio))
        draft_probs = draft_probs * alive.unsqueeze(-1).to(draft_probs.dtype)
    w = torch.cumprod(ratio, dim=1)
    # w_i = min(w_{i-1} * r_i, 1) is not a plain cumprod once it saturates; run the recursion
    ws = []
    cur = torch.ones(bs, device=dev)
    for i in range(g):
        cur = torch.minimum(cur * ratio[:, i], torch.ones_like(cur))
        ws.append(cur)
    w = torch.stack(ws, 1)                                                     # [bs, g] = w_1..w_g
    h = torch.empty(bs, g, device=dev)
    if g > 1:
        s = (w[:, :g - 1, None] * target_probs[:, 1:g].float() - draft_probs[:, 1:g].float()).clamp_min(0).sum(-1)
        # (1 - w) first: exact in FP32 for w in [0.5, 1] (Sterbenz), so S's digits below an ULP of 1
        # survive the sum. (s + 1) - w rounded S against 1: the effective (clipped) threshold moved by up
        # to ~0.5 on the rows tested; the raw value was unbounded (w = 1, S < 2^-24 rounds the denominator
        # to 0 and the 1e-30 floor). Positive S below 1e-30 still meets that inherited floor.
        h[:, :g - 1] = s / (s + (1 - w[:, :g - 1])).clamp_min(1e-30)
    h[:, g - 1] = w[:, g - 1]
    if eta is None:
        eta = torch.rand(bs, g, device=dev)
    ok = eta <= h                                                               # [bs, g]
    if live is not None:
        ok = ok & alive
    idx = torch.arange(1, g + 1, device=dev).view(1, -1)
    tau = torch.where(ok, idx, torch.zeros_like(idx)).max(dim=1).values        # [bs]
    # correction distribution at position tau (0-based row tau of target / draft)
    w_all = torch.cat([torch.ones(bs, 1, device=dev), w], dim=1)              # w_0..w_g
    wt = w_all.gather(1, tau.view(-1, 1))                                      # [bs, 1]
    rows = tau.view(-1, 1, 1).expand(-1, 1, target_probs.shape[-1])
    p_row = target_probs.gather(1, rows).squeeze(1).float()                    # [bs, V]
    q_pad = torch.cat([draft_probs.float(), torch.zeros_like(draft_probs[:, :1]).float()], dim=1)
    q_row = q_pad.gather(1, rows).squeeze(1)                                   # 0 for tau == g
    res = (wt * p_row - q_row).clamp_min(0)
    mass = res.sum(-1, keepdim=True)
    res = torch.where(mass > 0, res / mass.clamp_min(1e-30), p_row)            # degenerate: p == q
    if gumbel is None:
        gumbel = torch.empty_like(res).exponential_()
    corr = (res / gumbel).argmax(dim=-1)
    return tau, corr


try:  # DSV41_BLOCK_VERIFY_FOLD (adapter/block_verify_fold.py): the same rule in 4 Triton launches
    import block_verify_fold as _fold_mod
    _fold = _fold_mod if _fold_mod.ENABLED else None
except ImportError:  # pragma: no cover
    _fold_mod = _fold = None
try:  # DSV41_AB_VARIANTS in-boot A/B (test only): the fold is eager, so it switches per call
    import ab_variant as _ab
except ImportError:  # pragma: no cover
    _ab = None
_FOLD_CPU = os.environ.get("DSV41_BLOCK_VERIFY_FOLD_CPU", "0") == "1"   # tests only (Triton interpreter)


def _fold_mode():
    """None (torch path), "on" or "check" for this call."""
    if _fold_mod is not None and _ab is not None and getattr(_ab, "ACTIVE", False):
        v = _ab._fold_value(_ab.env("DSV41_BLOCK_VERIFY_FOLD", "0"))     # the value configure() validated
        return None if v == "off" else v
    if _fold is None:
        return None
    return "check" if _fold.CHECK else "on"


def _folded(candidates, target_probs, draft_probs, bs, gamma, stride, cutoff_verify_lens, check=False):
    """block_accept_probs through the fold kernels. The random draws are the ones block_accept_probs makes, in
    its order and shapes (eta [bs, gamma] uniform, then [bs, V] fp32 exponential), so the stream is unchanged."""
    tp = target_probs.view(bs, stride, -1)
    dp = draft_probs.view(bs, gamma, -1)
    dev = candidates.device
    eta = torch.rand(bs, gamma, device=dev)
    gumbel = torch.empty((bs, tp.shape[-1]), dtype=torch.float32, device=dev).exponential_()
    X = candidates.view(bs, stride)[:, 1:gamma + 1]
    cut = None if cutoff_verify_lens is None else cutoff_verify_lens[:bs]
    got = _fold_mod.fold(X, tp, dp, eta, gumbel, live=cut, live_sub=1)
    if not check:
        return got
    live = None if cut is None else cut.to(torch.int64) - 1
    ref = block_accept_probs(X.long(), tp, dp, eta=eta, gumbel=gumbel, live=live)
    _fold_mod.check_counts(got, ref)
    return ref


def install(accept_module):
    """sglang.kernels.ops.speculative.dspark.dspark_accept: swap AcceptSampling for sampled rows."""
    if not ENABLED:
        return
    cls = getattr(accept_module, "AcceptSampling", None)
    core = getattr(accept_module, "_accept_sampling_core", None)
    if cls is None or core is None:
        raise RuntimeError("DSV41_BLOCK_VERIFY: AcceptSampling/_accept_sampling_core gone; engine drifted")
    if getattr(cls, "_dsv41_block_verify", False):
        return
    cls._dsv41_block_verify = True
    softmax_temp = accept_module.SoftmaxTemp
    import sglang.srt.speculative.dflash_utils as du

    def execute(*, candidates, target_logits, draft_probs, sampling_info, draft_input, gamma,
                verify_num_draft_tokens, cutoff_verify_lens=None, seq_lens=None):
        # seq_lens: the request prefix lengths that the patch
        # sglang-dspark-seeded-sampling.patch adds to AcceptSampling.execute. The stock fallback
        # below uses them for seeded verify coins; the block verification path below still
        # draws its eta coins from the unseeded torch generator.
        if cutoff_verify_lens is not None and os.environ.get("DSV41_VERIFY_CAP", "").strip() in ("", "0", "off"):
            return cls._dsv41_orig_execute(candidates=candidates, target_logits=target_logits,
                                           draft_probs=draft_probs, sampling_info=sampling_info,
                                           draft_input=draft_input, gamma=gamma,
                                           verify_num_draft_tokens=verify_num_draft_tokens,
                                           cutoff_verify_lens=cutoff_verify_lens, seq_lens=seq_lens)
        bs = candidates.shape[0]
        if not sampling_info.need_top_k_sampling and not sampling_info.need_top_p_sampling:
            target_probs = softmax_temp.execute(logits=target_logits, temperatures=sampling_info.temperatures,
                                                rows_per_request=verify_num_draft_tokens
                                                ).view(bs, verify_num_draft_tokens, -1)
        else:
            target_probs = du.build_dflash_verify_target_probs(
                next_token_logits=target_logits, sampling_info=sampling_info,
                draft_token_num=verify_num_draft_tokens, bs=bs, max_top_k=draft_input.max_top_k,
                uniform_top_k_value=draft_input.uniform_top_k_value)
        mode = _fold_mode()
        if mode is not None and (candidates.is_cuda or _FOLD_CPU):
            tau, corr = _folded(candidates, target_probs, draft_probs, bs, gamma, verify_num_draft_tokens,
                                cutoff_verify_lens, check=mode == "check")
            _fold_mod.receipt("check" if mode == "check" else "fold", bs)
            return tau.to(torch.int32), corr, torch.zeros(bs, dtype=torch.int32, device=candidates.device)
        if _fold_mod is not None:
            _fold_mod.receipt("torch", bs)
        X = candidates.view(bs, verify_num_draft_tokens)[:, 1:gamma + 1].long()
        live = None if cutoff_verify_lens is None else (cutoff_verify_lens[:bs].to(torch.int64) - 1)
        tau, corr = block_accept_probs(X, target_probs.view(bs, verify_num_draft_tokens, -1),
                                       draft_probs.view(bs, gamma, -1), live=live)
        return tau.to(torch.int32), corr.to(torch.int64), torch.zeros(bs, dtype=torch.int32, device=candidates.device)

    cls._dsv41_orig_execute = cls.execute
    cls.execute = staticmethod(execute)
    fold_note = "" if _fold is None else f", fold {'check' if _fold.CHECK else 'on'} (DSV41_BLOCK_VERIFY_FOLD)"
    print(f"[block_verify] armed: block verification for sampled DSpark rows{fold_note}", flush=True)
