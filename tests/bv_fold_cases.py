"""Shared cases for the block_verify_fold tests (Triton interpreter on CPU, and the GPU test).

cases(device) yields (name, args) with args = dict(X, tp, dp, eta, gum, live, live_sub) on that device.
prod_debug() is block_verify.block_accept_probs with its intermediates exposed (S, h, mass, keys). MIRRORED_SHA pins
the production source it mirrors (a change there fails the tests until this copy is re-checked), and the tests also
require its outputs to equal block_accept_probs' on every case.
"""
import hashlib
import inspect

import torch

import block_verify as bv

G = 5
MIRRORED_SHA = "21e94b8c460e9ede8aa765f44296dba6339c71f01dd401cd00f5ce051d073d1b"   # sha256 of its source


def prod_source_sha():
    return hashlib.sha256(inspect.getsource(bv.block_accept_probs).encode()).hexdigest()


def prod_debug(X, target_probs, draft_probs, eta, gumbel, live=None):
    """block_accept_probs, line for line, returning the intermediates too."""
    bs, g = X.shape
    dev = X.device
    p_tok = target_probs[:, :g].gather(-1, X.unsqueeze(-1)).squeeze(-1).float()
    q_tok = draft_probs.gather(-1, X.unsqueeze(-1)).squeeze(-1).float()
    ratio = p_tok / q_tok.clamp_min(1e-30)
    if live is not None:
        alive = torch.arange(g, device=dev).view(1, -1) < live.view(-1, 1)
        ratio = torch.where(alive, ratio, torch.zeros_like(ratio))
        draft_probs = draft_probs * alive.unsqueeze(-1).to(draft_probs.dtype)
    ws = []
    cur = torch.ones(bs, device=dev)
    for i in range(g):
        cur = torch.minimum(cur * ratio[:, i], torch.ones_like(cur))
        ws.append(cur)
    w = torch.stack(ws, 1)
    h = torch.empty(bs, g, device=dev)
    s = None
    if g > 1:
        s = (w[:, :g - 1, None] * target_probs[:, 1:g].float() - draft_probs[:, 1:g].float()).clamp_min(0).sum(-1)
        h[:, :g - 1] = s / (s + (1 - w[:, :g - 1])).clamp_min(1e-30)
    h[:, g - 1] = w[:, g - 1]
    ok = eta <= h
    if live is not None:
        ok = ok & alive
    idx = torch.arange(1, g + 1, device=dev).view(1, -1)
    tau = torch.where(ok, idx, torch.zeros_like(idx)).max(dim=1).values
    w_all = torch.cat([torch.ones(bs, 1, device=dev), w], dim=1)
    wt = w_all.gather(1, tau.view(-1, 1))
    rows = tau.view(-1, 1, 1).expand(-1, 1, target_probs.shape[-1])
    p_row = target_probs.gather(1, rows).squeeze(1).float()
    q_pad = torch.cat([draft_probs.float(), torch.zeros_like(draft_probs[:, :1]).float()], dim=1)
    q_row = q_pad.gather(1, rows).squeeze(1)
    res = (wt * p_row - q_row).clamp_min(0)
    mass = res.sum(-1, keepdim=True)
    res2 = torch.where(mass > 0, res / mass.clamp_min(1e-30), p_row)
    keys = res2 / gumbel
    corr = keys.argmax(dim=-1)
    return tau, corr, {"s": s, "h": h, "mass": mass.squeeze(-1), "res": res, "keys": keys}


def probs(bs, rows, V, sharp, gen, device, topp=None):
    lg = torch.randn(bs, rows, V, generator=gen, device=gen.device) * sharp
    p = torch.softmax(lg, -1)
    if topp is not None:
        sp, si = p.sort(-1, descending=True)
        keep = (sp.cumsum(-1) - sp) < topp
        sp = torch.where(keep, sp, torch.zeros_like(sp))
        p = torch.zeros_like(p).scatter(-1, si, sp)
        p = p / p.sum(-1, keepdim=True)
    return p.contiguous().to(device)


def sample_X(dp, gen):
    src = dp if dp.device == gen.device else dp.to(gen.device)
    return torch.stack([torch.multinomial(src[:, i], 1, generator=gen).squeeze(-1) for i in range(src.shape[1])],
                       1).to(dp.device)


def random_case(bs, V, seed, device, sharp_p=2.0, sharp_q=2.0, topp=None, live_mode="none", tie=False, q_mix=0.6):
    # CPU cases draw on the CPU generator (the interpreter test's numbers); GPU cases draw on the device
    gen = torch.Generator(device=device).manual_seed(seed)
    tp = probs(bs, G + 1, V, sharp_p, gen, device, topp=topp)
    if tie:
        dp = tp[:, :G].clone().contiguous()
    else:
        noise = probs(bs, G, V, sharp_q, gen, device)
        dp = (q_mix * tp[:, :G] + (1 - q_mix) * noise)
        dp = (dp / dp.sum(-1, keepdim=True)).contiguous()
    X = sample_X(dp, gen)
    eta = torch.rand(bs, G, generator=gen, device=device)
    gum = torch.empty(bs, V, device=device).exponential_(generator=gen)
    live, live_sub = None, 0
    if live_mode == "cut":
        live, live_sub = torch.randint(2, G + 2, (bs,), generator=gen, device=device), 1
    elif live_mode == "drafts":
        live = torch.randint(1, G + 1, (bs,), generator=gen, device=device)
    return dict(X=X, tp=tp, dp=dp, eta=eta, gum=gum, live=live, live_sub=live_sub)


def cases(device, big=True, sweep=40):
    out = []
    for i, (bs, V) in enumerate([(1, 1000), (3, 4099), (2, 2048), (4, 777), (2, 8193)]):
        for lm in ("none", "cut", "drafts"):
            out.append((f"random bs={bs} V={V} live={lm}", random_case(bs, V, 100 + 10 * i + len(lm), device,
                                                                       live_mode=lm)))
    out.append(("top-p 0.9 target, sharp", random_case(3, 3000, 201, device, sharp_p=4.0, topp=0.9, live_mode="cut")))
    out.append(("top-p 0.5 target (p(X) = 0 rows)", random_case(3, 3000, 202, device, sharp_p=3.0, topp=0.5,
                                                                q_mix=0.1)))
    out.append(("p == q exactly (zero residual, degenerate mass)", random_case(2, 1500, 203, device, tie=True,
                                                                               live_mode="cut")))
    out.append(("flat target, sharp draft", random_case(2, 2500, 204, device, sharp_p=0.2, sharp_q=6.0, q_mix=0.0)))
    if big:
        out.append(("V = 129280, bs=1, cutoff, top-p 0.95", random_case(1, 129280, 301, device, sharp_p=3.0,
                                                                        topp=0.95, live_mode="cut")))
        out.append(("V = 129280, bs=2", random_case(2, 129280, 302, device, sharp_p=3.0)))
    # adversarial: q(X) = 0, one-hot target excluding X
    gen = torch.Generator().manual_seed(400)
    V = 1024
    tp = probs(2, G + 1, V, 2.0, gen, "cpu")
    dp = probs(2, G, V, 2.0, gen, "cpu")
    X = sample_X(dp, gen)
    dp[0, 1, X[0, 1]] = 0.0
    tp[1, 2] = 0.0
    tp[1, 2, (X[1, 2] + 1) % V] = 1.0
    eta = torch.rand(2, G, generator=gen)
    gum = torch.empty(2, V).exponential_(generator=gen)
    base = dict(X=X.to(device), tp=tp.contiguous().to(device), dp=dp.contiguous().to(device), live=None, live_sub=0)
    out.append(("q(X)=0 and one-hot target excluding X", dict(base, eta=eta.to(device), gum=gum.to(device))))
    _, _, d = prod_debug(base["X"].cpu(), tp, dp, eta, gum)
    h = d["h"].clamp(0, 1)
    out.append(("eta exactly equal to h", dict(base, eta=h.clone().to(device), gum=gum.to(device))))
    out.append(("eta one ulp above h", dict(base, eta=torch.nextafter(h, torch.full_like(h, 2.0)).to(device),
                                            gum=gum.to(device))))
    out.append(("eta one ulp below h", dict(base, eta=torch.nextafter(h, torch.full_like(h, -1.0)).clamp_min(0)
                                            .to(device), gum=gum.to(device))))
    out.append(("eta tiny -> full accept", dict(base, eta=torch.full((2, G), 1e-9).to(device), gum=gum.to(device))))
    out.append(("eta 1 -> nothing accepted unless h = 1", dict(base, eta=torch.ones(2, G).to(device),
                                                               gum=gum.to(device))))
    out.append(("live 0 (every draft dead)", dict(base, eta=eta.to(device), gum=gum.to(device),
                                                  live=torch.zeros(2, dtype=torch.int64).to(device))))
    # ties of the correction keys, NaN and inf keys
    tp_t = torch.zeros(1, G + 1, 512)
    tp_t[..., 10] = 0.25
    tp_t[..., 20] = 0.25
    tp_t[..., 30] = 0.5
    dp_t = torch.zeros(1, G, 512)
    dp_t[..., 30] = 1.0
    X_t = torch.full((1, G), 30)
    tb = dict(X=X_t.to(device), tp=tp_t.contiguous().to(device), dp=dp_t.contiguous().to(device), live=None,
              live_sub=0, eta=torch.ones(1, G).to(device))
    out.append(("equal correction keys -> first index", dict(tb, gum=torch.ones(1, 512).to(device))))
    g_n = torch.ones(1, 512)
    g_n[0, 5] = 0.0
    out.append(("NaN correction key (0 / 0)", dict(tb, gum=g_n.to(device))))
    g_i = torch.ones(1, 512)
    g_i[0, 20] = 0.0
    out.append(("inf correction key", dict(tb, gum=g_i.to(device))))
    # tiny residual mass (normalisation near the 1e-30 floor)
    tp_m = probs(1, G + 1, 700, 1.0, torch.Generator().manual_seed(9), "cpu")
    dp_m = (tp_m[:, :G] * (1 + 1e-7)).contiguous()
    dp_m = dp_m / dp_m.sum(-1, keepdim=True)
    out.append(("near-zero residual mass", dict(X=sample_X(dp_m, torch.Generator().manual_seed(3)).to(device),
                                                tp=tp_m.contiguous().to(device), dp=dp_m.contiguous().to(device),
                                                live=None, live_sub=0, eta=torch.full((1, G), 0.999).to(device),
                                                gum=torch.empty(1, 700).exponential_(
                                                    generator=torch.Generator().manual_seed(4)).to(device))))
    for s in range(sweep):
        out.append((f"sweep seed {s}", random_case(2, 600 + 37 * s, 1000 + s, device, sharp_p=0.5 + (s % 5),
                                                   topp=(None, 0.95, 0.8)[s % 3],
                                                   live_mode=("none", "cut", "drafts")[s % 3], q_mix=(s % 4) / 4)))
    return out
