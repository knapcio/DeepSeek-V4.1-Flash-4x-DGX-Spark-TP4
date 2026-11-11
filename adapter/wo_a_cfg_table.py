"""DSV41_WO_A_CFG_TABLE=1 (off by default): wo_a's fp8-twin kernels (adapter/wo_a_w8.py) at the row counts where a
measured sweep found a faster launch configuration that is bit-identical to production's; self-checked on the device at
load and again at the first eager call of every key, on every rank, TP-agreed, fail closed.

Measured 2026-11-11 (tier12o-kit window B, bench/woa: one GB10, production stopped; every config built from the image's
own kernel text, its outputs compared byte for byte with production's on two activation sets, eager and on a CUDA-graph
replay, with production's own A/A pair and a perturbed-twin negative control in every cell; 200 interleaved rounds,
warm = 4 back-to-back calls, cold = after a 64 MiB flush; us per call, medians with 95 % CIs):

    M = 6  (verify at c1: the DSpark block of 5 + 1 rows; the DROP route's 2..8-row partial kernel `_partial_w8`)
        production       BM 16, BN 64, BK 128, SPLIT_K 8, grid (16, 2, 8), 4 warps, 3 stages: warm 28.14
        p:64:2:2:b       the same tile, grid, k order and partial layout; 2 warps, 2 stages, the exponent applied once per
                         32 x 32 block (one exp2 per block, broadcast): warm 11.77 (-16.38 [-16.384, -16.376]),
                         cold -2.45 [-2.66, -2.11]
    M = 96 (verify at c16: `wo_a_mid`, production's _MID_CFG row (1, 32, 64, 4), 3 stages)
        mid:1:32:64:2:3  split_k 1, BM 32, BN 64, 2 warps, 3 stages: warm -19.18 [-19.29, -19.02], cold -0.96
    M = 24 (verify at c4): the best bitwise config gained -2.66 warm and +0.29 cold: no entry (production stays)
Production's c1 step runs ~43 wo_a calls at M = 6; production's wo_a is partly L2-prefetched (the WOA 6 MB window), so a
call's real state lies between the warm and the cold number: the engine A/B decides (tier15-kit window W).

Why bitwise. BK 128, SPLIT_K and BM are held at production's for that M, so every output element sums the same products
in the same k order through the same tl.dot shape per k-step; num_warps / num_stages change how the CTA splits its tile
over warps and how deep the loads are pipelined, not the accumulation. The block-scale form multiplies every e4m3 value by
the same exact power of two (e4m3 x 2^(e - 127) is exact in fp32) before the same bf16 rounding. The sweep found every
listed config bitwise on every cell; this module re-proves it on the device before serving anything.

Mechanics. adapter/wo_a_w8.py's `_partial_w8` (2..8 rows) and `wo_a_mid` (9..192 verify rows) ask `partial()` / `mid()`
first; None = production's own launch, unchanged. A call takes the table's launch only when
  1. DSV41_WO_A_CFG_TABLE is on for the variant being captured / replayed (ab_variant.env; plain os.environ otherwise);
  2. its row count M has an entry, qx is off (DSV41_WO_A_INPUT_FP8: the sweep measured production's profile) and, for a
     mid entry, production's _MID_CFG still selects the measured row at that M (a drift keeps production for that M);
  3. the load self-check passed for that M (TP-agreed): no load check = production (fail closed);
  4. the first eager call of its key (kind, M, activation row stride, eviction policy, dtype, device) ran BOTH launches on
     the real inputs, found the output bytes identical (uint8 views: +0.0 / -0.0 differ) and the TP group agreed (MIN).
     Inside a CUDA-graph capture an unchecked key takes production's launch and records nothing (SGLang's eager warm-up
     forward before each capture checks it); a key that failed keeps production for good.
Load self-check (`boot_check`, called by wo_a_w8 once the twins exist, before any capture): per entry, on the first
twinned wo_a layer's real e4m3 bytes and exponents and two seeded activation sets, production's launch and the table's,
output bytes compared; a negative control (the table's launch on that twin with ONE 32 x 32 exponent flipped by one) must
differ from production's bytes, else the compare proves nothing and the entry stays off. One all-reduce (MIN) for all
entries; an exception keeps the entry off and the all-reduce still runs (no rank waits alone).
Receipts (stdout, every rank): the armed line at install, one `load self-check` line per entry, one `first eager call`
line per key. Nothing is written to the environment, nothing cached across processes; the compile cache is Triton's own.
Under DSV41_AB_VARIANTS the gate is a union key (installed when any variant enables it, read per call).
"""
import os

import torch
import triton
import triton.language as tl

from wo_a_input_fp8 import qdq_tile  # noqa: F401  (the kernel's QX branch, compiled out: the table runs qx off only)

try:  # DSV41_AB_VARIANTS in-boot A/B (test only): per-variant value; plain os.environ when off
    from ab_variant import env as _env
except ImportError:
    _env = os.environ.get

ENV = "DSV41_WO_A_CFG_TABLE"
# exact row counts only; every other M takes production's launch. (kind, name, launch parameters)
TABLE = {
    6: ("partial", "p:64:2:2:b", {"BN": 64, "num_warps": 2, "num_stages": 2}),
    96: ("mid", "mid:1:32:64:2:3", {"split_k": 1, "BM": 32, "BN": 64, "num_warps": 2, "num_stages": 3}),
}
# production's config each entry was measured against (woa bench: woa_api.EXPECTED_PROD, the _MID_CFG row at that M)
PROD_NAME = {6: "p:64:4:3", 96: "mid:1:32:64:4:3"}
PROD_MID = {96: (1, 32, 64, 4)}
PARTIAL_ROWS = (2, 8)            # wo_a_w8's partial route
MID_ROWS = (9, 192)              # wo_a_w8's mid route
SEED = 20261111
_BOOT = {}                       # M -> True / False: the load self-check, TP-agreed (absent = not run: production)
_CHECKED = {}                    # key -> True (the table serves the key) / False (production kept)
_STATS = {"checked": 0, "same": 0, "diff": 0, "disagree": 0, "skipped_capture": 0, "no_boot": 0, "drift": 0}
_STATE = {"boot_ran": False}


def _truthy(v):
    return str(v or "0").strip().lower() in ("1", "on", "true", "yes")


def wanted_at_install():
    """Install-time union gate (the A/B harness unions the variants); the per-call value is read through ab_variant."""
    return _truthy(os.environ.get(ENV))


def _on():
    return _truthy(_env(ENV, "0"))


def _capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


# the 2..8-row partial kernel of adapter/wo_a_w8.py with BN a constexpr and the block-scale form: the text the woa bench
# generated from the image's `_wo_a_partial_w8` (woa_api.variant_source(..., blockscale=True)) and measured, renamed;
# tests/test_wo_a_cfg_table.py re-derives it from wo_a_w8's kernel text with the bench's anchored substitutions
@triton.jit
def _wo_a_partial_w8_bs(X, W8, S, P, M: tl.constexpr, SX: tl.constexpr, QX: tl.constexpr = False,
                        EV: tl.constexpr = "", BN: tl.constexpr = 64):
    tile, group, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = tl.arange(0, 16)
    n = tile * BN + tl.arange(0, BN)
    k = split * 512 + tl.arange(0, 128)
    acc = tl.zeros((16, BN), tl.float32)
    for i in range(4):
        offsets = k + i * 128
        x = tl.load(
            X + m[:, None] * SX + group * 4096 + offsets[None, :], m[:, None] < M, 0
        )
        if QX:  # 128 columns = four whole 32-blocks of the group's 4096
            x = qdq_tile(x, 16, 128)
        w8 = tl.load(W8 + (group * 1024 + n[None, :]) * 4096 + offsets[:, None], eviction_policy=EV)
        nb = tile * (BN // 32) + tl.arange(0, BN // 32)
        kb = (split * 512 + i * 128) // 32 + tl.arange(0, 4)
        e = tl.load(S + (group * 32 + nb[None, :]) * 128 + kb[:, None], eviction_policy=EV)
        sc = tl.exp2(e.to(tl.float32) - 127.0)
        w = tl.reshape(tl.reshape(w8.to(tl.float32), (4, 32, BN // 32, 32)) * sc[:, None, :, None], (128, BN)).to(tl.bfloat16)
        acc += tl.dot(x, w)
    tl.store(
        P + ((split * M + m[:, None]) * 2 + group) * 1024 + n[None, :],
        acc,
        m[:, None] < M,
    )


def launch_partial(x, twin, m, ev, cfg):
    """The table's 2..8-row launch: the woa bench's launch_partial with bn given (same partial layout [8, m, 2, 1024])."""
    w8, s = twin
    bn = cfg["BN"]
    partial = torch.empty((8, m, 2, 1024), dtype=torch.float32, device=x.device)
    _wo_a_partial_w8_bs[(1024 // bn, 2, 8)](x, w8, s, partial, m, x.stride(0), QX=False, EV=ev, BN=bn,
                                            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return partial


def launch_mid(kernel, o, twin, m, ev, cfg):
    """wo_a_w8's _wo_a_mid_kernel at the table's meta-parameters, launched and reduced exactly as wo_a_mid does."""
    w8, s = twin
    sk, bm, bn = cfg["split_k"], cfg["BM"], cfg["BN"]
    grid = (1024 // bn, 2, triton.cdiv(m, bm) * sk)
    if sk == 1:
        y = torch.empty((m, 2, 1024), dtype=torch.bfloat16, device=o.device)
        kernel[grid](o, w8, s, y, m, o.stride(0), 1, bm, bn, 128, QX=False, EV=ev, num_warps=cfg["num_warps"],
                     num_stages=cfg["num_stages"])
        return y
    y = torch.empty((sk, m, 2, 1024), dtype=torch.float32, device=o.device)
    kernel[grid](o, w8, s, y, m, o.stride(0), sk, bm, bn, 128, QX=False, EV=ev, num_warps=cfg["num_warps"],
                 num_stages=cfg["num_stages"])
    return y.sum(dim=0).to(torch.bfloat16)


def mid_row(mid_cfg, m):
    """The (split_k, BM, BN, warps) production's wo_a_mid selects at m rows (its own `next(...)` rule)."""
    return tuple(next(c for lim, c in mid_cfg if m <= lim))


def _bytewise_equal(a, b):
    """A byte comparison (torch.equal treats +0.0 and -0.0 as equal; the contract here is bitwise)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def _agree(flags, device):
    """TP-wide MIN of rank-local verdicts (adapter/mxfp8_b12x.py's pattern, one all-reduce for the whole list): every
    rank serves the same launch for a key, or none does. Rank-local only when no TP group exists (unit tests)."""
    flags = [bool(f) for f in flags]
    try:
        from sglang.srt.distributed import get_tp_group
        group = get_tp_group()
        if group is None or group.world_size <= 1:
            return flags
        t = torch.tensor([1 if f else 0 for f in flags], dtype=torch.int32, device=device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=group.device_group)
        return [bool(v) for v in t.tolist()]
    except Exception as exc:  # noqa: BLE001 - no group: the verdict stays rank-local and says so
        print(f"[wo_a_cfg_table] no TP agreement ({exc!r}): rank-local verdict", flush=True)
        return flags


def _drift(kind, m, mid_cfg):
    """None, or why production's own config at m is not the one the entry was measured against."""
    if kind == "mid":
        try:
            got = mid_row(mid_cfg, m)
        except StopIteration:
            return f"_MID_CFG has no row for M={m}"
        if got != PROD_MID[m]:
            return f"_MID_CFG selects {got} at M={m}, the sweep measured against {PROD_MID[m]}"
    return None


def _first_call(key, m, stock_fn, table_fn, device):
    """Eager: run both, compare the bytes, agree over TP; returns the output to use and records the key's verdict."""
    stock = stock_fn()
    cand = None
    try:
        cand = table_fn()
        same = _bytewise_equal(stock, cand)
    except Exception as exc:  # noqa: BLE001 - a launch failure keeps production (and still agrees over TP)
        print(f"[wo_a_cfg_table] first eager call M={m} raised {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        same = False
    agreed = _agree([same], device)[0]
    _CHECKED[key] = agreed
    _STATS["checked"] += 1
    _STATS["same" if same else "diff"] += 1
    if same and not agreed:
        _STATS["disagree"] += 1
    kind, _, sx, ev = key[0], key[1], key[2], key[3]
    print(f"[wo_a_cfg_table] first eager call {kind} M={m} SX={sx} ev={ev or 'none'} {TABLE[m][1]}: "
          + ("bit-identical to production " + PROD_NAME[m] + ", serves this key (TP-agreed)" if agreed else
             ("bytes differ from production " + PROD_NAME[m] + "; production kept" if not same else
              "bit-identical here but another rank differs; production kept on every rank")), flush=True)
    return cand if agreed else stock


def _gate(kind, m, qx, mid_cfg=None):
    """None when the table does not take this call; else the entry's launch parameters."""
    ent = TABLE.get(m)
    if ent is None or ent[0] != kind or qx or not _on():
        return None
    if _BOOT.get(m) is not True:
        _STATS["no_boot"] += 1
        return None
    if kind == "mid" and _drift(kind, m, mid_cfg) is not None:
        _STATS["drift"] += 1
        return None
    return ent[2]


def partial(x, twin, m, qx, ev, stock):
    """wo_a_w8._partial_w8's dispatch: the table's partials [8, m, 2, 1024] fp32, or None (production's launch).
    stock(): production's own launch on the same arguments."""
    cfg = _gate("partial", m, qx)
    if cfg is None:
        return None
    key = ("partial", m, int(x.stride(0)), ev, str(x.dtype), int(x.device.index or 0))
    verdict = _CHECKED.get(key)
    if verdict is None and _capturing():
        _STATS["skipped_capture"] += 1
        return None                     # never an unchecked launch inside a graph; not recorded
    if verdict is False:
        return None
    if verdict is True:
        return launch_partial(x, twin, m, ev, cfg)
    return _first_call(key, m, stock, lambda: launch_partial(x, twin, m, ev, cfg), x.device)


def mid(o, twin, qx, ev, kernel, mid_cfg, stock):
    """wo_a_w8.wo_a_mid's dispatch: the table's bf16 [m, 2, 1024], or None (production's launch). stock(): production's
    own wo_a_mid on the same arguments; kernel: wo_a_w8._wo_a_mid_kernel; mid_cfg: wo_a_w8._MID_CFG."""
    m = int(o.shape[0])
    cfg = _gate("mid", m, qx, mid_cfg)
    if cfg is None:
        return None
    key = ("mid", m, int(o.stride(0)), ev, str(o.dtype), int(o.device.index or 0))
    verdict = _CHECKED.get(key)
    if verdict is None and _capturing():
        _STATS["skipped_capture"] += 1
        return None
    if verdict is False:
        return None
    if verdict is True:
        return launch_mid(kernel, o, twin, m, ev, cfg)
    return _first_call(key, m, stock, lambda: launch_mid(kernel, o, twin, m, ev, cfg), o.device)


def _negative_twin(twin):
    """The twin with ONE 32 x 32 block exponent flipped by one (XOR 1: +-1, never wraps): 1024 weights x2 or /2."""
    w8, s = twin
    s2 = s.clone()
    s2.view(-1)[0] = s2.view(-1)[0] ^ 1
    return w8, s2


def boot_check(W, ev=None):
    """The load self-check (once per process, from wo_a_w8's load hook, every rank). W: the wo_a_w8 module (its twins,
    _MID_CFG, kernels and production launches). Returns {M: verdict}."""
    if _STATE["boot_ran"]:
        return dict(_BOOT)
    _STATE["boot_ran"] = True
    ev = W._ev() if ev is None else ev
    ms = sorted(TABLE)
    twin = next(iter(W._TWINS.values()), None)
    device = twin[0].device if twin is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    local, notes = [], []
    for m in ms:
        kind, name, cfg = TABLE[m]
        ok, why = False, ""
        try:
            if twin is None:
                why = "no twinned wo_a layer"
            elif kind == "mid" and _drift(kind, m, W._MID_CFG) is not None:
                why = "production drifted: " + _drift(kind, m, W._MID_CFG)
            else:
                g = torch.Generator(device=device).manual_seed(SEED + m)
                xs = [torch.randn((m, 2, 4096), generator=g, device=device).to(torch.bfloat16) for _ in range(2)]
                if kind == "partial":
                    def stock_fn(x, tw):
                        return W._partial_w8_stock(x, tw, m, qx=False)

                    def table_fn(x, tw):
                        return launch_partial(x, tw, m, ev, cfg)
                else:
                    def stock_fn(x, tw):
                        return W._wo_a_mid_stock(x, tw, qx=False)

                    def table_fn(x, tw):
                        return launch_mid(W._wo_a_mid_kernel, x, tw, m, ev, cfg)
                same = [_bytewise_equal(stock_fn(x, twin), table_fn(x, twin)) for x in xs]
                neg = _bytewise_equal(stock_fn(xs[0], twin), table_fn(xs[0], _negative_twin(twin)))
                ok = all(same) and not neg
                why = (f"{sum(same)}/{len(same)} inputs bit-identical, negative control "
                       + ("compared EQUAL (the compare proves nothing)" if neg else "differs"))
        except Exception as exc:  # noqa: BLE001 - the entry stays off; the all-reduce below still runs
            ok, why = False, f"raised {type(exc).__name__}: {str(exc)[:160]}"
        local.append(ok)
        notes.append(why)
    agreed = _agree(local, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()          # the check's few MB go back before SGLang sizes its pools
    for m, ok, ag, why in zip(ms, local, agreed, notes):
        _BOOT[m] = ag
        print(f"[wo_a_cfg_table] load self-check M={m} {TABLE[m][1]} vs production {PROD_NAME[m]} "
              f"(ev={ev or 'none'}): {why}; rank-local {ok} -> {'ON' if ag else 'OFF'}"
              + ("" if ok == ag else " (another rank differs)"), flush=True)
    return dict(_BOOT)


def stats():
    return dict(_STATS, keys=len(_CHECKED), on=sum(1 for v in _CHECKED.values() if v), boot=dict(_BOOT))


def armed_line():
    ents = ", ".join(f"M={m} {TABLE[m][1]}" for m in sorted(TABLE))
    return (f"[wo_a_cfg_table] armed ({ENV}, read per call): {ents}; every other M on production's launch; bitwise "
            "self-check at load and at the first eager call per key, TP-agreed, fail closed")
