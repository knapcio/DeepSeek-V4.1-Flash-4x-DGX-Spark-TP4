"""Fit per-position temperatures for the draft confidence head (STS), offline.

The verify-length rule in `adapter/verify_cap.py` is already close to optimal: a fit on
6,074 uncut steps found per-position thresholds worth only +0.4 % over the single 0.1, while
a *perfect* cut was worth +15.5 % ([docs/adapters.md](../docs/adapters.md)). What is left is
not the rule, it is the number the rule reads: the head's per-position survival is used as a
probability, and nothing ever checked that it is one.

Sequential temperature scaling fixes exactly that. For position i,

    p_i = sigmoid(logit(c_i) / T_i)

with one temperature per position, fitted by minimising the negative log-likelihood of "draft i
was accepted". It changes no decision that has been made and no token that is emitted: it
rescales the confidence the scheduler reads, so the same threshold 0.1 now cuts at the right
place. Upstream calls the same thing STS and applies it on the ragged-verify planner's path
(`srt/speculative/dspark_components/dspark_sts.py`, `--speculative-dspark-confidence-sts-path`);
this script produces the JSON for either consumer.

Two input formats, both already produced by the engine:

  --vcap-log PATH   the binary `DSV41_VERIFY_CAP_LOG` writes on rank 0: float32 rows of
                    [c_0..c_4, live, correct_len]. This is the tensor the cap actually consumes,
                    so it is the right thing to calibrate against. Rows are dropped unless the
                    step verified every draft (live == gamma + 1), because a cut step says
                    nothing about the positions it did not check.
  --sts-shards DIR  the `.pt` shards `SGLANG_DSPARK_STS_COLLECT_PATH` writes, `{logits,
                    prefix_mask}` of raw pre-sigmoid logits. Same fit, upstream's own format.

The fit for position i runs on the rows whose prefix survived to i-1, against "draft i was
accepted": confidence[i] is a *conditional* survival (the cap multiplies it), so fitting it
against the marginal acceptance rate would fold the earlier positions' failure into it and cut
every step too short.
Writes the `DSparkStsCalibration` JSON (`{"temperatures": [...]}` plus provenance) and prints,
before and after, the live-length distribution at a threshold -- the quantity that moves
throughput.

  python3 scripts/dspark_sts_fit.py --vcap-log /state/vcap-log.bin --gamma 5 \
      --threshold 0.1 --out /state/dspark_sts.json

Nothing here needs a GPU, the engine, or sglang: it reads a file and fits five scalars.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import torch

EPS = 1e-6
# Temperature search bracket, in log space: 0.05 .. 20 is far wider than any head needs.
LOG_T_MIN, LOG_T_MAX = math.log(0.05), math.log(20.0)


def logit(prob: torch.Tensor) -> torch.Tensor:
    """Log-odds of a probability, clamped away from 0 and 1 (a saturated head is a data point)."""
    p = prob.clamp(EPS, 1.0 - EPS)
    return torch.log(p) - torch.log1p(-p)


def apply_temperature(z: torch.Tensor, temperature) -> torch.Tensor:
    """sigmoid(z / T) with T broadcast per position. The one formula both consumers use."""
    return torch.sigmoid(z / torch.as_tensor(temperature, dtype=z.dtype))


def prefix_labels(counts: torch.Tensor, gamma: int) -> torch.Tensor:
    """[N, gamma] 1.0 where draft i survived: i < accepted count."""
    positions = torch.arange(gamma).view(1, -1)
    return (positions < counts.view(-1, 1)).to(torch.float32)


def read_vcap_log(path: str, gamma: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """[N, gamma] confidences and accepted counts from the `DSV41_VERIFY_CAP_LOG` stream.

    Only steps that verified every draft carry a label for every position; the rest are dropped
    and counted, because fitting on cut steps biases every temperature toward the cut.
    """
    raw = torch.from_numpy(_read_float32(path)).clone()
    width = gamma + 2
    if raw.numel() % width:
        raise SystemExit(
            f"{path}: {raw.numel()} floats is not a multiple of {width} "
            f"(gamma {gamma} confidences + live + correct_len)"
        )
    rows = raw.view(-1, width)
    conf = rows[:, :gamma].clone()
    live = rows[:, gamma]
    counts = rows[:, gamma + 1]
    full = live >= gamma + 1
    dropped = int((~full).sum())
    return conf[full], counts[full].round().clamp(0, gamma), dropped


def _read_float32(path: str) -> "torch.Tensor":
    import numpy as np

    with open(path, "rb") as handle:
        return np.frombuffer(handle.read(), dtype=np.float32)


def read_sts_shards(pattern: str, gamma: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Upstream's `{logits, prefix_mask}` shards -> confidences and counts.

    The shards hold pre-sigmoid logits, so a prefix_mask is the label directly: the accepted
    count is the length of the surviving prefix.
    """
    paths = sorted(glob.glob(os.path.join(pattern, "*.pt"))) if os.path.isdir(pattern) else sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no .pt shards under {pattern}")
    logits, masks = [], []
    for path in paths:
        shard = torch.load(path, map_location="cpu")
        logits.append(shard["logits"].to(torch.float32))
        masks.append(shard["prefix_mask"].to(torch.float32))
    z = torch.cat(logits, dim=0)[:, :gamma]
    mask = torch.cat(masks, dim=0)[:, :gamma]
    counts = mask.sum(dim=1)
    return torch.sigmoid(z), counts, 0


def fit_temperature(z: torch.Tensor, label: torch.Tensor, lo: float, hi: float) -> tuple[float, float]:
    """1-D NLL minimisation over log T by golden-section search. Convex in log T for this loss."""
    def nll(log_t: float) -> float:
        p = torch.sigmoid(z / math.exp(log_t))
        return float(-(label * torch.log(p.clamp(EPS, 1.0)) + (1 - label) * torch.log((1 - p).clamp(EPS, 1.0))).mean())

    a, b = lo, hi
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(200):
        if b - a < 1e-6:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = nll(d)
    log_t = (a + b) / 2.0
    return math.exp(log_t), nll(log_t)


def fit_positions(z: torch.Tensor, prefix: torch.Tensor) -> tuple[list[float], list[float], list[int]]:
    """One temperature per position, fitted on the rows whose prefix survived to it.

    confidence[i] is the survival of draft i given drafts < i accepted, and the cap multiplies
    it, so position i is fitted only where the prefix reached i-1 (position 0 on every row).
    A position whose subset is degenerate -- every row accepted, or none -- cannot identify a
    temperature: the NLL is then monotone in T and the search would return a bracket edge, so
    the temperature stays at 1.0 and the fit says so. The counts say how much data stands
    behind each temperature; a position fitted on a few hundred rows is not to be trusted.
    """
    temperatures, nlls, counts = [], [], []
    for position in range(z.shape[1]):
        subset = torch.ones(z.shape[0], dtype=torch.bool) if position == 0 else prefix[:, position - 1] > 0.5
        label = prefix[:, position][subset]
        counts.append(int(label.numel()))
        if label.numel() < 32 or label.min() == label.max():
            print(f"position {position}: {int(label.sum())}/{label.numel()} accepted on the "
                  "surviving prefix -- degenerate, temperature left at 1.0", file=sys.stderr)
            temperatures.append(1.0)
            nlls.append(float("nan"))
            continue
        temperature, value = fit_temperature(z[subset, position], label, LOG_T_MIN, LOG_T_MAX)
        temperatures.append(round(temperature, 6))
        nlls.append(value)
    return temperatures, nlls, counts


def per_position_ece(prob: torch.Tensor, prefix: torch.Tensor, bins: int = 10) -> list[float]:
    """ECE of each position on the same conditional subset the fit used (before vs after)."""
    out = []
    for position in range(prob.shape[1]):
        subset = torch.ones(prob.shape[0], dtype=torch.bool) if position == 0 else prefix[:, position - 1] > 0.5
        p, y = prob[subset, position], prefix[subset, position].float()
        edges = torch.linspace(0.0, 1.0, bins + 1)
        err = 0.0
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (p > low) & (p <= high) if low > 0 else (p >= low) & (p <= high)
            if mask.any():
                err += float(mask.float().mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
        out.append(err)
    return out


def live_lengths(prob: torch.Tensor, threshold: float, kmin: int = 1) -> torch.Tensor:
    """The cap's own rule: leading positions whose running survival product stays >= threshold."""
    cum = torch.cumprod(prob.clamp(0, 1), dim=1)
    k = (cum >= threshold).to(torch.int64).cumprod(dim=1).sum(dim=1).clamp(min=kmin)
    return k + 1


def distribution(values: torch.Tensor, width: int) -> list[float]:
    counts = torch.bincount(values.clamp(0, width).to(torch.int64), minlength=width + 1).to(torch.float64)
    return [float(v) for v in (counts / counts.sum()).tolist()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--vcap-log", help="DSV41_VERIFY_CAP_LOG binary (float32 rows of conf..., live, correct_len)")
    source.add_argument("--sts-shards", help="directory or glob of SGLANG_DSPARK_STS_COLLECT_PATH .pt shards")
    parser.add_argument("--gamma", type=int, default=5, help="drafts per verify step (DSPARK_BLOCK_SIZE)")
    parser.add_argument("--threshold", type=float, default=0.1, help="DSV41_VERIFY_CAP's conf:T value, for the report")
    parser.add_argument("--kmin", type=int, default=1, help="DSV41_VERIFY_CAP_MIN")
    parser.add_argument("--out", required=True, help="calibration JSON to write (DSparkStsCalibration schema)")
    parser.add_argument("--dataset", default="", help="provenance string recorded in the JSON")
    parser.add_argument("--min-samples", type=int, default=512, help="refuse to fit on less than this")
    args = parser.parse_args(argv)

    gamma = args.gamma
    if args.vcap_log:
        conf, counts, dropped = read_vcap_log(args.vcap_log, gamma)
        provenance = f"vcap-log:{os.path.basename(args.vcap_log)}"
    else:
        conf, counts, dropped = read_sts_shards(args.sts_shards, gamma)
        provenance = f"sts-shards:{args.sts_shards}"
    if conf.shape[0] < args.min_samples:
        raise SystemExit(
            f"{conf.shape[0]} usable steps (< --min-samples {args.min_samples}); "
            "collect on an uncut run (DSV41_VERIFY_CAP=5) before fitting"
        )
    if dropped:
        print(f"dropped {dropped} cut steps: their unverified positions carry no label", file=sys.stderr)

    z = logit(conf)
    prefix = prefix_labels(counts, gamma)
    temperatures, nll, fit_counts = fit_positions(z, prefix)
    before = torch.sigmoid(z)
    after = apply_temperature(z, torch.tensor(temperatures))
    ece_before = [round(v, 6) for v in per_position_ece(before, prefix)]
    ece_after = [round(v, 6) for v in per_position_ece(after, prefix)]

    calibration = {
        "temperatures": temperatures,
        "dataset": args.dataset or provenance,
        "num_samples": int(conf.shape[0]),
        "ece_before": ece_before,
        "ece_after": ece_after,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(calibration, handle, indent=2)
        handle.write("\n")

    print(f"steps           {conf.shape[0]}   (fitted per position: {fit_counts})")
    print(f"accepted/step   {float(counts.mean()):.3f}")
    print(f"temperatures    {temperatures}")
    print(f"NLL/position    {[round(v, 5) for v in nll]}")
    print(f"ECE   before    {[round(v, 4) for v in calibration['ece_before']]}")
    print(f"ECE   after     {[round(v, 4) for v in calibration['ece_after']]}")
    print(f"live  before    {[round(v, 4) for v in distribution(live_lengths(before, args.threshold, args.kmin), gamma)]}")
    print(f"live  after     {[round(v, 4) for v in distribution(live_lengths(after, args.threshold, args.kmin), gamma)]}")
    print(f"wrote {args.out}")
    print("Point DSV41_VERIFY_CAP_STS at it (or --speculative-dspark-confidence-sts-path for the ragged planner).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())