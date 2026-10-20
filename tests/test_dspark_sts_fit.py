"""CPU checks for scripts/dspark_sts_fit.py (no GPU, no engine, no sglang).

  python3 tests/test_dspark_sts_fit.py          # or: PYTHONPATH=scripts python3 tests/...

1. Recovery: a synthetic head whose per-position survival is sigmoid(z / T_true) is written out
   in the `DSV41_VERIFY_CAP_LOG` format and the fitter must recover T_true within 8 % on every
   position -- including the last one, which is only observed on the ~18 % of rows whose
   prefix survived that far;
2. the same temperatures recovered from `SGLANG_DSPARK_STS_COLLECT_PATH` shards;
3. cut steps (live < gamma + 1) are dropped, not fitted, and the dropped count is exact;
4. a degenerate position (prefix always survived, or never) keeps its temperature at 1.0
   instead of returning a bracket edge;
5. `live_lengths` is the cap's own rule (cumprod >= threshold, >= kmin) bit for bit;
6. the JSON lands in the `DSparkStsCalibration` schema (temperatures > 0, one per position)
   and calibration moves ECE down, not up.
"""
import json
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import dspark_sts_fit as fit  # noqa: E402

GAMMA = 5
N = 60_000
T_TRUE = [0.6, 0.8, 1.0, 1.4, 2.2]
MEANS = [0.3, 0.2, 0.1, 0.0, -0.1]        # later drafts, less confident, like a real head
SIGMAS = 1.2
SEED = 7


def synthetic(generator, n=N):
    """Raw head probabilities, the prefix they imply, and the counts a verify step would log.

    The head's raw logit z is a miscalibrated estimate of the true log-odds: the true
    conditional survival of position i is sigmoid(z_i / T_TRUE[i]), which is what the labels
    are drawn from. The cap is handed sigmoid(z_i), and the fitter has to get T back from that.
    """
    z = torch.randn(n, GAMMA, generator=generator) * SIGMAS + torch.tensor(MEANS)
    raw = torch.sigmoid(z)
    prefix = torch.zeros(n, GAMMA)
    counts = torch.full((n,), float(GAMMA))
    alive = torch.ones(n, dtype=torch.bool)
    for position in range(GAMMA):
        truth = torch.sigmoid(z[:, position] / T_TRUE[position])
        draw = torch.rand(n, generator=generator) < truth
        prefix[:, position] = alive & draw
        alive = alive & draw
    # the accepted count IS the surviving prefix length; deriving it here (rather than
    # tracking it alongside alive) is what keeps the log self-consistent
    return raw, prefix, prefix.sum(dim=1)


def write_vcap_log(path: Path, raw, counts, live):
    rows = torch.cat([raw, live.view(-1, 1), counts.view(-1, 1)], dim=1).to(torch.float32)
    path.write_bytes(rows.numpy().tobytes())
    return rows.shape[0]


def main():
    generator = torch.Generator().manual_seed(SEED)
    raw, prefix, counts = synthetic(generator)
    work = Path(__import__("tempfile").mkdtemp(prefix="sts-fit-"))

    # 1. recovery from the vcap log, including the tail position nobody sees very often
    log = work / "vcap-log.bin"
    write_vcap_log(log, raw, counts, torch.full((N,), float(GAMMA + 1)))
    out = work / "sts.json"
    argv = ["--vcap-log", str(log), "--gamma", str(GAMMA), "--threshold", "0.1",
            "--out", str(out), "--min-samples", "512"]
    fit.main(argv)
    got = json.loads(out.read_text())["temperatures"]
    print(f"recovered {got} vs true {T_TRUE}")
    for position, (estimate, truth) in enumerate(zip(got, T_TRUE)):
        assert abs(estimate - truth) / truth < 0.08, f"position {position}: {estimate} vs {truth}"

    # 2. the same temperatures from upstream's shard format (raw logits + prefix_mask)
    shards = work / "shards"
    shards.mkdir()
    torch.save({"logits": torch.log(raw / (1 - raw)),
                "prefix_mask": prefix}, shards / "0.pt")
    out2 = work / "sts-shards.json"
    fit.main(["--sts-shards", str(shards), "--gamma", str(GAMMA), "--out", str(out2)])
    got2 = json.loads(out2.read_text())["temperatures"]
    for position, (estimate, truth) in enumerate(zip(got2, T_TRUE)):
        assert abs(estimate - truth) / truth < 0.08, f"shards position {position}: {estimate} vs {truth}"

    # 3. cut steps are dropped and never fitted
    n_cut = 500
    rows_full = torch.cat([raw, torch.full((N, 1), float(GAMMA + 1)),
                           counts.view(-1, 1)], dim=1).to(torch.float32)
    rows_cut = torch.cat([raw[:n_cut], torch.full((n_cut, 1), 3.0),
                          counts[:n_cut].view(-1, 1)], dim=1).to(torch.float32)
    log3 = work / "vcap-log-cut.bin"
    log3.write_bytes(torch.cat([rows_full, rows_cut]).numpy().tobytes())
    conf, kept, dropped = fit.read_vcap_log(str(log3), GAMMA)
    assert conf.shape[0] == N == kept.shape[0], (conf.shape, N)
    assert dropped == n_cut, dropped
    log4 = work / "only-cut.bin"
    log4.write_bytes(rows_cut.numpy().tobytes())
    try:
        fit.read_vcap_log(str(log4), GAMMA)
        raise SystemExit("a log of nothing but cut steps should raise, not fit on them")
    except SystemExit as err:
        assert "not a multiple" not in str(err), err

    # 4. a degenerate position cannot be fitted and keeps 1.0 (prefix always survived at 0)
    easy = work / "degenerate.bin"
    always = raw[:4096].clone()
    counts_always = torch.full((4096,), float(GAMMA))
    prefix_always = torch.ones(4096, GAMMA)
    write_vcap_log(easy, always, counts_always, torch.full((4096,), float(GAMMA + 1)))
    z = fit.logit(always)
    temps, _, seen = fit.fit_positions(z, prefix_always)
    assert temps[0] == 1.0, temps                     # every row accepted: unidentifiable
    assert seen[0] == 4096
    never = prefix_always.clone()
    never[:, GAMMA - 1] = 0.0                        # last draft never survives
    temps2, _, _ = fit.fit_positions(z, never)
    assert temps2[GAMMA - 1] == 1.0, temps2

    # 5. live_lengths is the cap's own rule
    conf_matrix = torch.rand(64, GAMMA, generator=generator) * 0.9 + 0.05
    for threshold, kmin in ((0.1, 1), (0.35, 1), (0.02, 2)):
        cum = torch.cumprod(conf_matrix.clamp(0, 1), dim=1)
        expected = (cum >= threshold).to(torch.int64).cumprod(dim=1).sum(dim=1).clamp(min=kmin) + 1
        assert torch.equal(fit.live_lengths(conf_matrix, threshold, kmin), expected), (threshold, kmin)

    # 6. the JSON is a DSparkStsCalibration and calibration moved ECE down
    calibration = json.loads(out.read_text())
    assert len(calibration["temperatures"]) == GAMMA
    assert all(t > 0 for t in calibration["temperatures"])
    assert calibration["num_samples"] == N
    assert len(calibration["ece_before"]) == len(calibration["ece_after"]) == GAMMA
    before_mean = sum(calibration["ece_before"]) / GAMMA
    after_mean = sum(calibration["ece_after"]) / GAMMA
    print(f"ECE {before_mean:.4f} -> {after_mean:.4f}")
    assert after_mean <= before_mean + 1e-4, "calibration made ECE worse"

    # identity: T = 1 must be a no-op on already-raw confidences
    z_probe = torch.randn(11, GAMMA, generator=generator)
    assert torch.allclose(fit.apply_temperature(z_probe, torch.ones(GAMMA)), torch.sigmoid(z_probe))
    print("test_dspark_sts_fit: ok")


if __name__ == "__main__":
    main()